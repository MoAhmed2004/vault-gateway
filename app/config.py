"""Gateway configuration (§2.7, §3.9, PS-3, TM-S3, TM-R1, TM-R2, IP-8).

`Settings` reads process environment and an optional `.env` file. Nothing is
loaded at import time: callers build settings with `load_settings()` or, in
tests, with `Settings(_env_file=None, KEY=value, ...)`. Mode definitions are
imported from `contracts.modes` so importing `app.config` never loads `Faker`
or `phonenumbers`.

v3.3.3 flags (Step 5C): `ENABLE_RATE_LIMITING` (TM-R7) and
`ENABLE_OUTPUT_SANITIZER` (IP-10) default to None and are resolved by mode
during validation (rate limiting on in production and demo, the sanitizer on
in demo). Both are refused when true in `eval`, `shadow_ingress_only` and
`shadow_full_e2e`, so neither can touch a measured run, and the sanitizer is
also refused with `EGRESS_VARIANT=X0` [Project Decision, extends IP-10].
Step 5C.1 L-5 (TM-R7): with `UPSTREAM_MODE=stub` (the E4 and local load-rig
upstream) an unset `ENABLE_RATE_LIMITING` resolves to false in every mode,
and an explicit true is refused.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from contracts.modes import DETERMINISTIC_MODES, GatewayMode
from contracts.paths import synced_folder_component

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DB_PATH = REPO_ROOT / "data" / "logs" / "vault.db"
DEFAULT_CACHE_DB_PATH = REPO_ROOT / "data" / "cache" / "vault_cache.db"
DEFAULT_CHROMA_DB_PATH = REPO_ROOT / "data" / "chroma_db"
DEFAULT_MINILM_MODEL_DIR = REPO_ROOT / "models" / "minilm"
DEFAULT_L3_MODEL_DIR = REPO_ROOT / "models" / "deberta"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
SHADOW_MODES = frozenset({"shadow_ingress_only", "shadow_full_e2e"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")

DEFAULT_HMAC_SECRET = "vault_dev_secret_change_me"
MIN_HMAC_SECRET_CHARS = 16
DEFAULT_PROBE_CANDIDATES: tuple[str, ...] = (
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
)

GatewayModeName = GatewayMode
TimeoutPolicyName = Literal["P-open", "P-degrade", "P-closed"]
UpstreamModeName = Literal["groq", "ollama", "stub"]
EgressVariantName = Literal["X0", "X1", "X2", "X3"]
ReasoningEffortName = Literal["none", "low", "medium", "high"]
RateLimitKeyName = Literal["ip", "session"]
OutputSanitizerModeName = Literal["block", "neutralize"]

# Mode defaults for the two v3.3.3 flags when they are left unset (TM-R7, IP-10).
RATE_LIMIT_DEFAULT_ON: frozenset[str] = frozenset({"production", "demo"})
OUTPUT_SANITIZER_DEFAULT_ON: frozenset[str] = frozenset({"demo"})

PositiveInt = Annotated[int, Field(gt=0)]

_PLACEHOLDER_KEYS = frozenset({"changeme", "change_me", "placeholder", "todo", "xxx", "none", "null", "your_groq_api_key_here"})
_PLACEHOLDER_PATTERN = re.compile(r"^(your[_-].*|.*[_-]here|<.*>)$", re.IGNORECASE)
_UNSET: Any = object()


def is_placeholder_api_key(value: str) -> bool:
    """True for obvious template values such as `your_groq_api_key_here` (blank is not a placeholder)."""
    text = value.strip()
    if not text:
        return False
    return text.lower() in _PLACEHOLDER_KEYS or bool(_PLACEHOLDER_PATTERN.match(text))


def _split_candidates(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


_COMMA_LIST_FIELDS = frozenset({"PROBE_CANDIDATES", "OUTPUT_IMAGE_ALLOWLIST"})
_HOSTNAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*")
_HEADER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}")
# Headers the gateway already gives a meaning to; a session key must be a separate, opaque header (TM-R7).
_RESERVED_SESSION_HEADERS = frozenset({"host", "content-length", "content-type", "transfer-encoding", "x-trial-id"})


class _CommaListMixin:
    """Parse PROBE_CANDIDATES and OUTPUT_IMAGE_ALLOWLIST from a comma-separated string instead of JSON."""

    def prepare_field_value(self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool) -> Any:
        if field_name in _COMMA_LIST_FIELDS and isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                return super().prepare_field_value(field_name, field, value, value_is_complex)  # type: ignore[misc]
            return _split_candidates(text)
        return super().prepare_field_value(field_name, field, value, value_is_complex)  # type: ignore[misc]


class _EnvSource(_CommaListMixin, EnvSettingsSource):
    pass


class _DotEnvSource(_CommaListMixin, DotEnvSettingsSource):
    pass


class Settings(BaseSettings):
    """Validated gateway settings with the spec defaults."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        allow_inf_nan=False,
        frozen=True,
    )

    # API and target model (§2.7, §3.9)
    GROQ_API_KEY: str = Field(default="", repr=False)
    TARGET_MODEL: str = "openai/gpt-oss-20b"
    PROBE_CANDIDATES: tuple[str, ...] = DEFAULT_PROBE_CANDIDATES
    ALLOW_MODEL_FAILOVER: bool = False

    # Upstream client (§2.7: one client module; target parameters are part of the IP-5 response key)
    GROQ_BASE_URL: str = DEFAULT_GROQ_BASE_URL
    UPSTREAM_TIMEOUT_S: float = Field(default=20.0, gt=0.0, le=120.0, allow_inf_nan=False)
    TARGET_MAX_TOKENS: PositiveInt = 256
    TARGET_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0, allow_inf_nan=False)
    TARGET_SEED: int = 0
    TARGET_REASONING_EFFORT: ReasoningEffortName | None = None

    # Per-client rate limit (TM-R7, TM-R1, LLM10). None resolves by mode: true in production and demo, false in
    # eval and both shadow tiers; true is refused in those three measured modes (it would be scored as a block).
    ENABLE_RATE_LIMITING: bool | None = None
    RATE_LIMIT_RPM: int = Field(default=30, ge=1)
    RATE_LIMIT_KEY: RateLimitKeyName = "ip"
    RATE_LIMIT_SESSION_HEADER: str = "X-Session-ID"
    # Bound on the in-memory bucket store [Project Decision, extends TM-R7]: idle full buckets are evicted first.
    RATE_LIMIT_MAX_CLIENTS: int = Field(default=10_000, ge=1)

    # LLM05 output sanitizer (IP-10, IP-6), not an egress variant. None resolves by mode: true in demo, false in
    # production, eval and both shadow tiers; true is refused in the three measured modes and with EGRESS_VARIANT=X0.
    ENABLE_OUTPUT_SANITIZER: bool | None = None
    OUTPUT_SANITIZER_MODE: OutputSanitizerModeName = "block"
    # Image hosts an image URL may point to (exact lowercase hostnames); empty means every external image matches.
    OUTPUT_IMAGE_ALLOWLIST: tuple[str, ...] = ()

    # Egress (IP-6, §2.1): X0 none, X1 Presidio, X2 + exact canary and planted PII, X3 + decoded-text and leak check
    EGRESS_VARIANT: EgressVariantName = "X2"
    # X3 leak check thresholds: the O1 oracle defaults (§3.1) [Project Decision, extends IP-6]
    X3_SIMILARITY_THRESHOLD: float = Field(default=0.80, ge=-1.0, le=1.0, allow_inf_nan=False)
    X3_NGRAM_OVERLAP_THRESHOLD: float = Field(default=0.30, gt=0.0, le=1.0, allow_inf_nan=False)
    X3_NGRAM_SIZE: PositiveInt = 3

    # Ingress layers L2 (ChromaDB + MiniLM) and L3 (DeBERTa ONNX); artifacts from scripts/fetch_models.py
    L2_ENABLED: bool = True
    L3_ENABLED: bool = True
    MINILM_MODEL_DIR: Path = DEFAULT_MINILM_MODEL_DIR
    L3_MODEL_DIR: Path = DEFAULT_L3_MODEL_DIR
    L3_MALICIOUS_LABEL: str = "INJECTION"
    CHROMA_DB_PATH: Path = DEFAULT_CHROMA_DB_PATH
    # Thresholds are defaults until tuned on the validation split (§2.4, §4)
    L2_THRESHOLD: float = Field(default=0.80, ge=-1.0, le=1.0, allow_inf_nan=False)
    L3_THRESHOLD: float = Field(default=0.50, gt=0.0, le=1.0, allow_inf_nan=False)

    # Bounded layer pool (IP-7): 0 workers means one per physical core
    LAYER_POOL_WORKERS: int = Field(default=0, ge=0)
    LAYER_POOL_QUEUE_LIMIT: int = Field(default=32, ge=0)

    # Caches (IP-5, IP-8: local disk only). E4 runs with CACHE_ENABLED=false (no cache reads or writes,
    # TM-R5); shadow_full_e2e needs both flags on (IP-4 oracle join through the egress cache).
    CACHE_DB_PATH: Path = DEFAULT_CACHE_DB_PATH
    CACHE_ENABLED: bool = True
    CACHE_WRITES_ENABLED: bool = True

    # Objective whose canary and planted PII the gateway-built system prompt carries (PS-2, PS-3)
    OBJECTIVE_ID: str = "gateway-session"

    # Modes and policies
    GATEWAY_MODE: GatewayModeName = "production"
    TIMEOUT_POLICY: TimeoutPolicyName = "P-degrade"
    UPSTREAM_MODE: UpstreamModeName = "groq"
    STUB_DELAY_MS: int = Field(default=50, ge=0)

    # Secrets and storage
    HMAC_SECRET: str = Field(default=DEFAULT_HMAC_SECRET, repr=False)
    LOG_DB_PATH: Path = DEFAULT_LOG_DB_PATH

    # Caps (TM-R1, CA-1, IP-2)
    MAX_REQUEST_BODY_BYTES: PositiveInt = 131072
    MAX_PROMPT_CHARS: PositiveInt = 4000
    MAX_WINDOWS_PER_VIEW: PositiveInt = 16
    MAX_CONTEXT_DOCS: PositiveInt = 4
    MAX_CONTEXT_DOC_CHARS: PositiveInt = 1500

    # Time budgets (TM-R2) and P-degrade tightening (TM-R3)
    L1_BUDGET_MS: PositiveInt = 5
    L2_BUDGET_MS: PositiveInt = 50
    L3_BUDGET_MS: PositiveInt = 150
    L4_BUDGET_MS: PositiveInt = 150
    TOTAL_BUDGET_MS: PositiveInt = 400
    P_DEGRADE_THRESHOLD_DELTA: float = Field(default=0.05, ge=0.0, allow_inf_nan=False)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            _EnvSource(settings_cls),
            _DotEnvSource(
                settings_cls,
                env_file=getattr(dotenv_settings, "env_file", None),
                env_file_encoding=getattr(dotenv_settings, "env_file_encoding", None),
            ),
            file_secret_settings,
        )

    @field_validator("PROBE_CANDIDATES", mode="before")
    @classmethod
    def _parse_candidates(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _split_candidates(value)
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("PROBE_CANDIDATES")
    @classmethod
    def _candidates_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("PROBE_CANDIDATES must list at least one non-empty model id")
        return value

    @field_validator("ALLOW_MODEL_FAILOVER")
    @classmethod
    def _failover_must_be_off(cls, value: bool) -> bool:
        if value:
            raise ValueError("ALLOW_MODEL_FAILOVER must be false: runtime failover would mix models mid-experiment (§2.7, §3.9)")
        return value

    @field_validator("LOG_DB_PATH", "CACHE_DB_PATH", "CHROMA_DB_PATH", "MINILM_MODEL_DIR", "L3_MODEL_DIR")
    @classmethod
    def _anchor_log_path(cls, value: Path) -> Path:
        path = value.expanduser()
        return path if path.is_absolute() else REPO_ROOT / path

    @field_validator("TARGET_REASONING_EFFORT", mode="before")
    @classmethod
    def _blank_effort_is_none(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("ENABLE_RATE_LIMITING", "ENABLE_OUTPUT_SANITIZER", mode="before")
    @classmethod
    def _blank_flag_is_unset(cls, value: Any) -> Any:
        """A blank value (`ENABLE_RATE_LIMITING=` in `.env`) means "resolve by mode"."""
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("OUTPUT_IMAGE_ALLOWLIST", mode="before")
    @classmethod
    def _parse_allowlist(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _split_candidates(value)
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("OUTPUT_IMAGE_ALLOWLIST")
    @classmethod
    def _allowlist_is_hostnames(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Exact hostnames only, normalized to lowercase without a trailing dot (IP-10): no scheme, port or path."""
        hosts: list[str] = []
        for item in value:
            host = item.strip().lower().rstrip(".")
            if not _HOSTNAME.fullmatch(host):
                raise ValueError(f"OUTPUT_IMAGE_ALLOWLIST entries must be bare hostnames, got {item!r}")
            hosts.append(host)
        return tuple(dict.fromkeys(hosts))

    @field_validator("RATE_LIMIT_SESSION_HEADER")
    @classmethod
    def _session_header_is_a_token(cls, value: str) -> str:
        text = value.strip()
        if not _HEADER_NAME.fullmatch(text) or text.lower() in _RESERVED_SESSION_HEADERS:
            raise ValueError("RATE_LIMIT_SESSION_HEADER must be a dedicated header name (letters, digits, '-')")
        return text

    @field_validator("OBJECTIVE_ID")
    @classmethod
    def _objective_id_is_identifier(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("OBJECTIVE_ID must match the contracts identifier pattern (no '|', PS-3)")
        return value

    @field_validator("GROQ_BASE_URL")
    @classmethod
    def _base_url_is_https(cls, value: str) -> str:
        text = value.strip().rstrip("/")
        if not text.startswith("https://"):
            raise ValueError("GROQ_BASE_URL must be an https:// URL")
        return text

    def _resolve_mode_flags(self) -> None:
        """Fill the unset TM-R7 and IP-10 flags from the mode (the model is frozen, so write `__dict__` once here)."""
        if self.ENABLE_RATE_LIMITING is None:
            # The local fixed-delay stub upstream is the E4 and load-rig target: rate limiting stays off there (TM-R7).
            self.__dict__["ENABLE_RATE_LIMITING"] = (
                self.GATEWAY_MODE in RATE_LIMIT_DEFAULT_ON and self.UPSTREAM_MODE != "stub"
            )
        if self.ENABLE_OUTPUT_SANITIZER is None:
            self.__dict__["ENABLE_OUTPUT_SANITIZER"] = self.GATEWAY_MODE in OUTPUT_SANITIZER_DEFAULT_ON

    @model_validator(mode="after")
    def _startup_guards(self) -> Settings:
        self._resolve_mode_flags()
        if self.GATEWAY_MODE in DETERMINISTIC_MODES:
            if self.ENABLE_RATE_LIMITING:
                raise ValueError(
                    f"{self.GATEWAY_MODE} refuses ENABLE_RATE_LIMITING=true: a rate-limit 503 is indistinguishable "
                    "from a block and would distort E1 to E6, E4 and the throughput benchmarks (TM-R7)"
                )
            if self.ENABLE_OUTPUT_SANITIZER:
                raise ValueError(
                    f"{self.GATEWAY_MODE} refuses ENABLE_OUTPUT_SANITIZER=true: the LLM05 sanitizer is off in every "
                    "measured run so the frozen X0 to X3 variants and the E1 8 x 4 table are unchanged (IP-10, IP-6)"
                )
        if self.ENABLE_RATE_LIMITING and self.UPSTREAM_MODE == "stub":
            raise ValueError(
                "UPSTREAM_MODE=stub refuses ENABLE_RATE_LIMITING=true: the stub upstream is the E4 and local load-rig "
                "target, and a rate-limit 503 would be scored as a block or a timeout (TM-R7)"
            )
        # [Project Decision, extends IP-10] X0 performs no L4 scan, so the sanitizer has no egress scan to follow.
        if self.ENABLE_OUTPUT_SANITIZER and self.EGRESS_VARIANT == "X0":
            raise ValueError(
                "ENABLE_OUTPUT_SANITIZER=true needs EGRESS_VARIANT X1, X2 or X3: X0 performs no L4 scan, and the "
                "sanitizer runs after the egress-variant scan ([Project Decision, extends IP-10]); "
                "set ENABLE_OUTPUT_SANITIZER=false for X0"
            )
        if self.GATEWAY_MODE in DETERMINISTIC_MODES:
            secret = self.HMAC_SECRET.strip()
            if not secret or secret == DEFAULT_HMAC_SECRET or len(secret) < MIN_HMAC_SECRET_CHARS:
                raise ValueError(
                    f"{self.GATEWAY_MODE} mode refuses to start with an empty, default, or short "
                    f"(<{MIN_HMAC_SECRET_CHARS} chars) HMAC_SECRET (PS-3, TM-S3)"
                )
        if self.UPSTREAM_MODE == "groq" and is_placeholder_api_key(self.GROQ_API_KEY):
            raise ValueError("GROQ_API_KEY is a placeholder value; set a real key or leave it blank")
        stores = (("LOG_DB_PATH", self.LOG_DB_PATH), ("CACHE_DB_PATH", self.CACHE_DB_PATH), ("CHROMA_DB_PATH", self.CHROMA_DB_PATH))
        for name, path in stores:
            offender = synced_folder_component(path.resolve())
            if offender is not None:
                raise ValueError(f"{name} must be on local disk, not a synced folder (IP-8): component {offender!r}")
        if self.GATEWAY_MODE in SHADOW_MODES and not (self.L2_ENABLED and self.L3_ENABLED):
            raise ValueError(f"{self.GATEWAY_MODE} runs L1, L2 and L3 on every item (IP-4); enable L2 and L3")
        if self.GATEWAY_MODE == "shadow_full_e2e" and not (self.CACHE_ENABLED and self.CACHE_WRITES_ENABLED):
            raise ValueError(
                "shadow_full_e2e writes X0 to X3 to the IP-5 egress cache for the harness oracle join (IP-4, IP-5); "
                "set CACHE_ENABLED=true and CACHE_WRITES_ENABLED=true"
            )
        layer_sum = self.L1_BUDGET_MS + self.L2_BUDGET_MS + self.L3_BUDGET_MS + self.L4_BUDGET_MS
        if layer_sum > self.TOTAL_BUDGET_MS:
            raise ValueError(
                f"L1 to L4 budgets sum to {layer_sum} ms, above TOTAL_BUDGET_MS={self.TOTAL_BUDGET_MS} (TM-R2)"
            )
        return self


def load_settings(*, env_file: str | Path | None = _UNSET) -> Settings:
    """Build settings from the environment and optional `.env`. No import-time side effects."""
    if env_file is _UNSET:
        return Settings()
    return Settings(_env_file=env_file)
