"""Gateway configuration (§2.7, §3.9, PS-3, TM-S3, TM-R1, TM-R2, IP-8).

`Settings` reads process environment and an optional `.env` file. Nothing is
loaded at import time: callers build settings with `load_settings()` or, in
tests, with `Settings(_env_file=None, KEY=value, ...)`. Mode definitions are
imported from `contracts.modes` so importing `app.config` never loads `Faker`
or `phonenumbers`.
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


class _CommaListMixin:
    """Parse PROBE_CANDIDATES from a comma-separated string instead of JSON."""

    def prepare_field_value(self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool) -> Any:
        if field_name == "PROBE_CANDIDATES" and isinstance(value, str):
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

    @field_validator("LOG_DB_PATH")
    @classmethod
    def _anchor_log_path(cls, value: Path) -> Path:
        path = value.expanduser()
        return path if path.is_absolute() else REPO_ROOT / path

    @model_validator(mode="after")
    def _startup_guards(self) -> Settings:
        if self.GATEWAY_MODE in DETERMINISTIC_MODES:
            secret = self.HMAC_SECRET.strip()
            if not secret or secret == DEFAULT_HMAC_SECRET or len(secret) < MIN_HMAC_SECRET_CHARS:
                raise ValueError(
                    f"{self.GATEWAY_MODE} mode refuses to start with an empty, default, or short "
                    f"(<{MIN_HMAC_SECRET_CHARS} chars) HMAC_SECRET (PS-3, TM-S3)"
                )
        if self.UPSTREAM_MODE == "groq" and is_placeholder_api_key(self.GROQ_API_KEY):
            raise ValueError("GROQ_API_KEY is a placeholder value; set a real key or leave it blank")
        offender = synced_folder_component(self.LOG_DB_PATH.resolve())
        if offender is not None:
            raise ValueError(
                f"LOG_DB_PATH must be on local disk, not a synced folder (IP-8): component {offender!r}"
            )
        return self


def load_settings(*, env_file: str | Path | None = _UNSET) -> Settings:
    """Build settings from the environment and optional `.env`. No import-time side effects."""
    if env_file is _UNSET:
        return Settings()
    return Settings(_env_file=env_file)
