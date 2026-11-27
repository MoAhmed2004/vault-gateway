"""Unit tests for app/config.py (§2.7, §3.9, PS-3, TM-R1, TM-R2, IP-8)."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import contracts.modes
from app.config import (
    DEFAULT_HMAC_SECRET,
    DEFAULT_LOG_DB_PATH,
    DEFAULT_PROBE_CANDIDATES,
    MIN_HMAC_SECRET_CHARS,
    REPO_ROOT as CONFIG_REPO_ROOT,
    Settings,
    load_settings,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SETTING_KEYS = tuple(Settings.model_fields)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in SETTING_KEYS:
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(key.lower(), raising=False)


def make(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_defaults_match_spec():
    s = make()
    assert s.GROQ_API_KEY == ""
    assert s.TARGET_MODEL == "openai/gpt-oss-20b"
    assert s.PROBE_CANDIDATES == ("openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b")
    assert s.PROBE_CANDIDATES == DEFAULT_PROBE_CANDIDATES
    assert s.ALLOW_MODEL_FAILOVER is False
    assert s.GATEWAY_MODE == "production"
    assert s.TIMEOUT_POLICY == "P-degrade"
    assert s.UPSTREAM_MODE == "groq"
    assert s.STUB_DELAY_MS == 50
    assert s.HMAC_SECRET == DEFAULT_HMAC_SECRET == "vault_dev_secret_change_me"
    assert s.LOG_DB_PATH == DEFAULT_LOG_DB_PATH == REPO_ROOT / "data" / "logs" / "vault.db"
    assert CONFIG_REPO_ROOT == REPO_ROOT


def test_default_caps_and_budgets():
    s = make()
    assert (s.MAX_REQUEST_BODY_BYTES, s.MAX_PROMPT_CHARS, s.MAX_WINDOWS_PER_VIEW) == (131072, 4000, 16)
    assert (s.MAX_CONTEXT_DOCS, s.MAX_CONTEXT_DOC_CHARS) == (4, 1500)
    assert (s.L1_BUDGET_MS, s.L2_BUDGET_MS, s.L3_BUDGET_MS, s.TOTAL_BUDGET_MS) == (5, 50, 150, 400)
    assert s.P_DEGRADE_THRESHOLD_DELTA == 0.05


def test_every_env_example_key_is_a_setting():
    keys = set()
    for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.add(line.split("=", 1)[0])
    assert keys == set(SETTING_KEYS)


def test_env_example_values_equal_defaults_with_blank_groq_key():
    values = {}
    for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            values[k] = v
    assert values["GROQ_API_KEY"] == ""
    s = Settings(_env_file=None, **{k: v for k, v in values.items()})
    d = make()
    assert s.model_dump() == d.model_dump()


def test_probe_candidates_parsed_from_comma_env(monkeypatch):
    monkeypatch.setenv("PROBE_CANDIDATES", "a/x, b/y ,c/z")
    assert make().PROBE_CANDIDATES == ("a/x", "b/y", "c/z")


def test_probe_candidates_from_init_string_and_list():
    assert make(PROBE_CANDIDATES="a,b").PROBE_CANDIDATES == ("a", "b")
    assert make(PROBE_CANDIDATES=["a", "b"]).PROBE_CANDIDATES == ("a", "b")


def test_probe_candidates_empty_rejected():
    with pytest.raises(ValidationError):
        make(PROBE_CANDIDATES=" , ")


def test_dotenv_file_is_read(tmp_path):
    env = tmp_path / ".env"
    env.write_text("TARGET_MODEL=openai/gpt-oss-120b\nPROBE_CANDIDATES=m1,m2\nUNRELATED=1\n", encoding="utf-8")
    s = Settings(_env_file=env)
    assert s.TARGET_MODEL == "openai/gpt-oss-120b"
    assert s.PROBE_CANDIDATES == ("m1", "m2")


def test_unrelated_os_env_ignored(monkeypatch):
    monkeypatch.setenv("SOME_RANDOM_OS_VAR", "x")
    make()


def test_os_env_overrides(monkeypatch):
    monkeypatch.setenv("GATEWAY_MODE", "demo")
    monkeypatch.setenv("STUB_DELAY_MS", "7")
    s = make()
    assert s.GATEWAY_MODE == "demo" and s.STUB_DELAY_MS == 7


@pytest.mark.parametrize("mode", ["eval", "shadow_ingress_only", "shadow_full_e2e"])
def test_deterministic_modes_reject_default_secret(mode):
    with pytest.raises(ValidationError, match="HMAC_SECRET"):
        make(GATEWAY_MODE=mode)
    with pytest.raises(ValidationError, match="HMAC_SECRET"):
        make(GATEWAY_MODE=mode, HMAC_SECRET=DEFAULT_HMAC_SECRET)


@pytest.mark.parametrize("mode", ["eval", "shadow_ingress_only", "shadow_full_e2e"])
@pytest.mark.parametrize(
    "secret",
    [
        "",
        " ",
        "   \t\n",
        "short",
        "a" * 15,
        "   " + ("a" * 15) + "   ",  # raw length 21, stripped length 15 (< 16)
    ],
)
def test_deterministic_modes_reject_empty_blank_or_short_secret(mode, secret):
    assert MIN_HMAC_SECRET_CHARS == 16
    with pytest.raises(ValidationError, match="HMAC_SECRET"):
        make(GATEWAY_MODE=mode, HMAC_SECRET=secret)


@pytest.mark.parametrize("mode", ["eval", "shadow_ingress_only", "shadow_full_e2e"])
def test_deterministic_modes_accept_secret_at_least_16_chars(mode):
    assert make(GATEWAY_MODE=mode, HMAC_SECRET="a" * 16).GATEWAY_MODE == mode
    assert make(GATEWAY_MODE=mode, HMAC_SECRET="a-real-secret-value").GATEWAY_MODE == mode


@pytest.mark.parametrize("mode", ["production", "demo"])
def test_random_modes_allow_default_or_short_secret(mode):
    assert make(GATEWAY_MODE=mode).HMAC_SECRET == DEFAULT_HMAC_SECRET
    assert make(GATEWAY_MODE=mode, HMAC_SECRET="short").HMAC_SECRET == "short"


def test_secret_not_in_repr():
    s = make(HMAC_SECRET="super-secret-xyz", GROQ_API_KEY="gsk_hidden")
    text = repr(s) + str(s)
    assert "super-secret-xyz" not in text and "gsk_hidden" not in text


@pytest.mark.parametrize(
    "placeholder",
    ["your_groq_api_key_here", "YOUR_GROQ_API_KEY_HERE", "your-api-key", "put_key_here", "changeme", "<groq-key>"],
)
def test_placeholder_groq_api_key_rejected_in_groq_mode(placeholder):
    with pytest.raises(ValidationError, match="GROQ_API_KEY"):
        make(UPSTREAM_MODE="groq", GROQ_API_KEY=placeholder)


def test_blank_and_real_groq_api_key_accepted_in_groq_mode():
    assert make(UPSTREAM_MODE="groq", GROQ_API_KEY="").GROQ_API_KEY == ""
    assert make(UPSTREAM_MODE="groq", GROQ_API_KEY="gsk_real_token_12345").GROQ_API_KEY == "gsk_real_token_12345"


@pytest.mark.parametrize("upstream", ["stub", "ollama"])
def test_placeholder_groq_key_ignored_outside_groq_upstream(upstream):
    s = make(UPSTREAM_MODE=upstream, GROQ_API_KEY="your_groq_api_key_here")
    assert s.UPSTREAM_MODE == upstream


def test_failover_true_rejected():
    with pytest.raises(ValidationError, match="ALLOW_MODEL_FAILOVER"):
        make(ALLOW_MODEL_FAILOVER=True)
    assert make(ALLOW_MODEL_FAILOVER=False).ALLOW_MODEL_FAILOVER is False


def test_failover_true_from_env_rejected(monkeypatch):
    monkeypatch.setenv("ALLOW_MODEL_FAILOVER", "true")
    with pytest.raises(ValidationError):
        make()


@pytest.mark.parametrize(
    "field,value",
    [
        ("GATEWAY_MODE", "dev"),
        ("TIMEOUT_POLICY", "P-maybe"),
        ("UPSTREAM_MODE", "openai"),
        ("STUB_DELAY_MS", -1),
        ("MAX_REQUEST_BODY_BYTES", 0),
        ("MAX_PROMPT_CHARS", -5),
        ("MAX_WINDOWS_PER_VIEW", 0),
        ("MAX_CONTEXT_DOCS", 0),
        ("MAX_CONTEXT_DOC_CHARS", 0),
        ("L1_BUDGET_MS", 0),
        ("L2_BUDGET_MS", -1),
        ("L3_BUDGET_MS", 0),
        ("TOTAL_BUDGET_MS", 0),
        ("P_DEGRADE_THRESHOLD_DELTA", float("nan")),
        ("P_DEGRADE_THRESHOLD_DELTA", float("inf")),
        ("P_DEGRADE_THRESHOLD_DELTA", -0.1),
    ],
)
def test_invalid_values_rejected(field, value):
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_step_5b_defaults():
    s = make()
    assert (s.L2_ENABLED, s.L3_ENABLED) == (True, True)
    assert s.MINILM_MODEL_DIR == REPO_ROOT / "models" / "minilm" and s.L3_MODEL_DIR == REPO_ROOT / "models" / "deberta"
    assert s.CHROMA_DB_PATH == REPO_ROOT / "data" / "chroma_db"
    assert (s.L2_THRESHOLD, s.L3_THRESHOLD, s.L3_MALICIOUS_LABEL) == (0.80, 0.50, "INJECTION")
    assert (s.X3_SIMILARITY_THRESHOLD, s.X3_NGRAM_OVERLAP_THRESHOLD, s.X3_NGRAM_SIZE) == (0.80, 0.30, 3)
    assert (s.LAYER_POOL_WORKERS, s.LAYER_POOL_QUEUE_LIMIT) == (0, 32)
    assert make(EGRESS_VARIANT="X3").EGRESS_VARIANT == "X3"


@pytest.mark.parametrize(
    "field,value",
    [
        ("L2_THRESHOLD", 1.5),
        ("L3_THRESHOLD", 0.0),
        ("X3_SIMILARITY_THRESHOLD", float("nan")),
        ("X3_NGRAM_OVERLAP_THRESHOLD", 0.0),
        ("X3_NGRAM_SIZE", 0),
        ("LAYER_POOL_WORKERS", -1),
        ("LAYER_POOL_QUEUE_LIMIT", -1),
    ],
)
def test_invalid_step_5b_values_rejected(field, value):
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_synced_folder_chroma_path_rejected():
    with pytest.raises(ValidationError, match="CHROMA_DB_PATH"):
        make(CHROMA_DB_PATH="C:/Users/x/OneDrive/chroma_db")


@pytest.mark.parametrize("mode", ["shadow_ingress_only", "shadow_full_e2e"])
@pytest.mark.parametrize("disabled", ["L2_ENABLED", "L3_ENABLED"])
def test_shadow_modes_require_l2_and_l3(mode, disabled):
    with pytest.raises(ValidationError, match="IP-4"):
        make(GATEWAY_MODE=mode, HMAC_SECRET="real-secret-at-least-16", **{disabled: False})


@pytest.mark.parametrize("flag", ["CACHE_ENABLED", "CACHE_WRITES_ENABLED"])
def test_shadow_full_e2e_requires_active_cache_writes(flag):
    """IP-4, IP-5 (LM-4): X0 to X3 are written to the egress cache so the harness can join oracles offline."""
    with pytest.raises(ValidationError, match="oracle join"):
        make(GATEWAY_MODE="shadow_full_e2e", HMAC_SECRET="real-secret-at-least-16", **{flag: False})
    # The other tiers may run without the cache (E4 uses CACHE_ENABLED=false).
    assert make(GATEWAY_MODE="shadow_ingress_only", HMAC_SECRET="real-secret-at-least-16", **{flag: False})
    assert make(GATEWAY_MODE="eval", HMAC_SECRET="real-secret-at-least-16", **{flag: False})


@pytest.mark.parametrize("mode", ["production", "demo", "eval", "shadow_ingress_only", "shadow_full_e2e"])
def test_all_modes_valid(mode):
    assert make(GATEWAY_MODE=mode, HMAC_SECRET="real-secret-at-least-16").GATEWAY_MODE == mode


@pytest.mark.parametrize("policy", ["P-open", "P-degrade", "P-closed"])
def test_all_policies_valid(policy):
    assert make(TIMEOUT_POLICY=policy).TIMEOUT_POLICY == policy


@pytest.mark.parametrize(
    "path",
    [
        "C:/Users/x/OneDrive/data/vault.db",
        "C:/Users/x/OneDrive - Contoso/data/vault.db",
        "D:/Dropbox/vault.db",
        "D:/Google Drive/vault.db",
        "D:/GoogleDrive/vault.db",
        "/Users/x/iCloudDrive/vault.db",
        "/Users/x/Library/Mobile Documents/com~apple~CloudDocs/vault.db",
        "/home/x/ONEDRIVE/vault.db",
        "C:/Users/x/Box/vault.db",
        "C:/Users/x/Box Sync/vault.db",
    ],
)
def test_synced_folder_log_path_rejected(path):
    with pytest.raises(ValidationError, match="LOG_DB_PATH"):
        make(LOG_DB_PATH=path)


def test_local_and_sandbox_log_paths_accepted(tmp_path):
    assert make(LOG_DB_PATH=tmp_path / "logs" / "v.db").LOG_DB_PATH == tmp_path / "logs" / "v.db"
    assert make(LOG_DB_PATH=tmp_path / "sandbox" / "v.db").LOG_DB_PATH == tmp_path / "sandbox" / "v.db"


def test_relative_log_path_and_dotenv_anchored_to_repo_root(monkeypatch, tmp_path):
    assert Path(Settings.model_config["env_file"]) == REPO_ROOT / ".env"
    assert Path(Settings.model_config["env_file"]).is_absolute()

    # Use an isolated anchored .env in tmp_path instead of reading a developer's real local REPO_ROOT/.env.
    anchored_dir = tmp_path / "anchored_root"
    anchored_dir.mkdir()
    anchored_env = anchored_dir / ".env"
    anchored_env.write_text("STUB_DELAY_MS=77\n", encoding="utf-8")

    decoy_cwd = tmp_path / "decoy_cwd"
    decoy_cwd.mkdir()
    (decoy_cwd / ".env").write_text("TARGET_MODEL=decoy/wrong-model\nLOG_DB_PATH=decoy/v.db\n", encoding="utf-8")

    monkeypatch.setitem(Settings.model_config, "env_file", str(anchored_env))
    monkeypatch.chdir(decoy_cwd)

    loaded = load_settings()
    assert loaded.STUB_DELAY_MS == 77
    assert loaded.TARGET_MODEL == "openai/gpt-oss-20b"
    assert loaded.LOG_DB_PATH == REPO_ROOT / "data" / "logs" / "vault.db"

    explicit_env_loaded = load_settings(env_file=anchored_env)
    assert explicit_env_loaded.STUB_DELAY_MS == 77

    rel = make(LOG_DB_PATH=Path("data/logs/custom.db"))
    assert rel.LOG_DB_PATH == REPO_ROOT / "data" / "logs" / "custom.db"
    abs_path = tmp_path / "explicit.db"
    assert make(LOG_DB_PATH=abs_path).LOG_DB_PATH == abs_path


def test_settings_are_frozen():
    s = make()
    with pytest.raises(ValidationError):
        s.TARGET_MODEL = "other"


def test_load_settings_has_no_import_side_effect_and_builds(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TARGET_MODEL", "openai/gpt-oss-120b")
    assert load_settings(env_file=None).TARGET_MODEL == "openai/gpt-oss-120b"


def test_app_config_does_not_eagerly_import_secret_derivation_faker_or_phonenumbers():
    # 1. Static AST verification on app/config.py.
    source = (REPO_ROOT / "app" / "config.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {"app.secret_derivation", "faker", "phonenumbers"}
    assert not any(m == f or m.startswith(f + ".") for m in imported for f in forbidden)

    # 2. Runtime verification in a clean Python interpreter without pytest's Faker plugin.
    code = (
        "import sys; import app.config as cfg; "
        "bad = [m for m in ('app.secret_derivation', 'faker', 'phonenumbers') "
        "if m in sys.modules or any(k.startswith(m + '.') for k in sys.modules)]; "
        "assert not bad, f'eagerly imported: {bad}'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert contracts.modes.DETERMINISTIC_MODES == ("eval", "shadow_ingress_only", "shadow_full_e2e")
