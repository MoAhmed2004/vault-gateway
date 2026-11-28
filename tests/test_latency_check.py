"""Unit tests for the cap-sized TM-R2 check in `scripts/latency_check.py` (TM-R1, TM-R2, IP-1, IP-2; Step 5B.1).

The helpers run with the fake tokenizers here; the measurement itself needs the
real models and an L2 collection and is an operator step, not a unit test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import latency_check  # noqa: E402

from app.config import Settings  # noqa: E402
from app.inspection import InspectionConfig, build_inspection_bundle  # noqa: E402
from app.onnx_models import tokenizer_limit_for  # noqa: E402
from contracts.inspection import CapExceeded  # noqa: E402
from tests.fakes import FakeClassifier, FakeEmbedder, real_models_available, real_tokenizer_limit  # noqa: E402


def _check(config: InspectionConfig) -> None:
    prompts = latency_check.cap_prompts(config)
    assert set(prompts) == {"cap_plain", "cap_encoded"}
    for name, prompt in prompts.items():
        assert len(prompt) == latency_check.CAP_CHARS == 4000
        bundle = build_inspection_bundle(prompt, config)
        assert not isinstance(bundle, CapExceeded), name
        assert (bundle.decoded_view != bundle.raw_view) == (name == "cap_encoded")


def test_cap_prompts_are_4000_characters_and_pass_the_bundle_caps():
    _check(InspectionConfig(tokenizers=(tokenizer_limit_for(FakeEmbedder()), tokenizer_limit_for(FakeClassifier()))))


@pytest.mark.skipif(not real_models_available(), reason="run scripts/fetch_models.py for the real tokenizer test")
def test_cap_prompts_fit_the_real_tokenizers_after_sub_window_splitting():
    _check(InspectionConfig(tokenizers=(real_tokenizer_limit("minilm", 256), real_tokenizer_limit("deberta", 512))))


def test_cap_prompts_refuse_to_measure_less_when_nothing_fits():
    with pytest.raises(RuntimeError, match="caps"):
        latency_check.cap_prompts(InspectionConfig(max_windows_per_view=2))


def test_the_benchmark_runs_with_rate_limiting_and_the_output_sanitizer_off(tmp_path):
    """TM-R7: ENABLE_RATE_LIMITING MUST be false in every latency or throughput run; IP-10 is off in it too."""
    s = latency_check.gateway_settings(tmp_path / "chroma_db", tmp_path / "vault.db")
    assert (s.GATEWAY_MODE, s.ENABLE_RATE_LIMITING, s.ENABLE_OUTPUT_SANITIZER, s.CACHE_ENABLED) == (
        "production", False, False, False
    )


def test_budget_violations_use_the_configured_tm_r2_defaults():
    s = Settings(_env_file=None)
    budgets = {"L1": s.L1_BUDGET_MS, "L2": s.L2_BUDGET_MS, "L3": s.L3_BUDGET_MS, "L4": s.L4_BUDGET_MS}
    assert budgets == {"L1": 5, "L2": 50, "L3": 150, "L4": 150} and s.TOTAL_BUDGET_MS == 400  # unchanged in Step 5B.1
    fits = {"bundle": [3.0] * 20, "L1": [1.0] * 20, "L2": [40.0] * 20, "L3": [120.0] * 20, "L4": [30.0] * 20}
    assert latency_check.budget_violations(fits, budgets, s.TOTAL_BUDGET_MS) == []
    slow = dict(fits, L3=[120.0] * 18 + [400.0] * 2)  # p95 interpolates into the slow tail
    problems = latency_check.budget_violations(slow, budgets, s.TOTAL_BUDGET_MS)
    assert any(p.startswith("L3 p95") for p in problems)
    over_total = {"bundle": [100.0], "L1": [4.0], "L2": [49.0], "L3": [149.0], "L4": [149.0]}
    assert latency_check.budget_violations(over_total, budgets, s.TOTAL_BUDGET_MS) == [
        "bundle + L1..L4 p95 451.0 ms > TOTAL_BUDGET_MS 400 ms"
    ]
