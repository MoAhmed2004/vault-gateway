"""Unit tests for the neutral contracts package (IP-9, §3.2)."""

from __future__ import annotations

import ast
import hashlib
import re
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

import contracts
from contracts import (
    EgressVariantRecord,
    LayerResult,
    Outcome,
    RequestLogRecord,
    ShadowRecord,
    TelemetrySink,
    canonical_hash,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = REPO_ROOT / "contracts"
FORBIDDEN_PROJECT_PACKAGES = {"app", "telemetry", "attacker", "harness"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _valid_kwargs(**overrides):
    base = {
        "layer": "L2",
        "view": "decoded_view",
        "per_window_scores": [0.1, 0.4, 0.2],
        "overall_score": 0.4,
        "verdict": "allow",
        "latency_ms": 12.5,
    }
    base.update(overrides)
    return base


# LayerResult (IP-9)


def test_layer_result_valid_defaults():
    result = LayerResult(**_valid_kwargs())
    assert result.truncation_count == 0
    assert result.error_message is None
    assert result.per_window_scores == (0.1, 0.4, 0.2)


def test_layer_result_field_reassignment_fails():
    result = LayerResult(**_valid_kwargs())
    with pytest.raises(ValidationError):
        result.verdict = "block"
    with pytest.raises(ValidationError):
        result.per_window_scores = (1.0,)


def test_layer_result_per_window_scores_is_immutable_tuple():
    result = LayerResult(**_valid_kwargs())
    assert isinstance(result.per_window_scores, tuple)
    with pytest.raises(TypeError):
        result.per_window_scores[0] = 0.9  # type: ignore[index]
    assert not hasattr(result.per_window_scores, "append")


def test_layer_result_serializes_scores_as_json_array():
    assert '"per_window_scores":[0.1,0.4,0.2]' in LayerResult(**_valid_kwargs()).model_dump_json()


@pytest.mark.parametrize("verdict", ["allow", "block", "timeout", "error"])
def test_layer_result_accepts_spec_verdicts(verdict):
    assert LayerResult(**_valid_kwargs(verdict=verdict)).verdict == verdict


@pytest.mark.parametrize("verdict", ["ALLOW", "pass", "blocked", "", "unknown"])
def test_layer_result_rejects_invalid_verdict(verdict):
    with pytest.raises(ValidationError):
        LayerResult(**_valid_kwargs(verdict=verdict))


@pytest.mark.parametrize("count", [1, 2, -1, 100])
def test_layer_result_rejects_nonzero_truncation_count(count):
    with pytest.raises(ValidationError, match="truncation_count must be 0"):
        LayerResult(**_valid_kwargs(truncation_count=count))


def test_layer_result_rejects_negative_latency():
    with pytest.raises(ValidationError):
        LayerResult(**_valid_kwargs(latency_ms=-0.001))


def test_layer_result_accepts_zero_latency():
    assert LayerResult(**_valid_kwargs(latency_ms=0.0)).latency_ms == 0.0


def test_layer_result_rejects_extra_fields():
    with pytest.raises(ValidationError):
        LayerResult(**_valid_kwargs(threshold=0.5))


# Outcome (IP-9)


def test_outcome_values_match_spec_verbatim():
    assert Outcome.BLOCKED_INGRESS.value == "Blocked-ingress"
    assert Outcome.ADMITTED_REFUSED.value == "Admitted-refused"
    assert Outcome.EGRESS_SAVED.value == "Egress-saved"
    assert Outcome.SUCCESS.value == "Success"
    assert Outcome.ERROR.value == "Error"
    assert {o.value for o in Outcome} == {
        "Blocked-ingress",
        "Admitted-refused",
        "Egress-saved",
        "Success",
        "Error",
    }
    assert len(Outcome) == 5


def test_outcome_is_str_enum():
    assert isinstance(Outcome.SUCCESS, str)
    assert Outcome("Egress-saved") is Outcome.EGRESS_SAVED


# canonical_hash (IP-9, IP-5)


def test_canonical_hash_is_64_char_lowercase_hex():
    assert HEX64.match(canonical_hash({"a": 1}))


def test_canonical_hash_independent_of_key_insertion_order():
    first = {"b": 2, "a": 1, "nested": {"y": [1, 2], "x": {"q": True, "p": None}}}
    second = {"nested": {"x": {"p": None, "q": True}, "y": [1, 2]}, "a": 1, "b": 2}
    assert canonical_hash(first) == canonical_hash(second)


def test_canonical_hash_matches_sha256_of_compact_sorted_json():
    expected = hashlib.sha256(b'{"a":1,"b":[1,2]}').hexdigest()
    assert canonical_hash({"b": [1, 2], "a": 1}) == expected


def test_canonical_hash_unicode_is_deterministic_and_ascii_escaped():
    from contracts.hashing import canonical_json

    payload = {"text": "caf\u00e9 \u4e2d\u6587 \u00a72.7"}
    assert canonical_hash(payload) == canonical_hash(dict(payload))
    assert canonical_json(payload).isascii()
    assert "\\u00e9" in canonical_json(payload)
    assert canonical_hash({"text": "cafe"}) != canonical_hash({"text": "caf\u00e9"})


# F3: lone surrogates and NaN/Inf in canonical_hash


@pytest.mark.parametrize("text", ["\ud800", "a\udfffb", "\ud83d"])
def test_canonical_hash_lone_surrogates_hash_cleanly(text):
    first = canonical_hash({"t": text})
    assert HEX64.match(first)
    assert first == canonical_hash({"t": text})
    assert canonical_hash({text: 1}) == canonical_hash({text: 1})


def test_canonical_hash_distinguishes_different_surrogates():
    assert canonical_hash({"t": "\ud800"}) != canonical_hash({"t": "\ud801"})


def test_canonical_json_uses_ascii_and_disallows_nan():
    import inspect

    from contracts import hashing

    src = inspect.getsource(hashing.canonical_json)
    assert "ensure_ascii=True" in src
    assert "allow_nan=False" in src


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_canonical_hash_rejects_nan_and_inf(bad):
    with pytest.raises(ValueError):
        canonical_hash({"x": bad})
    with pytest.raises(ValueError):
        canonical_hash([bad])


# F4: integral floats normalize to ints


def test_canonical_hash_integral_float_equals_int():
    assert canonical_hash({"t": 0}) == canonical_hash({"t": 0.0})
    assert canonical_hash({"t": 0}) == canonical_hash({"t": -0.0})
    assert canonical_hash({"t": 3}) == canonical_hash({"t": 3.0})
    assert canonical_hash([1.0, 2.0]) == canonical_hash([1, 2])


def test_canonical_hash_non_integral_float_unchanged():
    assert canonical_hash({"t": 0.5}) != canonical_hash({"t": 0})
    assert canonical_hash({"t": 0.5}) == hashlib.sha256(b'{"t":0.5}').hexdigest()


def test_canonical_hash_bool_not_collapsed_to_int():
    assert canonical_hash({"t": True}) != canonical_hash({"t": 1})
    assert canonical_hash({"t": False}) != canonical_hash({"t": 0})
    assert canonical_hash({"t": True}) == hashlib.sha256(b'{"t":true}').hexdigest()


def test_canonical_hash_integral_float_inside_model():
    a = LayerResult(**_valid_kwargs(overall_score=1.0, per_window_scores=[0.0, 1.0], latency_ms=2.0))
    b = LayerResult(**_valid_kwargs(overall_score=1, per_window_scores=[0, 1], latency_ms=2))
    assert canonical_hash(a) == canonical_hash(b)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_layer_result_rejects_non_finite_overall_score(bad):
    with pytest.raises(ValidationError):
        LayerResult(**_valid_kwargs(overall_score=bad))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_layer_result_rejects_non_finite_window_scores(bad):
    with pytest.raises(ValidationError):
        LayerResult(**_valid_kwargs(per_window_scores=[0.1, bad]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_layer_result_rejects_non_finite_latency(bad):
    with pytest.raises(ValidationError):
        LayerResult(**_valid_kwargs(latency_ms=bad))


# F5: Outcome is a StrEnum


def test_outcome_is_strenum():
    import enum

    assert issubclass(Outcome, enum.StrEnum)


def test_outcome_str_and_format_give_value():
    assert str(Outcome.SUCCESS) == "Success"
    assert f"{Outcome.BLOCKED_INGRESS}" == "Blocked-ingress"
    assert "{}".format(Outcome.EGRESS_SAVED) == "Egress-saved"
    assert Outcome.ADMITTED_REFUSED == "Admitted-refused"


# F2: RequestLogRecord and ShadowRecord (RT-V6, IP-4)


def _layer(layer="L1", verdict="allow"):
    return LayerResult(**_valid_kwargs(layer=layer, verdict=verdict))


def _request_kwargs(**overrides):
    base = {
        "timestamp_utc": "2026-11-23T18:00:00Z",
        "gateway_mode": "production",
        "prompt_hash": canonical_hash({"p": "hello"}),
        "layer_results": (_layer("L1"), _layer("L2")),
        "target_called": True,
        "outcome": Outcome.SUCCESS,
        "total_latency_ms": 42.0,
    }
    base.update(overrides)
    return base


def _shadow_kwargs(**overrides):
    base = {
        "timestamp_utc": "2026-11-23T18:00:00Z",
        "shadow_tier": "shadow_full_e2e",
        "prompt_hash": canonical_hash({"p": "hello"}),
        "layer_results": (_layer("L1"), _layer("L2"), _layer("L3", "block")),
        "total_latency_ms": 120.5,
        "target_called": True,
        "cache_hit": False,
        "egress_variants": (
            EgressVariantRecord(variant="X0", verdict="allow", oracle_result=True),
            EgressVariantRecord(variant="X3", verdict="block", oracle_result=False),
        ),
    }
    base.update(overrides)
    return base


def test_request_log_record_field_names_cover_rt_v6():
    required = {
        "trial_id", "objective_id", "tier", "config", "seed", "query_index",
        "operator_chain", "layer_results", "target_called", "oracle_result", "judge_score",
        "outcome", "timestamp_utc", "gateway_mode", "prompt_hash", "total_latency_ms",
        "reasoning_findings",
    }
    assert set(RequestLogRecord.model_fields) == required


def test_request_log_record_production_defaults_are_none():
    rec = RequestLogRecord(**_request_kwargs())
    for name in ("trial_id", "objective_id", "tier", "config", "seed", "query_index",
                 "operator_chain", "oracle_result", "judge_score"):
        assert getattr(rec, name) is None
    assert rec.reasoning_findings == ()


def test_request_log_record_full_eval_record():
    rec = RequestLogRecord(
        **_request_kwargs(
            gateway_mode="eval", trial_id="t1", objective_id="O3-01", tier="T2", config="C2",
            seed=7, query_index=3, operator_chain=["base64", "role"], oracle_result=True,
            judge_score=0.75, reasoning_findings=["EMAIL_ADDRESS"],
        )
    )
    assert rec.operator_chain == ("base64", "role")
    assert rec.reasoning_findings == ("EMAIL_ADDRESS",)
    assert isinstance(rec.layer_results[0], LayerResult)
    assert HEX64.match(canonical_hash(rec))


def test_request_log_record_is_frozen_and_forbids_extra():
    rec = RequestLogRecord(**_request_kwargs())
    with pytest.raises(ValidationError):
        rec.outcome = Outcome.ERROR
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(surprise=1))


def test_request_log_record_rejects_bad_mode_and_outcome():
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(gateway_mode="dev"))
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(outcome="Passed"))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_request_log_record_rejects_non_finite_floats(bad):
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(judge_score=bad))
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(total_latency_ms=bad))


def test_request_log_record_rejects_negative_latency():
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(total_latency_ms=-1.0))


def test_shadow_record_field_names_cover_ip4():
    required = {
        "timestamp_utc", "shadow_tier", "prompt_hash", "layer_results", "total_latency_ms",
        "target_called", "cache_hit", "egress_variants", "layer_version", "bundle_version",
        "chroma_version", "trial_id", "objective_id",
    }
    assert set(ShadowRecord.model_fields) == required


def test_shadow_record_full_e2e_with_egress_variants():
    rec = ShadowRecord(**_shadow_kwargs())
    assert [v.variant for v in rec.egress_variants] == ["X0", "X3"]
    assert len(rec.layer_results) == 3
    assert HEX64.match(canonical_hash(rec))


def test_shadow_record_ingress_only_defaults():
    rec = ShadowRecord(**_shadow_kwargs(shadow_tier="shadow_ingress_only", target_called=False,
                                        cache_hit=None, egress_variants=()))
    assert rec.target_called is False and rec.egress_variants == ()


def test_shadow_record_ingress_only_rejects_groq_call_or_egress():
    with pytest.raises(ValidationError):
        ShadowRecord(**_shadow_kwargs(shadow_tier="shadow_ingress_only", egress_variants=()))
    with pytest.raises(ValidationError):
        ShadowRecord(**_shadow_kwargs(shadow_tier="shadow_ingress_only", target_called=False))


def test_shadow_record_rejects_duplicate_variant_and_bad_variant():
    dup = (
        EgressVariantRecord(variant="X1", verdict="allow"),
        EgressVariantRecord(variant="X1", verdict="block"),
    )
    with pytest.raises(ValidationError):
        ShadowRecord(**_shadow_kwargs(egress_variants=dup))
    with pytest.raises(ValidationError):
        EgressVariantRecord(variant="X4", verdict="allow")


def test_shadow_record_frozen_extra_and_finite():
    rec = ShadowRecord(**_shadow_kwargs())
    with pytest.raises(ValidationError):
        rec.cache_hit = True
    with pytest.raises(ValidationError):
        ShadowRecord(**_shadow_kwargs(extra_field=1))
    with pytest.raises(ValidationError):
        ShadowRecord(**_shadow_kwargs(total_latency_ms=float("nan")))


def test_records_reexported_from_contracts():
    assert contracts.RequestLogRecord is RequestLogRecord
    assert contracts.ShadowRecord is ShadowRecord


def test_canonical_hash_tuple_and_list_equivalent():
    assert canonical_hash({"s": (1, 2, 3)}) == canonical_hash({"s": [1, 2, 3]})


def test_canonical_hash_enum_uses_value():
    assert canonical_hash({"o": Outcome.SUCCESS}) == canonical_hash({"o": "Success"})


def test_canonical_hash_pydantic_model_deterministic():
    a = LayerResult(**_valid_kwargs())
    b = LayerResult(**dict(reversed(list(_valid_kwargs().items()))))
    assert canonical_hash(a) == canonical_hash(b)
    assert canonical_hash(a) == canonical_hash(a.model_dump())


def test_canonical_hash_nested_pydantic_model():
    class Wrapper(BaseModel):
        outcome: Outcome
        result: LayerResult

    w = Wrapper(outcome=Outcome.BLOCKED_INGRESS, result=LayerResult(**_valid_kwargs()))
    expected = {"outcome": "Blocked-ingress", "result": LayerResult(**_valid_kwargs()).model_dump()}
    assert canonical_hash(w) == canonical_hash(expected)


def test_canonical_hash_differs_on_value_change():
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})
    assert canonical_hash({"a": {"b": "x"}}) != canonical_hash({"a": {"b": "y"}})
    assert canonical_hash(LayerResult(**_valid_kwargs())) != canonical_hash(
        LayerResult(**_valid_kwargs(overall_score=0.41))
    )


def test_canonical_hash_rejects_unserializable_types():
    with pytest.raises(TypeError):
        canonical_hash({"s": {1, 2}})
    with pytest.raises(TypeError):
        canonical_hash({1: "non-string key"})


def test_hashing_module_never_calls_builtin_hash():
    tree = ast.parse((CONTRACTS_DIR / "hashing.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hash"
    ]
    assert calls == []


# TelemetrySink (§3.2)


def test_telemetry_sink_protocol_runtime_check():
    class MemorySink:
        def __init__(self):
            self.requests = []
            self.shadows = []

        def record_request(self, record):
            self.requests.append(record)

        def record_shadow(self, record):
            self.shadows.append(record)

    class Incomplete:
        def record_request(self, record):
            pass

    sink = MemorySink()
    assert isinstance(sink, TelemetrySink)
    assert not isinstance(Incomplete(), TelemetrySink)
    rec = RequestLogRecord(**_request_kwargs())
    sink.record_request(rec)
    assert sink.requests == [rec]
    shadow = ShadowRecord(**_shadow_kwargs())
    sink.record_shadow(shadow)
    assert sink.shadows == [shadow]


def test_telemetry_sink_signatures_use_typed_records():
    import typing

    hints_req = typing.get_type_hints(TelemetrySink.record_request)
    hints_shadow = typing.get_type_hints(TelemetrySink.record_shadow)
    assert hints_req["record"] is RequestLogRecord
    assert hints_shadow["record"] is ShadowRecord


def test_contracts_reexports():
    assert set(contracts.__all__) == {
        "EgressVariantRecord",
        "LayerResult",
        "Outcome",
        "RequestLogRecord",
        "ShadowRecord",
        "TelemetrySink",
        "canonical_hash",
    }


# Import boundary (§3.2, AGENTS.md Rule 5)


def _imported_top_levels(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
            elif node.level > 1:
                names.add("<relative-beyond-contracts>")
    return names


def test_contracts_never_imports_project_packages():
    files = sorted(CONTRACTS_DIR.rglob("*.py"))
    assert files, "contracts/ has no Python files"
    for path in files:
        bad = _imported_top_levels(path) & (FORBIDDEN_PROJECT_PACKAGES | {"<relative-beyond-contracts>"})
        assert not bad, f"{path.name} imports forbidden packages: {sorted(bad)}"


def test_contracts_imports_only_stdlib_pydantic_or_self():
    allowed = set(sys.stdlib_module_names) | {"pydantic", "contracts", "__future__"}
    for path in sorted(CONTRACTS_DIR.rglob("*.py")):
        extra = _imported_top_levels(path) - allowed
        assert not extra, f"{path.name} imports non-neutral modules: {sorted(extra)}"
