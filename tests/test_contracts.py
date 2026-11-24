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


def test_canonical_hash_unicode_is_deterministic_and_utf8():
    from contracts.hashing import canonical_json

    payload = {"text": "caf\u00e9 \u4e2d\u6587 \u00a72.7"}
    assert canonical_hash(payload) == canonical_hash(dict(payload))
    assert "caf\u00e9" in canonical_json(payload)
    expected = hashlib.sha256('{"text":"caf\u00e9"}'.encode("utf-8")).hexdigest()
    assert canonical_hash({"text": "caf\u00e9"}) == expected
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


def test_canonical_json_uses_utf8_and_disallows_nan():
    import inspect

    from contracts import hashing

    src = inspect.getsource(hashing.canonical_json)
    assert "ensure_ascii=False" in src
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


def _variant(name="X0", verdict="allow", **kw):
    if verdict == "block":
        kw.setdefault("egress_action", "blocked")
    return EgressVariantRecord(
        response_hash=canonical_hash({"r": "resp"}), manifest_id="M1", variant=name, verdict=verdict, **kw
    )


ALL_VARIANTS = tuple(_variant(n) for n in ("X0", "X1", "X2", "X3"))


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
        "egress_variants": ALL_VARIANTS,
        "layer_version": "l-1",
        "bundle_version": "b-1",
        "chroma_version": "c-1",
    }
    base.update(overrides)
    return base


def test_request_log_record_field_names_cover_rt_v6():
    required = {
        "trial_id", "objective_id", "tier", "config", "seed", "query_index",
        "operator_chain", "layer_results", "target_called", "oracle_result", "judge_score",
        "outcome", "timestamp_utc", "gateway_mode", "prompt_hash", "total_latency_ms",
        "reasoning_findings", "blocked_by", "egress_variant", "egress_action",
        "timeout_policy", "cache_hit", "model_id",
    }
    assert set(RequestLogRecord.model_fields) == required


def test_request_log_record_production_defaults_are_none():
    rec = RequestLogRecord(**_request_kwargs())
    for name in ("trial_id", "objective_id", "tier", "config", "seed", "query_index",
                 "operator_chain", "oracle_result", "judge_score"):
        assert getattr(rec, name) is None
    assert rec.reasoning_findings == ()
    assert rec.egress_action == "none"


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
    assert [v.variant for v in rec.egress_variants] == ["X0", "X1", "X2", "X3"]
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
    dup = (_variant("X1", "allow"), _variant("X1", "block"))
    with pytest.raises(ValidationError):
        ShadowRecord(**_shadow_kwargs(egress_variants=dup))
    with pytest.raises(ValidationError):
        _variant("X4", "allow")


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
        "ALL_GATEWAY_MODES",
        "DETERMINISTIC_MODES",
        "DerivationKind",
        "EgressAction",
        "EgressVariantRecord",
        "GatewayMode",
        "LayerResult",
        "Outcome",
        "RANDOM_MODES",
        "RequestLogRecord",
        "SYNCED_FOLDER_MARKERS",
        "ShadowRecord",
        "TelemetrySink",
        "canonical_hash",
        "resolve_derivation_mode",
        "synced_folder_component",
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


# Step 3: Findings 1 and 2

SURROGATE_PAIR = "\ud83d\ude00"


def test_canonical_hash_astral_does_not_collide_with_surrogate_pair():
    assert canonical_hash("\U0001F600") != canonical_hash(SURROGATE_PAIR)
    assert canonical_hash({"t": "\U0001F600"}) != canonical_hash({"t": SURROGATE_PAIR})


def test_canonical_hash_lone_surrogate_scalar_succeeds():
    assert HEX64.match(canonical_hash("\ud800"))
    assert canonical_hash("\ud800") == canonical_hash("\ud800")


def test_canonical_hash_astral_matches_utf8_digest():
    expected = hashlib.sha256('"\U0001F600"'.encode("utf-8")).hexdigest()
    assert canonical_hash("\U0001F600") == expected


@pytest.mark.parametrize(
    "bad", ["", "ABC", "a" * 63, "a" * 65, "A" * 64, "g" * 64, " " + "a" * 63, "a" * 64 + "\n"]
)
def test_prompt_hash_must_be_64_lowercase_hex(bad):
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(prompt_hash=bad))
    with pytest.raises(ValidationError):
        ShadowRecord(**_shadow_kwargs(prompt_hash=bad))


def test_egress_variant_record_requires_valid_response_hash_and_manifest():
    with pytest.raises(ValidationError):
        EgressVariantRecord(response_hash="XYZ", manifest_id="M1", variant="X1", verdict="allow")
    with pytest.raises(ValidationError):
        EgressVariantRecord(response_hash="A" * 64, manifest_id="M1", variant="X1", verdict="allow")
    with pytest.raises(ValidationError):
        EgressVariantRecord(response_hash="a" * 64, manifest_id="", variant="X1", verdict="allow")
    with pytest.raises(ValidationError):
        EgressVariantRecord(response_hash="a" * 64, variant="X1", verdict="allow")
    with pytest.raises(ValidationError):
        EgressVariantRecord(manifest_id="M1", variant="X1", verdict="allow")


@pytest.mark.parametrize(
    "stage",
    ["L1", "L2", "L3", "body_cap", "schema_validation", "decoded_cap", "window_cap"],
)
def test_request_log_blocked_by_accepts_all_ingress_and_prelayer_stages(stage):
    layers = (_layer(stage, "block"),) if stage in ("L1", "L2", "L3") else ()
    rec = RequestLogRecord(
        **_request_kwargs(
            blocked_by=stage, outcome=Outcome.BLOCKED_INGRESS, target_called=False, layer_results=layers
        )
    )
    assert rec.blocked_by == stage


def test_request_log_blocked_by_uses_canonical_l4_and_rejects_duplicate_label():
    rec = RequestLogRecord(
        **_request_kwargs(
            outcome=Outcome.EGRESS_SAVED,
            egress_variant="X1",
            egress_action="blocked",
            blocked_by="L4",
            target_called=True,
        )
    )
    assert rec.blocked_by == "L4"
    with pytest.raises(ValidationError):
        RequestLogRecord(
            **_request_kwargs(
                outcome=Outcome.EGRESS_SAVED,
                egress_variant="X1",
                egress_action="blocked",
                blocked_by="L4_egress",
                target_called=True,
            )
        )


def test_request_log_blocked_by_rejects_unknown_stage():
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(blocked_by="L9", outcome=Outcome.BLOCKED_INGRESS, target_called=False))


def test_request_log_context_fields_validated():
    rec = RequestLogRecord(
        **_request_kwargs(egress_variant="X2", timeout_policy="P-closed", cache_hit=True, model_id="m")
    )
    assert (rec.egress_variant, rec.timeout_policy, rec.cache_hit, rec.model_id) == ("X2", "P-closed", True, "m")
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(egress_variant="X9"))
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(timeout_policy="P-weird"))


def test_request_log_config_disallows_inf_nan():
    assert RequestLogRecord.model_config["allow_inf_nan"] is False


@pytest.mark.parametrize("variant", ["X1", "X2", "X3"])
def test_egress_saved_valid_with_blocked_by(variant):
    rec = RequestLogRecord(
        **_request_kwargs(
            outcome=Outcome.EGRESS_SAVED,
            egress_variant=variant,
            egress_action="blocked",
            blocked_by="L4",
            target_called=True,
        )
    )
    assert rec.outcome == Outcome.EGRESS_SAVED
    assert rec.egress_action == "blocked"


def test_egress_saved_valid_with_l4_block_layer_result():
    rec = RequestLogRecord(
        **_request_kwargs(
            outcome=Outcome.EGRESS_SAVED,
            egress_variant="X1",
            egress_action="blocked",
            target_called=True,
            layer_results=(_layer("L1"), _layer("L4", "block")),
        )
    )
    assert rec.blocked_by is None
    assert rec.egress_action == "blocked"


@pytest.mark.parametrize("variant", [None, "X0"])
def test_egress_saved_rejects_missing_or_x0_variant(variant):
    with pytest.raises(ValidationError):
        RequestLogRecord(
            **_request_kwargs(
                outcome=Outcome.EGRESS_SAVED,
                egress_variant=variant,
                egress_action="blocked",
                blocked_by="L4",
                target_called=True,
            )
        )


def test_egress_saved_rejects_missing_blocking_evidence():
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(outcome=Outcome.EGRESS_SAVED, egress_variant="X2", target_called=True))
    with pytest.raises(ValidationError):
        RequestLogRecord(
            **_request_kwargs(
                outcome=Outcome.EGRESS_SAVED,
                egress_variant="X2",
                egress_action="blocked",
                blocked_by="L3",
                target_called=True,
            )
        )
    with pytest.raises(ValidationError):
        RequestLogRecord(
            **_request_kwargs(
                outcome=Outcome.EGRESS_SAVED,
                egress_variant="X2",
                egress_action="blocked",
                target_called=True,
                layer_results=(_layer("L4", "allow"),),
            )
        )
    with pytest.raises(ValidationError):
        RequestLogRecord(
            **_request_kwargs(
                outcome=Outcome.EGRESS_SAVED,
                egress_variant="X2",
                egress_action="blocked",
                blocked_by="L1",
                target_called=True,
                layer_results=(_layer("L4", "block"),),
            )
        )


# Step 3.5: Cross-field consistency and shared synced-folder paths (Findings 6 and 8)


@pytest.mark.parametrize("outcome", [Outcome.SUCCESS, Outcome.ADMITTED_REFUSED])
def test_success_and_admitted_refused_require_no_block_and_target_called(outcome):
    rec = RequestLogRecord(**_request_kwargs(outcome=outcome, blocked_by=None, target_called=True))
    assert rec.outcome == outcome
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(outcome=outcome, blocked_by="L1", target_called=True))
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(outcome=outcome, blocked_by="L4", target_called=True))
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(outcome=outcome, blocked_by=None, target_called=False))


def test_blocked_ingress_rejects_contradictory_states():
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(outcome=Outcome.BLOCKED_INGRESS, blocked_by=None, target_called=False))
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(outcome=Outcome.BLOCKED_INGRESS, blocked_by="L4", target_called=False))
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(outcome=Outcome.BLOCKED_INGRESS, blocked_by="L1", target_called=True))


def test_egress_saved_rejects_target_not_called():
    with pytest.raises(ValidationError):
        RequestLogRecord(
            **_request_kwargs(
                outcome=Outcome.EGRESS_SAVED,
                egress_variant="X2",
                egress_action="blocked",
                blocked_by="L4",
                target_called=False,
            )
        )


def test_error_outcome_allows_either_target_called_state():
    a = RequestLogRecord(**_request_kwargs(outcome=Outcome.ERROR, blocked_by="p_closed", target_called=False))
    b = RequestLogRecord(**_request_kwargs(outcome=Outcome.ERROR, blocked_by="timeout", target_called=True))
    assert (a.target_called, b.target_called) == (False, True)


@pytest.mark.parametrize("field", ["layer_version", "bundle_version", "chroma_version"])
def test_shadow_record_versions_required_and_non_empty(field):
    kwargs = _shadow_kwargs()
    del kwargs[field]
    with pytest.raises(ValidationError):
        ShadowRecord(**kwargs)
    with pytest.raises(ValidationError):
        ShadowRecord(**_shadow_kwargs(**{field: ""}))
    with pytest.raises(ValidationError):
        ShadowRecord(**_shadow_kwargs(**{field: None}))


@pytest.mark.parametrize("names", [(), ("X0",), ("X0", "X1", "X2"), ("X1", "X2", "X3")])
def test_shadow_full_e2e_rejects_partial_variant_sets(names):
    with pytest.raises(ValidationError):
        ShadowRecord(**_shadow_kwargs(egress_variants=tuple(_variant(n) for n in names)))


def test_shadow_full_e2e_accepts_all_four_in_any_order():
    rec = ShadowRecord(**_shadow_kwargs(egress_variants=tuple(_variant(n) for n in ("X3", "X1", "X0", "X2"))))
    assert len(rec.egress_variants) == 4


def test_shadow_full_e2e_requires_target_called_and_shared_response_and_manifest():
    with pytest.raises(ValidationError):
        ShadowRecord(**_shadow_kwargs(target_called=False))
    diff_hash = (
        _variant("X0"),
        _variant("X1"),
        _variant("X2"),
        EgressVariantRecord(
            response_hash=canonical_hash({"r": "other"}), manifest_id="M1", variant="X3", verdict="block",
            egress_action="blocked",
        ),
    )
    with pytest.raises(ValidationError):
        ShadowRecord(**_shadow_kwargs(egress_variants=diff_hash))
    diff_manifest = (
        _variant("X0"),
        _variant("X1"),
        _variant("X2"),
        EgressVariantRecord(
            response_hash=canonical_hash({"r": "resp"}), manifest_id="M2", variant="X3", verdict="block",
            egress_action="blocked",
        ),
    )
    with pytest.raises(ValidationError):
        ShadowRecord(**_shadow_kwargs(egress_variants=diff_manifest))


@pytest.mark.parametrize(
    "path",
    [
        Path("C:/Users/x/OneDrive/vault.db"),
        Path("C:/Users/x/OneDrive - Org/vault.db"),
        Path("D:/Dropbox/vault.db"),
        Path("D:/Google Drive/vault.db"),
        Path("D:/GoogleDrive/vault.db"),
        Path("/Users/x/iCloudDrive/vault.db"),
        Path("/Users/x/Library/Mobile Documents/com~apple~CloudDocs/vault.db"),
        Path("C:/Users/x/Box/vault.db"),
        Path("C:/Users/x/Box Sync/vault.db"),
    ],
)
def test_synced_folder_component_detects_cloud_folders(path):
    assert contracts.synced_folder_component(path) is not None


@pytest.mark.parametrize(
    "path",
    [
        Path("C:/Users/x/sandbox/vault.db"),
        Path("C:/Users/x/toolbox/vault.db"),
        Path("C:/Users/x/inbox/vault.db"),
        Path("C:/projects/vault-gateway/data/logs/vault.db"),
    ],
)
def test_synced_folder_component_avoids_box_false_positives(path):
    assert contracts.synced_folder_component(path) is None


# Step 3.6: EgressAction masking, strict §3.2 outcome mapping, and contracts/modes.py


@pytest.mark.parametrize("variant", ["X1", "X2", "X3"])
def test_egress_saved_accepts_masked_action(variant):
    rec = RequestLogRecord(
        **_request_kwargs(
            outcome=Outcome.EGRESS_SAVED,
            egress_variant=variant,
            egress_action="masked",
            blocked_by=None,
            target_called=True,
            layer_results=(_layer("L1"), _layer("L4", "allow")),
        )
    )
    assert rec.outcome == Outcome.EGRESS_SAVED
    assert rec.egress_action == "masked"
    assert rec.blocked_by is None


def test_egress_saved_masked_rejects_block_metadata_and_none_action_rejected():
    with pytest.raises(ValidationError):
        RequestLogRecord(
            **_request_kwargs(
                outcome=Outcome.EGRESS_SAVED,
                egress_variant="X1",
                egress_action="masked",
                blocked_by="L4",
                target_called=True,
            )
        )
    with pytest.raises(ValidationError):
        RequestLogRecord(
            **_request_kwargs(
                outcome=Outcome.EGRESS_SAVED,
                egress_variant="X1",
                egress_action="masked",
                target_called=True,
                layer_results=(_layer("L4", "block"),),
            )
        )
    with pytest.raises(ValidationError):
        RequestLogRecord(
            **_request_kwargs(
                outcome=Outcome.EGRESS_SAVED,
                egress_variant="X1",
                egress_action="none",
                blocked_by="L4",
                target_called=True,
            )
        )


def test_egress_variant_record_egress_action_consistency():
    v_none = _variant("X0", "allow")
    assert v_none.egress_action == "none"
    v_masked = _variant("X1", "allow", egress_action="masked")
    assert v_masked.egress_action == "masked"
    v_blocked = _variant("X2", "block", egress_action="blocked")
    assert v_blocked.egress_action == "blocked"
    with pytest.raises(ValidationError):
        _variant("X0", "allow", egress_action="masked")
    with pytest.raises(ValidationError):
        _variant("X0", "block", egress_action="blocked")
    with pytest.raises(ValidationError):
        _variant("X1", "allow", egress_action="blocked")
    with pytest.raises(ValidationError):
        _variant("X1", "block", egress_action="masked")


def test_benign_overmasking_allowed_on_admitted_refused_but_blocked_action_rejected():
    rec = RequestLogRecord(
        **_request_kwargs(
            outcome=Outcome.ADMITTED_REFUSED,
            egress_variant="X1",
            egress_action="masked",
            blocked_by=None,
            target_called=True,
        )
    )
    assert rec.egress_action == "masked"
    with pytest.raises(ValidationError):
        RequestLogRecord(
            **_request_kwargs(
                outcome=Outcome.ADMITTED_REFUSED,
                egress_variant="X1",
                egress_action="blocked",
                blocked_by=None,
                target_called=True,
            )
        )
    with pytest.raises(ValidationError):
        RequestLogRecord(
            **_request_kwargs(
                outcome=Outcome.ADMITTED_REFUSED,
                egress_variant="X0",
                egress_action="masked",
                blocked_by=None,
                target_called=True,
            )
        )


@pytest.mark.parametrize("fault", ["load_shed", "p_closed", "timeout", "upstream_error", "internal_error"])
def test_operational_faults_rejected_under_blocked_ingress_and_accepted_under_error(fault):
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(outcome=Outcome.BLOCKED_INGRESS, blocked_by=fault, target_called=False))
    rec = RequestLogRecord(**_request_kwargs(outcome=Outcome.ERROR, blocked_by=fault, target_called=False))
    assert rec.outcome == Outcome.ERROR and rec.blocked_by == fault


@pytest.mark.parametrize(
    "security_or_cap",
    ["L1", "L2", "L3", "L4", "body_cap", "schema_validation", "decoded_cap", "window_cap", None],
)
def test_layer_blocks_and_caps_rejected_under_error_outcome(security_or_cap):
    for tc in (False, True):
        with pytest.raises(ValidationError):
            RequestLogRecord(**_request_kwargs(outcome=Outcome.ERROR, blocked_by=security_or_cap, target_called=tc))


def test_p_closed_rejects_target_called_true():
    with pytest.raises(ValidationError):
        RequestLogRecord(**_request_kwargs(outcome=Outcome.ERROR, blocked_by="p_closed", target_called=True))


@pytest.mark.parametrize("routing_miss", ["not_found", "method_not_allowed"])
@pytest.mark.parametrize("outcome", list(Outcome))
def test_routing_misses_removed_from_blocked_by(routing_miss, outcome):
    for tc in (False, True):
        with pytest.raises(ValidationError):
            RequestLogRecord(**_request_kwargs(outcome=outcome, blocked_by=routing_miss, target_called=tc))


def test_cache_hit_with_target_called_true_is_valid_across_admitted_outcomes():
    for outcome in (Outcome.SUCCESS, Outcome.ADMITTED_REFUSED):
        rec = RequestLogRecord(**_request_kwargs(outcome=outcome, target_called=True, cache_hit=True))
        assert rec.target_called is True and rec.cache_hit is True
    saved = RequestLogRecord(
        **_request_kwargs(
            outcome=Outcome.EGRESS_SAVED,
            egress_variant="X2",
            egress_action="masked",
            target_called=True,
            cache_hit=True,
        )
    )
    assert saved.target_called is True and saved.cache_hit is True
    shadow = ShadowRecord(**_shadow_kwargs(target_called=True, cache_hit=True))
    assert shadow.target_called is True and shadow.cache_hit is True


def test_contracts_modes_module_is_pure_stdlib_and_matches_spec():
    assert contracts.DETERMINISTIC_MODES == ("eval", "shadow_ingress_only", "shadow_full_e2e")
    assert contracts.RANDOM_MODES == ("production", "demo")
    assert contracts.ALL_GATEWAY_MODES == (
        "eval", "shadow_ingress_only", "shadow_full_e2e", "production", "demo"
    )
    for m in contracts.DETERMINISTIC_MODES:
        assert contracts.resolve_derivation_mode(m) == "deterministic"
    for m in contracts.RANDOM_MODES:
        assert contracts.resolve_derivation_mode(m) == "random"
    with pytest.raises(ValueError):
        contracts.resolve_derivation_mode("unknown")
    imports = _imported_top_levels(CONTRACTS_DIR / "modes.py")
    assert imports <= (set(sys.stdlib_module_names) | {"__future__"})


# Step 4 (Part A2): oracle_result vs outcome, ingress layer evidence, EgressVariantRecord (§3.2)


def test_success_requires_true_oracle_result_when_recorded():
    assert RequestLogRecord(**_request_kwargs(outcome=Outcome.SUCCESS, oracle_result=True)).oracle_result is True
    assert RequestLogRecord(**_request_kwargs(outcome=Outcome.SUCCESS, oracle_result=None)).oracle_result is None
    with pytest.raises(ValidationError, match="Success requires oracle_result"):
        RequestLogRecord(**_request_kwargs(outcome=Outcome.SUCCESS, oracle_result=False))


def test_admitted_refused_requires_false_oracle_result_when_recorded():
    rec = RequestLogRecord(**_request_kwargs(outcome=Outcome.ADMITTED_REFUSED, oracle_result=False))
    assert rec.oracle_result is False
    with pytest.raises(ValidationError, match="Admitted-refused requires oracle_result"):
        RequestLogRecord(**_request_kwargs(outcome=Outcome.ADMITTED_REFUSED, oracle_result=True))


@pytest.mark.parametrize("action", ["masked", "blocked"])
def test_egress_saved_requires_false_oracle_result_when_recorded(action):
    extra = {"blocked_by": "L4"} if action == "blocked" else {}
    base = _request_kwargs(
        outcome=Outcome.EGRESS_SAVED, egress_variant="X2", egress_action=action, target_called=True, **extra
    )
    assert RequestLogRecord(**{**base, "oracle_result": False}).oracle_result is False
    with pytest.raises(ValidationError, match="Egress-saved requires oracle_result"):
        RequestLogRecord(**{**base, "oracle_result": True})


@pytest.mark.parametrize("oracle", [True, False, None])
def test_blocked_ingress_and_error_do_not_constrain_oracle_result(oracle):
    # A blocked or failed trial has no client-visible content; any recorded value is the harness's own label.
    a = RequestLogRecord(**_request_kwargs(outcome=Outcome.ERROR, blocked_by="timeout", oracle_result=oracle))
    b = RequestLogRecord(
        **_request_kwargs(
            outcome=Outcome.BLOCKED_INGRESS, blocked_by="body_cap", target_called=False, oracle_result=oracle
        )
    )
    assert a.oracle_result is b.oracle_result is oracle


@pytest.mark.parametrize("layer", ["L1", "L2", "L3"])
def test_blocked_ingress_layer_requires_matching_block_evidence(layer):
    ok = RequestLogRecord(
        **_request_kwargs(
            outcome=Outcome.BLOCKED_INGRESS, blocked_by=layer, target_called=False,
            layer_results=(_layer("L1"), _layer(layer, "block")) if layer != "L1" else (_layer("L1", "block"),),
        )
    )
    assert ok.blocked_by == layer
    others = [name for name in ("L1", "L2", "L3") if name != layer]
    contradictions = [
        (),  # no evidence at all
        (_layer(layer, "allow"),),  # the named layer allowed
        (_layer(layer, "timeout"),),  # the named layer timed out (that is an Error, not a block)
        (_layer(layer, "error"),),
        tuple(_layer(name, "block") for name in others),  # a different layer blocked
        (_layer("L4", "block"),),
    ]
    for layers in contradictions:
        with pytest.raises(ValidationError, match="requires a L[123] LayerResult"):
            RequestLogRecord(
                **_request_kwargs(
                    outcome=Outcome.BLOCKED_INGRESS, blocked_by=layer, target_called=False, layer_results=layers
                )
            )


@pytest.mark.parametrize("cap", ["body_cap", "schema_validation", "decoded_cap", "window_cap"])
def test_blocked_ingress_input_caps_need_no_layer_evidence(cap):
    rec = RequestLogRecord(
        **_request_kwargs(outcome=Outcome.BLOCKED_INGRESS, blocked_by=cap, target_called=False, layer_results=())
    )
    assert rec.layer_results == ()


def test_blocked_ingress_evidence_matches_any_view():
    decoded_block = LayerResult(**_valid_kwargs(layer="L2", view="decoded_view", verdict="block"))
    rec = RequestLogRecord(
        **_request_kwargs(
            outcome=Outcome.BLOCKED_INGRESS, blocked_by="L2", target_called=False,
            layer_results=(_layer("L1"), LayerResult(**_valid_kwargs(layer="L2", verdict="allow")), decoded_block),
        )
    )
    assert rec.blocked_by == "L2"


@pytest.mark.parametrize("action", ["none", "masked"])
@pytest.mark.parametrize("variant", ["X1", "X2", "X3"])
def test_egress_variant_block_verdict_requires_blocked_action(variant, action):
    with pytest.raises(ValidationError, match="verdict='block' requires egress_action='blocked'"):
        EgressVariantRecord(
            response_hash="a" * 64, manifest_id="M1", variant=variant, verdict="block", egress_action=action
        )
    ok = EgressVariantRecord(
        response_hash="a" * 64, manifest_id="M1", variant=variant, verdict="block", egress_action="blocked"
    )
    assert ok.egress_action == "blocked"


@pytest.mark.parametrize(
    ("verdict", "action"),
    [
        ("block", "blocked"), ("block", "none"), ("block", "masked"),
        ("timeout", "none"), ("error", "none"), ("allow", "masked"), ("allow", "blocked"),
    ],
)
def test_x0_runs_no_scan_so_only_allow_none_is_valid(verdict, action):
    with pytest.raises(ValidationError, match="X0 performs no egress scan"):
        EgressVariantRecord(response_hash="a" * 64, manifest_id="M1", variant="X0", verdict=verdict, egress_action=action)
    ok = EgressVariantRecord(response_hash="a" * 64, manifest_id="M1", variant="X0", verdict="allow")
    assert (ok.verdict, ok.egress_action) == ("allow", "none")
