"""Unit tests for the ingress timeout and error policy (TM-R3).

The policy functions are tested directly; the pipeline cases below drive the
real L2 layer (fake embedder, real ChromaDB) with a slow L3 or L1 to check
P-open, P-degrade (L2 threshold lowered by 0.05 when deciding from completed
layers, P-closed when no layer completed) and P-closed end to end. Single-
layer cases are in tests/test_gateway_shell.py.
"""

from __future__ import annotations

import time

import pytest

from app.timeout_policy import TIMEOUT_POLICIES, admit_after_faults, tighten_after_fault
from contracts import Outcome
from contracts.layer_result import LayerResult
from tests.fakes import FakeClassifier, FakeEmbedder
from tests.test_gateway_shell import _SlowL1, assert_uniform
from tests.test_pipeline_modes import L2_SEED, L3_TRIGGER, Stack

# L2_SEED has 10 distinct words. Adding k distinct words gives the fake embedder's cosine sqrt(10 / (10 + k)).
IN_BAND = L2_SEED + " alpha bravo charlie delta echo foxtrot"  # k=6: 0.791, in [0.75, 0.80)
BELOW_BAND = IN_BAND + " golf hotel india juliet kilo"  # k=11: 0.690, below 0.75


def slow_l3() -> FakeClassifier:
    return FakeClassifier(triggers=(L3_TRIGGER,), delay_s=0.06)


def l2_scores(record) -> float:
    return max(r.overall_score for r in record.layer_results if r.layer == "L2")


@pytest.mark.parametrize("policy", TIMEOUT_POLICIES)
def test_no_fault_always_admits(policy):
    assert admit_after_faults(policy, completed_layers=1, faulted_layers=0)
    assert admit_after_faults(policy, completed_layers=0, faulted_layers=0)


@pytest.mark.parametrize(
    ("policy", "completed", "expected"),
    [
        ("P-open", 0, True),
        ("P-open", 2, True),
        ("P-closed", 0, False),
        ("P-closed", 2, False),
        ("P-degrade", 0, False),  # no layer completed: behave as P-closed
        ("P-degrade", 1, True),  # decide from the completed layers
    ],
)
def test_policy_after_a_fault(policy, completed, expected):
    assert admit_after_faults(policy, completed_layers=completed, faulted_layers=1) is expected


def test_only_p_degrade_tightens_later_layers_and_only_after_a_fault():
    assert tighten_after_fault("P-degrade", True) is True
    assert tighten_after_fault("P-degrade", False) is False
    assert tighten_after_fault("P-open", True) is False
    assert tighten_after_fault("P-closed", True) is False


def test_unknown_policy_and_negative_counts_are_errors():
    with pytest.raises(ValueError):
        admit_after_faults("P-maybe", completed_layers=0, faulted_layers=1)
    with pytest.raises(ValueError):
        tighten_after_fault("p-open", True)
    with pytest.raises(ValueError):
        admit_after_faults("P-open", completed_layers=-1, faulted_layers=0)


# Pipeline: L3 times out after L1 and L2 completed


def test_p_degrade_lowers_the_l2_threshold_by_0_05_when_deciding_from_completed_layers(tmp_path):
    stack = Stack(tmp_path, classifier=slow_l3(), TIMEOUT_POLICY="P-degrade", L3_BUDGET_MS=10)
    (response,) = stack.chat(IN_BAND)
    assert_uniform(response)
    (record,) = stack.sink.records
    assert 0.75 <= l2_scores(record) < 0.80  # allowed at 0.80, blocked at 0.75
    assert (record.outcome, record.blocked_by, record.timeout_policy) == (Outcome.BLOCKED_INGRESS, "L2", "P-degrade")
    assert [(r.layer, r.verdict) for r in record.layer_results if r.layer != "L1"] == [
        ("L2", "block"), ("L2", "block"), ("L3", "timeout"),
    ]
    assert stack.fake.chat_requests == []


def test_p_degrade_admits_when_the_completed_layers_allow_under_the_tightened_threshold(tmp_path):
    stack = Stack(tmp_path, classifier=slow_l3(), TIMEOUT_POLICY="P-degrade", L3_BUDGET_MS=10)
    (response,) = stack.chat(BELOW_BAND)
    assert response.status_code == 200
    (record,) = stack.sink.records
    assert l2_scores(record) < 0.75 and record.outcome == Outcome.SUCCESS
    assert any(r.layer == "L3" and r.verdict == "timeout" for r in record.layer_results)


@pytest.mark.parametrize(("policy", "admitted"), [("P-open", True), ("P-closed", False)])
def test_p_open_and_p_closed_do_not_tighten(tmp_path, policy, admitted):
    stack = Stack(tmp_path, classifier=slow_l3(), TIMEOUT_POLICY=policy, L3_BUDGET_MS=10)
    (response,) = stack.chat(IN_BAND)
    (record,) = stack.sink.records
    assert all(r.verdict == "allow" for r in record.layer_results if r.layer == "L2")  # threshold stays 0.80
    if admitted:
        assert response.status_code == 200 and record.outcome == Outcome.SUCCESS
    else:
        assert_uniform(response)
        assert (record.outcome, record.blocked_by, record.target_called) == (Outcome.ERROR, "p_closed", False)


def test_p_degrade_behaves_as_p_closed_when_no_layer_completed(tmp_path):
    stack = Stack(
        tmp_path, embedder=FakeEmbedder(delay_s=0.08), classifier=slow_l3(), extra_layers=(_SlowL1(),),
        TIMEOUT_POLICY="P-degrade", L2_BUDGET_MS=10, L3_BUDGET_MS=10,
    )
    (response,) = stack.chat("What is the capital of France?")
    assert_uniform(response)
    (record,) = stack.sink.records
    assert [r.layer for r in record.layer_results] == ["L1", "L2", "L3"]
    assert {r.verdict for r in record.layer_results} == {"timeout"}
    assert (record.outcome, record.blocked_by) == (Outcome.ERROR, "p_closed")


class _LateBlockL1:
    """An L1 stand-in that blocks after 20 ms: over the 5 ms budget, inside the 50 ms hang grace."""

    name, version, enabled = "L1", "late-block-test-layer", True
    chroma_version = None

    def inspect(self, bundle, *, degraded=False):
        time.sleep(0.02)
        return tuple(
            LayerResult(layer="L1", view=view, per_window_scores=(1.0,), overall_score=1.0, verdict="block", latency_ms=20.0)
            for view in ("raw_view", "decoded_view")
        )


@pytest.mark.parametrize("policy", TIMEOUT_POLICIES)
def test_a_block_that_finishes_over_budget_is_kept_under_every_policy(tmp_path, policy):
    """M-2 (TM-R3): a late block is never downgraded to a timeout, so P-open cannot admit it."""
    stack = Stack(tmp_path, extra_layers=(_LateBlockL1(),), TIMEOUT_POLICY=policy)
    (response,) = stack.chat("What is the capital of France?")
    assert_uniform(response)
    (record,) = stack.sink.records
    assert (record.outcome, record.blocked_by, record.target_called) == (Outcome.BLOCKED_INGRESS, "L1", False)
    assert [(r.layer, r.verdict) for r in record.layer_results] == [("L1", "block"), ("L1", "block")]
    assert all("over the 5 ms budget" in r.error_message and "block kept" in r.error_message for r in record.layer_results)
    assert stack.fake.chat_requests == []


def test_a_late_allow_is_still_a_timeout(tmp_path):
    stack = Stack(tmp_path, extra_layers=(_SlowL1(),), TIMEOUT_POLICY="P-open")
    stack.chat("What is the capital of France?")
    (record,) = stack.sink.records
    assert record.layer_results[0].verdict == "timeout" and record.outcome == Outcome.SUCCESS


def test_after_an_l1_fault_p_degrade_runs_l2_with_the_tightened_threshold(tmp_path):
    stack = Stack(tmp_path, extra_layers=(_SlowL1(),), TIMEOUT_POLICY="P-degrade")
    (response,) = stack.chat(IN_BAND)
    assert_uniform(response)
    (record,) = stack.sink.records
    assert record.layer_results[0].verdict == "timeout"
    assert (record.outcome, record.blocked_by) == (Outcome.BLOCKED_INGRESS, "L2")
    assert not any(r.layer == "L3" for r in record.layer_results)  # the tightened L2 block short-circuits L3
