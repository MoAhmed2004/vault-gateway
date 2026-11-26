"""Unit tests for the ingress timeout and error policy (TM-R3). End-to-end cases are in tests/test_gateway_shell.py."""

from __future__ import annotations

import pytest

from app.timeout_policy import TIMEOUT_POLICIES, admit_after_faults, tighten_after_fault


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
