"""Ingress timeout and error policy (TM-R3), applied by `app/pipeline.py`.

A layer whose call times out (measured latency above its TM-R2 budget, or a
hung call) or raises gives a `timeout` or `error` `LayerResult` instead of a
verdict. The configured `TIMEOUT_POLICY` then decides:

- `P-open`: admit the request.
- `P-degrade` (default): decide from the layers that completed; the layers
  that run after the fault use tightened thresholds (L2 threshold lowered by
  `P_DEGRADE_THRESHOLD_DELTA`). If no layer completed, behave as `P-closed`.
- `P-closed`: refuse with the uniform 503, logged as `Error` / `p_closed`.

A block verdict from any completed layer always wins and is handled before
this policy is consulted. Egress is never governed here: an L4 timeout or
error always fails closed (TM-R4).
"""

from __future__ import annotations

from typing import Literal

TimeoutPolicy = Literal["P-open", "P-degrade", "P-closed"]
TIMEOUT_POLICIES: tuple[str, ...] = ("P-open", "P-degrade", "P-closed")


def _check(policy: str) -> None:
    if policy not in TIMEOUT_POLICIES:
        raise ValueError(f"unknown TIMEOUT_POLICY {policy!r}")


def tighten_after_fault(policy: str, faulted: bool) -> bool:
    """True when later layers must use tightened thresholds (P-degrade after a fault)."""
    _check(policy)
    return faulted and policy == "P-degrade"


def admit_after_faults(policy: str, *, completed_layers: int, faulted_layers: int) -> bool:
    """Whether a request with no block verdict is admitted, given how many layers completed or faulted."""
    _check(policy)
    if completed_layers < 0 or faulted_layers < 0:
        raise ValueError("layer counts must be non-negative")
    if faulted_layers == 0:
        return True
    if policy == "P-open":
        return True
    if policy == "P-closed":
        return False
    return completed_layers > 0
