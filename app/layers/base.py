"""Ingress layer plug-in interface (IP-3, IP-9, §5.2 layer interface: verdict, score and latency per view).

An ingress layer reads one frozen `InspectionBundle` and returns one frozen
`contracts.LayerResult` per view it scored (`raw_view`, `decoded_view`). The
pipeline (`app/pipeline.py`) calls `inspect` in a worker thread (IP-7), checks
the measured latency against the layer's TM-R2 budget, caches allow and block
results under the IP-5 layer-verdict key, and applies the TM-R3 policy to
timeouts and errors.

`version` is the layer version in the IP-5 key. `chroma_version` is the
ChromaDB collection version and must be None for every layer except L2 (the
key builder enforces this).

`degraded=True` asks the layer to decide with tightened thresholds (TM-R3
P-degrade, L2 threshold lowered by `P_DEGRADE_THRESHOLD_DELTA`); layers
without a threshold ignore it.

`DisabledLayer` stands in for a layer switched off by configuration
(`L2_ENABLED=false`, `L3_ENABLED=false`). It never scores anything and the
pipeline skips it, so no detector is faked: a request admitted with a layer
disabled carries no evidence from that layer in its log record.

Optional extensions (Step 5B, additive; the frozen `IngressLayer` protocol is
unchanged):

- `PinnableLayer.pin()` returns an `IngressLayer` bound to the state read at
  that moment. L2 uses it so each request reads the ChromaDB active-version
  pointer once and uses that exact version for its queries and its IP-5 key
  (IP-7). The pipeline pins every pinnable layer at request start.
- `TightenableLayer.tighten(results)` re-decides completed results with the
  TM-R3 P-degrade thresholds (L2 threshold lowered by
  `P_DEGRADE_THRESHOLD_DELTA`) from the scores alone, without re-running the
  model.
- `Deadline` lets a windowed layer stop between windows once its TM-R2 budget
  is spent by raising `LayerBudgetExceeded`, which the pipeline records as a
  `timeout`. A stopped call frees its pool thread early (IP-7); its partial
  scores are never returned or cached. Windowed layers check a window's block
  before the deadline, so a block found in time is never lost (Step 5B.1 M-2).
- `view_results` turns window scores into per-view results for the windowed
  layers (L2, L3), including the production early-stop case.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from contracts.inspection import InspectionBundle, TextWindow
from contracts.layer_result import LayerResult


class LayerBudgetExceeded(Exception):
    """A layer stopped between windows because its TM-R2 budget was spent (recorded as `timeout`)."""


class Deadline:
    """Cooperative TM-R2 budget check inside one layer call. `budget_ms=None` never expires."""

    def __init__(self, budget_ms: float | None) -> None:
        self.budget_ms = budget_ms
        self.started = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0

    def check(self) -> None:
        if self.budget_ms is not None and self.elapsed_ms() > self.budget_ms:
            raise LayerBudgetExceeded(f"over the {self.budget_ms:g} ms budget after {self.elapsed_ms():.1f} ms")


@runtime_checkable
class IngressLayer(Protocol):
    """One ingress layer (L1, L2 or L3)."""

    name: str
    version: str
    enabled: bool

    @property
    def chroma_version(self) -> str | None: ...

    def inspect(self, bundle: InspectionBundle, *, degraded: bool = False) -> tuple[LayerResult, ...]: ...


@runtime_checkable
class PinnableLayer(Protocol):
    """A layer whose verdict depends on mutable state (L2: the active ChromaDB version)."""

    def pin(self) -> IngressLayer: ...


@runtime_checkable
class TightenableLayer(Protocol):
    """A layer whose completed results can be re-decided with P-degrade thresholds (TM-R3)."""

    def tighten(self, results: tuple[LayerResult, ...]) -> tuple[LayerResult, ...]: ...


def pin_layer(layer: IngressLayer) -> IngressLayer:
    """The request-scoped view of `layer` (IP-7): pinned when it supports pinning, else the layer itself."""
    return layer.pin() if isinstance(layer, PinnableLayer) and layer.enabled else layer


def view_results(
    layer: str,
    views: tuple[tuple[str, tuple[TextWindow, ...]], ...],
    scores: Mapping[str, float],
    threshold: float,
    latency_ms: float,
) -> tuple[LayerResult, ...]:
    """Per-view results of a windowed layer from its window scores (keyed by window text).

    A view is reported when every one of its windows was scored, or when the
    scored windows already reach `threshold` (the production early stop, IP-3):
    a view left incomplete without a block is not reported, so no partial scan
    is ever logged as `allow`.
    """
    results = []
    for view, windows in views:
        per_window = tuple(scores[w.text] for w in windows if w.text in scores)
        if not per_window:
            continue
        overall = max(per_window)
        blocks = overall >= threshold
        if blocks or len(per_window) == len(windows):
            results.append(
                LayerResult(
                    layer=layer,
                    view=view,
                    per_window_scores=per_window,
                    overall_score=overall,
                    verdict="block" if blocks else "allow",
                    latency_ms=latency_ms,
                )
            )
    return tuple(results)


@dataclass(frozen=True)
class DisabledLayer:
    """Placeholder for a layer switched off by configuration. The pipeline skips it."""

    name: str
    version: str = "disabled"
    enabled: bool = False

    @property
    def chroma_version(self) -> str | None:
        return None

    def inspect(self, bundle: InspectionBundle, *, degraded: bool = False) -> tuple[LayerResult, ...]:
        raise RuntimeError(f"{self.name} is disabled by configuration and must not be called")
