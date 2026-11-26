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

`DisabledLayer` is the Step 5B placeholder for L2 (ChromaDB and MiniLM) and
L3 (DeBERTa ONNX). It never scores anything and the pipeline skips it, so no
detector is faked: a request admitted with L2 and L3 disabled carries only L1
evidence in its log record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from contracts.inspection import InspectionBundle
from contracts.layer_result import LayerResult


@runtime_checkable
class IngressLayer(Protocol):
    """One ingress layer (L1, L2 or L3)."""

    name: str
    version: str
    enabled: bool

    @property
    def chroma_version(self) -> str | None: ...

    def inspect(self, bundle: InspectionBundle, *, degraded: bool = False) -> tuple[LayerResult, ...]: ...


@dataclass(frozen=True)
class DisabledLayer:
    """Placeholder for a layer that is not implemented yet (Step 5B: L2, L3). The pipeline skips it."""

    name: str
    version: str = "disabled"
    enabled: bool = False

    @property
    def chroma_version(self) -> str | None:
        return None

    def inspect(self, bundle: InspectionBundle, *, degraded: bool = False) -> tuple[LayerResult, ...]:
        raise RuntimeError(f"{self.name} is disabled until Step 5B and must not be called")
