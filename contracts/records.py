"""Frozen telemetry record schemas (RT-V6, IP-4, IP-9, §3.2).

`RequestLogRecord` carries the RT-V6 accounting fields plus gateway fields.
Evaluation-only fields (`trial_id`, `objective_id`, `tier`, `config`, `seed`,
`query_index`, `operator_chain`, `oracle_result`, `judge_score`) default to
`None` so production and demo traffic can omit them.

`ShadowRecord` carries the IP-4 shadow-mode fields: L1, L2, L3 results with no
short-circuit, and for `shadow_full_e2e` the per-variant egress verdicts (X0 to
X3) with the oracle result on each variant's client-visible output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from contracts.layer_result import FiniteFloat, LayerResult, NonNegativeFiniteFloat, Verdict
from contracts.outcome import Outcome

GatewayMode = Literal["production", "demo", "eval", "shadow_ingress_only", "shadow_full_e2e"]
ShadowTier = Literal["shadow_ingress_only", "shadow_full_e2e"]
EgressVariant = Literal["X0", "X1", "X2", "X3"]


class RequestLogRecord(BaseModel):
    """One logged gateway request (RT-V6)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Gateway fields, always present.
    timestamp_utc: str
    gateway_mode: GatewayMode
    prompt_hash: str
    layer_results: tuple[LayerResult, ...]
    target_called: bool
    outcome: Outcome
    total_latency_ms: NonNegativeFiniteFloat
    reasoning_findings: tuple[str, ...] = ()

    # RT-V6 evaluation fields, None outside evaluation traffic.
    trial_id: str | None = None
    objective_id: str | None = None
    tier: str | None = None
    config: str | None = None
    seed: int | None = None
    query_index: int | None = None
    operator_chain: tuple[str, ...] | None = None
    oracle_result: bool | None = None
    judge_score: FiniteFloat | None = None


class EgressVariantRecord(BaseModel):
    """Egress verdict for one variant on one response (IP-4 shadow_full_e2e)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: EgressVariant
    verdict: Verdict
    oracle_result: bool | None = None


class ShadowRecord(BaseModel):
    """One shadow-mode (evaluate-all) record (IP-4, IP-5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp_utc: str
    shadow_tier: ShadowTier
    prompt_hash: str
    layer_results: tuple[LayerResult, ...]
    total_latency_ms: NonNegativeFiniteFloat
    target_called: bool = False
    cache_hit: bool | None = None
    egress_variants: tuple[EgressVariantRecord, ...] = ()

    # Versions that determine an ingress verdict (IP-5).
    layer_version: str | None = None
    bundle_version: str | None = None
    chroma_version: str | None = None

    trial_id: str | None = None
    objective_id: str | None = None

    @model_validator(mode="after")
    def _ingress_only_has_no_target_or_egress(self) -> ShadowRecord:
        if self.shadow_tier == "shadow_ingress_only" and (self.target_called or self.egress_variants):
            raise ValueError("shadow_ingress_only makes zero Groq calls and has no egress variants (IP-4)")
        variants = [v.variant for v in self.egress_variants]
        if len(set(variants)) != len(variants):
            raise ValueError("egress_variants must not repeat a variant")
        return self
