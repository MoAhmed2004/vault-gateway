"""Frozen telemetry record schemas (RT-V6, IP-4, IP-5, IP-9, §2.5, §3.2).

`RequestLogRecord` carries the RT-V6 accounting fields plus gateway fields.
Evaluation-only fields (`trial_id`, `objective_id`, `tier`, `config`, `seed`,
`query_index`, `operator_chain`, `oracle_result`, `judge_score`) default to
`None` so production and demo traffic can omit them.

`target_called` semantics (§3.2, IP-4, IP-5): `target_called=True` means the
request was admitted past ingress to the target stage, EVEN when the target
response is served from cache (`cache_hit=True`). `cache_hit` records whether
the target output came from the response cache (`True`, zero upstream API calls)
or a live upstream provider call (`False`).

`Outcome` vs `blocked_by` (§3.2):
- Security layer blocks (`L1`, `L2`, `L3`) and input/view/window caps
  (`body_cap`, `schema_validation`, `decoded_cap`, `window_cap`) map ONLY to
  `Outcome.BLOCKED_INGRESS` (`target_called=False`).
- Operational and policy faults (`load_shed`, `p_closed`, `timeout`,
  `upstream_error`, `internal_error`) map ONLY to `Outcome.ERROR`, so overload
  shedding and timeouts never inflate block rate.
- HTTP routing misses (404/405) are not gateway attack queries and are excluded
  from `BlockedBy`.
- `Outcome.EGRESS_SAVED` requires `target_called=True`, `egress_variant` in
  `X1..X3`, and `egress_action` in `("masked", "blocked")` (§3.2 "L4 blocked or
  masked the result", §2.5 egress over-masking).
- Layer evidence: `Blocked-ingress` with `blocked_by` in `L1..L3` requires a
  `LayerResult` for that exact layer with `verdict="block"`, mirroring the L4
  evidence rule for a blocked `Egress-saved`.
- `oracle_result`, when recorded, must agree with the outcome: `Success`
  requires `True`; `Admitted-refused` and `Egress-saved` require `False`.
- `EgressVariantRecord`: `verdict="block"` if and only if
  `egress_action="blocked"`; `X0` runs no scan, so it is always
  `verdict="allow"` with `egress_action="none"`.

`ShadowRecord` carries the IP-4 shadow-mode fields: L1, L2, L3 results with no
short-circuit, and for `shadow_full_e2e` the per-variant egress verdicts (X0 to
X3) with `egress_action` and the oracle result on each variant's client-visible
output.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from contracts.hashing import canonical_hash
from contracts.layer_result import FiniteFloat, LayerResult, NonNegativeFiniteFloat, Verdict
from contracts.modes import GatewayMode
from contracts.outcome import Outcome

ShadowTier = Literal["shadow_ingress_only", "shadow_full_e2e"]
EgressVariant = Literal["X0", "X1", "X2", "X3"]
EgressAction = Literal["none", "masked", "blocked"]
TimeoutPolicyName = Literal["P-open", "P-degrade", "P-closed"]

# L4 is the single egress stage label (spec layers are named L1 to L4).
EGRESS_STAGE = "L4"
INGRESS_LAYERS = frozenset({"L1", "L2", "L3"})
INGRESS_BLOCK_REASONS = frozenset(
    {"L1", "L2", "L3", "body_cap", "schema_validation", "decoded_cap", "window_cap"}
)
ERROR_FAULT_REASONS = frozenset(
    {"load_shed", "p_closed", "timeout", "upstream_error", "internal_error"}
)
BlockedBy = Literal[
    "L1", "L2", "L3", "L4",
    "body_cap", "schema_validation", "decoded_cap", "window_cap",
    "load_shed", "p_closed", "timeout", "upstream_error", "internal_error",
]
ALL_EGRESS_VARIANTS = frozenset({"X0", "X1", "X2", "X3"})

# `prompt_hash` of a request refused before a prompt was parsed (body cap, schema
# violation, or a fault before parsing). A fixed, recognizable digest, distinct
# from the hash of any real raw view (those hash {"raw_view": <str>}).
UNPARSED_PROMPT_HASH = canonical_hash({"raw_view": None})

# SHA-256 digests: exactly 64 lowercase hex characters.
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class RequestLogRecord(BaseModel):
    """One logged gateway request (RT-V6, §3.2).

    `target_called=True` means admitted to the target stage (including response
    cache hits); `cache_hit` records whether an upstream API call was avoided.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, protected_namespaces=())

    # Gateway fields, always present.
    timestamp_utc: str
    gateway_mode: GatewayMode
    prompt_hash: Sha256Hex
    layer_results: tuple[LayerResult, ...]
    target_called: bool
    outcome: Outcome
    total_latency_ms: NonNegativeFiniteFloat
    reasoning_findings: tuple[str, ...] = ()

    # Policy and egress context (None when not applicable; egress_action defaults to "none").
    blocked_by: BlockedBy | None = None
    egress_variant: EgressVariant | None = None
    egress_action: EgressAction = "none"
    timeout_policy: TimeoutPolicyName | None = None
    cache_hit: bool | None = None
    model_id: str | None = None

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

    @model_validator(mode="after")
    def _outcome_is_consistent(self) -> RequestLogRecord:
        if self.oracle_result is not None:
            # §3.2: Success means the oracle fired on the client-visible response;
            # Admitted-refused and Egress-saved mean it did not.
            if self.outcome == Outcome.SUCCESS and self.oracle_result is not True:
                raise ValueError("Success requires oracle_result to be True when it is recorded")
            if self.outcome in (Outcome.ADMITTED_REFUSED, Outcome.EGRESS_SAVED) and self.oracle_result is not False:
                raise ValueError(f"{self.outcome.value} requires oracle_result to be False when it is recorded")
        if self.egress_variant == "X0" and self.egress_action != "none":
            raise ValueError("X0 performs no egress scan; egress_action must be 'none'")
        if not self.target_called and self.egress_action != "none":
            raise ValueError("egress_action must be 'none' when target_called is False")

        if self.outcome in (Outcome.SUCCESS, Outcome.ADMITTED_REFUSED):
            if self.blocked_by is not None:
                raise ValueError(f"{self.outcome.value} requires blocked_by to be None")
            if not self.target_called:
                raise ValueError(f"{self.outcome.value} requires target_called to be True")
            if self.egress_action == "blocked":
                raise ValueError(f"{self.outcome.value} cannot have egress_action='blocked'")
            if self.egress_action == "masked" and self.egress_variant not in ("X1", "X2", "X3"):
                raise ValueError("egress_action='masked' requires egress_variant in X1, X2, X3")
        elif self.outcome == Outcome.BLOCKED_INGRESS:
            if self.blocked_by not in INGRESS_BLOCK_REASONS:
                raise ValueError(
                    "Blocked-ingress requires a security layer or input-cap blocked_by "
                    f"({sorted(INGRESS_BLOCK_REASONS)}), got {self.blocked_by!r}"
                )
            if self.target_called:
                raise ValueError("Blocked-ingress requires target_called to be False")
            if self.blocked_by in INGRESS_LAYERS and not any(
                r.layer == self.blocked_by and r.verdict == "block" for r in self.layer_results
            ):
                raise ValueError(
                    f"Blocked-ingress with blocked_by={self.blocked_by!r} requires a {self.blocked_by} "
                    "LayerResult with verdict='block' in layer_results"
                )
        elif self.outcome == Outcome.EGRESS_SAVED:
            if not self.target_called:
                raise ValueError("Egress-saved requires target_called to be True")
            if self.egress_variant not in ("X1", "X2", "X3"):
                raise ValueError("Egress-saved requires egress_variant in X1, X2, X3 (X0 performs no egress scan)")
            if self.egress_action not in ("masked", "blocked"):
                raise ValueError("Egress-saved requires egress_action to be 'masked' or 'blocked' (§3.2)")
            l4_block = any(r.layer == EGRESS_STAGE and r.verdict == "block" for r in self.layer_results)
            if self.egress_action == "masked":
                if self.blocked_by is not None or l4_block:
                    raise ValueError("masked Egress-saved requires blocked_by=None and no L4 block verdict")
            else:
                if self.blocked_by not in (None, EGRESS_STAGE):
                    raise ValueError("blocked Egress-saved allows only blocked_by None or L4")
                if self.blocked_by != EGRESS_STAGE and not l4_block:
                    raise ValueError("blocked Egress-saved requires L4 egress blocking evidence")
        elif self.outcome == Outcome.ERROR:
            if self.blocked_by not in ERROR_FAULT_REASONS:
                raise ValueError(
                    "Error requires an operational or policy fault blocked_by "
                    f"({sorted(ERROR_FAULT_REASONS)}), got {self.blocked_by!r}"
                )
            if self.blocked_by == "p_closed" and self.target_called:
                raise ValueError("p_closed is an ingress policy fault and requires target_called to be False")
            if self.egress_action != "none":
                raise ValueError("Error requires egress_action to be 'none'")
        return self


class EgressVariantRecord(BaseModel):
    """Egress verdict and action for one variant on one response (IP-4 shadow_full_e2e, §2.5)."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    response_hash: Sha256Hex
    manifest_id: NonEmptyStr
    variant: EgressVariant
    verdict: Verdict
    egress_action: EgressAction = "none"
    oracle_result: bool | None = None

    @model_validator(mode="after")
    def _variant_action_is_consistent(self) -> EgressVariantRecord:
        if self.variant == "X0" and (self.verdict != "allow" or self.egress_action != "none"):
            raise ValueError("X0 performs no egress scan; verdict must be 'allow' and egress_action must be 'none'")
        if self.egress_action == "blocked" and self.verdict != "block":
            raise ValueError("egress_action='blocked' requires verdict='block'")
        if self.verdict == "block" and self.egress_action != "blocked":
            raise ValueError("verdict='block' requires egress_action='blocked'")
        if self.egress_action == "masked" and self.verdict != "allow":
            raise ValueError("egress_action='masked' requires verdict='allow'")
        return self


class ShadowRecord(BaseModel):
    """One shadow-mode (evaluate-all) record (IP-4, IP-5).

    In `shadow_full_e2e`, `target_called=True` means admitted to the target
    stage even on a response-cache hit (`cache_hit=True`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    timestamp_utc: str
    shadow_tier: ShadowTier
    prompt_hash: Sha256Hex
    layer_results: tuple[LayerResult, ...]
    total_latency_ms: NonNegativeFiniteFloat
    target_called: bool = False
    cache_hit: bool | None = None
    egress_variants: tuple[EgressVariantRecord, ...] = ()

    # Versions that determine an ingress verdict (IP-5).
    layer_version: NonEmptyStr
    bundle_version: NonEmptyStr
    chroma_version: NonEmptyStr

    trial_id: str | None = None
    objective_id: str | None = None

    @model_validator(mode="after")
    def _ingress_only_has_no_target_or_egress(self) -> ShadowRecord:
        if self.shadow_tier == "shadow_ingress_only" and (self.target_called or self.egress_variants):
            raise ValueError("shadow_ingress_only makes zero Groq calls and has no egress variants (IP-4)")
        variants = [v.variant for v in self.egress_variants]
        if len(set(variants)) != len(variants):
            raise ValueError("egress_variants must not repeat a variant")
        if self.shadow_tier == "shadow_full_e2e":
            if set(variants) != ALL_EGRESS_VARIANTS:
                raise ValueError("shadow_full_e2e requires all four egress variants X0, X1, X2, X3 (IP-4)")
            if not self.target_called:
                raise ValueError("shadow_full_e2e requires target_called to be True")
            if len({v.response_hash for v in self.egress_variants}) != 1:
                raise ValueError("shadow_full_e2e variants must share one response_hash")
            if len({v.manifest_id for v in self.egress_variants}) != 1:
                raise ValueError("shadow_full_e2e variants must share one manifest_id")
        return self
