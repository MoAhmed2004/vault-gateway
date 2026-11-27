"""Request pipeline for `POST /v1/chat` (IP-1, IP-3, IP-4, IP-5, IP-6, IP-7, TM-R2, TM-R3, TM-R4, TM-R6, RT-V6, §3.2).

Request start (IP-7): every pinnable layer is pinned once, so L2 reads the
ChromaDB active-version pointer exactly once and uses that version for all
its scores and its IP-5 layer-verdict key. All CPU work (bundle build, L1 to
L4) runs on the bounded layer pool (`app/executor.py`); when the pool is full
the request is refused at once as `Error` / `load_shed` with the uniform 503.

Production and `eval` (IP-3):

1. InspectionBundle (IP-1, IP-2); a decoded-view or window cap breach is
   `Blocked-ingress` (`decoded_cap`, `window_cap`). Sub-window splits are
   logged, never counted as truncation. The bundle build runs on the pool
   with no budget of its own; it is not a TM-R2 layer (see Budgets below).
2. Ingress layers L1, L2, L3 in order, short-circuiting on the first block
   (`Blocked-ingress`, `blocked_by` the layer). A layer switched off by
   configuration is a `DisabledLayer` and is skipped, never faked. Allow and
   block results are cached under the IP-5 layer-verdict key and replayed.
   L2 and L3 stop scanning at the first window that blocks (early stop).
3. TM-R3 on timeouts and errors: P-open admits; P-closed refuses
   (`Error` / `p_closed`); P-degrade runs the layers after a fault with the
   tightened L2 threshold, then re-decides every completed tightenable layer
   (L2) from its scores with the threshold lowered by
   `P_DEGRADE_THRESHOLD_DELTA`, and behaves as P-closed if no layer completed.
   Tightened verdicts are never cached. A layer that finishes over its budget
   with a `block` keeps the block (Step 5B.1 M-2): a block is never turned
   into a timeout, so it blocks under every policy, P-open included.
4. `aegis_context` documents need the TM-S1 to TM-S5 sanitizer, which is not
   implemented yet, so a request that carries documents fails closed
   (`Error` / `internal_error`, target not called) [Project Decision].
5. Target: the gateway-built `[system, user]` message list (PS-4, CA-3), the
   raw view forwarded untouched (IP-1), through the IP-5 target-response cache
   and the single Groq client module. `target_called=True` from here on.
6. L4 egress (IP-6) in `EGRESS_VARIANT`, through the IP-5 egress cache. An L4
   timeout or error always fails closed (TM-R4): `Error` / `timeout` or
   `internal_error`, uniform 503.

Post-target guard (TM-R6, RT-V6; Step 5B.1 LM-1): any unexpected exception
after the target call returned (cache write, L4 bookkeeping, record building)
is caught and logged as `Error` / `internal_error` with `target_called=True`,
the real `prompt_hash` and the actual `cache_hit`, and the client gets the
uniform 503. It never escapes to the global handler, which would log the
request as unparsed with `target_called=False`.

Shadow modes (IP-4), deterministic like `eval`:

- `shadow_ingress_only`: L1, L2 and L3 all run on every item with no
  short-circuit and no TM-R3 decision; zero Groq calls (the app builds no
  Groq client in this tier). The result is one `ShadowRecord` (verdict, score
  and latency per layer and view). Nothing is released, so the client gets
  the uniform 503.
- `shadow_full_e2e`: L1, L2 and L3 all run with no short-circuit, the target
  (or its cache) is called for every item, and every egress variant X0 to X3
  is evaluated, cached in the IP-5 egress cache (X0 included) and logged
  independently on the same response. The client-visible response is the one
  the production cascade would release under the configured `EGRESS_VARIANT`
  and `TIMEOUT_POLICY` [Project Decision, extends IP-4], so a shadow run never
  shows more than production would.
- Oracle join (IP-4, Step 5B.1 LM-4): `harness/` fills
  `EgressVariantRecord.oracle_result` either in process, through
  `run_shadow_full_e2e_detail()` (each variant's client-visible output, its
  egress cache key and the shared `response_hash` and `manifest_id`), or
  offline, by rebuilding `egress_verdict_key(response_hash, manifest_id,
  variant)` from the logged `ShadowRecord` and reading the cached decision.
  Variant outputs are never logged. The HTTP response is unchanged.
- No TM-R2 cut-off (Step 5B.1 H2, IP-4, IP-5, §1.9, §2.5): the shadow tiers
  build L2 and L3 with no deadline and run every layer, L4 included, under a
  `SHADOW_SAFETY_CEILING_MS` (30 s) hang guard only, never under the TM-R2
  budgets. Every logged latency is the true measured one, so verdicts are
  deterministic (and cacheable) and p95 and p99 are never capped by a budget.

Budgets (TM-R2): a layer call is timed inside its pool thread and a result
measured above the budget is a timeout (a `block` excepted, see step 3);
windowed layers also stop between windows once their budget is spent
(`LayerBudgetExceeded`). The event loop stops waiting after budget plus
`HANG_GRACE_MS` for a hung call, because asyncio timers on Windows have about
15.6 ms resolution. A timed-out call keeps its pool slot until it returns
(IP-7). `TOTAL_BUDGET_MS` is checked at startup against the sum of the L1 to
L4 budgets; it is not a runtime deadline, and the bundle build is outside it.
The whole gateway time (bundle build included) is `total_latency_ms`.

Cache hits (IP-5, RT-V6; Step 5B.1 LM-2): a replayed `LayerResult` or
`EgressDecision` carries the measured cache lookup time as `latency_ms`, not
the latency stored with the original computation.

Outcome labelling: the gateway runs no oracle (PS-2), so a delivered,
unaltered response is logged `Success` with `oracle_result=None`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from app.cache import VerdictCache
from app.config import SHADOW_MODES, Settings
from app.executor import BoundedLayerPool, LoadShed
from app.groq_client import GroqClient, UpstreamError
from app.inspection import InspectionConfig, build_inspection_bundle, prompt_hash
from app.layers.base import DisabledLayer, IngressLayer, LayerBudgetExceeded, TightenableLayer, pin_layer
from app.layers.l4_egress import EgressDecision, L4EgressScanner
from app.prompting import BuiltPrompt, build_messages, build_user_content
from app.schemas import GatewayRequest
from app.timeout_policy import admit_after_faults, tighten_after_fault
from contracts.cache_keys import egress_verdict_key, layer_verdict_key, raw_response_hash, target_response_key
from contracts.hashing import canonical_hash
from contracts.inspection import CapExceeded, InspectionBundle
from contracts.layer_result import LayerResult
from contracts.outcome import Outcome
from contracts.records import EgressVariantRecord

logger = logging.getLogger("vault.pipeline")

HANG_GRACE_MS = 50.0
SHADOW_SAFETY_CEILING_MS = 30_000.0
EGRESS_VARIANTS = ("X0", "X1", "X2", "X3")
T = TypeVar("T")


@dataclass
class GatewayRuntime:
    """Everything one request needs, built once by the lifespan."""

    settings: Settings
    inspection_config: InspectionConfig
    layers: tuple[IngressLayer, ...]
    l4: L4EgressScanner
    prompt: BuiltPrompt
    cache: VerdictCache | None = None
    upstream: GroqClient | None = None
    pool: BoundedLayerPool | None = None

    @property
    def budgets_ms(self) -> dict[str, float]:
        s = self.settings
        return {"L1": s.L1_BUDGET_MS, "L2": s.L2_BUDGET_MS, "L3": s.L3_BUDGET_MS, "L4": s.L4_BUDGET_MS}

    @property
    def shadow(self) -> bool:
        return self.settings.GATEWAY_MODE in SHADOW_MODES

    def budget_for(self, layer: str) -> float:
        """The TM-R2 budget of `layer`, or the shadow safety ceiling in a shadow tier (H2)."""
        return SHADOW_SAFETY_CEILING_MS if self.shadow else self.budgets_ms[layer]


@dataclass(frozen=True)
class ShadowData:
    """Everything a `ShadowRecord` holds except the timestamp, latency, trial and objective ids (added by `app.audit`)."""

    shadow_tier: str
    prompt_hash: str
    layer_results: tuple[LayerResult, ...]
    layer_version: str
    bundle_version: str
    chroma_version: str
    target_called: bool = False
    cache_hit: bool | None = None
    egress_variants: tuple[EgressVariantRecord, ...] = ()


@dataclass(frozen=True)
class VariantOutput:
    """One egress variant's result on the shared response, for the harness oracles (IP-4, LM-4). Never logged.

    `content` is the variant's client-visible output (None when it blocked or
    faulted). `cache_key` is the IP-5 egress-verdict key the decision is
    stored under (`egress_verdict_key(response_hash, manifest_id, variant)`).
    """

    variant: str
    verdict: str
    egress_action: str
    content: str | None
    cache_key: str | None
    cache_hit: bool
    latency_ms: float


@dataclass
class ChatResult:
    """What the route logs and returns. `content` is set only when a response is released.

    In shadow modes `shadow` is set and the route writes a `ShadowRecord`
    instead of a `RequestLogRecord`; `outcome` is then None.
    """

    prompt_hash: str
    outcome: Outcome | None
    content: str | None = None
    blocked_by: str | None = None
    layer_results: tuple[LayerResult, ...] = ()
    target_called: bool = False
    cache_hit: bool | None = None
    model_id: str | None = None
    egress_variant: str | None = None
    egress_action: str = "none"
    reasoning_findings: tuple[str, ...] = ()
    timeout_policy: str | None = None
    shadow: ShadowData | None = None
    variant_outputs: Mapping[str, str | None] = field(default_factory=dict)
    response_hash: str | None = None
    manifest_id: str | None = None
    variant_details: tuple[VariantOutput, ...] = ()

    @property
    def released(self) -> bool:
        return self.content is not None


@dataclass(frozen=True)
class ShadowE2EDetail:
    """In-process `shadow_full_e2e` result for `harness/` (IP-4, LM-4).

    `variants` is empty when the request ended before the target stage
    (cap, load shed, upstream error); `result` then carries the outcome and
    `blocked_by` that were logged.
    """

    result: ChatResult
    response_hash: str | None
    manifest_id: str | None
    variants: tuple[VariantOutput, ...]

    def output(self, variant: str) -> str | None:
        """The client-visible output of `variant` (None when it blocked, faulted or did not run)."""
        return next((v.content for v in self.variants if v.variant == variant), None)


@dataclass(frozen=True)
class Budgeted:
    status: str  # "ok", "timeout", "error" or "shed"
    value: Any
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True)
class EgressRun:
    """One L4 evaluation: the decision or the fault, and how the IP-5 egress cache took part."""

    decision: EgressDecision | None
    fault: Budgeted | None
    cache_key: str | None = None
    cache_hit: bool = False


async def _in_pool(pool: BoundedLayerPool | None, fn: Callable[[], T]) -> T:
    """Run CPU work off the event loop: on the bounded pool, or a plain worker thread without one."""
    if pool is None:
        return await asyncio.to_thread(fn)
    return await pool.run(fn)


async def run_budgeted(fn: Callable[[], T], budget_ms: float, pool: BoundedLayerPool | None = None) -> Budgeted:
    """Run `fn` on the pool; a call measured above `budget_ms`, stopped by its deadline, or hung, is a timeout (TM-R2).

    A call that returned but was measured above the budget keeps its value in
    the `timeout` outcome, so the caller can honour a late `block` (M-2).
    """

    def timed() -> tuple[T, float]:
        started = time.perf_counter()
        value = fn()
        return value, (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    try:
        future = asyncio.wrap_future(pool.submit(timed)) if pool is not None else asyncio.to_thread(timed)
    except LoadShed:
        return Budgeted("shed", None, 0.0, "LoadShed")
    try:
        value, inner_ms = await asyncio.wait_for(future, timeout=(budget_ms + HANG_GRACE_MS) / 1000.0)
    except (TimeoutError, LayerBudgetExceeded):
        return Budgeted("timeout", None, (time.perf_counter() - started) * 1000.0)
    except Exception as exc:  # noqa: BLE001  a layer fault is a verdict, never a crash
        return Budgeted("error", None, (time.perf_counter() - started) * 1000.0, type(exc).__name__)
    if inner_ms > budget_ms:
        return Budgeted("timeout", value, inner_ms)
    return Budgeted("ok", value, inner_ms)


def _fault_result(layer: str, view: str, outcome: Budgeted, budget_ms: float) -> LayerResult:
    message = f"over the {budget_ms:g} ms budget" if outcome.status == "timeout" else f"layer raised {outcome.error}"
    return LayerResult(
        layer=layer,
        view=view,
        per_window_scores=(),
        overall_score=0.0,
        verdict="timeout" if outcome.status == "timeout" else "error",
        latency_ms=outcome.latency_ms,
        error_message=message,
    )


def _faulted(results: Sequence[LayerResult]) -> bool:
    return any(r.verdict in ("timeout", "error") for r in results)


def _blocked(results: Sequence[LayerResult]) -> bool:
    return any(r.verdict == "block" for r in results)


def _with_latency(results: Sequence[LayerResult], latency_ms: float) -> tuple[LayerResult, ...]:
    return tuple(r.model_copy(update={"latency_ms": latency_ms}) for r in results)


def _late_block(layer: IngressLayer, outcome: Budgeted, budget: float) -> tuple[LayerResult, ...] | None:
    """The results of a call that finished over its budget with a `block`, annotated (M-2); else None."""
    if outcome.status != "timeout" or outcome.value is None:
        return None
    results = tuple(outcome.value)
    if not results or any(r.layer != layer.name for r in results) or not _blocked(results):
        return None
    note = f"block finished over the {budget:g} ms budget after {outcome.latency_ms:.1f} ms; block kept (TM-R3, M-2)"
    return tuple(r.model_copy(update={"error_message": note}) if r.verdict == "block" else r for r in results)


async def _run_ingress_layer(
    runtime: GatewayRuntime, layer: IngressLayer, bundle: InspectionBundle, *, degraded: bool
) -> tuple[LayerResult, ...]:
    """One layer's results, from the IP-5 cache or a budgeted pool call. Raises LoadShed when the pool is full."""
    cache = runtime.cache if runtime.settings.CACHE_ENABLED else None
    key = None
    if cache is not None and not degraded:  # a tightened-threshold verdict is never cached or replayed
        key = layer_verdict_key(
            prompt_hash=bundle.prompt_hash,
            layer=layer.name,
            layer_version=layer.version,
            bundle_version=bundle.bundle_version,
            chroma_version=layer.chroma_version,
        )
        started = time.perf_counter()
        cached = await cache.get_layer(key)
        if cached is not None:
            return _with_latency(cached, (time.perf_counter() - started) * 1000.0)  # LM-2
    budget = runtime.budget_for(layer.name)
    outcome = await run_budgeted(lambda: layer.inspect(bundle, degraded=degraded), budget, runtime.pool)
    if outcome.status == "shed":
        raise LoadShed(f"{layer.name} could not be queued")
    late = _late_block(layer, outcome, budget)
    if late is not None:
        logger.warning("%s blocked after %.2f ms, over its %g ms budget; the block is kept (M-2)", layer.name, outcome.latency_ms, budget)
        return late  # not cached: only in-budget results are replayed
    if outcome.status != "ok":
        logger.warning("%s %s after %.2f ms", layer.name, outcome.status, outcome.latency_ms)
        return (_fault_result(layer.name, "bundle", outcome, budget),)
    results = tuple(outcome.value)
    if not results or any(r.layer != layer.name for r in results):
        return (_fault_result(layer.name, "bundle", Budgeted("error", None, outcome.latency_ms, "BadLayerResult"), budget),)
    if key is not None and cache is not None:
        cache.put_layer(key, layer.name, results)
    return results


Runs = list[tuple[IngressLayer, tuple[LayerResult, ...]]]


def decide_ingress(policy: str, runs: Runs) -> tuple[str | None, Runs]:
    """(blocked_by, runs) for completed ingress runs: a layer name, `p_closed`, or None (admit) (IP-3, TM-R3).

    The first blocking layer in L1, L2, L3 order wins. With faults and
    P-degrade, every completed tightenable layer (L2) is re-decided with the
    lowered threshold and may block; the returned runs carry those results.
    """
    for layer, results in runs:
        if _blocked(results):
            return layer.name, runs
    faulted = sum(1 for _, results in runs if _faulted(results))
    completed = len(runs) - faulted
    if faulted and tighten_after_fault(policy, True):
        runs = [
            (layer, layer.tighten(results) if isinstance(layer, TightenableLayer) and not _faulted(results) else results)
            for layer, results in runs
        ]
        for layer, results in runs:
            if _blocked(results):
                logger.info("%s blocks under the P-degrade tightened threshold (TM-R3)", layer.name)
                return layer.name, runs
    if not admit_after_faults(policy, completed_layers=completed, faulted_layers=faulted):
        return "p_closed", runs
    return None, runs


def _flatten(runs: Runs) -> tuple[LayerResult, ...]:
    return tuple(r for _, results in runs for r in results)


def _messages(runtime: GatewayRuntime, payload: GatewayRequest) -> tuple[dict[str, str], ...]:
    return build_messages(runtime.prompt.system_prompt, build_user_content(payload.user_prompt))


async def _target_response(runtime: GatewayRuntime, payload: GatewayRequest) -> tuple[dict[str, Any], bool, str | None]:
    """(raw provider JSON, cache_hit, key to store a fresh response under). Raises UpstreamError on any upstream failure.

    The fresh response is stored by `_store_response`, after the target call,
    inside the post-target guard.
    """
    upstream = runtime.upstream
    if upstream is None:
        raise UpstreamError("not_configured")
    messages = _messages(runtime, payload)
    cache = runtime.cache if runtime.settings.CACHE_ENABLED else None
    key = target_response_key(model_id=upstream.model, parameters=dict(upstream.parameters), messages=messages)
    if cache is not None:
        cached = await cache.get_response(key)
        if cached is not None:
            return cached, True, None
    raw = await upstream.complete(messages)
    return raw, False, key if cache is not None else None


def _store_response(runtime: GatewayRuntime, key: str | None, raw: Mapping[str, Any]) -> None:
    if key is not None and runtime.cache is not None and runtime.upstream is not None:
        runtime.cache.put_response(key, runtime.upstream.model, raw)


async def _egress(runtime: GatewayRuntime, raw: dict[str, Any], variant: str) -> EgressRun:
    """One L4 evaluation through the IP-5 egress cache. Malformed provider JSON is an L4 error (TM-R4).

    X0 is cached only in `shadow_full_e2e`, where all four variants are kept for
    the oracle join (LM-4). A shadow tier runs L4 under the safety ceiling (H2).
    """
    manifest = runtime.prompt.manifest
    cache = runtime.cache if runtime.settings.CACHE_ENABLED else None
    cache_x0 = runtime.settings.GATEWAY_MODE == "shadow_full_e2e"
    key = None
    if cache is not None and (variant != "X0" or cache_x0):
        key = egress_verdict_key(response_hash=raw_response_hash(raw), manifest_id=manifest.manifest_id, variant=variant)
        started = time.perf_counter()
        cached = await cache.get_egress(key, runtime.l4.version)
        if cached is not None:
            try:
                decision = EgressDecision.from_cache(cached, latency_ms=(time.perf_counter() - started) * 1000.0)
                return EgressRun(decision, None, key, cache_hit=True)
            except Exception:  # noqa: BLE001  a bad cache row is re-scanned, never trusted
                logger.error("ignoring an unreadable egress cache row")
    outcome = await run_budgeted(lambda: runtime.l4.scan(raw, manifest, variant), runtime.budget_for("L4"), runtime.pool)
    if outcome.status != "ok":
        logger.error("L4 %s %s after %.2f ms (fails closed, TM-R4)", variant, outcome.status, outcome.latency_ms)
        return EgressRun(None, outcome, key)
    decision: EgressDecision = outcome.value
    if key is not None and cache is not None:
        cache.put_egress(key, runtime.l4.version, decision.to_cache())
    return EgressRun(decision, None, key)


def _log_splits(bundle: InspectionBundle) -> None:
    for plan in bundle.window_plans:
        if plan.split_count:
            logger.info(
                "IP-2 sub-window split (not truncation): tokenizer=%s windows_split=%d raw_windows=%d decoded_windows=%d",
                plan.tokenizer, plan.split_count, len(plan.raw_windows), len(plan.decoded_windows),
            )


async def run_chat(runtime: GatewayRuntime, payload: GatewayRequest) -> ChatResult:
    """Run one admitted-to-parse request in the configured gateway mode."""
    digest = prompt_hash(payload.user_prompt)
    layers = tuple(pin_layer(layer) for layer in runtime.layers)  # IP-7: the L2 pointer is read once, here
    try:
        bundle = await _in_pool(runtime.pool, lambda: build_inspection_bundle(payload.user_prompt, runtime.inspection_config))
    except LoadShed:
        return ChatResult(prompt_hash=digest, outcome=Outcome.ERROR, blocked_by="load_shed")
    if isinstance(bundle, CapExceeded):
        return ChatResult(prompt_hash=bundle.prompt_hash, outcome=Outcome.BLOCKED_INGRESS, blocked_by=bundle.blocked_by)
    _log_splits(bundle)
    enabled = tuple(layer for layer in layers if layer.enabled)
    if runtime.shadow:
        return await _run_shadow(runtime, payload, bundle, enabled)
    return await _run_production(runtime, payload, bundle, enabled)


async def run_shadow_full_e2e_detail(runtime: GatewayRuntime, payload: GatewayRequest) -> ShadowE2EDetail:
    """`shadow_full_e2e` in process: the logged result plus every variant's output and cache key (IP-4, LM-4).

    `harness/` calls this instead of `POST /v1/chat` to score each variant's
    client-visible output with the oracles and fill
    `EgressVariantRecord.oracle_result`. The caller logs `detail.result`.
    """
    if runtime.settings.GATEWAY_MODE != "shadow_full_e2e":
        raise RuntimeError("run_shadow_full_e2e_detail needs GATEWAY_MODE=shadow_full_e2e")
    result = await run_chat(runtime, payload)
    return ShadowE2EDetail(
        result=result,
        response_hash=result.response_hash,
        manifest_id=result.manifest_id,
        variants=result.variant_details,
    )


def _post_target_fault(base: Mapping[str, Any], ingress: tuple[LayerResult, ...], cache_hit: bool, exc: Exception, **extra: Any) -> ChatResult:
    """`Error` / `internal_error` for an unexpected exception after the target returned (LM-1). Logs the type only."""
    logger.error("unexpected %s after the target call; the request fails closed", type(exc).__name__)
    return ChatResult(outcome=Outcome.ERROR, blocked_by="internal_error", layer_results=ingress, cache_hit=cache_hit, **extra, **base)


async def _run_production(
    runtime: GatewayRuntime, payload: GatewayRequest, bundle: InspectionBundle, layers: tuple[IngressLayer, ...]
) -> ChatResult:
    settings = runtime.settings
    policy = settings.TIMEOUT_POLICY
    base = dict(prompt_hash=bundle.prompt_hash, timeout_policy=policy)
    runs: Runs = []
    faulted = 0
    for layer in layers:
        try:
            results = await _run_ingress_layer(runtime, layer, bundle, degraded=tighten_after_fault(policy, faulted > 0))
        except LoadShed:
            return ChatResult(outcome=Outcome.ERROR, blocked_by="load_shed", layer_results=_flatten(runs), **base)
        runs.append((layer, results))
        if _blocked(results):  # IP-3: short-circuit, later layers never run
            return ChatResult(outcome=Outcome.BLOCKED_INGRESS, blocked_by=layer.name, layer_results=_flatten(runs), **base)
        faulted += _faulted(results)
    blocked_by, runs = decide_ingress(policy, runs)
    ingress = _flatten(runs)
    if blocked_by is not None:
        outcome = Outcome.ERROR if blocked_by == "p_closed" else Outcome.BLOCKED_INGRESS
        return ChatResult(outcome=outcome, blocked_by=blocked_by, layer_results=ingress, **base)

    if payload.aegis_context:
        logger.warning("aegis_context refused: the TM-S1 to TM-S5 sanitizer is not implemented yet")
        return ChatResult(outcome=Outcome.ERROR, blocked_by="internal_error", layer_results=ingress, **base)
    if runtime.upstream is None:
        return ChatResult(outcome=Outcome.ERROR, blocked_by="upstream_error", layer_results=ingress, **base)

    admitted = dict(target_called=True, model_id=runtime.upstream.model, **base)
    try:
        raw, cache_hit, store_key = await _target_response(runtime, payload)
    except UpstreamError as exc:
        logger.error("upstream call failed: %s", exc.kind)
        return ChatResult(outcome=Outcome.ERROR, blocked_by="upstream_error", layer_results=ingress, cache_hit=False, **admitted)
    except Exception as exc:  # noqa: BLE001  a target-stage fault fails closed with target_called=True
        return _post_target_fault(admitted, ingress, False, exc)
    try:
        return await _release_production(runtime, raw, cache_hit, store_key, ingress, admitted)
    except Exception as exc:  # noqa: BLE001  LM-1: nothing after the target call may escape unlogged
        return _post_target_fault(admitted, ingress, cache_hit, exc, egress_variant=settings.EGRESS_VARIANT)


async def _release_production(
    runtime: GatewayRuntime,
    raw: dict[str, Any],
    cache_hit: bool,
    store_key: str | None,
    ingress: tuple[LayerResult, ...],
    admitted: dict[str, Any],
) -> ChatResult:
    settings = runtime.settings
    _store_response(runtime, store_key, raw)
    variant = settings.EGRESS_VARIANT
    admitted = dict(admitted, cache_hit=cache_hit, egress_variant=variant)
    run = await _egress(runtime, raw, variant)
    decision = run.decision
    if decision is None:
        fault = run.fault or Budgeted("error", None, 0.0, "MissingDecision")
        if fault.status == "shed":
            return ChatResult(outcome=Outcome.ERROR, blocked_by="load_shed", layer_results=ingress, **admitted)
        l4 = _fault_result("L4", "content", fault, settings.L4_BUDGET_MS)
        blocked_by = "timeout" if fault.status == "timeout" else "internal_error"
        return ChatResult(outcome=Outcome.ERROR, blocked_by=blocked_by, layer_results=ingress + (l4,), **admitted)

    l4_result = decision.layer_result()
    layer_results = ingress + ((l4_result,) if l4_result is not None else ())
    common = dict(layer_results=layer_results, reasoning_findings=decision.reasoning_findings, egress_action=decision.action)
    if decision.action == "blocked":
        return ChatResult(outcome=Outcome.EGRESS_SAVED, blocked_by="L4", **common, **admitted)
    outcome = Outcome.EGRESS_SAVED if decision.action == "masked" else Outcome.SUCCESS
    return ChatResult(outcome=outcome, content=decision.content, **common, **admitted)


def combined_layer_version(layers: Sequence[IngressLayer]) -> str:
    """One version over every ingress layer's version, for `ShadowRecord.layer_version` (IP-4, IP-5)."""
    return canonical_hash({"component": "ingress-layers", "versions": {layer.name: layer.version for layer in layers}})


async def _run_shadow(
    runtime: GatewayRuntime, payload: GatewayRequest, bundle: InspectionBundle, layers: tuple[IngressLayer, ...]
) -> ChatResult:
    settings = runtime.settings
    tier = settings.GATEWAY_MODE
    policy = settings.TIMEOUT_POLICY
    base = dict(prompt_hash=bundle.prompt_hash, timeout_policy=policy)
    if payload.aegis_context:
        logger.warning("aegis_context refused: the TM-S1 to TM-S5 sanitizer is not implemented yet")
        return ChatResult(outcome=Outcome.ERROR, blocked_by="internal_error", **base)
    runs: Runs = []
    for layer in layers:  # IP-4: every layer, no short-circuit, untightened thresholds, no TM-R2 cut-off (H2)
        try:
            runs.append((layer, await _run_ingress_layer(runtime, layer, bundle, degraded=False)))
        except LoadShed:
            return ChatResult(outcome=Outcome.ERROR, blocked_by="load_shed", layer_results=_flatten(runs), **base)
    chroma = next((layer.chroma_version for layer in layers if layer.name == "L2"), None)
    if not chroma:
        raise RuntimeError("shadow modes need L2 and its ChromaDB version (IP-4)")
    shadow = dict(
        shadow_tier=tier,
        prompt_hash=bundle.prompt_hash,
        layer_results=_flatten(runs),
        layer_version=combined_layer_version(layers),
        bundle_version=bundle.bundle_version,
        chroma_version=chroma,
    )
    if tier == "shadow_ingress_only":  # zero Groq calls; nothing is released
        return ChatResult(outcome=None, shadow=ShadowData(**shadow), layer_results=_flatten(runs), **base)

    blocked_by, decided = decide_ingress(policy, runs)
    if runtime.upstream is None:
        return ChatResult(outcome=Outcome.ERROR, blocked_by="upstream_error", layer_results=_flatten(runs), **base)
    admitted = dict(target_called=True, model_id=runtime.upstream.model, **base)
    try:
        raw, cache_hit, store_key = await _target_response(runtime, payload)
    except UpstreamError as exc:
        logger.error("upstream call failed: %s", exc.kind)
        return ChatResult(outcome=Outcome.ERROR, blocked_by="upstream_error", layer_results=_flatten(runs), cache_hit=False, **admitted)
    except Exception as exc:  # noqa: BLE001  a target-stage fault fails closed with target_called=True
        return _post_target_fault(admitted, _flatten(runs), False, exc)
    try:
        return await _release_shadow(runtime, raw, cache_hit, store_key, runs, decided, blocked_by, shadow, admitted)
    except Exception as exc:  # noqa: BLE001  LM-1: nothing after the target call may escape unlogged
        return _post_target_fault(admitted, _flatten(runs), cache_hit, exc, egress_variant=settings.EGRESS_VARIANT)


async def _release_shadow(
    runtime: GatewayRuntime,
    raw: dict[str, Any],
    cache_hit: bool,
    store_key: str | None,
    runs: Runs,
    decided: Runs,
    blocked_by: str | None,
    shadow: dict[str, Any],
    admitted: dict[str, Any],
) -> ChatResult:
    settings = runtime.settings
    _store_response(runtime, store_key, raw)
    response_hash = raw_response_hash(raw)
    manifest_id = runtime.prompt.manifest.manifest_id
    records: list[EgressVariantRecord] = []
    details: list[VariantOutput] = []
    decisions: dict[str, EgressDecision | None] = {}
    for variant in EGRESS_VARIANTS:  # IP-4: every variant on the same response, each logged independently
        run = await _egress(runtime, raw, variant)
        if run.fault is not None and run.fault.status == "shed":
            return ChatResult(outcome=Outcome.ERROR, blocked_by="load_shed", layer_results=_flatten(runs), cache_hit=cache_hit, **admitted)
        decision = run.decision
        decisions[variant] = decision
        if decision is not None:
            record = decision.variant_record(response_hash=response_hash, manifest_id=manifest_id)
            latency = decision.latency_ms
        else:
            verdict = "timeout" if run.fault is not None and run.fault.status == "timeout" else "error"
            record = EgressVariantRecord(response_hash=response_hash, manifest_id=manifest_id, variant=variant, verdict=verdict)
            latency = run.fault.latency_ms if run.fault is not None else 0.0
        records.append(record)
        details.append(
            VariantOutput(
                variant=variant,
                verdict=record.verdict,
                egress_action=record.egress_action,
                content=decision.content if decision is not None else None,
                cache_key=run.cache_key,
                cache_hit=run.cache_hit,
                latency_ms=latency,
            )
        )
    shadow = dict(shadow, target_called=True, cache_hit=cache_hit, egress_variants=tuple(records))
    outputs = {v.variant: v.content for v in details}
    active = decisions.get(settings.EGRESS_VARIANT)
    released = None if blocked_by is not None or active is None else active.content
    return ChatResult(
        outcome=None,
        content=released,
        shadow=ShadowData(**shadow),
        variant_outputs=outputs,
        layer_results=_flatten(decided),
        cache_hit=cache_hit,
        egress_variant=settings.EGRESS_VARIANT,
        response_hash=response_hash,
        manifest_id=manifest_id,
        variant_details=tuple(details),
        **admitted,
    )


def default_ingress_layers(l1: IngressLayer, extra: Sequence[IngressLayer] = ()) -> tuple[IngressLayer, ...]:
    """L1, L2, L3 in order (IP-3). A layer not supplied in `extra` is a `DisabledLayer`.

    `extra` may also replace L1 (tests use a slow stand-in to drive the TM-R3 policy).
    """
    by_name = {layer.name: layer for layer in extra}
    unknown = set(by_name) - {"L1", "L2", "L3"}
    if unknown:
        raise ValueError(f"ingress layers must be named L1, L2 or L3, got {sorted(unknown)}")
    return (by_name.get("L1", l1), by_name.get("L2", DisabledLayer("L2")), by_name.get("L3", DisabledLayer("L3")))
