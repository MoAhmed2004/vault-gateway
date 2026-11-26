"""Production request pipeline for `POST /v1/chat` (IP-1, IP-3, IP-5, IP-6, IP-7, TM-R2, TM-R3, TM-R4, §3.2).

Order (IP-3, production and the deterministic `eval` mode):

1. InspectionBundle (IP-1, IP-2); a decoded-view or window cap breach is
   `Blocked-ingress` (`decoded_cap`, `window_cap`).
2. Ingress layers L1, L2, L3 in order, short-circuiting on the first block
   (`Blocked-ingress`, `blocked_by` the layer). L2 and L3 are `DisabledLayer`
   hooks until Step 5B and are skipped, never faked. Each enabled layer runs
   in a worker thread (IP-7) and its allow or block results are cached under
   the IP-5 layer-verdict key; a cached verdict is replayed without calling
   the layer. A timeout or error goes to the TM-R3 policy (`p_closed` when it
   refuses).
3. `aegis_context` documents need the TM-S1 to TM-S5 sanitizer, which is not
   implemented yet, so a request that carries documents fails closed
   (`Error` / `internal_error`, target not called) instead of forwarding
   unsanitized retrieved text [Project Decision].
4. Target: the gateway-built `[system, user]` message list (PS-4, CA-3), with
   the raw view forwarded untouched (IP-1), is looked up in the IP-5 target
   response cache; on a miss the single Groq client module is called and the
   raw provider JSON is stored. `target_called=True` from here on, with
   `cache_hit` telling a cache hit (`True`) from an upstream API call
   (`False`). Any upstream failure is `Error` / `upstream_error`. With no
   upstream client configured (`UPSTREAM_MODE` `stub` or `ollama` before
   their clients exist), the request fails closed the same way without
   calling a target.
5. L4 egress (IP-6) in the configured variant, through the IP-5 egress cache.
   A canary block is `Egress-saved` / `blocked` with the uniform 503; a mask is
   `Egress-saved` / `masked` with the masked content; a clean response is
   `Success`. An L4 timeout is `Error` / `timeout` and an L4 error is
   `Error` / `internal_error`, both with the uniform 503 (TM-R4).

Outcome labelling: the gateway does not run oracles (they live in
`harness/`, PS-2), so a delivered, unaltered response is logged `Success`
with `oracle_result=None`; the harness relabels it `Success` or
`Admitted-refused` from the oracle on the client-visible content (§3.2).

Budgets (TM-R2): a layer call is timed inside its worker thread and a result
measured above the layer's budget counts as a timeout. The event loop adds a
second, coarser deadline (budget plus `HANG_GRACE_MS`) only to stop waiting
for a hung call: asyncio timers on Windows have about 15.6 ms resolution, so
a 5 ms asyncio deadline would fire before any thread could answer. A timed-out
thread keeps running until it returns (IP-7); the bounded pool and load
shedding of IP-7 (`app/executor.py`) are not part of this step.

Shadow modes (IP-4) need the evaluate-all pipeline and are refused at app
creation until Step 5B.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from app.cache import VerdictCache
from app.config import Settings
from app.groq_client import GroqClient, UpstreamError
from app.inspection import InspectionConfig, build_inspection_bundle
from app.layers.base import DisabledLayer, IngressLayer
from app.layers.l4_egress import EgressDecision, L4EgressScanner
from app.prompting import BuiltPrompt, build_messages, build_user_content
from app.schemas import GatewayRequest
from app.timeout_policy import admit_after_faults, tighten_after_fault
from contracts.cache_keys import egress_verdict_key, layer_verdict_key, raw_response_hash, target_response_key
from contracts.inspection import CapExceeded, InspectionBundle
from contracts.layer_result import LayerResult
from contracts.outcome import Outcome

logger = logging.getLogger("vault.pipeline")

HANG_GRACE_MS = 50.0
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

    @property
    def budgets_ms(self) -> dict[str, float]:
        s = self.settings
        return {"L1": s.L1_BUDGET_MS, "L2": s.L2_BUDGET_MS, "L3": s.L3_BUDGET_MS, "L4": s.L4_BUDGET_MS}


@dataclass
class ChatResult:
    """What the route logs and returns. `content` is set only when a response is released."""

    prompt_hash: str
    outcome: Outcome
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

    @property
    def released(self) -> bool:
        return self.content is not None


@dataclass(frozen=True)
class Budgeted:
    status: str  # "ok", "timeout" or "error"
    value: Any
    latency_ms: float
    error: str | None = None


async def run_budgeted(fn: Callable[[], T], budget_ms: float) -> Budgeted:
    """Run `fn` in a worker thread; a call measured above `budget_ms`, or hung, is a timeout (TM-R2)."""

    def timed() -> tuple[T, float]:
        started = time.perf_counter()
        value = fn()
        return value, (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    try:
        value, inner_ms = await asyncio.wait_for(asyncio.to_thread(timed), timeout=(budget_ms + HANG_GRACE_MS) / 1000.0)
    except TimeoutError:
        return Budgeted("timeout", None, (time.perf_counter() - started) * 1000.0)
    except Exception as exc:  # noqa: BLE001  a layer fault is a verdict, never a crash
        return Budgeted("error", None, (time.perf_counter() - started) * 1000.0, type(exc).__name__)
    if inner_ms > budget_ms:
        return Budgeted("timeout", None, inner_ms)
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


async def _run_ingress_layer(
    runtime: GatewayRuntime, layer: IngressLayer, bundle: InspectionBundle, *, degraded: bool
) -> tuple[LayerResult, ...]:
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
        cached = await cache.get_layer(key)
        if cached is not None:
            return cached
    budget = runtime.budgets_ms[layer.name]
    outcome = await run_budgeted(lambda: layer.inspect(bundle, degraded=degraded), budget)
    if outcome.status != "ok":
        logger.warning("%s %s after %.2f ms", layer.name, outcome.status, outcome.latency_ms)
        return (_fault_result(layer.name, "bundle", outcome, budget),)
    results = tuple(outcome.value)
    if not results or any(r.layer != layer.name for r in results):
        return (_fault_result(layer.name, "bundle", Budgeted("error", None, outcome.latency_ms, "BadLayerResult"), budget),)
    if key is not None and cache is not None:
        cache.put_layer(key, layer.name, results)
    return results


def _messages(runtime: GatewayRuntime, payload: GatewayRequest) -> tuple[dict[str, str], ...]:
    return build_messages(runtime.prompt.system_prompt, build_user_content(payload.user_prompt))


async def _target_response(runtime: GatewayRuntime, payload: GatewayRequest) -> tuple[dict[str, Any], bool]:
    """(raw provider JSON, cache_hit). Raises UpstreamError on any upstream failure."""
    upstream = runtime.upstream
    if upstream is None:
        raise UpstreamError("not_configured")
    messages = _messages(runtime, payload)
    cache = runtime.cache if runtime.settings.CACHE_ENABLED else None
    key = target_response_key(model_id=upstream.model, parameters=dict(upstream.parameters), messages=messages)
    if cache is not None:
        cached = await cache.get_response(key)
        if cached is not None:
            return cached, True
    raw = await upstream.complete(messages)
    if cache is not None:
        cache.put_response(key, upstream.model, raw)
    return raw, False


async def _egress(runtime: GatewayRuntime, raw: dict[str, Any]) -> tuple[EgressDecision | None, Budgeted | None]:
    """(decision, None) or (None, fault). Malformed provider JSON is an L4 error (TM-R4)."""
    variant = runtime.settings.EGRESS_VARIANT
    manifest = runtime.prompt.manifest
    cache = runtime.cache if runtime.settings.CACHE_ENABLED else None
    key = None
    if cache is not None and variant != "X0":
        key = egress_verdict_key(response_hash=raw_response_hash(raw), manifest_id=manifest.manifest_id, variant=variant)
        cached = await cache.get_egress(key, runtime.l4.version)
        if cached is not None:
            try:
                return EgressDecision.from_cache(cached), None
            except Exception:  # noqa: BLE001  a bad cache row is re-scanned, never trusted
                logger.error("ignoring an unreadable egress cache row")
    outcome = await run_budgeted(lambda: runtime.l4.scan(raw, manifest, variant), runtime.settings.L4_BUDGET_MS)
    if outcome.status != "ok":
        logger.error("L4 %s after %.2f ms (fails closed, TM-R4)", outcome.status, outcome.latency_ms)
        return None, outcome
    decision: EgressDecision = outcome.value
    if key is not None and cache is not None:
        cache.put_egress(key, runtime.l4.version, decision.to_cache())
    return decision, None


async def run_chat(runtime: GatewayRuntime, payload: GatewayRequest) -> ChatResult:
    """Run one admitted-to-parse request through the production pipeline (IP-3)."""
    settings = runtime.settings
    bundle = await asyncio.to_thread(build_inspection_bundle, payload.user_prompt, runtime.inspection_config)
    if isinstance(bundle, CapExceeded):
        return ChatResult(prompt_hash=bundle.prompt_hash, outcome=Outcome.BLOCKED_INGRESS, blocked_by=bundle.blocked_by)

    results: list[LayerResult] = []
    policy = settings.TIMEOUT_POLICY
    completed = faulted = 0
    for layer in runtime.layers:
        if not layer.enabled:
            continue  # Step 5B hook (L2, L3): skipped, never faked
        layer_results = await _run_ingress_layer(
            runtime, layer, bundle, degraded=tighten_after_fault(policy, faulted > 0)
        )
        results.extend(layer_results)
        if any(r.verdict == "block" for r in layer_results):
            return ChatResult(
                prompt_hash=bundle.prompt_hash, outcome=Outcome.BLOCKED_INGRESS, blocked_by=layer.name,
                layer_results=tuple(results), timeout_policy=policy,
            )
        if any(r.verdict in ("timeout", "error") for r in layer_results):
            faulted += 1
        else:
            completed += 1
    ingress = tuple(results)
    if not admit_after_faults(policy, completed_layers=completed, faulted_layers=faulted):
        return ChatResult(
            prompt_hash=bundle.prompt_hash, outcome=Outcome.ERROR, blocked_by="p_closed",
            layer_results=ingress, timeout_policy=policy,
        )

    if payload.aegis_context:
        logger.warning("aegis_context refused: the TM-S1 to TM-S5 sanitizer is not implemented yet")
        return ChatResult(
            prompt_hash=bundle.prompt_hash, outcome=Outcome.ERROR, blocked_by="internal_error",
            layer_results=ingress, timeout_policy=policy,
        )

    if runtime.upstream is None:
        return ChatResult(
            prompt_hash=bundle.prompt_hash, outcome=Outcome.ERROR, blocked_by="upstream_error",
            layer_results=ingress, timeout_policy=policy,
        )

    admitted = dict(prompt_hash=bundle.prompt_hash, target_called=True, model_id=runtime.upstream.model, timeout_policy=policy)
    try:
        raw, cache_hit = await _target_response(runtime, payload)
    except UpstreamError as exc:
        logger.error("upstream call failed: %s", exc.kind)
        return ChatResult(outcome=Outcome.ERROR, blocked_by="upstream_error", layer_results=ingress, cache_hit=False, **admitted)

    variant = settings.EGRESS_VARIANT
    admitted.update(cache_hit=cache_hit, egress_variant=variant)
    decision, fault = await _egress(runtime, raw)
    if decision is None:
        fault = fault or Budgeted("error", None, 0.0, "MissingDecision")
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


def default_ingress_layers(l1: IngressLayer, extra: Sequence[IngressLayer] = ()) -> tuple[IngressLayer, ...]:
    """L1, L2, L3 in order (IP-3). L2 and L3 are the Step 5B hooks unless `extra` supplies them.

    `extra` may also replace L1 (tests use a slow stand-in to drive the TM-R3 policy).
    """
    by_name = {layer.name: layer for layer in extra}
    unknown = set(by_name) - {"L1", "L2", "L3"}
    if unknown:
        raise ValueError(f"ingress layers must be named L1, L2 or L3, got {sorted(unknown)}")
    return (by_name.get("L1", l1), by_name.get("L2", DisabledLayer("L2")), by_name.get("L3", DisabledLayer("L3")))
