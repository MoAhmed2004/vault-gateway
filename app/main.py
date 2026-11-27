"""Gateway application factory and lifespan (startup composition, §5.2, IP-7, IP-8, TM-R1, TM-R6).

This is one of the two `app/` modules allowed to import `telemetry/` (startup
composition only). Nothing runs at import time; start the server with:

    uvicorn app.main:create_app --factory --workers 1 --no-server-header --no-date-header

`--workers 1` keeps one in-memory queue and one SQLite writer (AGENTS.md Rule
1). The two `--no-*-header` flags keep the uniform 503 headers identical
across responses (TM-R6).

`create_app(settings=None, telemetry_sink=None)` validates settings when the
factory runs (fail fast, §2.7, PS-3). The lifespan opens the telemetry sink
on startup and, on shutdown, flushes it and closes it when the app owns it
(an injected sink is flushed but left for its owner to close). Each shutdown
step (upstream client, cache, layer pool, L2 store, telemetry) runs in its
own `try`/`except`, so a failure in one is logged and never skips the
telemetry flush (Step 5B.1, closes Step 5A L3).

Startup composition, in order, before the first request is accepted (any
failure aborts startup, fail fast):
1. The telemetry sink.
2. The IP-5 cache at `CACHE_DB_PATH` (when `CACHE_ENABLED`), schema verified.
3. The bounded layer pool (IP-7): `LAYER_POOL_WORKERS` threads (0 means one
   per physical core) and `LAYER_POOL_QUEUE_LIMIT` queued calls.
4. The models, off the event loop: the MiniLM embedder when L2 is enabled or
   an X3 scan can run (`EGRESS_VARIANT=X3` or `shadow_full_e2e`), the L2
   threat memory at `CHROMA_DB_PATH` (its active collection version must
   exist, otherwise startup stops: build one with
   `scripts/build_l2_collection.py`), and the DeBERTa classifier when L3 is
   enabled. A missing artifact stops startup with its path named. Each
   layer's tokenizer limit is registered with the InspectionBundle so IP-2
   sub-window splitting runs before the window cap. In production, demo and
   `eval`, L2 and L3 get their TM-R2 budgets and stop at the first blocking
   window; in the shadow tiers they get no deadline and scan every window
   (IP-4, Step 5B.1 H2, M-2). In a deterministic mode (`eval` and both
   shadow tiers), an active L2 version labelled `bootstrap`, or descending
   from one, stops startup: that seed is development and demo data only
   (§2.4).
5. L4 (`L4EgressScanner`, pattern-only Presidio, warmed up, with the embedder
   for X3), and every enabled ingress layer warmed up once.
6. The upstream: with `UPSTREAM_MODE=groq` the single Groq client module,
   then the model check (§2.7, §3.9): `TARGET_MODEL` and every
   `PROBE_CANDIDATES` id must be on Groq's active model list. A missing or
   inactive id, or an unreadable list, stops startup. `stub` and `ollama`
   have no client yet, so admitted requests fail closed (`upstream_error`).
   `shadow_ingress_only` never calls the target, so it builds and probes no
   Groq client and needs no `GROQ_API_KEY` (zero network calls, IP-4).
7. The gateway-built system prompt and its canary/PII manifest for
   `OBJECTIVE_ID` (PS-2, PS-3), once per process: in production and demo
   mode the secrets are fresh per gateway session, in deterministic modes
   they are derived from `HMAC_SECRET`. The system prompt is registered with
   L4 as the X3 leak reference.

`upstream`, `l4`, `extra_layers`, `embedder`, `classifier`, `store` and
`pool` are injection points for tests. The app owns and closes whatever
upstream client, store and pool it uses.

Middleware order, outermost first: Starlette's own `ServerErrorMiddleware`
(always added by Starlette, outside every user middleware; its handler is the
registered global `Exception` handler, so it also answers with the uniform
503), `FailClosedMiddleware`, `BodyCapMiddleware`, then FastAPI's exception
middleware and router. Trailing-slash redirects are disabled
(`redirect_slashes=False`) so a path such as `/v1/chat/` is a routing miss
with the uniform 503, not a 307 (TM-R6).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI

from app.audit import record_refusal
from app.cache import VerdictCache
from app.config import SHADOW_MODES, Settings, load_settings
from app.errors import install_exception_handlers
from app.executor import BoundedLayerPool
from app.groq_client import GroqClient, target_parameters
from app.inspection import InspectionConfig, TokenizerLimit, build_inspection_bundle
from app.layers.base import IngressLayer, LayerBudgetExceeded
from app.layers.l1_regex import L1RegexLayer
from app.layers.l2_threat_memory import L2ThreatMemoryLayer
from app.layers.l3_classifier import L3ClassifierLayer
from app.layers.l4_egress import L4EgressScanner
from app.middleware import BodyCapMiddleware, FailClosedMiddleware
from app.onnx_models import Classifier, DebertaClassifier, Embedder, MiniLMEmbedder
from app.pipeline import GatewayRuntime, default_ingress_layers
from app.prompting import build_system_prompt
from app.routes import router
from app.vector_store import BOOTSTRAP_LABEL, ThreatMemoryStore
from contracts.inspection import InspectionBundle
from contracts.modes import DETERMINISTIC_MODES
from contracts.telemetry_sink import TelemetrySink
from telemetry.writer import SQLiteTelemetrySink

logger = logging.getLogger("vault.gateway")

SHUTDOWN_FLUSH_TIMEOUT_S = 5.0
WARM_UP_PROMPT = "warm-up prompt for the ingress rules"


def build_groq_client(settings: Settings) -> GroqClient:
    """The production upstream: the single Groq client module pinned to `TARGET_MODEL` (§2.7)."""
    return GroqClient(
        api_key=settings.GROQ_API_KEY,
        model=settings.TARGET_MODEL,
        base_url=settings.GROQ_BASE_URL,
        timeout_s=settings.UPSTREAM_TIMEOUT_S,
        parameters=target_parameters(
            temperature=settings.TARGET_TEMPERATURE,
            seed=settings.TARGET_SEED,
            max_tokens=settings.TARGET_MAX_TOKENS,
            reasoning_effort=settings.TARGET_REASONING_EFFORT,
        ),
    )


def _warm_up_layers(layers: Sequence[IngressLayer], config: InspectionConfig) -> None:
    """Run every enabled ingress layer once so the first request is not charged for lazy setup (TM-R2).

    Budgets apply per request, not at startup: a warm-up call that runs past its
    budget (a cold first ONNX run) is logged, never fatal. Any other error stops startup.
    """
    bundle = build_inspection_bundle(WARM_UP_PROMPT, config)
    if isinstance(bundle, InspectionBundle):
        for layer in layers:
            if layer.enabled:
                try:
                    layer.inspect(bundle)
                except LayerBudgetExceeded as exc:
                    logger.warning("%s warm-up ran past its budget (%s); budgets are enforced per request", layer.name, exc)


def _merge_tokenizers(explicit: Sequence[TokenizerLimit], layers: Sequence[IngressLayer]) -> tuple[TokenizerLimit, ...]:
    """Explicit limits plus each layer's own `tokenizer_limit` (IP-2), one per tokenizer name."""
    merged = {limit.name: limit for limit in explicit}
    for layer in layers:
        limit = getattr(layer, "tokenizer_limit", None)
        if layer.enabled and isinstance(limit, TokenizerLimit):
            merged.setdefault(limit.name, limit)
    return tuple(merged.values())


@dataclass
class _Models:
    """Model-backed components built (or injected) at startup."""

    layers: list[IngressLayer]
    embedder: Embedder | None = None
    store: ThreatMemoryStore | None = None
    owns_store: bool = False


def refuse_dev_seed(settings: Settings, store: ThreatMemoryStore) -> None:
    """Deterministic modes refuse an active L2 version that is, or descends from, the `bootstrap` dev seed (§2.4)."""
    if settings.GATEWAY_MODE not in DETERMINISTIC_MODES:
        return
    active = store.active()
    if active is None:
        return
    for version in store.lineage(active):
        if version.label == BOOTSTRAP_LABEL:
            raise RuntimeError(
                f"{settings.GATEWAY_MODE} refuses the L2 collection version {active.version[:12]}: it is or descends "
                f"from the '{BOOTSTRAP_LABEL}' development seed (§2.4: measured runs populate L2 from S1-train only)"
            )


def _build_models(
    settings: Settings,
    extra_names: set[str],
    *,
    embedder: Embedder | None,
    classifier: Classifier | None,
    store: ThreatMemoryStore | None,
    needs_x3: bool,
) -> _Models:
    """Load MiniLM, the L2 store and DeBERTa as configured (blocking; run off the event loop).

    Shadow tiers build L2 and L3 with no deadline and no early stop (IP-4, H2, M-2).
    """
    built = _Models(layers=[])
    shadow = settings.GATEWAY_MODE in SHADOW_MODES
    want_l2 = settings.L2_ENABLED and "L2" not in extra_names
    if not (want_l2 or needs_x3):
        built.embedder = None
    else:
        minilm = embedder if embedder is not None else MiniLMEmbedder(settings.MINILM_MODEL_DIR)
        built.embedder = minilm
        if want_l2:
            if store is None:
                store = ThreatMemoryStore(settings.CHROMA_DB_PATH, embedder_version=minilm.version)
                built.owns_store = True
                store.open()
            built.store = store
            if store.active() is None:
                raise RuntimeError(
                    f"L2 is enabled but {store.path} has no active collection version "
                    "(build one with scripts/build_l2_collection.py, or set L2_ENABLED=false)"
                )
            refuse_dev_seed(settings, store)
            built.layers.append(
                L2ThreatMemoryLayer(
                    minilm, store, threshold=settings.L2_THRESHOLD,
                    degrade_delta=settings.P_DEGRADE_THRESHOLD_DELTA,
                    budget_ms=None if shadow else settings.L2_BUDGET_MS,
                    early_stop=not shadow,
                )
            )
    if settings.L3_ENABLED and "L3" not in extra_names:
        model = classifier if classifier is not None else DebertaClassifier(
            settings.L3_MODEL_DIR, malicious_label=settings.L3_MALICIOUS_LABEL
        )
        built.layers.append(
            L3ClassifierLayer(
                model, threshold=settings.L3_THRESHOLD,
                budget_ms=None if shadow else settings.L3_BUDGET_MS,
                early_stop=not shadow,
            )
        )
    return built


async def _shutdown_step(label: str, step: Callable[[], Any]) -> None:
    """Run one shutdown step; a failure is logged and never stops the steps after it (closes Step 5A L3)."""
    try:
        result = step()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # noqa: BLE001  every later shutdown step must still run
        logger.error("shutdown step %s failed: %s", label, type(exc).__name__)


def _shutdown_sink(sink: Any, *, owned: bool) -> None:
    flush = getattr(sink, "flush", None)
    try:
        if callable(flush):
            flush(timeout=SHUTDOWN_FLUSH_TIMEOUT_S)
    except Exception:  # noqa: BLE001  still close below; the error is reported, not hidden
        logger.exception("telemetry flush failed during shutdown")
    if owned:
        sink.close(timeout=SHUTDOWN_FLUSH_TIMEOUT_S)


def create_app(
    settings: Settings | None = None,
    telemetry_sink: TelemetrySink | None = None,
    *,
    tokenizers: Sequence[TokenizerLimit] = (),
    upstream: GroqClient | None = None,
    l4: L4EgressScanner | None = None,
    extra_layers: Sequence[IngressLayer] = (),
    embedder: Embedder | None = None,
    classifier: Classifier | None = None,
    store: ThreatMemoryStore | None = None,
    pool: BoundedLayerPool | None = None,
) -> FastAPI:
    """Build the gateway. `tokenizers` adds tokenizer limits for IP-2 sub-window splitting to the layers' own."""
    resolved = settings if settings is not None else load_settings()
    if not isinstance(resolved, Settings):
        raise TypeError("settings must be an app.config.Settings instance")
    if telemetry_sink is not None and not isinstance(telemetry_sink, TelemetrySink):
        raise TypeError("telemetry_sink must implement contracts.TelemetrySink")
    calls_target = resolved.GATEWAY_MODE != "shadow_ingress_only"  # IP-4: zero Groq calls in that tier
    if calls_target and upstream is None and resolved.UPSTREAM_MODE == "groq" and not resolved.GROQ_API_KEY.strip():
        raise ValueError("UPSTREAM_MODE=groq requires GROQ_API_KEY (§2.7 startup model check)")
    extra_names = {layer.name for layer in extra_layers}
    needs_x3 = l4 is None and (resolved.EGRESS_VARIANT == "X3" or resolved.GATEWAY_MODE == "shadow_full_e2e")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = telemetry_sink is None
        sink = SQLiteTelemetrySink(resolved.LOG_DB_PATH) if owned else telemetry_sink
        app.state.telemetry = sink
        cache: VerdictCache | None = None
        client = upstream
        models: _Models | None = None
        layer_pool = pool if pool is not None else BoundedLayerPool(resolved.LAYER_POOL_WORKERS, resolved.LAYER_POOL_QUEUE_LIMIT)
        try:
            if resolved.CACHE_ENABLED:
                cache = VerdictCache(resolved.CACHE_DB_PATH, writes_enabled=resolved.CACHE_WRITES_ENABLED)
                await asyncio.to_thread(cache.open)
            models = await asyncio.to_thread(
                _build_models, resolved, extra_names, embedder=embedder, classifier=classifier, store=store, needs_x3=needs_x3
            )
            layers = default_ingress_layers(L1RegexLayer(), [*models.layers, *extra_layers])
            inspection_config = InspectionConfig(
                max_decoded_chars=resolved.MAX_PROMPT_CHARS,
                max_windows_per_view=resolved.MAX_WINDOWS_PER_VIEW,
                tokenizers=_merge_tokenizers(tokenizers, layers),
            )
            app.state.inspection_config = inspection_config
            if l4 is not None:
                scanner = l4
            else:
                x3_embedder = models.embedder if needs_x3 else None
                scanner = await asyncio.to_thread(
                    lambda: L4EgressScanner(
                        embedder=x3_embedder,
                        similarity_threshold=resolved.X3_SIMILARITY_THRESHOLD,
                        ngram_threshold=resolved.X3_NGRAM_OVERLAP_THRESHOLD,
                        ngram_size=resolved.X3_NGRAM_SIZE,
                    )
                )
            await asyncio.to_thread(_warm_up_layers, layers, inspection_config)
            if calls_target and client is None and resolved.UPSTREAM_MODE == "groq":
                client = build_groq_client(resolved)
            if calls_target and client is not None:
                await client.verify_models([resolved.TARGET_MODEL, *resolved.PROBE_CANDIDATES])
            prompt = build_system_prompt(
                secret=resolved.HMAC_SECRET, objective_id=resolved.OBJECTIVE_ID, mode=resolved.GATEWAY_MODE
            )
            register = getattr(scanner, "register_reference", None)
            if callable(register):
                await asyncio.to_thread(register, prompt.manifest.manifest_id, prompt.system_prompt)
            app.state.runtime = GatewayRuntime(
                settings=resolved,
                inspection_config=inspection_config,
                layers=layers,
                l4=scanner,
                prompt=prompt,
                cache=cache,
                upstream=client if calls_target else None,
                pool=layer_pool,
            )
            logger.info(
                "gateway started: mode=%s upstream=%s egress=%s layers=%s pool=%d+%d", resolved.GATEWAY_MODE,
                client.model if client is not None and calls_target else "none", resolved.EGRESS_VARIANT,
                ",".join(layer.name for layer in layers if layer.enabled), layer_pool.workers, layer_pool.queue_limit,
            )
            yield
        finally:
            app.state.runtime = None
            app.state.telemetry = None
            if client is not None:
                await _shutdown_step("upstream client", client.aclose)
            if cache is not None:
                await _shutdown_step("cache", lambda: asyncio.to_thread(cache.close, SHUTDOWN_FLUSH_TIMEOUT_S))
            await _shutdown_step("layer pool", lambda: layer_pool.close(wait=True))
            if models is not None and models.owns_store and models.store is not None:
                await _shutdown_step("L2 store", models.store.close)
            await _shutdown_step("telemetry", lambda: _shutdown_sink(sink, owned=owned))
            logger.info("gateway stopped")

    # redirect_slashes=False: `/v1/chat/` gets the uniform 503, never a 307 redirect (TM-R6).
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)
    app.state.settings = resolved
    app.state.inspection_config = None
    app.state.telemetry = None
    app.state.runtime = None

    install_exception_handlers(app, lambda request, blocked_by: record_refusal(request.app, request.scope, blocked_by))
    app.include_router(router)

    def on_refusal(scope: Any, blocked_by: str) -> None:
        record_refusal(app, scope, blocked_by)

    # add_middleware inserts at the front, so the last call is the outermost layer.
    app.add_middleware(BodyCapMiddleware, max_body_bytes=resolved.MAX_REQUEST_BODY_BYTES, on_refusal=on_refusal)
    app.add_middleware(FailClosedMiddleware, on_refusal=on_refusal)
    return app
