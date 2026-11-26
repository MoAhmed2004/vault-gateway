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
(an injected sink is flushed but left for its owner to close).

Startup composition, in order, before the first request is accepted (any
failure aborts startup, fail fast):
1. The telemetry sink.
2. The IP-5 cache at `CACHE_DB_PATH` (when `CACHE_ENABLED`), schema verified.
3. L4 (`L4EgressScanner`, pattern-only Presidio, warmed up) and L1 (warmed up).
4. The upstream: with `UPSTREAM_MODE=groq` the single Groq client module,
   then the model check (§2.7, §3.9): `TARGET_MODEL` and every
   `PROBE_CANDIDATES` id must be on Groq's active model list. A missing or
   inactive id, or an unreadable list, stops startup. `stub` and `ollama`
   have no client yet, so admitted requests fail closed (`upstream_error`).
5. The gateway-built system prompt and its canary/PII manifest for
   `OBJECTIVE_ID` (PS-2, PS-3), once per process: in production and demo
   mode the secrets are fresh per gateway session, in `eval` they are derived
   from `HMAC_SECRET`.

`upstream`, `l4`, and `extra_layers` are injection points for tests and for
Step 5B (real L2 and L3 layers). The app owns and closes whatever upstream
client it is given. Shadow modes (IP-4) are refused until Step 5B.

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
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.audit import record_refusal
from app.cache import VerdictCache
from app.config import Settings, load_settings
from app.errors import install_exception_handlers
from app.groq_client import GroqClient, target_parameters
from app.inspection import InspectionConfig, TokenizerLimit, build_inspection_bundle
from app.layers.base import IngressLayer
from app.layers.l1_regex import L1RegexLayer
from app.layers.l4_egress import L4EgressScanner
from app.middleware import BodyCapMiddleware, FailClosedMiddleware
from app.pipeline import GatewayRuntime, default_ingress_layers
from app.prompting import build_system_prompt
from app.routes import router
from contracts.inspection import InspectionBundle
from contracts.telemetry_sink import TelemetrySink
from telemetry.writer import SQLiteTelemetrySink

logger = logging.getLogger("vault.gateway")

SHUTDOWN_FLUSH_TIMEOUT_S = 5.0
SHADOW_MODES = frozenset({"shadow_ingress_only", "shadow_full_e2e"})


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


def _warm_up_l1(layer: IngressLayer, config: InspectionConfig) -> None:
    bundle = build_inspection_bundle("warm-up prompt for the ingress rules", config)
    if isinstance(bundle, InspectionBundle):
        layer.inspect(bundle)


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
) -> FastAPI:
    """Build the gateway. `tokenizers` registers layer tokenizer limits for IP-2 sub-window splitting."""
    resolved = settings if settings is not None else load_settings()
    if not isinstance(resolved, Settings):
        raise TypeError("settings must be an app.config.Settings instance")
    if telemetry_sink is not None and not isinstance(telemetry_sink, TelemetrySink):
        raise TypeError("telemetry_sink must implement contracts.TelemetrySink")
    if resolved.GATEWAY_MODE in SHADOW_MODES:
        raise ValueError(f"{resolved.GATEWAY_MODE} needs the IP-4 evaluate-all pipeline, planned for Step 5B")
    if upstream is None and resolved.UPSTREAM_MODE == "groq" and not resolved.GROQ_API_KEY.strip():
        raise ValueError("UPSTREAM_MODE=groq requires GROQ_API_KEY (§2.7 startup model check)")
    inspection_config = InspectionConfig(
        max_decoded_chars=resolved.MAX_PROMPT_CHARS,
        max_windows_per_view=resolved.MAX_WINDOWS_PER_VIEW,
        tokenizers=tuple(tokenizers),
    )
    layers = default_ingress_layers(L1RegexLayer(), extra_layers)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = telemetry_sink is None
        sink = SQLiteTelemetrySink(resolved.LOG_DB_PATH) if owned else telemetry_sink
        app.state.telemetry = sink
        cache: VerdictCache | None = None
        client = upstream
        try:
            if resolved.CACHE_ENABLED:
                cache = VerdictCache(resolved.CACHE_DB_PATH, writes_enabled=resolved.CACHE_WRITES_ENABLED)
                await asyncio.to_thread(cache.open)
            scanner = l4 if l4 is not None else await asyncio.to_thread(L4EgressScanner)
            await asyncio.to_thread(_warm_up_l1, layers[0], inspection_config)
            if client is None and resolved.UPSTREAM_MODE == "groq":
                client = build_groq_client(resolved)
            if client is not None:
                await client.verify_models([resolved.TARGET_MODEL, *resolved.PROBE_CANDIDATES])
            prompt = build_system_prompt(
                secret=resolved.HMAC_SECRET, objective_id=resolved.OBJECTIVE_ID, mode=resolved.GATEWAY_MODE
            )
            app.state.runtime = GatewayRuntime(
                settings=resolved,
                inspection_config=inspection_config,
                layers=layers,
                l4=scanner,
                prompt=prompt,
                cache=cache,
                upstream=client,
            )
            logger.info(
                "gateway started: mode=%s upstream=%s egress=%s", resolved.GATEWAY_MODE,
                client.model if client is not None else "none", resolved.EGRESS_VARIANT,
            )
            yield
        finally:
            app.state.runtime = None
            app.state.telemetry = None
            if client is not None:
                await client.aclose()
            if cache is not None:
                await asyncio.to_thread(cache.close, SHUTDOWN_FLUSH_TIMEOUT_S)
            _shutdown_sink(sink, owned=owned)
            logger.info("gateway stopped")

    # redirect_slashes=False: `/v1/chat/` gets the uniform 503, never a 307 redirect (TM-R6).
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)
    app.state.settings = resolved
    app.state.inspection_config = inspection_config
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
