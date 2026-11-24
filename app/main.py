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

Middleware order, outermost first: `FailClosedMiddleware`, `BodyCapMiddleware`,
then FastAPI's exception middleware and router.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.audit import record_refusal
from app.config import Settings, load_settings
from app.errors import install_exception_handlers
from app.inspection import InspectionConfig, TokenizerLimit
from app.middleware import BodyCapMiddleware, FailClosedMiddleware
from app.routes import router
from contracts.telemetry_sink import TelemetrySink
from telemetry.writer import SQLiteTelemetrySink

logger = logging.getLogger("vault.gateway")

SHUTDOWN_FLUSH_TIMEOUT_S = 5.0


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
) -> FastAPI:
    """Build the gateway. `tokenizers` registers layer tokenizer limits for IP-2 sub-window splitting."""
    resolved = settings if settings is not None else load_settings()
    if not isinstance(resolved, Settings):
        raise TypeError("settings must be an app.config.Settings instance")
    if telemetry_sink is not None and not isinstance(telemetry_sink, TelemetrySink):
        raise TypeError("telemetry_sink must implement contracts.TelemetrySink")
    inspection_config = InspectionConfig(
        max_decoded_chars=resolved.MAX_PROMPT_CHARS,
        max_windows_per_view=resolved.MAX_WINDOWS_PER_VIEW,
        tokenizers=tuple(tokenizers),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = telemetry_sink is None
        sink = SQLiteTelemetrySink(resolved.LOG_DB_PATH) if owned else telemetry_sink
        app.state.telemetry = sink
        logger.info("gateway started: mode=%s", resolved.GATEWAY_MODE)
        try:
            yield
        finally:
            app.state.telemetry = None
            _shutdown_sink(sink, owned=owned)
            logger.info("gateway stopped")

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = resolved
    app.state.inspection_config = inspection_config
    app.state.telemetry = None

    install_exception_handlers(app, lambda request, blocked_by: record_refusal(request.app, request.scope, blocked_by))
    app.include_router(router)

    def on_refusal(scope: Any, blocked_by: str) -> None:
        record_refusal(app, scope, blocked_by)

    # add_middleware inserts at the front, so the last call is the outermost layer.
    app.add_middleware(BodyCapMiddleware, max_body_bytes=resolved.MAX_REQUEST_BODY_BYTES, on_refusal=on_refusal)
    app.add_middleware(FailClosedMiddleware, on_refusal=on_refusal)
    return app
