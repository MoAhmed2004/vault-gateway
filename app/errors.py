"""Uniform refusal and global exception handlers (TM-R6, F1).

Every block and failure returns one identical response: status 503, the fixed
header set `UNIFORM_REFUSAL_HEADERS`, and the fixed body
`{"detail": "Service Unavailable. Request blocked by security policy."}`
(the spec text, byte for byte). Nothing about the cause reaches the client:
no stack trace, exception class, field name, layer, or `X-Trial-ID` echo.
The cause is logged server-side only.

Handlers are registered for `RequestValidationError` (the default 422),
`StarletteHTTPException` (404, 405 and any other HTTP error) and `Exception`.
`app.middleware.FailClosedMiddleware` additionally catches anything raised
below it, so an unhandled exception never reaches Starlette's default 500
page. Production servers must run uvicorn with `--no-server-header` and
`--no-date-header` so the server adds no varying headers.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

UNIFORM_REFUSAL_STATUS = 503
UNIFORM_REFUSAL_DETAIL = "Service Unavailable. Request blocked by security policy."
UNIFORM_REFUSAL_BODY = b'{"detail": "Service Unavailable. Request blocked by security policy."}'
UNIFORM_REFUSAL_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"content-type", b"application/json"),
    (b"content-length", str(len(UNIFORM_REFUSAL_BODY)).encode("ascii")),
    (b"cache-control", b"no-store"),
)

logger = logging.getLogger("vault.gateway")

Send = Callable[[dict[str, Any]], Awaitable[None]]
RefusalHook = Callable[[Request, str], None]


def uniform_refusal_response() -> Response:
    """A fresh Starlette response carrying exactly the uniform status, headers and body."""
    response = Response(content=UNIFORM_REFUSAL_BODY, status_code=UNIFORM_REFUSAL_STATUS)
    response.raw_headers = list(UNIFORM_REFUSAL_HEADERS)
    return response


async def send_uniform_refusal(send: Send) -> None:
    """Emit the uniform refusal directly on an ASGI `send` (used by middleware before routing)."""
    await send({"type": "http.response.start", "status": UNIFORM_REFUSAL_STATUS, "headers": list(UNIFORM_REFUSAL_HEADERS)})
    await send({"type": "http.response.body", "body": UNIFORM_REFUSAL_BODY, "more_body": False})


ROUTING_MISS_STATUSES = frozenset({404, 405})


def install_exception_handlers(app: FastAPI, on_refusal: RefusalHook) -> None:
    """Register the TM-R6 handlers. `on_refusal(request, blocked_by)` logs server-side."""

    async def validation_handler(request: Request, exc: Exception) -> Response:
        logger.info("request refused: schema_validation (%d errors)", len(getattr(exc, "errors", lambda: [])()))
        on_refusal(request, "schema_validation")
        return uniform_refusal_response()

    async def http_handler(request: Request, exc: Exception) -> Response:
        status = getattr(exc, "status_code", None)
        if status in ROUTING_MISS_STATUSES:
            # Routing misses are not attack outcomes and are excluded from BlockedBy (§3.2).
            logger.info("request refused: routing miss %s %s", status, request.url.path)
        else:
            logger.info("request refused: http error %s", status)
            on_refusal(request, "schema_validation")
        return uniform_refusal_response()

    async def unhandled_handler(request: Request, exc: Exception) -> Response:
        logger.error("request refused: unhandled exception", exc_info=exc)
        on_refusal(request, "internal_error")
        return uniform_refusal_response()

    app.add_exception_handler(RequestValidationError, validation_handler)
    app.add_exception_handler(StarletteHTTPException, http_handler)
    app.add_exception_handler(Exception, unhandled_handler)
