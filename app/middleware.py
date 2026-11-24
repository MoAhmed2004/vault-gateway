"""Pure ASGI middleware: fail-closed guard and the 128 KiB streaming body cap (TM-R1, TM-R6).

Neither class subclasses `BaseHTTPMiddleware`, so no request body is ever
buffered beyond what the cap allows.

`BodyCapMiddleware` (TM-R1): before any JSON parsing, a request whose
`Content-Length` exceeds the cap (or is malformed: not ASCII digits, more
than 20 digits, or conflicting) is refused at once as `body_cap` without
reading a single body byte. Otherwise the middleware counts bytes as chunks
arrive and stops reading the instant the running total exceeds the cap, also
when the client sends no `Content-Length` (chunked transfer) or lies in it. A
body within the cap (at most `max_body_bytes`, so memory stays bounded) is
replayed to the application unchanged. Exactly `max_body_bytes` is accepted;
one byte more is refused.

`FailClosedMiddleware` (TM-R6): the outermost middleware registered by the
gateway. It records the start time and `X-Trial-ID` (server-side only) and
turns any exception raised below it into the uniform 503 when the response
has not started yet, so a fault never reaches a default error page.

Starlette itself always wraps the user middleware stack in its own
`ServerErrorMiddleware`, so that class sits outside `FailClosedMiddleware`.
It only sees an exception that escapes `FailClosedMiddleware`, and because
the gateway registers a global `Exception` handler (`app.errors`), Starlette
uses that handler there too: the response is still the uniform 503, never
the default 500 page.

Out of scope for TM-R6: uvicorn answers a request it cannot parse as HTTP
(for example a malformed request line or header) with its own plain-text
400 before the ASGI app is called, so no gateway code can shape that reply.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from app.audit import STATE_STARTED, STATE_TRIAL_ID, parse_trial_id, scope_state
from app.errors import send_uniform_refusal

logger = logging.getLogger("vault.gateway")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]
RefusalLogger = Callable[[Scope, str], None]

MAX_REQUEST_BODY_BYTES = 128 * 1024  # 131,072 bytes (TM-R1)


class _ClientDisconnected(Exception):
    pass


def _content_length(headers: list[tuple[bytes, bytes]]) -> int | None | str:
    """Parsed Content-Length, None when absent, or "invalid" for malformed or conflicting values."""
    values = [value for name, value in headers if name.lower() == b"content-length"]
    if not values:
        return None
    parsed = set()
    for value in values:
        text = value.decode("latin-1").strip()
        # ASCII digits only (latin-1 superscripts such as b"\xb2" pass str.isdigit but not int()),
        # and at most 20 digits so int() never meets Python's integer string conversion limit.
        if not (text.isascii() and text.isdigit()) or len(text) > 20:
            return "invalid"
        parsed.add(int(text))
    if len(parsed) != 1:
        return "invalid"
    return parsed.pop()


class BodyCapMiddleware:
    """Streaming request-body cap that refuses with the uniform 503 (TM-R1)."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int = MAX_REQUEST_BODY_BYTES, on_refusal: RefusalLogger) -> None:
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.on_refusal = on_refusal

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = _content_length(scope.get("headers", []))
        if declared == "invalid" or (isinstance(declared, int) and declared > self.max_body_bytes):
            self.on_refusal(scope, "body_cap")
            await send_uniform_refusal(send)
            return
        try:
            body = await self._read_capped(receive)
        except _ClientDisconnected:
            return
        if body is None:
            self.on_refusal(scope, "body_cap")
            await send_uniform_refusal(send)
            return
        await self.app(scope, _replay(body, receive), send)

    async def _read_capped(self, receive: Receive) -> bytes | None:
        """The whole body, or None the moment the running total exceeds the cap (nothing more is read)."""
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                raise _ClientDisconnected
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_body_bytes:
                return None
            chunks.append(chunk)
            if not message.get("more_body", False):
                return b"".join(chunks)


def _replay(body: bytes, receive: Receive) -> Receive:
    sent = False

    async def replay() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay


class FailClosedMiddleware:
    """Outermost guard: request context plus a uniform 503 for any exception below (TM-R6)."""

    def __init__(self, app: ASGIApp, *, on_refusal: RefusalLogger) -> None:
        self.app = app
        self.on_refusal = on_refusal

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        state = scope_state(scope)
        state[STATE_STARTED] = time.perf_counter()
        state[STATE_TRIAL_ID] = parse_trial_id(scope.get("headers", []))
        started = False

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception:  # noqa: BLE001  fail closed on every fault
            logger.exception("request refused: unhandled exception below FailClosedMiddleware")
            self.on_refusal(scope, "internal_error")
            if not started:
                await send_uniform_refusal(send)
