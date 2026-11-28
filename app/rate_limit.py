"""Per-client rate limiting: an in-memory token bucket as the outermost ASGI middleware (TM-R7, TM-R1, TM-R6, LLM10).

Spec rule (TM-R7): with `ENABLE_RATE_LIMITING=true`, each client may send at
most `RATE_LIMIT_RPM` requests per minute. The limiter runs before the 128 KiB
body cap (TM-R1) and before any JSON parsing or ingress layer (§1.9), so a
rate-limited request is never read, parsed or scanned. It answers with the
uniform 503 (TM-R6): the same status, headers and body as every other refusal,
so there is no `Retry-After` header. The reason `rate_limit` is logged
server-side only (`rate_limit_logs`, with `X-Trial-ID` when sent and
`target_called=False`).

Mode defaults and guards live in `app/config.py`: on by default in
`production` and `demo`, off by default in `eval` and both shadow tiers, and
refused there when set to true, so it never distorts E1 to E6, E4 or the
throughput benchmarks.

Token bucket: capacity `RATE_LIMIT_RPM` tokens, refilled continuously at
`RATE_LIMIT_RPM / 60` tokens per second on a monotonic clock (injectable for
tests). A new client starts with a full bucket; each admitted request takes one
token; a refused request takes none. All state changes happen under one
`threading.Lock` with no `await` inside, so the limiter is safe for concurrent
coroutines and threads. The clock is read inside that lock and a bucket's last
update time never moves backwards (Step 5C.1 L-1), so concurrent or
out-of-order clock readings never credit the same interval twice.

Client key (`RATE_LIMIT_KEY`):

- `ip` (default): the socket peer address from the ASGI scope (`client[0]`).
  Forwarding headers such as `X-Forwarded-For` are never read: they are client
  controlled.
- `session`: the single, well-formed value of the `RATE_LIMIT_SESSION_HEADER`
  header (default `X-Session-ID`; 1 to 128 visible ASCII characters). It is
  never read from the body (PS-1 forbids extra fields, and the limiter runs
  before the body is read). A missing, repeated or malformed header falls back
  to the peer address bucket, so leaving the header out never escapes the
  limit [Project Decision, extends TM-R7]. The session value is chosen by the
  client, so `session` mode limits a cooperating front end (the demo UI), not
  an attacker who mints new values; `ip` stays the default.
- No peer address at all (a server without `client` in the scope): one shared
  deterministic `anonymous` bucket.

Bounded memory [Project Decision, extends TM-R7]: at most
`RATE_LIMIT_MAX_CLIENTS` buckets. Buckets are kept in least-recently-seen
order; on every call, idle buckets at the old end that have refilled to
capacity are dropped (a full bucket is the same as no bucket), and when the
store is still full the least recently seen bucket is evicted.

Scope [Project Decision, extends TM-R7]: every HTTP request is limited
(`POST /v1/chat`, routing misses and trailing-slash variants included, because
each of them would otherwise reach the body cap), except `GET` and `HEAD` on
the operational probes `/healthz` and `/readyz`.

Logging: the client key is stored only as an HMAC-SHA256 under a key drawn
once per process, so the log never holds a raw IP address or session value but
one client's refusals still share one hash within a run.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from app.audit import request_context
from app.errors import send_uniform_refusal

logger = logging.getLogger("vault.gateway")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]
Clock = Callable[[], float]
RateLimitLogger = Callable[[Scope, str, str], None]  # (scope, key_kind, client_key_hash)
RefusalLogger = Callable[[Scope, str], None]

DEFAULT_MAX_CLIENTS = 10_000
EXEMPT_PROBES = frozenset({("GET", "/healthz"), ("HEAD", "/healthz"), ("GET", "/readyz"), ("HEAD", "/readyz")})
ANONYMOUS_KEY = "anonymous"
_SESSION_VALUE = re.compile(r"[!-~]{1,128}")


class TokenBucketLimiter:
    """Thread-safe in-memory token buckets, one per client key, with a bounded key store (TM-R7)."""

    def __init__(self, rpm: int, *, max_clients: int = DEFAULT_MAX_CLIENTS, clock: Clock = time.monotonic) -> None:
        if isinstance(rpm, bool) or not isinstance(rpm, int) or rpm < 1:
            raise ValueError("rpm must be an int >= 1")
        if max_clients < 1:
            raise ValueError("max_clients must be >= 1")
        self.rpm = rpm
        self.capacity = float(rpm)
        self.refill_per_second = rpm / 60.0
        self.max_clients = max_clients
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()  # key -> (tokens, last update)

    def client_count(self) -> int:
        """Buckets currently held (not `__len__`, so an empty limiter is never falsy)."""
        with self._lock:
            return len(self._buckets)

    def _refilled(self, tokens: float, last: float, now: float) -> float:
        return min(self.capacity, tokens + max(0.0, now - last) * self.refill_per_second)

    def _evict_idle(self, now: float) -> None:
        """Drop buckets at the least-recently-seen end that have refilled to capacity (equivalent to absent)."""
        while self._buckets:
            key, (tokens, last) = next(iter(self._buckets.items()))
            if self._refilled(tokens, last, now) < self.capacity:
                return
            del self._buckets[key]

    def allow(self, key: str) -> bool:
        """Take one token for `key` and return True, or return False (no token taken) when the bucket is empty.

        Step 5C.1 L-1: the clock is read inside the lock, and a bucket's `last` never moves backwards
        (`now = max(clock(), last)`), so a stale or out-of-order reading can neither rewind `last` nor
        credit the same interval twice.
        """
        with self._lock:
            now = self._clock()
            self._evict_idle(now)
            current = self._buckets.pop(key, None)
            if current is None:
                while len(self._buckets) >= self.max_clients:
                    self._buckets.popitem(last=False)  # least recently seen
                tokens = self.capacity
            else:
                now = max(now, current[1])
                tokens = self._refilled(current[0], current[1], now)
            allowed = tokens >= 1.0
            if allowed:
                tokens -= 1.0
            self._buckets[key] = (tokens, now)  # re-inserted at the most-recently-seen end
            return allowed

    def tokens(self, key: str) -> float:
        """The tokens `key` would have now (capacity for an unknown key); for tests and diagnostics."""
        with self._lock:
            now = self._clock()
            current = self._buckets.get(key)
            return self.capacity if current is None else self._refilled(current[0], current[1], now)


def _single_header(headers: list[tuple[bytes, bytes]], name: bytes) -> str | None:
    values = [value for key, value in headers if key.lower() == name]
    if len(values) != 1:
        return None
    try:
        text = values[0].decode("ascii")
    except UnicodeDecodeError:
        return None
    return text if _SESSION_VALUE.fullmatch(text) else None


def client_key(scope: Scope, *, key_mode: str, session_header: bytes) -> tuple[str, str]:
    """(key kind, key) for one request: `session` header, else the peer address, else the anonymous bucket."""
    if key_mode == "session":
        value = _single_header(scope.get("headers", []), session_header)
        if value is not None:
            return "session", value
    client = scope.get("client")
    if client and client[0]:
        return "ip", str(client[0])
    return "anonymous", ANONYMOUS_KEY


class RateLimitMiddleware:
    """Outermost pure ASGI middleware: refuse over-limit clients with the uniform 503 before any body byte is read."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: TokenBucketLimiter,
        key_mode: str = "ip",
        session_header: str = "X-Session-ID",
        on_rate_limit: RateLimitLogger,
        on_refusal: RefusalLogger,
    ) -> None:
        if key_mode not in ("ip", "session"):
            raise ValueError("key_mode must be 'ip' or 'session'")
        self.app = app
        self.limiter = limiter
        self.key_mode = key_mode
        self.session_header = session_header.lower().encode("ascii")
        self.on_rate_limit = on_rate_limit
        self.on_refusal = on_refusal
        self._hash_key = secrets.token_bytes(32)  # per process: log rows never hold a raw IP or session value

    def key_hash(self, kind: str, key: str) -> str:
        return hmac.new(self._hash_key, f"{kind}|{key}".encode("utf-8", "surrogatepass"), hashlib.sha256).hexdigest()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_context(scope)  # outermost: the request clock starts here
        if (scope.get("method"), scope.get("path")) in EXEMPT_PROBES:
            await self.app(scope, receive, send)
            return
        try:
            kind, key = client_key(scope, key_mode=self.key_mode, session_header=self.session_header)
            allowed = self.limiter.allow(f"{kind}:{key}")
        except Exception:  # noqa: BLE001  a limiter fault fails closed, before any body byte is read
            logger.exception("rate limiter fault; the request is refused")
            self.on_refusal(scope, "internal_error")
            await send_uniform_refusal(send)
            return
        if not allowed:
            logger.info("request refused: rate_limit (%s key, %d rpm)", kind, self.limiter.rpm)
            self.on_rate_limit(scope, kind, self.key_hash(kind, key))
            await send_uniform_refusal(send)  # identical status, headers and body; no Retry-After (TM-R6)
            return
        await self.app(scope, receive, send)
