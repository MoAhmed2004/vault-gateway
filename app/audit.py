"""Server-side outcome logging for gateway requests (RT-V6, §3.2, TM-R6 F1).

Request handlers, exception handlers and middleware log through the
`contracts.TelemetrySink` stored on `app.state.telemetry`; this module never
imports `telemetry/`. Logging never changes a response: any sink failure is
reported to the Python logger and swallowed, and each request is logged at
most once.

Per-request context lives in the ASGI scope `state` dict, set by
`app.middleware.FailClosedMiddleware`: the start time, and the `X-Trial-ID`
header value, which is read for server-side logging only and never echoed.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import MutableMapping
from datetime import UTC, datetime
from typing import Any

from contracts.layer_result import LayerResult
from contracts.outcome import Outcome
from contracts.records import INGRESS_BLOCK_REASONS, UNPARSED_PROMPT_HASH, RequestLogRecord

logger = logging.getLogger("vault.gateway")

STATE_STARTED = "vault_started"
STATE_TRIAL_ID = "vault_trial_id"
STATE_LOGGED = "vault_logged"

TRIAL_ID_HEADER = b"x-trial-id"
_TRIAL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


def parse_trial_id(headers: list[tuple[bytes, bytes]]) -> str | None:
    """The single well-formed `X-Trial-ID` value, else None (malformed or repeated values are ignored)."""
    values = [value for name, value in headers if name.lower() == TRIAL_ID_HEADER]
    if len(values) != 1:
        return None
    try:
        text = values[0].decode("ascii")
    except UnicodeDecodeError:
        return None
    return text if _TRIAL_ID.fullmatch(text) else None


def scope_state(scope: MutableMapping[str, Any]) -> dict[str, Any]:
    state = scope.setdefault("state", {})
    return state


def outcome_for(blocked_by: str) -> Outcome:
    """§3.2: security layers and input caps are Blocked-ingress; every fault is Error."""
    return Outcome.BLOCKED_INGRESS if blocked_by in INGRESS_BLOCK_REASONS else Outcome.ERROR


def record_refusal(
    app: Any,
    scope: MutableMapping[str, Any],
    blocked_by: str,
    *,
    prompt_hash: str = UNPARSED_PROMPT_HASH,
    layer_results: tuple[LayerResult, ...] = (),
) -> None:
    """Log one refused request (target never called) to the configured sink, at most once per request."""
    state = scope_state(scope)
    if state.get(STATE_LOGGED):
        return
    state[STATE_LOGGED] = True
    sink = getattr(getattr(app, "state", None), "telemetry", None)
    settings = getattr(getattr(app, "state", None), "settings", None)
    if sink is None or settings is None:
        logger.error("refusal %s not logged: telemetry sink or settings missing", blocked_by)
        return
    started = state.get(STATE_STARTED, time.perf_counter())
    try:
        record = RequestLogRecord(
            timestamp_utc=datetime.now(UTC).isoformat(),
            gateway_mode=settings.GATEWAY_MODE,
            prompt_hash=prompt_hash,
            layer_results=layer_results,
            target_called=False,
            outcome=outcome_for(blocked_by),
            total_latency_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
            blocked_by=blocked_by,
            trial_id=state.get(STATE_TRIAL_ID),
        )
        sink.record_request(record)
    except Exception:  # noqa: BLE001  logging must never change the response
        logger.exception("failed to log refusal %s", blocked_by)
