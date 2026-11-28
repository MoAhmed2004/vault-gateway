"""Server-side outcome logging for gateway requests (RT-V6, §3.2, TM-R6 F1).

Request handlers, exception handlers and middleware log through the
`contracts.TelemetrySink` stored on `app.state.telemetry`; this module never
imports `telemetry/`. Logging never changes a response: any sink failure is
reported to the Python logger and swallowed, and each request is logged at
most once.

Per-request context lives in the ASGI scope `state` dict, set by
`app.middleware.FailClosedMiddleware`: the start time, and the `X-Trial-ID`
header value, which is read for server-side logging only and never echoed.

`objective_id` (RT-V6, IP-4; Step 5B.1, closes Step 5A L1): in the
deterministic modes (`eval`, `shadow_ingress_only`, `shadow_full_e2e`) every
`RequestLogRecord` and `ShadowRecord` carries `settings.OBJECTIVE_ID`, the
objective whose canary and planted PII the gateway-built system prompt holds,
so `harness/` can join records to the objective file. Production and demo
records leave it None (their secrets are per session).

`total_latency_ms` is measured from the request start in the outermost
middleware to the moment the record is built, so it covers the body read,
the InspectionBundle build, every layer and the target call.

Step 5C adds two server-side-only rows that do not fit the frozen
`RequestLogRecord` (and `contracts/` stays untouched): `record_rate_limit`
(TM-R7: reason `rate_limit`, `target_called=False`, `X-Trial-ID` when sent,
the client key as a keyed hash, never the raw IP or session value) and
`record_output_sanitizer` (IP-10: version, mode, verdict, matched families
and latency, a record apart from the X0 to X3 egress verdicts). They reach
the sink through the `RateLimitSink` and `OutputSanitizerSink` Protocols
defined here, which `telemetry.writer.SQLiteTelemetrySink` implements.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from contracts.layer_result import LayerResult
from contracts.modes import DETERMINISTIC_MODES
from contracts.outcome import Outcome
from contracts.records import INGRESS_BLOCK_REASONS, UNPARSED_PROMPT_HASH, EgressVariantRecord, RequestLogRecord, ShadowRecord

logger = logging.getLogger("vault.gateway")

STATE_STARTED = "vault_started"
STATE_TRIAL_ID = "vault_trial_id"
STATE_LOGGED = "vault_logged"
STATE_REQUEST_ID = "vault_request_id"
STATE_SANITIZER_LOGGED = "vault_sanitizer_logged"


@runtime_checkable
class RateLimitSink(Protocol):
    """Optional sink extension for TM-R7 refusals (`telemetry.writer.SQLiteTelemetrySink` implements it).

    Lives in `app/`, not `contracts/` (the W2 freeze): the record is a plain
    mapping with the `rate_limit_logs` columns, validated by the sink.
    """

    def record_rate_limit(self, record: Mapping[str, Any]) -> None: ...


@runtime_checkable
class OutputSanitizerSink(Protocol):
    """Optional sink extension for IP-10 verdicts, a record apart from every X0 to X3 egress verdict."""

    def record_output_sanitizer(self, record: Mapping[str, Any]) -> None: ...


def request_context(scope: MutableMapping[str, Any]) -> dict[str, Any]:
    """Set the per-request context once, in whichever gateway middleware runs first (outermost).

    The start time (for `total_latency_ms`), the `X-Trial-ID` value (server-side
    only, never echoed) and a random `request_id` that joins the Step 5C log
    rows of one request.
    """
    state = scope_state(scope)
    state.setdefault(STATE_STARTED, time.perf_counter())
    if STATE_TRIAL_ID not in state:
        state[STATE_TRIAL_ID] = parse_trial_id(scope.get("headers", []))
    state.setdefault(STATE_REQUEST_ID, uuid.uuid4().hex)
    return state

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


def objective_id_for(settings: Any) -> str | None:
    """`OBJECTIVE_ID` in a deterministic mode, else None (RT-V6)."""
    if settings is not None and getattr(settings, "GATEWAY_MODE", None) in DETERMINISTIC_MODES:
        return str(settings.OBJECTIVE_ID)
    return None


def record_request(
    app: Any,
    scope: MutableMapping[str, Any],
    *,
    outcome: Outcome,
    prompt_hash: str,
    blocked_by: str | None = None,
    layer_results: tuple[LayerResult, ...] = (),
    target_called: bool = False,
    **fields: Any,
) -> None:
    """Log one request to the configured sink, at most once per request (RT-V6).

    `fields` are the remaining optional `RequestLogRecord` fields (`cache_hit`,
    `model_id`, `egress_variant`, `egress_action`, `reasoning_findings`,
    `timeout_policy`). The record validates the §3.2 outcome rules; a record
    that fails them is reported to the Python logger and never changes the
    response.
    """
    state = scope_state(scope)
    if state.get(STATE_LOGGED):
        return
    state[STATE_LOGGED] = True
    sink = getattr(getattr(app, "state", None), "telemetry", None)
    settings = getattr(getattr(app, "state", None), "settings", None)
    if sink is None or settings is None:
        logger.error("request (%s, %s) not logged: telemetry sink or settings missing", outcome, blocked_by)
        return
    started = state.get(STATE_STARTED, time.perf_counter())
    fields.setdefault("objective_id", objective_id_for(settings))
    try:
        record = RequestLogRecord(
            timestamp_utc=datetime.now(UTC).isoformat(),
            gateway_mode=settings.GATEWAY_MODE,
            prompt_hash=prompt_hash,
            layer_results=layer_results,
            target_called=target_called,
            outcome=outcome,
            total_latency_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
            blocked_by=blocked_by,
            trial_id=state.get(STATE_TRIAL_ID),
            **fields,
        )
        sink.record_request(record)
    except Exception:  # noqa: BLE001  logging must never change the response
        logger.exception("failed to log request (%s, %s)", outcome, blocked_by)


def record_shadow(
    app: Any,
    scope: MutableMapping[str, Any],
    *,
    shadow_tier: str,
    prompt_hash: str,
    layer_results: tuple[LayerResult, ...],
    layer_version: str,
    bundle_version: str,
    chroma_version: str,
    target_called: bool = False,
    cache_hit: bool | None = None,
    egress_variants: tuple[EgressVariantRecord, ...] = (),
) -> None:
    """Log one shadow-mode record (IP-4) to the configured sink, at most once per request.

    Shares the once-per-request guard with `record_request`, so a shadow
    request is logged either as a `ShadowRecord` or, when it ends before the
    shadow record is complete (cap, load shed, upstream failure), as a
    `RequestLogRecord`, never both.
    """
    state = scope_state(scope)
    if state.get(STATE_LOGGED):
        return
    state[STATE_LOGGED] = True
    sink = getattr(getattr(app, "state", None), "telemetry", None)
    if sink is None:
        logger.error("shadow record (%s) not logged: telemetry sink missing", shadow_tier)
        return
    started = state.get(STATE_STARTED, time.perf_counter())
    settings = getattr(getattr(app, "state", None), "settings", None)
    try:
        record = ShadowRecord(
            timestamp_utc=datetime.now(UTC).isoformat(),
            shadow_tier=shadow_tier,
            prompt_hash=prompt_hash,
            layer_results=layer_results,
            total_latency_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
            target_called=target_called,
            cache_hit=cache_hit,
            egress_variants=egress_variants,
            layer_version=layer_version,
            bundle_version=bundle_version,
            chroma_version=chroma_version,
            trial_id=state.get(STATE_TRIAL_ID),
            objective_id=objective_id_for(settings),
        )
        sink.record_shadow(record)
    except Exception:  # noqa: BLE001  logging must never change the response
        logger.exception("failed to log shadow record (%s)", shadow_tier)


def record_refusal(
    app: Any,
    scope: MutableMapping[str, Any],
    blocked_by: str,
    *,
    prompt_hash: str = UNPARSED_PROMPT_HASH,
    layer_results: tuple[LayerResult, ...] = (),
) -> None:
    """Log one refused request (target never called) to the configured sink, at most once per request."""
    record_request(
        app, scope, outcome=outcome_for(blocked_by), prompt_hash=prompt_hash, blocked_by=blocked_by,
        layer_results=layer_results,
    )


def _scope_text(scope: MutableMapping[str, Any], key: str) -> str | None:
    value = scope.get(key)
    return value if isinstance(value, str) else None


def record_rate_limit(
    app: Any, scope: MutableMapping[str, Any], *, key_kind: str, client_key_hash: str, rate_limit_rpm: int
) -> None:
    """Log one TM-R7 refusal as a `rate_limit_logs` row, server-side only (reason `rate_limit`, target not called).

    The request is marked logged, so nothing else logs it. A sink without
    `record_rate_limit` (a custom `TelemetrySink`) is reported to the Python
    logger; logging never changes the response.
    """
    state = request_context(scope)
    if state.get(STATE_LOGGED):
        return
    state[STATE_LOGGED] = True
    sink = getattr(getattr(app, "state", None), "telemetry", None)
    settings = getattr(getattr(app, "state", None), "settings", None)
    if not isinstance(sink, RateLimitSink) or settings is None:
        logger.error("rate-limit refusal not logged: telemetry sink without record_rate_limit or settings missing")
        return
    try:
        sink.record_rate_limit(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "gateway_mode": settings.GATEWAY_MODE,
                "reason": "rate_limit",
                "request_id": state[STATE_REQUEST_ID],
                "key_kind": key_kind,
                "client_key_hash": client_key_hash,
                "rate_limit_rpm": rate_limit_rpm,
                "target_called": False,
                "method": _scope_text(scope, "method"),
                "path": _scope_text(scope, "path"),
                "trial_id": state.get(STATE_TRIAL_ID),
                "total_latency_ms": max(0.0, (time.perf_counter() - state[STATE_STARTED]) * 1000.0),
            }
        )
    except Exception:  # noqa: BLE001  logging must never change the response
        logger.exception("failed to log a rate-limit refusal")


def record_output_sanitizer(
    app: Any,
    scope: MutableMapping[str, Any],
    *,
    prompt_hash: str,
    response_hash: str,
    egress_variant: str,
    sanitizer_version: str,
    mode: str,
    verdict: str,
    matched_families: tuple[str, ...],
    latency_ms: float,
    error_message: str | None = None,
) -> None:
    """Log one IP-10 verdict as an `output_sanitizer_logs` row, apart from the request record, at most once."""
    state = request_context(scope)
    if state.get(STATE_SANITIZER_LOGGED):
        return
    state[STATE_SANITIZER_LOGGED] = True
    sink = getattr(getattr(app, "state", None), "telemetry", None)
    settings = getattr(getattr(app, "state", None), "settings", None)
    if not isinstance(sink, OutputSanitizerSink) or settings is None:
        logger.error("output-sanitizer verdict not logged: telemetry sink without record_output_sanitizer")
        return
    try:
        sink.record_output_sanitizer(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "gateway_mode": settings.GATEWAY_MODE,
                "request_id": state[STATE_REQUEST_ID],
                "trial_id": state.get(STATE_TRIAL_ID),
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
                "egress_variant": egress_variant,
                "sanitizer_version": sanitizer_version,
                "mode": mode,
                "verdict": verdict,
                "matched_families": tuple(matched_families),
                "latency_ms": latency_ms,
                "error_message": error_message,
            }
        )
    except Exception:  # noqa: BLE001  logging must never change the response
        logger.exception("failed to log an output-sanitizer verdict")
