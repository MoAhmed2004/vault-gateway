"""Gateway HTTP endpoints (PS-1, IP-1, IP-2, TM-R1, TM-R6, IP-6).

`POST /v1/chat` [Project Decision: the spec names a single public gateway
endpoint but not its path] accepts the canonical single-turn request and, on
success, returns strictly `{"content": str}` (IP-6). Every refusal is the
uniform 503 (TM-R6), with the cause logged server-side.

Pipeline, in order:
1. Body cap (middleware) and schema validation (exception handlers).
2. Config field caps (document count and length).
3. `app.pipeline.run_chat`: InspectionBundle and its caps, L1, L2, L3 (IP-3
   cascade, or IP-4 evaluate-all in the shadow modes), the TM-R3 policy, the
   target through the IP-5 response cache and the single Groq client module
   (§2.7), then L4 egress (IP-6, TM-R4) and, when enabled, the IP-10 output
   sanitizer. Documents in `aegis_context` fail closed until the sanitizer
   (TM-S1 to TM-S5) exists.

The TM-R7 rate limit (`app/rate_limit.py`) runs before all of this, as the
outermost middleware, so a rate-limited request never reaches this module.

A released response is exactly `{"content": str}`; every other result is the
uniform 503. Each request is logged once (RT-V6, §3.2): a `RequestLogRecord`
with its outcome, `blocked_by`, layer results, `target_called`, `cache_hit`,
`model_id`, egress variant and action, and the reasoning finding labels; or,
in a shadow mode, a `ShadowRecord` (IP-4). Both shadow tiers answer every
request with the uniform 503 (Step 5C F3: `shadow_full_e2e` releases nothing
over HTTP; `harness/` reads variant outputs in process or from the egress
cache). When the IP-10 output sanitizer ran on the released content, its
verdict is logged as a separate `output_sanitizer_logs` row.

`GET /healthz` (liveness) and `GET /readyz` (readiness: settings loaded and the
telemetry sink healthy) are [Project Decision] operational probes that return
constant bodies; an unready gateway answers `/readyz` with the uniform 503.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse, Response

from app.audit import record_output_sanitizer, record_refusal, record_request, record_shadow
from app.errors import uniform_refusal_response
from app.inspection import prompt_hash
from app.pipeline import GatewayRuntime, run_chat
from app.schemas import GatewayRequest, GatewayResponse, field_cap_violation

GATEWAY_PATH = "/v1/chat"

router = APIRouter()


@router.get("/healthz")
async def healthz() -> Response:
    return JSONResponse({"status": "ok"})


@router.get("/readyz")
async def readyz(request: Request) -> Response:
    state = request.app.state
    sink = getattr(state, "telemetry", None)
    ready = (
        getattr(state, "settings", None) is not None
        and getattr(state, "runtime", None) is not None
        and sink is not None
        and getattr(sink, "healthy", True)
    )
    return JSONResponse({"status": "ready"}) if ready else uniform_refusal_response()


@router.post(GATEWAY_PATH)
async def chat(payload: GatewayRequest, request: Request) -> Response:
    app = request.app
    settings = app.state.settings
    if field_cap_violation(
        payload,
        max_prompt_chars=settings.MAX_PROMPT_CHARS,
        max_context_docs=settings.MAX_CONTEXT_DOCS,
        max_context_doc_chars=settings.MAX_CONTEXT_DOC_CHARS,
    ):
        # The prompt is already parsed here, so the record carries its real hash (RT-V6).
        record_refusal(app, request.scope, "schema_validation", prompt_hash=prompt_hash(payload.user_prompt))
        return uniform_refusal_response()

    runtime = getattr(app.state, "runtime", None)
    if not isinstance(runtime, GatewayRuntime):
        raise RuntimeError("gateway runtime is not initialized")  # uniform 503 via the global handler
    result = await run_chat(runtime, payload)
    if result.shadow is not None:
        # Both shadow tiers release nothing over HTTP (IP-4; Step 5C F3 for shadow_full_e2e).
        record_shadow(app, request.scope, **vars(result.shadow))
        return uniform_refusal_response()
    if result.outcome is None:
        raise RuntimeError("pipeline returned neither an outcome nor a shadow record")  # uniform 503
    record_request(
        app,
        request.scope,
        outcome=result.outcome,
        prompt_hash=result.prompt_hash,
        blocked_by=result.blocked_by,
        layer_results=result.layer_results,
        target_called=result.target_called,
        cache_hit=result.cache_hit,
        model_id=result.model_id,
        egress_variant=result.egress_variant,
        egress_action=result.egress_action,
        reasoning_findings=result.reasoning_findings,
        timeout_policy=result.timeout_policy,
    )
    if result.sanitizer_log is not None:  # IP-10: a separate record, apart from the X0 to X3 verdicts
        record_output_sanitizer(app, request.scope, prompt_hash=result.prompt_hash, **vars(result.sanitizer_log))
    if result.content is None:
        return uniform_refusal_response()
    return JSONResponse(GatewayResponse(content=result.content).model_dump())
