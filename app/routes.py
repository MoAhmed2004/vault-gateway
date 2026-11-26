"""Gateway HTTP endpoints (PS-1, IP-1, IP-2, TM-R1, TM-R6, IP-6).

`POST /v1/chat` [Project Decision: the spec names a single public gateway
endpoint but not its path] accepts the canonical single-turn request and, on
success, returns strictly `{"content": str}` (IP-6). Every refusal is the
uniform 503 (TM-R6), with the cause logged server-side.

Pipeline, in order:
1. Body cap (middleware) and schema validation (exception handlers).
2. Config field caps (document count and length).
3. `app.pipeline.run_chat`: InspectionBundle and its caps, L1 (L2 and L3 are
   Step 5B hooks), the TM-R3 policy, the target through the IP-5 response
   cache and the single Groq client module (§2.7), then L4 egress (IP-6,
   TM-R4). Documents in `aegis_context` fail closed until the sanitizer
   (TM-S1 to TM-S5) exists.

A released response is exactly `{"content": str}`; every other result is the
uniform 503. Each request is logged once with its outcome, `blocked_by`,
layer results, `target_called`, `cache_hit`, `model_id`, egress variant and
action, and the reasoning finding labels (RT-V6, §3.2).

`GET /healthz` (liveness) and `GET /readyz` (readiness: settings loaded and the
telemetry sink healthy) are [Project Decision] operational probes that return
constant bodies; an unready gateway answers `/readyz` with the uniform 503.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse, Response

from app.audit import record_refusal, record_request
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
    if result.content is None:
        return uniform_refusal_response()
    return JSONResponse(GatewayResponse(content=result.content).model_dump())
