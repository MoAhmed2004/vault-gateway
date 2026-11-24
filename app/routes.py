"""Gateway HTTP endpoints (PS-1, IP-1, IP-2, TM-R1, TM-R6, IP-6).

`POST /v1/chat` [Project Decision: the spec names a single public gateway
endpoint but not its path] accepts the canonical single-turn request and, on
success, returns strictly `{"content": str}` (IP-6). Every refusal is the
uniform 503 (TM-R6), with the cause logged server-side.

Step 4 state of the pipeline, in order:
1. Body cap (middleware) and schema validation (exception handlers).
2. Config field caps (document count and length).
3. InspectionBundle for the user prompt, refusing on the decoded-view or window cap.
4. Step 5+ hooks, not yet wired: ingress layers L1 to L3 (IP-3, IP-4), the
   per-document sanitizer (TM-S1 to TM-S5), the single Groq client module
   (§2.7) and L4 egress (IP-6, TM-R4). No detector is faked: with no upstream
   client, an admitted request fails closed as `Error` / `upstream_error`.

`GET /healthz` (liveness) and `GET /readyz` (readiness: settings loaded and the
telemetry sink healthy) are [Project Decision] operational probes that return
constant bodies; an unready gateway answers `/readyz` with the uniform 503.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response

from app.audit import record_refusal
from app.errors import uniform_refusal_response
from app.inspection import build_inspection_bundle, prompt_hash
from app.schemas import GatewayRequest, field_cap_violation
from contracts.inspection import CapExceeded

GATEWAY_PATH = "/v1/chat"

router = APIRouter()


@router.get("/healthz")
async def healthz() -> Response:
    return JSONResponse({"status": "ok"})


@router.get("/readyz")
async def readyz(request: Request) -> Response:
    state = request.app.state
    sink = getattr(state, "telemetry", None)
    ready = getattr(state, "settings", None) is not None and sink is not None and getattr(sink, "healthy", True)
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

    bundle = await run_in_threadpool(build_inspection_bundle, payload.user_prompt, app.state.inspection_config)
    if isinstance(bundle, CapExceeded):
        record_refusal(app, request.scope, bundle.blocked_by, prompt_hash=bundle.prompt_hash)
        return uniform_refusal_response()

    # Step 5+ hook point: ingress layers, sanitizer, upstream client and L4 egress attach here.
    record_refusal(app, request.scope, "upstream_error", prompt_hash=bundle.prompt_hash)
    return uniform_refusal_response()
