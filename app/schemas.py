"""Canonical single-turn request and response schemas (PS-1, CA-1, CA-2, CA-3, IP-6, TM-R1).

The gateway accepts exactly one request shape (`extra="forbid"`):
`user_prompt` (at most 4,000 characters, the Pydantic `max_length`), optional
`aegis_context` documents `{id, source, text}`, and `stream` (only `false` is
accepted, CA-2). Unknown fields, multi-turn history, client-supplied `system`
or `tool` roles and `stream=true` fail validation and get the uniform 503
(TM-R6). Types are strict, so `"true"` or `1` never coerce to a boolean and
numbers never coerce to strings.

The document count and per-document length caps are config defaults (4
documents, 1,500 characters), checked by `field_cap_violation` against the
running `Settings`.

Project decisions [extends PS-1, PS-3]: document ids use the `contracts.datasets`
identifier pattern (no `|`, which the PS-3 nonce message uses as its separator)
and must be unique within one request, since each id derives its own nonce.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, StringConstraints, model_validator

MAX_USER_PROMPT_CHARS = 4000

DocumentId = Annotated[StrictStr, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]


class ContextDocument(BaseModel):
    """One retrieved document (CA-1). The only channel for retrieved content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: DocumentId
    source: Annotated[StrictStr, Field(min_length=1, max_length=512)]
    text: StrictStr


class GatewayRequest(BaseModel):
    """The canonical single-turn request (PS-1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_prompt: Annotated[StrictStr, Field(max_length=MAX_USER_PROMPT_CHARS)]
    aegis_context: list[ContextDocument] | None = None
    stream: StrictBool = False

    @model_validator(mode="after")
    def _single_turn_non_streaming(self) -> GatewayRequest:
        if self.stream:
            raise ValueError("stream=true is not supported (CA-2)")
        if self.aegis_context:
            ids = [doc.id for doc in self.aegis_context]
            if len(set(ids)) != len(ids):
                raise ValueError("aegis_context document ids must be unique")
        return self


class GatewayResponse(BaseModel):
    """The only client-visible success shape (IP-6)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str


def field_cap_violation(
    request: GatewayRequest,
    *,
    max_prompt_chars: int,
    max_context_docs: int,
    max_context_doc_chars: int,
) -> str | None:
    """Return a server-side reason when a config field cap is broken (TM-R1), else None."""
    if len(request.user_prompt) > max_prompt_chars:
        return "user_prompt above MAX_PROMPT_CHARS"
    docs = request.aegis_context or []
    if len(docs) > max_context_docs:
        return "more aegis_context documents than MAX_CONTEXT_DOCS"
    if any(len(doc.text) > max_context_doc_chars for doc in docs):
        return "aegis_context document above MAX_CONTEXT_DOC_CHARS"
    return None
