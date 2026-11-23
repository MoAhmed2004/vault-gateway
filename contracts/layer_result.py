"""Frozen per-layer result schema shared by every detection layer (IP-9, §3.2)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Verdict = Literal["allow", "block", "timeout", "error"]

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class LayerResult(BaseModel):
    """One layer's decision on one view of an InspectionBundle.

    The model is frozen and rejects unknown fields. `per_window_scores` is a
    tuple so the instance cannot be mutated in place; it serializes as a JSON
    array. `truncation_count` must be 0 because IP-2 forbids truncation. NaN
    and infinity are rejected in every float field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer: str
    view: str
    per_window_scores: tuple[FiniteFloat, ...]
    overall_score: FiniteFloat
    verdict: Verdict
    latency_ms: NonNegativeFiniteFloat
    truncation_count: int = 0
    error_message: str | None = None

    @field_validator("truncation_count")
    @classmethod
    def _truncation_must_be_zero(cls, value: int) -> int:
        if value != 0:
            raise ValueError("truncation_count must be 0 (IP-2: never truncate)")
        return value
