"""Frozen InspectionBundle schema (IP-1, IP-2, TM-R1, §5.2 interface freeze).

One `InspectionBundle` is built per inspected text. It holds the untouched
`raw_view`, the normalized and decoded `decoded_view`, the shared base windows
of both views (600 characters, stride 300), and one `WindowPlan` per
registered tokenizer with the windows after sub-window splitting. Layers only
read the bundle.

The schema enforces "nothing is truncated or left unscanned" (IP-2): every
window's text is exactly its slice of the view, the windows of each view (and
of each plan) cover the whole view, and no view holds more than
`max_windows_per_view` windows. A bundle whose views would break a cap is
never built; the builder returns `CapExceeded` instead, which maps to
`Outcome.BLOCKED_INGRESS` with `blocked_by` `"decoded_cap"` or `"window_cap"`
and the uniform 503 (TM-R6).

Imports only stdlib and pydantic.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

ViewName = Literal["raw_view", "decoded_view"]
CapReason = Literal["decoded_cap", "window_cap"]
EncodingName = Literal["base64", "hex"]

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
TokenizerName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class TextWindow(BaseModel):
    """One character window of one view. `parent_index` is set only on sub-windows."""

    model_config = _FROZEN

    view: ViewName
    index: NonNegativeInt
    start: NonNegativeInt
    end: NonNegativeInt
    text: str
    parent_index: NonNegativeInt | None = None

    @model_validator(mode="after")
    def _span_matches_text(self) -> TextWindow:
        if self.end < self.start:
            raise ValueError("window end must be >= start")
        if len(self.text) != self.end - self.start:
            raise ValueError("window text length must equal end - start")
        return self


class WindowPlan(BaseModel):
    """Windows of both views after sub-window splitting for one tokenizer limit (IP-2).

    `split_count` is the number of base windows that were split. Splits are
    logged but are not truncation.
    """

    model_config = _FROZEN

    tokenizer: TokenizerName
    token_limit: PositiveInt
    raw_windows: tuple[TextWindow, ...]
    decoded_windows: tuple[TextWindow, ...]
    split_count: NonNegativeInt


class DecodeStep(BaseModel):
    """One Base64 or hex substitution made while building the decoded view (IP-1).

    `start` and `end` index the text of the pass in which the substitution was
    made; `joined` marks whitespace-separated adjacent candidates decoded together.
    """

    model_config = _FROZEN

    depth: Annotated[int, Field(ge=1, le=2)]
    encoding: EncodingName
    joined: bool
    start: NonNegativeInt
    end: NonNegativeInt
    decoded_chars: NonNegativeInt


def _check_view_windows(view_name: str, view: str, windows: tuple[TextWindow, ...], limit: int) -> None:
    if not windows:
        raise ValueError(f"{view_name} must have at least one window")
    if len(windows) > limit:
        raise ValueError(f"{view_name} has {len(windows)} windows, above max_windows_per_view={limit}")
    covered = 0
    for position, window in enumerate(windows):
        if window.view != view_name:
            raise ValueError(f"window {position} belongs to {window.view}, expected {view_name}")
        if window.index != position:
            raise ValueError(f"{view_name} window indexes must be 0..n-1 in order")
        if window.end > len(view) or view[window.start:window.end] != window.text:
            raise ValueError(f"{view_name} window {position} text is not its slice of the view")
        if window.start > covered:
            raise ValueError(f"{view_name} windows leave characters {covered}..{window.start} unscanned")
        covered = max(covered, window.end)
    if covered != len(view):
        raise ValueError(f"{view_name} windows leave characters {covered}..{len(view)} unscanned")


class InspectionBundle(BaseModel):
    """Frozen multi-view inspection input shared by every layer (IP-1, IP-2)."""

    model_config = _FROZEN

    bundle_version: Sha256Hex
    prompt_hash: Sha256Hex
    raw_view: str
    decoded_view: str
    max_decoded_chars: PositiveInt
    max_windows_per_view: PositiveInt
    raw_windows: tuple[TextWindow, ...]
    decoded_windows: tuple[TextWindow, ...]
    window_plans: tuple[WindowPlan, ...] = ()
    decode_steps: tuple[DecodeStep, ...] = ()
    truncation_count: int = 0

    @model_validator(mode="after")
    def _nothing_truncated_or_unscanned(self) -> InspectionBundle:
        if self.truncation_count != 0:
            raise ValueError("truncation_count must be 0 (IP-2: never truncate)")
        if len(self.decoded_view) > self.max_decoded_chars:
            raise ValueError("decoded_view exceeds max_decoded_chars; build CapExceeded instead (TM-R1)")
        limit = self.max_windows_per_view
        _check_view_windows("raw_view", self.raw_view, self.raw_windows, limit)
        _check_view_windows("decoded_view", self.decoded_view, self.decoded_windows, limit)
        names = [plan.tokenizer for plan in self.window_plans]
        if len(set(names)) != len(names):
            raise ValueError("window_plans must not repeat a tokenizer")
        for plan in self.window_plans:
            _check_view_windows("raw_view", self.raw_view, plan.raw_windows, limit)
            _check_view_windows("decoded_view", self.decoded_view, plan.decoded_windows, limit)
        return self

    def plan_for(self, tokenizer: str) -> WindowPlan:
        """Return the window plan registered for `tokenizer` (KeyError if none)."""
        for plan in self.window_plans:
            if plan.tokenizer == tokenizer:
                return plan
        raise KeyError(f"no window plan for tokenizer {tokenizer!r}")


class CapExceeded(BaseModel):
    """Structured refusal when a view breaks the decoded-view or window cap (TM-R1, IP-2).

    Maps to `Outcome.BLOCKED_INGRESS` with `blocked_by` equal to `blocked_by`
    here. `tokenizer` names the plan that overflowed (None for base windows).
    """

    model_config = _FROZEN

    blocked_by: CapReason
    view: ViewName
    observed: NonNegativeInt
    limit: PositiveInt
    tokenizer: TokenizerName | None = None
    bundle_version: Sha256Hex
    prompt_hash: Sha256Hex

    @model_validator(mode="after")
    def _observed_breaks_limit(self) -> CapExceeded:
        if self.observed <= self.limit:
            raise ValueError("CapExceeded requires observed > limit")
        if self.blocked_by == "decoded_cap" and (self.view != "decoded_view" or self.tokenizer is not None):
            raise ValueError("decoded_cap applies to decoded_view only and names no tokenizer")
        return self
