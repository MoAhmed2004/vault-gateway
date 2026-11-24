"""InspectionBundle builder: normalized views, decoding, windows, sub-window splitting (IP-1, IP-2, TM-R1).

Decoded view (IP-1), built from the raw view, which is never altered:

1. Unicode NFKC, then strip every format character (category `Cf`: zero-width
   characters, bidirectional controls, the tag block).
2. Base64 and hex candidate substitution, depth 2 (at most two passes). A
   candidate is a run of Base64-alphabet characters that is at least 20
   characters long or carries explicit `=` padding. Hex (even length, at least
   20 hex digits) is tried before Base64. A candidate is substituted only if
   its decoded bytes are valid UTF-8 with at least 85% printable characters.
   Adjacent candidates separated only by whitespace are first tried joined,
   under the same checks. Every decoded segment is normalized again (NFKC and
   `Cf` strip) before substitution, and each pass ends with a full re-normalization.

This module implements only the transforms IP-1 specifies. It does not fold
cross-script homoglyphs (that is the optional H-NORM variant, §3.6) and does
not decode URL percent-encoding or HTML entities.

Windows (IP-2): each view is sliced into overlapping character windows (600
characters, stride 300), never by whitespace. For every registered tokenizer
limit (256 for MiniLM, 512 for DeBERTa), a window whose token count exceeds the
limit is split into the fewest equal contiguous sub-windows that all fit,
BEFORE the per-view window cap (16) is checked; identical sub-windows from
overlapping parents are counted once. The builder refuses (returns
`CapExceeded`) only if the decoded view exceeds 4,000 characters or a view
exceeds 16 windows in total after splitting for any tokenizer. Nothing is
truncated.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from contracts.hashing import canonical_hash
from contracts.inspection import (
    CapExceeded,
    DecodeStep,
    InspectionBundle,
    TextWindow,
    ViewName,
    WindowPlan,
)

BUNDLE_ALGORITHM_REVISION = 1
WINDOW_CHARS = 600
WINDOW_STRIDE = 300
MAX_WINDOWS_PER_VIEW = 16
MAX_DECODED_CHARS = 4000
DECODE_DEPTH = 2
MIN_CANDIDATE_CHARS = 20
MIN_PRINTABLE_RATIO = 0.85
MINILM_TOKEN_LIMIT = 256
DEBERTA_TOKEN_LIMIT = 512

_TOKEN = re.compile(r"[A-Za-z0-9+/]+={0,2}")
_HEX = re.compile(r"[0-9A-Fa-f]+")
_PRINTABLE_WHITESPACE = frozenset("\n\r\t")


@dataclass(frozen=True)
class TokenizerLimit:
    """A consumer's tokenizer limit. `count_tokens` must include any special tokens the model adds."""

    name: str
    token_limit: int
    count_tokens: Callable[[str], int] = field(compare=False)

    def __post_init__(self) -> None:
        if self.token_limit <= 0:
            raise ValueError("token_limit must be positive")


@dataclass(frozen=True)
class InspectionConfig:
    """Bundle parameters. Defaults are the spec values (TM-R1, IP-1, IP-2)."""

    max_decoded_chars: int = MAX_DECODED_CHARS
    max_windows_per_view: int = MAX_WINDOWS_PER_VIEW
    window_chars: int = WINDOW_CHARS
    window_stride: int = WINDOW_STRIDE
    tokenizers: tuple[TokenizerLimit, ...] = ()

    def __post_init__(self) -> None:
        if min(self.max_decoded_chars, self.max_windows_per_view, self.window_chars, self.window_stride) <= 0:
            raise ValueError("inspection caps, window size and stride must be positive")
        if self.window_stride > self.window_chars:
            raise ValueError("window_stride must not exceed window_chars (windows must overlap or touch)")
        names = [t.name for t in self.tokenizers]
        if len(set(names)) != len(names):
            raise ValueError("tokenizer names must be unique")

    @property
    def bundle_version(self) -> str:
        """Deterministic version of every parameter that changes a bundle (IP-5 layer cache key)."""
        return canonical_hash(
            {
                "component": "InspectionBundle",
                "revision": BUNDLE_ALGORITHM_REVISION,
                "unicode_version": unicodedata.unidata_version,
                "max_decoded_chars": self.max_decoded_chars,
                "max_windows_per_view": self.max_windows_per_view,
                "window_chars": self.window_chars,
                "window_stride": self.window_stride,
                "decode_depth": DECODE_DEPTH,
                "min_candidate_chars": MIN_CANDIDATE_CHARS,
                "min_printable_ratio": MIN_PRINTABLE_RATIO,
                "tokenizers": sorted([t.name, t.token_limit] for t in self.tokenizers),
            }
        )


def prompt_hash(raw_view: str) -> str:
    """IP-5 prompt hash of the untouched raw view."""
    return canonical_hash({"raw_view": raw_view})


# Normalization and decoding (IP-1)


def normalize_text(text: str) -> str:
    """Unicode NFKC, then remove every format (`Cf`) character."""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")


def _is_candidate(token: str) -> bool:
    return len(token) >= MIN_CANDIDATE_CHARS or token.endswith("=")


def _printable_utf8(data: bytes) -> str | None:
    if not data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    printable = sum(1 for ch in text if ch.isprintable() or ch in _PRINTABLE_WHITESPACE)
    if printable < MIN_PRINTABLE_RATIO * len(text):
        return None
    return text


def _decode_hex(token: str) -> bytes | None:
    if len(token) < MIN_CANDIDATE_CHARS or len(token) % 2 or not _HEX.fullmatch(token):
        return None
    return bytes.fromhex(token)


def _decode_base64(token: str) -> bytes | None:
    body = token.rstrip("=")
    if "=" in body or len(body) % 4 == 1:
        return None
    padded = token if token.endswith("=") else token + "=" * (-len(token) % 4)
    if len(padded) % 4:
        return None
    try:
        return base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None


def _try_decode(candidate: str) -> tuple[str, str] | None:
    """Return (encoding, normalized decoded text) when the candidate passes the IP-1 checks."""
    for encoding, decoder in (("hex", _decode_hex), ("base64", _decode_base64)):
        raw = decoder(candidate)
        if raw is None:
            continue
        text = _printable_utf8(raw)
        if text is not None:
            return encoding, normalize_text(text)
    return None


def _candidate_groups(text: str) -> list[list[re.Match[str]]]:
    """Group candidate tokens that are separated only by whitespace."""
    groups: list[list[re.Match[str]]] = []
    for match in _TOKEN.finditer(text):
        if not _is_candidate(match.group()):
            continue
        if groups:
            previous = groups[-1][-1]
            gap = text[previous.end():match.start()]
            if gap and gap.isspace():
                groups[-1].append(match)
                continue
        groups.append([match])
    return groups


def _decode_pass(text: str, depth: int) -> tuple[str, list[DecodeStep]]:
    pieces: list[str] = []
    steps: list[DecodeStep] = []
    cursor = 0

    def substitute(start: int, end: int, candidate: str, joined: bool) -> bool:
        nonlocal cursor
        result = _try_decode(candidate)
        if result is None:
            return False
        encoding, decoded = result
        pieces.append(text[cursor:start])
        pieces.append(decoded)
        cursor = end
        steps.append(
            DecodeStep(
                depth=depth, encoding=encoding, joined=joined, start=start, end=end, decoded_chars=len(decoded)
            )
        )
        return True

    for group in _candidate_groups(text):
        if len(group) > 1:
            joined = "".join(m.group() for m in group)
            if substitute(group[0].start(), group[-1].end(), joined, True):
                continue
        for match in group:
            substitute(match.start(), match.end(), match.group(), False)
    pieces.append(text[cursor:])
    return "".join(pieces), steps


def build_decoded_view(raw_view: str) -> tuple[str, tuple[DecodeStep, ...]]:
    """Return the IP-1 decoded view and the substitutions made, depth 2."""
    text = normalize_text(raw_view)
    steps: list[DecodeStep] = []
    for depth in range(1, DECODE_DEPTH + 1):
        decoded, pass_steps = _decode_pass(text, depth)
        if not pass_steps:
            break
        steps.extend(pass_steps)
        text = normalize_text(decoded)
    return text, tuple(steps)


# Windows and sub-window splitting (IP-2)


def slice_windows(view: ViewName, text: str, window_chars: int, stride: int) -> tuple[TextWindow, ...]:
    """Overlapping character windows covering all of `text` (one empty window for empty text)."""
    windows: list[TextWindow] = []
    start = 0
    while True:
        end = min(start + window_chars, len(text))
        windows.append(TextWindow(view=view, index=len(windows), start=start, end=end, text=text[start:end]))
        if end >= len(text):
            return tuple(windows)
        start += stride


def _split_window(window: TextWindow, text: str, tokenizer: TokenizerLimit) -> list[tuple[int, int]]:
    """Fewest equal contiguous pieces of `window` that each fit the tokenizer limit."""
    length = window.end - window.start
    for pieces in range(2, length + 1):
        bounds = [
            (window.start + (i * length) // pieces, window.start + ((i + 1) * length) // pieces)
            for i in range(pieces)
        ]
        if all(tokenizer.count_tokens(text[s:e]) <= tokenizer.token_limit for s, e in bounds):
            return bounds
    raise ValueError(
        f"tokenizer {tokenizer.name!r} reports a single character above its {tokenizer.token_limit}-token limit"
    )


def split_for_tokenizer(
    view: ViewName,
    text: str,
    windows: Sequence[TextWindow],
    tokenizer: TokenizerLimit,
) -> tuple[tuple[TextWindow, ...], int]:
    """Split every window above the tokenizer limit; return (deduplicated windows, split_count)."""
    spans: dict[tuple[int, int], int | None] = {}
    split_count = 0
    for window in windows:
        if tokenizer.count_tokens(window.text) <= tokenizer.token_limit:
            spans.setdefault((window.start, window.end), None)
            continue
        split_count += 1
        for span in _split_window(window, text, tokenizer):
            spans.setdefault(span, window.index)
    ordered = sorted(spans.items())
    result = tuple(
        TextWindow(view=view, index=i, start=s, end=e, text=text[s:e], parent_index=parent)
        for i, ((s, e), parent) in enumerate(ordered)
    )
    return result, split_count


def build_inspection_bundle(raw_view: str, config: InspectionConfig | None = None) -> InspectionBundle | CapExceeded:
    """Build the frozen bundle, or a `CapExceeded` refusal (never a truncated bundle)."""
    config = config or InspectionConfig()
    version = config.bundle_version
    digest = prompt_hash(raw_view)
    decoded_view, steps = build_decoded_view(raw_view)

    def refuse(reason: str, view: ViewName, observed: int, limit: int, tokenizer: str | None = None) -> CapExceeded:
        return CapExceeded(
            blocked_by=reason, view=view, observed=observed, limit=limit, tokenizer=tokenizer,
            bundle_version=version, prompt_hash=digest,
        )

    if len(decoded_view) > config.max_decoded_chars:
        return refuse("decoded_cap", "decoded_view", len(decoded_view), config.max_decoded_chars)

    limit = config.max_windows_per_view
    views: dict[ViewName, tuple[str, tuple[TextWindow, ...]]] = {}
    for name, text in (("raw_view", raw_view), ("decoded_view", decoded_view)):
        windows = slice_windows(name, text, config.window_chars, config.window_stride)
        if len(windows) > limit:
            return refuse("window_cap", name, len(windows), limit)
        views[name] = (text, windows)

    plans: list[WindowPlan] = []
    for tokenizer in config.tokenizers:
        split: dict[ViewName, tuple[TextWindow, ...]] = {}
        splits = 0
        for name, (text, windows) in views.items():
            sub_windows, count = split_for_tokenizer(name, text, windows, tokenizer)
            if len(sub_windows) > limit:
                return refuse("window_cap", name, len(sub_windows), limit, tokenizer.name)
            split[name] = sub_windows
            splits += count
        plans.append(
            WindowPlan(
                tokenizer=tokenizer.name,
                token_limit=tokenizer.token_limit,
                raw_windows=split["raw_view"],
                decoded_windows=split["decoded_view"],
                split_count=splits,
            )
        )

    return InspectionBundle(
        bundle_version=version,
        prompt_hash=digest,
        raw_view=raw_view,
        decoded_view=decoded_view,
        max_decoded_chars=config.max_decoded_chars,
        max_windows_per_view=limit,
        raw_windows=views["raw_view"][1],
        decoded_windows=views["decoded_view"][1],
        window_plans=tuple(plans),
        decode_steps=steps,
    )
