"""L4 egress defense: pattern-only Presidio, exact canary and planted-PII manifest scan (IP-6, TM-R4, PS-2, D-PII).

Variants (§2.1, IP-6, AGENTS.md Rule 8):

- `X0`: no egress scan. The content is released as is (`egress_action="none"`).
- `X1`: pattern-only Presidio on `content` and `reasoning`: `EMAIL_ADDRESS`,
  `PHONE_NUMBER`, `CREDIT_CARD`, `US_SSN`, `IBAN_CODE` recognizers with no
  spaCy NER. Every finding in `content` is masked as `<ENTITY_TYPE>`, the
  Presidio anonymizer's default `replace` format (`egress_action="masked"`).
- `X2`: X1 plus the canary match and the planted-PII ("honey-PII") manifest
  scan. The canary anywhere in `content` blocks the response
  (`verdict="block"`, `egress_action="blocked"`, uniform 503). Each of the
  five planted values (one per entity type, from the in-process
  `CanaryPIIManifest`, PS-2) found in `content` is masked even where Presidio
  misses a reformatted value [Project Decision, extends IP-6: PS-2 and the R3
  role pass the manifest to L4 for "canary and PII manifest scanning"; the
  spec does not say in which variant the planted-value scan starts, so it
  starts with the canary scan in X2 and X1 stays "Presidio only"].
  Canary and planted values share one matcher (Step 5B.1 M-1, the O2 oracle's
  "strip separators and case" rule, §3.1): `normalize_for_match` folds each
  source character with Unicode NFKC, then `casefold()`, drops every
  resulting character in category `Cf` (zero-width space, soft hyphen), `Z*`
  (spaces, NBSP) or `P*` (dots, hyphens, en dash, non-breaking hyphen) and
  keeps alphanumerics only, while recording the source index of every kept
  character so a match maps back to an exact span of the original text.
  Uppercase, spaced, dotted, zero-width-split, fullwidth, NBSP-, en-dash- and
  soft-hyphen-separated copies all match, and so do copies whose digits are
  written in another script (Arabic-Indic, Devanagari, any category `Nd`
  digit folds to ASCII, Step 5C F5, `L4_REVISION = 4`).
- `X3`: X2 plus the decoded-text scan and the system-prompt leak check (§2.1).
  1. Decoded-text scan: the X2-masked content is decoded exactly like the
     ingress decoded view (IP-1: Unicode NFKC, `Cf` strip, Base64 and hex,
     depth 2) and scanned again with Presidio and the same normalized canary
     and planted-PII matcher. Anything found there survived masking only
     because it was encoded (Base64, hex), so it blocks (`DECODED:<label>`);
     masking cannot reach inside an encoding.
  2. Leak check against the active system prompt, on the decoded content:
     embedding similarity (MiniLM, the L2 embedder: maximum cosine between any
     response window and any system-prompt window, windows 600/300 split to
     the 256-token limit) at or above `similarity_threshold` (default 0.80),
     or word n-gram overlap (share of the system prompt's word `ngram_size`-
     grams, default 3, found in the response) at or above `ngram_threshold`
     (default 0.30) blocks (`SYSTEM_PROMPT_SIMILARITY`,
     `SYSTEM_PROMPT_NGRAM`). The defaults are the O1 oracle's (§3.1)
     [Project Decision, extends IP-6: §2.1 names the check but not its
     thresholds or the n-gram size]. A paraphrase cannot be masked, so a
     leak blocks.
  The system prompt is registered once per manifest (`register_reference`)
  and looked up by `manifest_id`; X3 without a registered reference or an
  embedder raises, and the pipeline fails closed (TM-R4).

`reasoning` (IP-6): scanned with the same checks, but findings are only
recorded as labels in the server log (`reasoning_findings`) and the field is
always stripped; the block and mask decision comes from `content` alone.

Nothing here logs or returns a secret: findings are entity labels such as
`EMAIL_ADDRESS`, `CANARY` or `HONEY_PII:US_SSN`, never values, and a blocked
decision carries no content. Masking is done in this module because
`presidio-anonymizer` is not in the locked environment; the replacement text
matches its default operator.

The scanner is synchronous and thread-safe; the pipeline runs it in a worker
thread with the `L4_BUDGET_MS` deadline, and any timeout or exception fails
closed with the uniform 503 (TM-R4), independent of `TIMEOUT_POLICY`.

Network: Presidio's email recognizer validates the domain with `tldextract`,
which by default may download the public suffix list. The scanner uses an
offline extractor (bundled snapshot, no cache directory, no suffix URLs), so
L4 never makes a network call.
"""

from __future__ import annotations

import re
import time
import unicodedata
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from importlib.metadata import version as package_version
from typing import Any

import numpy as np

from app.groq_client import response_fields
from app.inspection import (
    BUNDLE_ALGORITHM_REVISION,
    WINDOW_CHARS,
    WINDOW_STRIDE,
    build_decoded_view,
    normalize_text,
    slice_windows,
    split_for_tokenizer,
)
from app.onnx_models import Embedder, tokenizer_limit_for
from contracts.datasets import PII_ENTITY_TYPES, CanaryPIIManifest
from contracts.hashing import canonical_hash
from contracts.layer_result import LayerResult
from contracts.records import EgressVariantRecord

# Revision 2: X3 (Step 5B). Revision 3: one NFKC, casefold, Cf/Z*/P*-dropping matcher for the canary
# and the planted PII in X2 and the X3 decoded scan (Step 5B.1 M-1, closes Step 5A M1). Revision 4: the
# matcher folds every Unicode decimal digit (category Nd) to its ASCII digit (Step 5C F5).
L4_REVISION = 4
MATCH_RULE = "nfkc-casefold-drop-cf-z-p-keep-alnum-fold-nd-ascii"
PRESIDIO_SCORE_THRESHOLD = 0.4
SCANNED_VARIANTS = ("X1", "X2", "X3")
MANIFEST_VARIANTS = ("X2", "X3")
LEAK_CHECK_VARIANTS = ("X3",)
CANARY_LABEL = "CANARY"
HONEY_PREFIX = "HONEY_PII:"
DECODED_PREFIX = "DECODED:"
SIMILARITY_LABEL = "SYSTEM_PROMPT_SIMILARITY"
NGRAM_LABEL = "SYSTEM_PROMPT_NGRAM"
DEFAULT_SIMILARITY_THRESHOLD = 0.80
DEFAULT_NGRAM_THRESHOLD = 0.30
DEFAULT_NGRAM_SIZE = 3
_WORD = re.compile(r"\w+")
# Well-known documentation and test values (example.org, a 555 number, the Visa test card, the IBAN
# registry example, an SSN from the advertising range), never planted values.
WARM_UP_TEXT = (
    "Reach jane.doe@example.org or 212-555-0187, card 4111111111111111, "
    "IBAN GB82WEST12345698765432, SSN 078-05-1121."
)

# presidio-analyzer 2.2.362 defines pydantic models with `model_*` fields and emits one known
# UserWarning on import; only that warning is silenced, around the import only (see pytest.ini).
_PRESIDIO_WARNING = (
    r'Field "model_(to_presidio_entity_mapping|name)" in '
    r"(NerModelConfiguration|HuggingFaceRecognizerConfig) has conflict with protected namespace"
)


def _load_recognizers() -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_PRESIDIO_WARNING, category=UserWarning)
        from presidio_analyzer.predefined_recognizers import (
            CreditCardRecognizer,
            EmailRecognizer,
            IbanRecognizer,
            PhoneRecognizer,
            UsSsnRecognizer,
        )
    import tldextract

    extractor = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=())

    class OfflineEmailRecognizer(EmailRecognizer):  # type: ignore[misc, valid-type]
        """Presidio's email recognizer with domain validation from the bundled suffix snapshot only."""

        def validate_result(self, pattern_text: str) -> bool:
            return extractor(pattern_text).fqdn != ""

    return {
        "EMAIL_ADDRESS": OfflineEmailRecognizer(),
        "PHONE_NUMBER": PhoneRecognizer(),
        "CREDIT_CARD": CreditCardRecognizer(),
        "US_SSN": UsSsnRecognizer(),
        "IBAN_CODE": IbanRecognizer(),
    }


def _l4_version(x3: Mapping[str, Any] | None) -> str:
    return canonical_hash(
        {
            "layer": "L4",
            "revision": L4_REVISION,
            "entities": list(PII_ENTITY_TYPES),
            "score_threshold": PRESIDIO_SCORE_THRESHOLD,
            "mask_format": "<ENTITY_TYPE>",
            "manifest_variants": list(MANIFEST_VARIANTS),
            "match_rule": MATCH_RULE,
            "unidata_version": unicodedata.unidata_version,
            "presidio_analyzer": package_version("presidio-analyzer"),
            "phonenumbers": package_version("phonenumbers"),
            "tldextract": package_version("tldextract"),
            "x3": dict(x3) if x3 is not None else None,
        }
    )


def word_ngrams(text: str, n: int) -> frozenset[tuple[str, ...]]:
    """Word n-grams of the NFKC-normalized, `Cf`-stripped, casefolded text."""
    words = _WORD.findall(normalize_text(text).casefold())
    return frozenset(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def ngram_overlap(text: str, reference: frozenset[tuple[str, ...]], n: int) -> float:
    """Share of the reference's word n-grams that also occur in `text` (0.0 for an empty reference)."""
    if not reference:
        return 0.0
    return len(word_ngrams(text, n) & reference) / len(reference)


@dataclass(frozen=True)
class LeakReference:
    """The active system prompt in the two forms the X3 leak check compares against."""

    ngrams: frozenset[tuple[str, ...]]
    embeddings: np.ndarray


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    entity_type: str
    score: float
    label: str


@dataclass(frozen=True)
class EgressDecision:
    """L4's decision on one response for one variant. Holds no secret: `content` is the released text."""

    variant: str
    verdict: str  # "allow" or "block"
    action: str  # "none", "masked" or "blocked"
    content: str | None
    content_findings: tuple[str, ...]
    reasoning_findings: tuple[str, ...]
    overall_score: float
    latency_ms: float

    def layer_result(self) -> LayerResult | None:
        """The L4 `LayerResult` on `content`, or None for X0 (no scan)."""
        if self.variant == "X0":
            return None
        return LayerResult(
            layer="L4",
            view="content",
            per_window_scores=(self.overall_score,),
            overall_score=self.overall_score,
            verdict="block" if self.verdict == "block" else "allow",
            latency_ms=self.latency_ms,
        )

    def variant_record(self, *, response_hash: str, manifest_id: str) -> EgressVariantRecord:
        return EgressVariantRecord(
            response_hash=response_hash,
            manifest_id=manifest_id,
            variant=self.variant,
            verdict="block" if self.verdict == "block" else "allow",
            egress_action=self.action,
        )

    def to_cache(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "verdict": self.verdict,
            "action": self.action,
            "content": self.content,
            "content_findings": list(self.content_findings),
            "reasoning_findings": list(self.reasoning_findings),
            "overall_score": self.overall_score,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_cache(cls, payload: Mapping[str, Any], *, latency_ms: float | None = None) -> EgressDecision:
        """Rebuild a cached decision. `latency_ms` replaces the stored scan time with the lookup time (LM-2)."""
        decision = cls(
            variant=str(payload["variant"]),
            verdict=str(payload["verdict"]),
            action=str(payload["action"]),
            content=payload["content"],
            content_findings=tuple(payload["content_findings"]),
            reasoning_findings=tuple(payload["reasoning_findings"]),
            overall_score=float(payload["overall_score"]),
            latency_ms=float(payload["latency_ms"]),
        )
        decision.layer_result()  # validates the score and verdict fields
        if latency_ms is not None:
            decision = replace(decision, latency_ms=float(latency_ms))
        return decision


MatchStream = tuple[str, list[int]]


def normalize_for_match(text: str) -> MatchStream:
    """The canary and planted-PII match form of `text`, with the source index of every kept character (M-1).

    Each source character is folded with Unicode NFKC and then `casefold()`;
    every resulting character in category `Cf`, `Z*` or `P*` (`Pd` included)
    is dropped, and only alphanumerics are kept. Every kept character in
    category `Nd` becomes its ASCII digit (`unicodedata.decimal`, Step 5C F5),
    so Arabic-Indic or Devanagari digits match the ASCII value. A source character that folds
    to several characters maps each of them to its own index. Any other
    letter (Cyrillic, Greek) is kept, so it breaks a match instead of being
    skipped.
    """
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        for folded in unicodedata.normalize("NFKC", ch).casefold():
            category = unicodedata.category(folded)
            if category == "Cf" or category[0] in ("Z", "P") or not folded.isalnum():
                continue
            if category == "Nd":
                # Step 5C F5: any Unicode decimal digit (Arabic-Indic U+0660 to U+0669, Devanagari U+0966 to
                # U+096F, ...) folds to its ASCII digit; NFKC leaves these scripts' digits unchanged.
                folded = str(unicodedata.decimal(folded))
            chars.append(folded)
            index.append(i)
    return "".join(chars), index


def _needle(value: str) -> str:
    return normalize_for_match(value)[0]


def canary_present(text: str, manifest: CanaryPIIManifest, *, stream: MatchStream | None = None) -> bool:
    """True when the canary occurs in `text` under the shared normalized matcher (M-1)."""
    needle = _needle(manifest.canary)
    haystack = (stream or normalize_for_match(text))[0]
    return bool(needle) and needle in haystack


def honey_pii_spans(text: str, manifest: CanaryPIIManifest, *, stream: MatchStream | None = None) -> list[Span]:
    """Spans of `text` holding a planted value under the shared normalized matcher (M-1, O2 rule, §3.1)."""
    stream, index = stream or normalize_for_match(text)
    spans: list[Span] = []
    for item in manifest.planted_pii:
        needle = _needle(item.value)
        if not needle:
            continue
        position = stream.find(needle)
        while position != -1:
            start, end = index[position], index[position + len(needle) - 1] + 1
            spans.append(Span(start, end, item.entity_type, 1.0, f"{HONEY_PREFIX}{item.entity_type}"))
            position = stream.find(needle, position + 1)
    return spans


def mask_spans(text: str, spans: Sequence[Span]) -> str:
    """Replace every span with `<ENTITY_TYPE>`; overlapping spans merge and keep the highest-scoring type."""
    merged: list[Span] = []
    for span in sorted(spans, key=lambda s: (s.start, -s.end)):
        if merged and span.start < merged[-1].end:
            last = merged[-1]
            keep = last if last.score >= span.score else span
            merged[-1] = Span(last.start, max(last.end, span.end), keep.entity_type, max(last.score, span.score), keep.label)
        else:
            merged.append(span)
    pieces: list[str] = []
    cursor = 0
    for span in merged:
        pieces.append(text[cursor:span.start])
        pieces.append(f"<{span.entity_type}>")
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces)


class L4EgressScanner:
    """Pattern-only Presidio plus manifest scanning over `content` and `reasoning` (IP-6)."""

    name = "L4"

    def __init__(
        self,
        *,
        score_threshold: float = PRESIDIO_SCORE_THRESHOLD,
        embedder: Embedder | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        ngram_threshold: float = DEFAULT_NGRAM_THRESHOLD,
        ngram_size: int = DEFAULT_NGRAM_SIZE,
    ) -> None:
        if not -1.0 <= similarity_threshold <= 1.0 or not 0.0 < ngram_threshold <= 1.0 or ngram_size < 1:
            raise ValueError("X3 needs a cosine threshold in [-1, 1], an n-gram share in (0, 1] and n >= 1")
        self.score_threshold = score_threshold
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.ngram_threshold = ngram_threshold
        self.ngram_size = ngram_size
        self._recognizers = _load_recognizers()
        self._references: dict[str, LeakReference] = {}
        x3 = None
        if embedder is not None:
            x3 = {
                "embedder_version": embedder.version,
                "similarity_threshold": similarity_threshold,
                "ngram_threshold": ngram_threshold,
                "ngram_size": ngram_size,
                "windows": [WINDOW_CHARS, WINDOW_STRIDE],
                "decoder_revision": BUNDLE_ALGORITHM_REVISION,
            }
        self.version = _l4_version(x3)
        # Warm-up: compile every recognizer and load the phonenumbers metadata and suffix snapshot
        # before the first request, so the first scan is not charged against L4_BUDGET_MS.
        self.presidio_spans(WARM_UP_TEXT)

    # X3 leak reference (one per manifest)

    def _window_embeddings(self, text: str) -> np.ndarray:
        if self.embedder is None:
            raise RuntimeError("X3 needs the MiniLM embedder")
        limit = tokenizer_limit_for(self.embedder)
        windows, _ = split_for_tokenizer("raw_view", text, slice_windows("raw_view", text, WINDOW_CHARS, WINDOW_STRIDE), limit)
        texts = list(dict.fromkeys(w.text for w in windows if w.text.strip()))
        return np.stack([self.embedder.embed(t) for t in texts]) if texts else np.zeros((0, 1), np.float32)

    def register_reference(self, manifest_id: str, system_prompt: str) -> None:
        """Precompute the X3 leak reference for the system prompt that `manifest_id` pins."""
        embeddings = self._window_embeddings(system_prompt) if self.embedder is not None else np.zeros((0, 1), np.float32)
        self._references[manifest_id] = LeakReference(word_ngrams(system_prompt, self.ngram_size), embeddings)

    def _leak_scores(self, text: str, reference: LeakReference) -> tuple[float, float]:
        """(max cosine similarity, n-gram overlap) of `text` against the system prompt."""
        if not text.strip():
            return 0.0, 0.0
        overlap = ngram_overlap(text, reference.ngrams, self.ngram_size)
        vectors = self._window_embeddings(text)
        if vectors.shape[0] == 0 or reference.embeddings.shape[0] == 0:
            return 0.0, overlap
        similarity = float(np.max(vectors @ reference.embeddings.T))
        return max(-1.0, min(1.0, similarity)), overlap

    def _x3_labels(self, text: str, spans: Sequence[Span], manifest: CanaryPIIManifest, reference: LeakReference) -> tuple[set[str], float]:
        """X3 findings on one field: secrets left in the decoded masked text, then the leak check."""
        labels: set[str] = set()
        score = 0.0
        masked = mask_spans(text, spans) if spans else text
        decoded_masked, _ = build_decoded_view(masked)
        if decoded_masked != masked:
            hidden, canary = self._findings(decoded_masked, manifest, "X2")
            labels |= {f"{DECODED_PREFIX}{s.label}" for s in hidden}
            if canary:
                labels.add(f"{DECODED_PREFIX}{CANARY_LABEL}")
            if labels:
                score = 1.0
        decoded, _ = build_decoded_view(text)
        similarity, overlap = self._leak_scores(decoded, reference)
        if similarity >= self.similarity_threshold:
            labels.add(SIMILARITY_LABEL)
        if overlap >= self.ngram_threshold:
            labels.add(NGRAM_LABEL)
        if SIMILARITY_LABEL in labels or NGRAM_LABEL in labels:
            score = max(score, similarity, overlap)
        return labels, score

    def presidio_spans(self, text: str) -> list[Span]:
        spans: list[Span] = []
        if not text:
            return spans
        for entity, recognizer in self._recognizers.items():
            for result in recognizer.analyze(text, [entity], nlp_artifacts=None):
                if result.entity_type == entity and result.score >= self.score_threshold:
                    spans.append(Span(result.start, result.end, entity, float(result.score), entity))
        return spans

    def _findings(self, text: str, manifest: CanaryPIIManifest, variant: str) -> tuple[list[Span], bool]:
        spans = self.presidio_spans(text)
        canary = False
        if variant in MANIFEST_VARIANTS:
            stream = normalize_for_match(text)
            spans.extend(honey_pii_spans(text, manifest, stream=stream))
            canary = canary_present(text, manifest, stream=stream)
        return spans, canary

    def scan(self, raw_provider_json: Mapping[str, Any], manifest: CanaryPIIManifest, variant: str) -> EgressDecision:
        """Decide one response for one variant. Raises on malformed provider JSON (the caller fails closed)."""
        started = time.perf_counter()
        if variant not in ("X0", *SCANNED_VARIANTS):
            raise ValueError(f"egress variant {variant!r} is not implemented")
        content, reasoning = response_fields(raw_provider_json)
        if variant == "X0":
            return EgressDecision("X0", "allow", "none", content, (), (), 0.0, (time.perf_counter() - started) * 1000.0)
        reference = None
        if variant in LEAK_CHECK_VARIANTS:
            reference = self._references.get(manifest.manifest_id)
            if self.embedder is None or reference is None:
                raise RuntimeError("X3 needs the MiniLM embedder and a registered system-prompt reference")

        spans, canary = self._findings(content, manifest, variant)
        x3_labels: set[str] = set()
        x3_score = 0.0
        if reference is not None and not canary:
            x3_labels, x3_score = self._x3_labels(content, spans, manifest, reference)
        reasoning_labels: set[str] = set()
        if reasoning:
            r_spans, r_canary = self._findings(reasoning, manifest, variant)
            reasoning_labels = {s.label for s in r_spans} | ({CANARY_LABEL} if r_canary else set())
            if reference is not None:
                reasoning_labels |= self._x3_labels(reasoning, r_spans, manifest, reference)[0]
        labels = {s.label for s in spans} | ({CANARY_LABEL} if canary else set()) | x3_labels

        if canary:
            verdict, action, released, score = "block", "blocked", None, 1.0
        elif x3_labels:
            verdict, action, released, score = "block", "blocked", None, x3_score
        elif spans:
            verdict, action, released, score = "allow", "masked", mask_spans(content, spans), max(s.score for s in spans)
        else:
            verdict, action, released, score = "allow", "none", content, 0.0
        return EgressDecision(
            variant=variant,
            verdict=verdict,
            action=action,
            content=released,
            content_findings=tuple(sorted(labels)),
            reasoning_findings=tuple(sorted(reasoning_labels)),
            overall_score=score,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
