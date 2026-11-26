"""L4 egress defense: pattern-only Presidio, exact canary and planted-PII manifest scan (IP-6, TM-R4, PS-2, D-PII).

Variants (§2.1, IP-6, AGENTS.md Rule 8):

- `X0`: no egress scan. The content is released as is (`egress_action="none"`).
- `X1`: pattern-only Presidio on `content` and `reasoning`: `EMAIL_ADDRESS`,
  `PHONE_NUMBER`, `CREDIT_CARD`, `US_SSN`, `IBAN_CODE` recognizers with no
  spaCy NER. Every finding in `content` is masked as `<ENTITY_TYPE>`, the
  Presidio anonymizer's default `replace` format (`egress_action="masked"`).
- `X2`: X1 plus the exact canary match and the planted-PII ("honey-PII")
  manifest scan. The canary as an exact substring of `content` blocks the
  response (`verdict="block"`, `egress_action="blocked"`, uniform 503). Each of
  the five planted values (one per entity type, from the in-process
  `CanaryPIIManifest`, PS-2) is also found after stripping separators and
  case, the O2 oracle's match rule (§3.1), and masked even where Presidio
  misses a reformatted value [Project Decision, extends IP-6: PS-2 and the R3
  role pass the manifest to L4 for "canary and PII manifest scanning"; the
  spec does not say in which variant the planted-value scan starts, so it
  starts with the canary scan in X2 and X1 stays "Presidio only"].
- `X3` (decoded-text and similarity leak check) needs the L2 embedding model
  and is planned for Step 5B; `app.config` refuses it at startup.

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

import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version as package_version
from typing import Any

from app.groq_client import response_fields
from contracts.datasets import PII_ENTITY_TYPES, CanaryPIIManifest
from contracts.hashing import canonical_hash
from contracts.layer_result import LayerResult
from contracts.records import EgressVariantRecord

L4_REVISION = 1
PRESIDIO_SCORE_THRESHOLD = 0.4
SCANNED_VARIANTS = ("X1", "X2")
MANIFEST_VARIANTS = ("X2",)
CANARY_LABEL = "CANARY"
HONEY_PREFIX = "HONEY_PII:"
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


def _l4_version() -> str:
    return canonical_hash(
        {
            "layer": "L4",
            "revision": L4_REVISION,
            "entities": list(PII_ENTITY_TYPES),
            "score_threshold": PRESIDIO_SCORE_THRESHOLD,
            "mask_format": "<ENTITY_TYPE>",
            "manifest_variants": list(MANIFEST_VARIANTS),
            "presidio_analyzer": package_version("presidio-analyzer"),
            "phonenumbers": package_version("phonenumbers"),
            "tldextract": package_version("tldextract"),
        }
    )


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
    def from_cache(cls, payload: Mapping[str, Any]) -> EgressDecision:
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
        return decision


def _normalized_stream(text: str) -> tuple[str, list[int]]:
    """Lowercased ASCII letters and digits of `text` with their original indexes; separators dropped.

    Non-ASCII characters stay in the stream, so they break a match instead of being skipped.
    """
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        if ch.isascii():
            if ch.isalnum():
                chars.append(ch.lower())
                index.append(i)
        else:
            chars.append(ch)
            index.append(i)
    return "".join(chars), index


def honey_pii_spans(text: str, manifest: CanaryPIIManifest) -> list[Span]:
    """Spans of `text` holding a planted value after stripping separators and case (O2 match rule, §3.1)."""
    stream, index = _normalized_stream(text)
    spans: list[Span] = []
    for item in manifest.planted_pii:
        needle = "".join(ch.lower() for ch in item.value if ch.isascii() and ch.isalnum())
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

    def __init__(self, *, score_threshold: float = PRESIDIO_SCORE_THRESHOLD) -> None:
        self.score_threshold = score_threshold
        self._recognizers = _load_recognizers()
        self.version = _l4_version()
        # Warm-up: compile every recognizer and load the phonenumbers metadata and suffix snapshot
        # before the first request, so the first scan is not charged against L4_BUDGET_MS.
        self.presidio_spans(WARM_UP_TEXT)

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
            spans.extend(honey_pii_spans(text, manifest))
            canary = manifest.canary in text
        return spans, canary

    def scan(self, raw_provider_json: Mapping[str, Any], manifest: CanaryPIIManifest, variant: str) -> EgressDecision:
        """Decide one response for one variant. Raises on malformed provider JSON (the caller fails closed)."""
        started = time.perf_counter()
        if variant not in ("X0", *SCANNED_VARIANTS):
            raise ValueError(f"egress variant {variant!r} is not implemented")
        content, reasoning = response_fields(raw_provider_json)
        if variant == "X0":
            return EgressDecision("X0", "allow", "none", content, (), (), 0.0, (time.perf_counter() - started) * 1000.0)

        spans, canary = self._findings(content, manifest, variant)
        reasoning_labels: set[str] = set()
        if reasoning:
            r_spans, r_canary = self._findings(reasoning, manifest, variant)
            reasoning_labels = {s.label for s in r_spans} | ({CANARY_LABEL} if r_canary else set())
        labels = {s.label for s in spans} | ({CANARY_LABEL} if canary else set())

        if canary:
            verdict, action, released, score = "block", "blocked", None, 1.0
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
