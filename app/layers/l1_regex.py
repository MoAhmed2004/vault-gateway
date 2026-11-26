"""L1 deterministic ingress rules (IP-1, IP-3, IP-9, TM-R2, §5.3 W3 "regex set").

The spec fixes where L1 runs (on both views of the `InspectionBundle`, first
in the cascade, 5 ms budget) but not its rule text, so the rule set below is
a [Project Decision, extends IP-1, IP-3]. It targets the attack shapes the spec
names: instruction override (SG1), system-prompt and canary extraction (SG4,
O1), persona jailbreaks (JailBreakV `template` family), chat-template and
`<EXTERNAL_DATA` delimiter spoofing (TM-S2, H5 delimiter breakout), and
"decode this, then obey it" encoding triggers (§3.4 Base64, hex and ROT13
wrap operators).

Views and obfuscation:
- `raw_view` is scanned as received, `decoded_view` after NFKC, `Cf` strip
  and Base64/hex decoding (IP-1), so zero-width insertion and Base64/hex
  wrapping cannot hide a rule match. A block on either view blocks.
- IP-1 does not decode ROT13, so L1 also runs every rule on the ROT13
  transform of `decoded_view` [Project Decision]; such a match is reported on
  the `decoded_view` result. ROT13 keeps every character position, so match
  spans map to the same windows.
- Homoglyphs are not folded (IP-1, optional H-NORM in §3.6) and are expected
  to bypass L1.

Scoring: every rule is a hard rule with weight 1.0, so a view scores 1.0 if
any rule matches and 0.0 otherwise, and the verdict is `block` at 1.0.
`per_window_scores` gives, for each base window of the view (600/300, IP-2),
the highest weight of any match that overlaps it. Matching is case-insensitive
except `DAN`, which is matched case-sensitively so the given name "Dan" is not
a jailbreak. Word gaps use bounded `(?:\\w+\\W+){0,n}?` runs of disjoint
character classes, so no pattern can backtrack catastrophically; the 4,000
character cap (TM-R1) bounds the work per view.

`L1_VERSION` is a SHA-256 over the revision, every pattern and its flags, and
the ROT13 setting, so editing any rule changes the IP-5 layer-verdict key.
"""

from __future__ import annotations

import codecs
import re
import time
from dataclasses import dataclass

from contracts.hashing import canonical_hash
from contracts.inspection import InspectionBundle, TextWindow
from contracts.layer_result import LayerResult

L1_RULES_REVISION = 1
BLOCK_THRESHOLD = 1.0
SCAN_ROT13 = True


def _gap(words: int) -> str:
    return rf"(?:\w+\W+){{0,{words}}}?"


_OVERRIDE_TARGET = (
    r"(?:instructions?|rules?|guidelines?|directives?|prompts?|guardrails?|polic(?:y|ies)|"
    r"restrictions?|constraints?|programming|filters?|safeguards?)"
)
_EXTRACTION_TARGET = (
    r"(?:system\s+(?:prompt|message|instructions?)|hidden\s+(?:prompt|instructions?|rules)|"
    r"(?:initial|original|secret|developer)\s+(?:prompt|instructions?|message)|"
    r"canary(?:\s+token)?|confidential\s+data|employee\s+record)"
)
_PERSONA = (
    r"(?:(?-i:DAN)|unrestricted|unfiltered|uncensored|jailbroken|amoral|"
    r"without\s+(?:any\s+)?(?:restrictions|filters|rules|limits|guidelines|ethics)|"
    r"no\s+(?:restrictions|rules|limits|filters|guidelines|ethics)|developer\s+mode)"
)

_EXTRACTION_ANCHORS = ("system", "hidden", "initial", "original", "secret", "developer", "canary", "confidential", "employee")

# (name, anchors, pattern). Every match of a rule contains at least one of its anchors (single words,
# compared against the casefolded text), so a rule whose anchors are all absent cannot match and is skipped.
RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "instruction_override",
        ("ignore", "disregard", "forget", "override", "bypass"),
        r"\b(?:ignore|disregard|forget|override|bypass)\W+" + _gap(2)
        + r"(?:previous|prior|above|earlier|preceding|initial|original|system|safety|your|developer)\W+"
        + _gap(2) + _OVERRIDE_TARGET + r"\b",
    ),
    (
        "instruction_override_all",
        ("ignore", "disregard"),
        r"\b(?:ignore|disregard)\W+(?:all|any|every)\W+" + _gap(1) + _OVERRIDE_TARGET + r"\b",
    ),
    (
        "prompt_extraction",
        _EXTRACTION_ANCHORS,
        r"\b(?:reveal|print|show|repeat|output|display|dump|leak|disclose|recite|expose|share|"
        r"tell\s+me|give\s+me|write\s+out)\W+" + _gap(4) + _EXTRACTION_TARGET + r"\b",
    ),
    (
        "prompt_extraction_question",
        _EXTRACTION_ANCHORS,
        r"\bwhat\s+(?:is|are|was|were)\s+(?:your|the)\s+" + _EXTRACTION_TARGET + r"\b",
    ),
    (
        "repeat_above",
        ("above", "before"),
        r"\b(?:repeat|print|output|recite|copy)\W+" + _gap(3)
        + r"(?:everything|all|text|words|content|instructions)\W+(?:above|before\s+this)\b",
    ),
    (
        "persona_jailbreak",
        ("dan", "unrestricted", "unfiltered", "uncensored", "jailbroken", "amoral", "without", "no", "developer"),
        r"\b(?:you\s+are\s+now|from\s+now\s+on,?\s+you(?:\s+are|'re|\s+will\s+be)?|act\s+as|"
        r"pretend\s+(?:to\s+be|that\s+you\s+are|you\s+are)|role-?play\s+as|simulate)\W+" + _gap(4) + _PERSONA + r"\b",
    ),
    (
        "jailbreak_mode",
        ("anything", "dan", "developer", "jailbreak", "god"),
        r"\b(?:do\s+anything\s+now|(?-i:DAN)\s+mode|developer\s+mode\s+(?:enabled|on|activated)|"
        r"jailbreak(?:ed)?\s+mode|god\s+mode\s+(?:enabled|on))\b",
    ),
    (
        "chat_template_token",
        ("<|", "inst]", "sys>>"),
        r"<\|\s*(?:im_start|im_end|system|user|assistant|endoftext|start_header_id|end_header_id|eot_id)\s*\|>"
        r"|\[/?INST\]|<</?SYS>>",
    ),
    ("external_data_delimiter", ("external_data",), r"</?\s*EXTERNAL_DATA"),
    (
        "system_override_header",
        ("system", "developer"),
        r"(?m)^\s*\[(?:system|developer)(?:\s+(?:override|message))?\]\s*:?"
        r"|(?m:^\s*(?:system|developer)\s+override\s*:)"
        r"|\bBEGIN\s+(?:SYSTEM|DEVELOPER)\s+(?:PROMPT|MESSAGE|INSTRUCTIONS)\b",
    ),
    (
        "decode_and_execute",
        ("follow", "execute", "obey", "carry", "act", "comply"),
        r"\b(?:decode|decrypt|deobfuscate|translate|unscramble|reverse|convert)\b\W+" + _gap(6)
        + r"(?:and|then)\W+" + _gap(2)
        + r"(?:follow|execute|obey|carry\s+out|act\s+on|comply\s+with)\W+"
        + r"(?:it|them|the\s+(?:instructions?|commands?)|its\s+instructions|what\s+it\s+says)\b",
    ),
)

_FLAGS = re.IGNORECASE
_COMPILED: tuple[tuple[str, tuple[str, ...], re.Pattern[str]], ...] = tuple(
    (name, anchors, re.compile(p, _FLAGS)) for name, anchors, p in RULES
)

L1_VERSION = canonical_hash(
    {
        "layer": "L1",
        "revision": L1_RULES_REVISION,
        "flags": int(_FLAGS),
        "rot13": SCAN_ROT13,
        "threshold": BLOCK_THRESHOLD,
        "rules": [[name, list(anchors), pattern] for name, anchors, pattern in RULES],
    }
)


@dataclass(frozen=True)
class RuleMatch:
    rule: str
    start: int
    end: int
    rot13: bool = False


# Non-ASCII characters that `re.IGNORECASE` (simple case mapping) treats as an ASCII letter. `str.casefold`
# uses full mapping and keeps U+0131 or yields "i" plus U+0307 for U+0130, so without this table the anchor
# prefilter could skip a rule the regex would match (for example a dotless "ignore"). The table is complete
# for Python 3.11; `tests/test_layer_l1.py` re-derives it over every code point.
# U+0130 dotted capital I, U+0131 dotless small i, U+017F long s, U+212A Kelvin sign.
REGEX_ASCII_EQUIVALENTS = str.maketrans({chr(0x0130): "i", chr(0x0131): "i", chr(0x017F): "s", chr(0x212A): "k"})


def _scan(text: str, *, rot13: bool) -> list[RuleMatch]:
    folded = text.translate(REGEX_ASCII_EQUIVALENTS).casefold()
    return [
        RuleMatch(name, m.start(), m.end(), rot13=rot13)
        for name, anchors, pattern in _COMPILED
        if any(anchor in folded for anchor in anchors)
        for m in pattern.finditer(text)
    ]


def find_matches(text: str, *, rot13: bool = SCAN_ROT13) -> tuple[RuleMatch, ...]:
    """Every rule match on `text` (and on its ROT13 transform when `rot13`), in rule order."""
    matches = _scan(text, rot13=False)
    if rot13:
        matches.extend(_scan(codecs.encode(text, "rot13"), rot13=True))
    return tuple(matches)


def _window_scores(windows: tuple[TextWindow, ...], matches: tuple[RuleMatch, ...]) -> tuple[float, ...]:
    scores = []
    for window in windows:
        hit = any(m.start < window.end and m.end > window.start for m in matches) or (
            window.start == window.end and bool(matches)
        )
        scores.append(1.0 if hit else 0.0)
    return tuple(scores)


class L1RegexLayer:
    """Deterministic rule layer over both bundle views (L1)."""

    name = "L1"
    version = L1_VERSION
    enabled = True

    @property
    def chroma_version(self) -> str | None:
        return None

    def _result(self, view: str, windows: tuple[TextWindow, ...], matches: tuple[RuleMatch, ...], started: float) -> LayerResult:
        per_window = _window_scores(windows, matches)
        overall = max(per_window, default=0.0)
        return LayerResult(
            layer=self.name,
            view=view,
            per_window_scores=per_window,
            overall_score=overall,
            verdict="block" if overall >= BLOCK_THRESHOLD else "allow",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def inspect(self, bundle: InspectionBundle, *, degraded: bool = False) -> tuple[LayerResult, ...]:
        """One result per view (raw, then decoded). L1 has no tunable threshold, so `degraded` changes nothing.

        When the decoded view equals the raw view (nothing was normalized or decoded), the plain rule
        scan of that text runs once and serves both views; the ROT13 scan applies to the decoded view only.
        """
        started = time.perf_counter()
        decoded_plain = tuple(_scan(bundle.decoded_view, rot13=False))
        decoded_rot = tuple(_scan(codecs.encode(bundle.decoded_view, "rot13"), rot13=True)) if SCAN_ROT13 else ()
        decoded = self._result("decoded_view", bundle.decoded_windows, decoded_plain + decoded_rot, started)
        started = time.perf_counter()
        same = bundle.raw_view == bundle.decoded_view
        raw_plain = decoded_plain if same else tuple(_scan(bundle.raw_view, rot13=False))
        raw = self._result("raw_view", bundle.raw_windows, raw_plain, started)
        return raw, decoded
