"""L4 output sanitizer for downstream-dangerous payloads (IP-10, LLM05).

Spec rule (IP-10, IP-6, TM-R4, TM-R6): when `ENABLE_OUTPUT_SANITIZER=true`,
L4 checks the client-visible `content` for payloads that are dangerous to a
downstream interpreter, after the egress-variant scan (X1 to X3) and before
release. It is not an egress variant: its verdict is never written to the
IP-5 egress cache, so X0 to X3 results are identical whether the flag is on
or off, and it is logged as a separate record (`output_sanitizer_logs`). It
runs in the bounded pool (IP-7), shares the L4 time budget (TM-R2) and fails
closed on a timeout or an error (TM-R4). The pipeline wiring is in
`app/pipeline.py`.

Matching form (since `out_san_v2`, Step 5C.1 M-1): Unicode NFKC on every
source character, then one round of case-sensitive HTML-entity decoding
(named references as HTML5 defines them, so `&Tab;` and `&NewLine;` decode;
numeric references with any number of digits, so zero padding cannot hide
one, and anything above U+10FFFF becomes U+FFFD), then NFKC and `casefold()`
on every character. `&lt;SCRIPT`, `&#x6a;avascript:`, `java&Tab;script:` and
a fullwidth less-than sign (U+FF1C) before `script` all match. Every character
of the matching form keeps the source span it came from, so a match maps back
to an exact span of the original content.

Tags are read with the WHATWG start-tag rules (Step 5C.3 B-1), with no
length cap (Step 5C.1 H-1). A tag name is an ASCII letter followed by any
characters but whitespace (tab, LF, FF, CR, space), `/` and `>`, with no
length cap (Step 5C.4 B-2), so `<x_y`, `<x.y`, `<a#` and a name with an
accented letter are tags, and `<img<img src=...>` is, for a browser, a tag
named `img<img` that fetches nothing. A tag with no attribute after its name
has nothing to scan. After `<` and the tag name, the body is a run of items,
each HTML whitespace (tab, LF, FF, CR, space; not every Unicode space), a `/`,
or an attribute: a name (a first character that is not whitespace, `/` or
`>`, so it may be `=`, then any characters but whitespace, `/`, `>` and `=`),
then optionally `=` and a value, double quoted, single quoted (either may hold
`>` and `<`) or unquoted. An unquoted value runs to whitespace or `>`, so it
holds `<`, `=` and quotes (`<img src=x=" onerror=...">` has an `onerror`
attribute, `<img src=x<1 onerror=...>` too, `<img/src=x/onerror=...>` has
none). A quote therefore opens a value only right after `=` (Step 5C.2 F-1);
a quote with no closing quote starts an unquoted value [Project Decision,
extends IP-10: the browser reads to the end of the input and drops the tag;
reading on keeps the rest of the response scanned]. Inline event handlers,
`javascript:` in a tag attribute value and the HTML image elements are read
from the attributes of every tag start, also one inside another tag's value (a
tag that the browser does not see, inside a comment or a `<textarea>`, cannot
hide a real one). The attributes that follow an item boundary do not depend on
where the tag started, so the scan memoizes them per boundary and stays
linear.

Inner tag starts (Step 5C.5 B-3, version `out_san_v4`): a Markdown renderer
can turn `<x<img src=//attacker.example/p.png>` into `&lt;x<img src=...>`
with the same URL, whose inner `<img>` is a live image. Besides the
browser-accurate start, every `<` followed by an ASCII letter inside a tag
name is therefore also indexed as a tag start, so `<x<img src=...>`,
`<x<input type=image src=...>` and `<img<img src=...>` are scanned as image
tags too. The inner name is only kept when it is no longer than the longest
image element name (`_LONGEST_IMAGE_TAG`), which keeps the work per start
constant. Inner starts share the item boundary of the outer one, so the scan
stays linear.

Structure-preserving view (Step 5C.2, extended in Step 5C.3): a browser
decodes a character reference inside a tag only in an attribute value, after
tokenization, and never applies NFKC, so `&quot;`, `&#39;`, `&gt;`,
`&equals;`, `&sol;`, `&#32;`, `&Tab;` or a fullwidth equals sign (U+FF1D)
there is ordinary text, never a delimiter. The tag scans therefore also
tokenize a copy of the matching form in which every delimiter in
`_STRUCTURAL` (`<`, `>`, `"`, `'`, `=`, `/` and HTML whitespace) that came
from a reference or from an NFKC mapping is an inert placeholder; attribute
values are then read from the decoded matching form at the same offsets, as
the browser decodes them, so `java&Tab;script:` and `/&#9;/attacker.example`
are read as the browser reads them (this replaces the Step 5C.2 F-2 special
case). The decoded-view pass still runs, so a fully encoded tag
(`&lt;script&gt;`) is still caught; findings of the two passes are merged, so
matches only add.

Pattern families (version `OUTPUT_SANITIZER_VERSION`; each pattern is tagged
with its family and its `neutralize` action):

(a) `xss`: a `<script` tag; an inline event-handler attribute (`on<name>=`)
    inside a tag; a `javascript:` URI inside a tag attribute; a `javascript:`
    URI anywhere else (Markdown link target, bare link, and inside `<...>`
    even with spaces, `[x](<javascript: alert(1)>)`). Browsers ignore tab,
    CR and LF inside a URL scheme, so `java&#x09;script:` matches too. Prose
    that only names the attribute or the scheme (`the onerror attribute`,
    `javascript: URLs`) does not match.
(b) `image_exfil`: a Markdown image (inline `![alt](url)` or reference style
    `![alt][ref]` with its `[ref]: url` definition, the URL possibly on the
    next line, also inside a block quote or list item; the alt text may hold
    escaped brackets, brackets nested to any depth and single line breaks,
    read by a linear bracket counter, as in CommonMark) or an HTML element
    that fetches an image without a click: `<img>` and SVG `<image>` (`src`,
    `srcset`, `href`, `xlink:href`), `<source>` (`src`, `srcset`), `<video>`
    (`poster`) and `<input type=image>` (`src`), also as an inner tag start
    (`<x<img src=...>`), whose URL is absolute (`http:`, `https:`, any number
    of slashes) or protocol relative (`//host`, `\\host`) and whose hostname
    is not in `OUTPUT_IMAGE_ALLOWLIST` (exact, lowercase; empty by default, so
    every external image matches). Tab, CR and LF inside a URL attribute
    value (quoted, or from a character reference in an unquoted one, decoded
    as the browser decodes a value) are dropped before the scheme and host are
    read, as the URL parser does. Relative image paths and ordinary Markdown
    links are not images and do not match. CSS `url()`, `data:text/html`
    frames and the media fetches of `<video src>` and `<audio src>` are
    outside IP-10's named list.
(c) `sqli_cmdi`, tuned so ordinary SQL, shell and code samples do not match:
    `UNION [ALL|DISTINCT] SELECT` right after a closing quote or a bare
    number (`id=1 UNION SELECT`, not `SELECT 1 UNION SELECT 2`), optionally
    followed by closing parens, or followed by `NULL`, `@@`,
    `information_schema` or a fingerprint call; a stacked destructive
    statement on the same line (`; DROP TABLE`, `;DROP/**/TABLE`,
    `; DELETE FROM`, `; TRUNCATE TABLE`, `; UPDATE ... SET`); a tautology
    after a quote (`' OR '1'='1`, `admin'or'1'='1`, `" OR "1"="1`,
    `' OR 1=1`) or `OR n=n`; shell command substitution `$(...)` of a recon,
    fetch or destructive command, with an optional `sudo`, `env`, `command`
    or `exec` wrapper (itself possibly with a directory path,
    `/usr/bin/env bash`, and sudo flags with one argument, `sudo -u root`),
    a directory path (`$(/usr/bin/id)`) and flags (`$(id -u)`); backtick
    substitution of such a command with arguments (`` `cat /etc/passwd` ``)
    or, for a bare command (`` `id` ``, `` `uname -r` ``), in an execution
    context (`=`, `(`, `$`, a quote, `;`, `|`, `&`, or after `echo`, `eval`,
    `exec`, `system`); and a pipe into a shell (`| sh`, `| /bin/bash`,
    `| sudo -E bash`, `| zsh`, `| cmd`, `| powershell`, `| pwsh`, `| iex`) or
    into an interpreter that runs its input (`| python3`, `| perl`, `| ruby`,
    `| php`, `| node`, with nothing after it but `-`) that is not a Markdown
    table cell. Inline code such as `` `user_prompt` ``, `` `rm -rf build/` ``
    or `$(date)` does not match.

Actions (`OUTPUT_SANITIZER_MODE`):

- `block` (default): any match blocks the response (uniform 503, TM-R6).
- `neutralize`: each matched span is HTML-escaped and every external image URL
  and every `javascript:` URI outside a tag is removed before release.
  [Project Decision, extends IP-10] The escape writes numeric character
  references not only for `& < > " '` (`html.escape`) but also for the shell
  and SQL metacharacters `` $ ` ( ) ; | `` inside a matched span, because
  `html.escape` leaves `$(whoami)`, `` `id` `` and `; DROP TABLE` unchanged;
  an HTML renderer shows the same text. A matched span that this escape
  would leave unchanged (`1 UNION SELECT`) is written entirely as numeric
  character references (Step 5C.2 F-5), so neutralized output never holds an
  intact matched phrase. A `javascript:` URI outside a tag is removed rather
  than escaped, because Markdown renderers decode character references in
  link destinations. Where an escape span and a removal span overlap, the
  removal wins.

The sanitizer is synchronous, stateless after construction and thread-safe.
It never logs content; the result carries the family names only.
"""

from __future__ import annotations

import html
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from urllib.parse import unquote

from contracts.hashing import canonical_hash

OUTPUT_SANITIZER_VERSION = "out_san_v4"
FAMILIES = ("xss", "image_exfil", "sqli_cmdi")
MODES = ("block", "neutralize")
ESCAPE = "escape"
REMOVE = "remove"

# One round of HTML character references, matched case-sensitively on the NFKC form (Step 5C.1 M-1): numeric
# (decimal, hex, any number of digits, so zero padding cannot hide one) and named, with or without the `;`.
_ENTITY = re.compile(r"&(?:#[xX][0-9a-fA-F]+;?|#[0-9]+;?|[A-Za-z][A-Za-z0-9]{1,31};?)")
_MAX_CODEPOINT = 0x10FFFF
_MAX_CODEPOINT_DIGITS = 8  # more significant digits than this is above U+10FFFF in base 10 and base 16
_URL_NOISE = "[\\t\\n\\r]*"  # browsers drop tab, CR and LF inside a URL (WHATWG URL parser)
_JS_SCHEME = _URL_NOISE.join("javascript") + _URL_NOISE + ":"
# WHATWG start-tag grammar (Step 5C.3 B-1). HTML whitespace is tab, LF, FF, CR and space only: `\s` would also
# split a value at U+001C or U+2028, which the browser keeps inside it, and a quote after that split would then
# swallow a real handler (`<img a=b<U+001C>c=" onerror=...">`).
_WS = r"[\t\n\f\r ]"
# An attribute name: the first character may be `=` (the tokenizer starts a name with it), the rest stops at `=`.
_ATTR_NAME = r"[^\t\n\f\r />][^\t\n\f\r />=]*+"
# One body item: whitespace, a `/`, or an attribute with an optional value. An unquoted value runs to whitespace
# or `>`, so it holds `<`, `=` and quotes; a quote opens a value only right after `=` (Step 5C.2 F-1), and one
# with no closing quote starts an unquoted value. Groups: 1 name, 2 double-quoted, 3 single-quoted, 4 unquoted.
# Atomic and possessive, so an item has one reading and a tag costs one pass.
_TAG_ITEM = (
    r"(?>" + _WS + r"++|/|(" + _ATTR_NAME + r")(?:" + _WS + r"*+=" + _WS
    + r"""*+(?:"([^"]*)"|'([^']*)'|([^\t\n\f\r >]++)))?)"""
)
_TAG_STEP = re.compile(_TAG_ITEM)
# A tag start (Step 5C.4 B-2): `<`, an ASCII letter, then any characters but HTML whitespace, `/` and `>`, with no
# length cap, as the WHATWG tag-name state reads it (`<x_y`, `<x.y`, an accented name and `<a#` are tag names, and
# so is `img<img`). Possessive, so a name has one reading. The character after the name must be whitespace or `/`
# (a tag with no attribute has nothing to scan); `TagChains` checks it after the match, because a lookahead here
# makes `<a<a<a...` quadratic.
_TAG_START = re.compile(r"<([a-z][^\t\n\f\r />]*+)")
_TAG_NAME_END = ("\t", "\n", "\f", "\r", " ", "/")
# An inner `<` followed by an ASCII letter inside a tag name (Step 5C.5 B-3). A Markdown renderer can output
# `<x<img src=...>` as `&lt;x<img src=...>`, whose inner tag is a live `<img>`, so `<img<img ...>` is indexed as
# an image start as well, besides the browser-accurate start with the name `img<img`.
_INNER_TAG_START = re.compile(r"<(?=[a-z])")
_EXTERNAL_URL = r"(?:https?:[/\\]*|[/\\]{2})[^\s\"'<>()\[\],]+"
_SEP = r"(?:\s|/\*[^*\n]{0,50}\*/)+"  # SQL whitespace, including /**/ comments
# The same, possessive (Step 5C.3 N-1): after `select` the separator was split two ways with the probe's
# `[^;\n]{0,200}?`, about 430 ms for the probe on `union select ` followed by 80 KB of spaces (about 3 ms now).
# The set of matches is unchanged.
_SEP_POSSESSIVE = r"(?:\s|/\*[^*\n]{0,50}\*/)++"
_SEP_OPT = r"(?:\s|/\*[^*\n]{0,50}\*/)*"
_SQL_GAP = r"(?:[ \t]|/\*[^*\n]{0,50}\*/)"  # same-line SQL whitespace, including /**/ comments

# An optional `sudo [-flags]`, `env`, `command` or `exec` wrapper and an optional directory path before a command
# name (`/usr/bin/id`, `sudo -E bash`), Step 5C.1 M-3. The wrapper may carry its own directory path
# (`/usr/bin/env bash`) and a sudo flag may take one argument that does not start with `-` (`sudo -u root id`,
# `sudo -g wheel`, `sudo -C 3`), Step 5C.2 F-6. Arguments never start with `-`, so the flag loop stays linear.
_CMD_PREFIX = (
    r"(?:(?:/?[\w.-]+/)*+(?:sudo(?:\s+-\S+(?:\s+(?!-)[^\s`)]+)?)*|env|command|exec)\s+)?(?:/?[\w.-]+/)*+"
)
# Commands whose substitution is a recon, fetch, persistence or destructive step.
_CMD_WITH_ARGS = (
    _CMD_PREFIX
    + r"(?:cat\s+(?:/etc/(?:passwd|shadow|group|hosts|sudoers)|~?/?\.ssh/\S*|/proc/self/\S*)"
    r"|curl\s+\S|wget\s+\S|nc\s+\S|ncat\s+\S|netcat\s+\S|telnet\s+\S"
    r"|bash\s+-[a-z]*[ci]|sh\s+-[a-z]*c|zsh\s+-[a-z]*c"
    r"|rm\s+-[a-z]*[rf][a-z]*\s+(?:/|~|\*)|chmod\s+[0-7]*7[0-7]{2}\s|chmod\s+\+s\s"
    r"|base64\s+(?:-d|--decode)|python[0-9.]*\s+-c|perl\s+-e|ruby\s+-e|php\s+-r"
    r"|powershell\b|pwsh\b|certutil\b)"
)
# A bare recon command, with any flags (`id -u`, `uname -r`), closing the substitution.
_CMD_BARE = _CMD_PREFIX + r"(?:id|whoami|uname|env|printenv|hostname)(?:\s+-[\w-]+)*(?=\s*[`)])"
_SHELL_CONTEXT = r"(?:[=($\"';|&]\s*|\b(?:echo|eval|exec|system|print)\s+)"
_SHELLS = r"(?:sh|bash|zsh|ksh|dash|cmd(?:\.exe)?|powershell(?:\.exe)?|pwsh|iex|invoke-expression)\b"
# An interpreter that runs its standard input as a program: nothing after it but an optional `-` (stdin), so
# `| python -m json.tool` or `| perl -pe 's/a/b/'` (stdin read as data) do not match.
_INTERPRETERS = r"(?:python(?:[23](?:\.[0-9]+)?)?|perl|ruby|php|node)(?=[ \t]*(?:-[ \t]*)?(?:$|[\n;&|)]))"


TEXT = "text"  # the pattern runs over the whole matching form
ATTRIBUTE = "attribute"  # the pattern runs at each attribute boundary of a tag, on the delimiter view
VALUE = "value"  # the pattern runs at the start of each attribute value, on the decoded matching form


@dataclass(frozen=True)
class _Pattern:
    family: str
    name: str
    regex: re.Pattern[str]
    action: str
    group: int = 0  # the regex group whose span is the finding (0: the whole match)
    scope: str = TEXT  # tag-scoped findings span from the tag start to the end of the match


def _p(family: str, name: str, pattern: str, action: str = ESCAPE, group: int = 0, scope: str = TEXT) -> _Pattern:
    return _Pattern(family, name, re.compile(pattern, re.DOTALL), action, group, scope)


# The versioned pattern list (OUTPUT_SANITIZER_VERSION). Patterns run on the casefolded matching form.
PATTERNS: tuple[_Pattern, ...] = (
    _p("xss", "script_tag", r"<script(?=[\s/>]|$)"),
    # An attribute whose name is `on<letters>` (Step 5C.3 B-1: tried at every item boundary of the tag, so a
    # handler right after a closing quote, `<img src="x"onerror=...>`, is a second attribute, and a name inside a
    # value never is). The lookbehind is the character before an attribute: whitespace, `/` or a closing quote.
    _p("xss", "event_handler", r"(?<=[\t\n\f\r /\"'])on[a-z]{2,}" + _WS + r"*=", scope=ATTRIBUTE),
    # A value whose URL scheme is `javascript:`, read from the decoded value (the browser decodes references in a
    # value) after the leading C0 controls and spaces the URL parser strips (Step 5C.3 B-1; possessive, F-4).
    _p("xss", "javascript_uri_in_tag", r"[\x00-\x20]*+" + _JS_SCHEME, scope=VALUE),
    # Inside `<...>` (a Markdown angle-bracket link destination) the URI may hold spaces (Step 5C.1 LM-2); elsewhere
    # a `javascript:` followed by whitespace is prose ("javascript: a language").
    _p("xss", "javascript_uri", r"(?:(?<=<)[ \t]*" + _JS_SCHEME + r"[^<>\n]*(?=>)|" + _JS_SCHEME
       + r"(?=\S)[^\s\"'<>)]*)", REMOVE),
    # `\s*+\)*+` is possessive (Step 5C.2 F-4, same shape): `\s*\)*` then `_SEP_OPT` split one run of spaces two
    # ways, quadratic on a quote or a number followed by 80 KB of spaces. The set of matches is unchanged.
    _p("sqli_cmdi", "union_select_after_quote", r"[\"']\s*+\)*+" + _SEP_OPT + r"union(?:" + _SEP
       + r"(?:all|distinct))?" + _SEP + r"select\b"),
    # A bare number (a numeric parameter such as `id=1`) right before UNION SELECT, but not `SELECT 1 UNION SELECT 2`.
    _p("sqli_cmdi", "union_select_after_number", r"(?:^|(?<=[=(,\s]))(?<!select\s)-?\d+\s*+\)*+" + _SEP_OPT
       + r"union(?:" + _SEP + r"(?:all|distinct))?" + _SEP + r"select\b"),
    _p("sqli_cmdi", "union_select_probe", r"\bunion(?:" + _SEP + r"(?:all|distinct))?" + _SEP + r"select"
       + _SEP_POSSESSIVE + r"(?:null\b|@@|[^;\n]{0,200}?\b(?:information_schema|load_file\s*\(|version\s*\(\s*\)|current_user\b"
       + r"|database\s*\(\s*\)|user\s*\(\s*\)))"),
    _p("sqli_cmdi", "stacked_statement", r";" + _SQL_GAP + r"*(?:drop" + _SQL_GAP + r"+(?:table|database|schema)\b"
       + r"|delete" + _SQL_GAP + r"+from\b|truncate" + _SQL_GAP + r"+table\b|update" + _SQL_GAP + r"+[^\s;]+"
       + _SQL_GAP + r"+set\b)"),
    # `or` needs no space before a quote: `admin'or'1'='1` (Step 5C.1 M-3).
    _p("sqli_cmdi", "quoted_tautology", r"[\"']\s*+\)?\s*+or(?:\s+|(?=[\"']))([\"']?)([a-z0-9_]+)\1\s*=\s*\1\2"
       r"(?![a-z0-9_=])"),
    _p("sqli_cmdi", "numeric_tautology", r"\bor\s+([0-9]+)\s*=\s*\1(?![0-9=])"),
    _p("sqli_cmdi", "command_substitution", r"\$\(\s*(?:" + _CMD_WITH_ARGS + "|" + _CMD_BARE + r")[^)\n]{0,200}\)?"),
    _p("sqli_cmdi", "backtick_command", r"`" + _CMD_WITH_ARGS + r"[^`\n]{0,200}`"),
    _p("sqli_cmdi", "backtick_bare_command", _SHELL_CONTEXT + r"(`" + _CMD_BARE + r"[^`\n]{0,40}`)", group=1),
    _p("sqli_cmdi", "pipe_to_shell", r"[^\s|]\s*(\|\s*" + _CMD_PREFIX + r"(?:" + _SHELLS + "|" + _INTERPRETERS
       + r"))(?!\s*\|)", group=1),
)
_TEXT_PATTERNS = tuple(p for p in PATTERNS if p.scope == TEXT)
_TAG_PATTERNS = tuple(p for p in PATTERNS if p.scope != TEXT)

# Markdown image alt text as CommonMark reads it (Step 5C.1 M-2, Step 5C.2 F-3): one linear pass over these tokens
# with a bracket stack (`markdown_images`), so escaped characters (`\]`), brackets nested to any depth
# (`a[b[c]d]e`) and single line breaks stay inside the alt text, and a blank line ends every open bracket.
_MD_TOKEN = re.compile(r"\\[^\n]|!\[|[\[\]]|\n[ \t]*\n")
# What follows the closing bracket of an image: an inline target, or a full or collapsed reference label.
_MD_INLINE_TARGET = re.compile(r"\(\s*+<?\s*+(" + _EXTERNAL_URL + ")")
_MD_REF_LABEL = re.compile(r"\[([^\]\n]{0,1000})\]")
# A reference definition, also inside block quotes and list items (`> [px]: ...`, `- [px]: ...`, `1. [px]: ...`,
# Step 5C.2 F-3), may put its URL on the next line (`[px]:\n  https://...`, `> [px]:\n> https://...`). Possessive,
# so the spaces around container markers have one reading.
_MD_CONTAINER = r"(?:[ \t]{0,3}+(?:>|[-+*]|\d{1,9}[.)])[ \t]*+)*+"
_MD_REF_DEFINITION = re.compile(
    r"^" + _MD_CONTAINER + r"[ \t]{0,3}+\[([^\]\n]{1,1000})\]:[ \t]*+(?:\n[ \t]*+(?:>[ \t]*+)*+)?<?("
    + _EXTERNAL_URL + ")",
    re.MULTILINE,
)
# HTML elements that fetch an image without a click, and the attributes that name the URL (Step 5C.1 LM-1).
# `<input>` fetches only with `type=image`. CSS `url()`, `data:text/html` frames and the media fetches of
# `<video src>` and `<audio src>` are outside IP-10's list (they fetch media, not an image).
IMAGE_URL_ATTRIBUTES: dict[str, frozenset[str]] = {
    "img": frozenset({"src", "srcset", "href", "xlink:href"}),
    "image": frozenset({"src", "srcset", "href", "xlink:href"}),
    "source": frozenset({"src", "srcset"}),
    "video": frozenset({"poster"}),
    "input": frozenset({"src"}),
}
# An inner tag name longer than this cannot be an image element, so it is not sliced (constant work per start).
_LONGEST_IMAGE_TAG = max(map(len, IMAGE_URL_ATTRIBUTES))
_IMAGE_FINDING_NAME = {"img": "html_image", "image": "html_image", "source": "html_source",
                       "video": "html_video_poster", "input": "html_input_image"}
_URL_IN_VALUE = re.compile(_EXTERNAL_URL)
_URL_CONTROLS = re.compile(r"[\t\n\r]")

# Structure-preserving view: a tag delimiter that came from a character reference or an NFKC mapping becomes this
# inert character (not a quote, angle bracket, whitespace, `/`, `=` or tag-name character), so it never opens,
# closes or splits a tag or an attribute (Step 5C.2; `=`, `/` and whitespace added in Step 5C.3 B-1).
_STRUCTURAL = frozenset("<>\"'=/\t\n\f\r ")
_INERT = "�"

# Escape table for neutralize: html.escape's five characters plus the shell and SQL metacharacters.
_ESCAPES = str.maketrans({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#x27;",
    "$": "&#36;", "`": "&#96;", "(": "&#40;", ")": "&#41;", ";": "&#59;", "|": "&#124;",
})


@dataclass(frozen=True)
class Finding:
    family: str
    name: str
    start: int  # source span, end exclusive
    end: int
    action: str


@dataclass(frozen=True)
class SanitizerResult:
    """The sanitizer's verdict on one `content` string. `content` is what may be released (None when blocked)."""

    verdict: str  # "allow", "block" or "neutralized"
    families: tuple[str, ...]
    content: str | None
    latency_ms: float
    findings: tuple[Finding, ...] = ()


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def decode_entity(entity: str) -> str:
    """One HTML character reference, decoded as a browser does; never raises, whatever the number of digits.

    Numeric references above U+10FFFF (and surrogates, NUL) become U+FFFD; leading zeros are ignored, so
    `&#x000000006a;` is `j`. Named references are case sensitive (`&Tab;` is a tab, `&tab;` is not a reference).
    """
    if entity.startswith("&#"):
        body = entity[2:].rstrip(";")
        digits, base = (body[1:], 16) if body[:1] in ("x", "X") else (body, 10)
        digits = digits.lstrip("0") or "0"
        if len(digits) > _MAX_CODEPOINT_DIGITS:
            return "�"
        codepoint = int(digits, base)
        if not 0 <= codepoint <= _MAX_CODEPOINT:
            return "�"
        return html.unescape(f"&#{codepoint};")
    return html.unescape(entity)


def matching_form(text: str) -> tuple[str, list[tuple[int, int]]]:
    """NFKC, one round of case-sensitive HTML-entity decoding, then NFKC and casefold; with the source span per char.

    Step 5C.1 M-1: the first stage applies NFKC only, so named references keep their case until they are
    decoded (`&Tab;` and `&NewLine;` are not `&tab;` and `&newline;`); `casefold()` runs after the decode.
    """
    first: list[str] = []
    first_map: list[tuple[int, int]] = []
    for i, ch in enumerate(text):
        for normal in unicodedata.normalize("NFKC", ch):
            first.append(normal)
            first_map.append((i, i + 1))
    stage = "".join(first)
    decoded_chars: list[str] = []
    decoded_map: list[tuple[int, int]] = []
    cursor = 0
    for match in _ENTITY.finditer(stage):
        decoded = decode_entity(match.group())
        if decoded == match.group():
            continue
        decoded_chars.extend(stage[cursor:match.start()])
        decoded_map.extend(first_map[cursor:match.start()])
        source = (first_map[match.start()][0], first_map[match.end() - 1][1])
        decoded_chars.extend(decoded)
        decoded_map.extend([source] * len(decoded))
        cursor = match.end()
    decoded_chars.extend(stage[cursor:])
    decoded_map.extend(first_map[cursor:])
    chars: list[str] = []
    spans: list[tuple[int, int]] = []
    for ch, span in zip(decoded_chars, decoded_map, strict=True):
        for folded in _fold(ch):
            chars.append(folded)
            spans.append(span)
    return "".join(chars), spans


def structure_form(form: str, spans: Sequence[tuple[int, int]], source: str | None = None) -> str:
    """The matching form with every derived tag delimiter (`_STRUCTURAL`) replaced by an inert character.

    HTML tokenization runs before character references are decoded and never applies NFKC, so a reference or a
    compatibility character is never a delimiter (`<img alt=&quot; onerror=alert(1)> say "hi"` and
    `<img alt&equals;" onerror=alert(1)//">` both have an `onerror` attribute). A character is derived when its
    source span is longer than one character (a reference) or, with `source`, when its source character is not
    the same ASCII character (NFKC, Step 5C.3 B-1). Same length and spans as `form`.
    """

    def derived(ch: str, span: tuple[int, int]) -> bool:
        if span[1] - span[0] > 1:
            return True
        if source is None:
            return False
        original = source[span[0]]
        return original != ch and not (original.isascii() and original.lower() == ch)

    return "".join(
        _INERT if ch in _STRUCTURAL and derived(ch, span) else ch for ch, span in zip(form, spans, strict=True)
    )


def url_host(url: str) -> str:
    """The lowercase hostname of an absolute or protocol-relative URL in matching form ('' when there is none).

    Tab, CR and LF are removed first, as the WHATWG URL parser does (Step 5C.1 LM-3), so
    `https://cdn.example.com<TAB>.attacker.example/` names `cdn.example.com.attacker.example`.
    """
    url = _URL_CONTROLS.sub("", url)
    rest = re.sub(r"^(?:https?:)?[/\\]*", "", url)
    authority = re.split(r"[/\\?#]", rest, maxsplit=1)[0]
    host = unquote(authority.rsplit("@", 1)[-1])
    if host.startswith("["):
        host = host[1:].split("]", 1)[0]
    else:
        host = host.split(":", 1)[0]
    return _fold(host).strip().rstrip(".")


# (next boundary, attribute name or None, value start, value end; -1 when there is no value)
_Item = tuple[int, str | None, int, int]
_END: _Item = (-1, None, -1, -1)  # `>` or the end of the view: the tag ends


class TagChains:
    """The body items of every tag start in one view, tokenized once per item boundary (Step 5C.3 B-1).

    `delimiters` is the view the tokenizer reads (the matching form, or the structure-preserving view); attribute
    values are read from `values`, the decoded matching form, at the same offsets. Every tag start is scanned, also
    one inside another tag's value, as the old per-start regex did. The items after a boundary do not depend on
    where the tag started, so two starts whose items reach the same boundary share everything after it: each
    boundary is tokenized once and each scan stops where it meets a boundary already scanned, so the cost is linear
    even for `<img a=b` repeated 10,000 times (each tag runs to the end of the input there).
    """

    def __init__(self, delimiters: str, values: str) -> None:
        self.delimiters = delimiters
        self.values = values
        # (tag start, first boundary after the tag name, tag name). Besides the browser-accurate start, every `<`
        # followed by a letter inside the name is an inner start with the rest of the name (Step 5C.5 B-3), kept only
        # when short enough to be an image element name; it shares the boundary of the outer start.
        starts: list[tuple[int, int, str]] = []
        for m in _TAG_START.finditer(delimiters):
            if delimiters[m.end():m.end() + 1] not in _TAG_NAME_END:
                continue
            starts.append((m.start(), m.end(), m.group(1)))
            for inner in _INNER_TAG_START.finditer(delimiters, m.start() + 1, m.end()):
                i = inner.start()
                inner_name = delimiters[i + 1:m.end()] if m.end() - i - 1 <= _LONGEST_IMAGE_TAG else ""
                starts.append((i, m.end(), inner_name))
        self.starts = starts
        self._items: dict[int, _Item] = {}

    def item(self, pos: int) -> _Item:
        cached = self._items.get(pos)
        if cached is None:
            match = _TAG_STEP.match(self.delimiters, pos)
            if match is None:
                cached = _END
            else:
                group = 2 if match.start(2) >= 0 else 3 if match.start(3) >= 0 else 4
                cached = (match.end(), match.group(1), match.start(group), match.end(group))
            self._items[pos] = cached
        return cached

    def first_hits(self, test: Callable[[int, _Item], int]) -> list[tuple[int, int]]:
        """(tag start, hit end) for each tag start whose items include one that `test` accepts (the first such item).

        `test` returns the end of its match at the boundary, or -1. Like `finditer`, a tag start inside an earlier
        hit's span is skipped, so the spans do not overlap.
        """
        memo: dict[int, int] = {}  # boundary -> end of the first hit at or after it (-1: none before the tag ends)
        found: list[tuple[int, int]] = []
        last = -1
        items, parse = self._items, self.item
        for start, boundary, _name in self.starts:
            if start < last:
                continue
            path: list[int] = []
            pos = boundary
            while pos not in memo:
                item = items.get(pos) or parse(pos)
                if item is _END:
                    memo[pos] = -1
                    break
                hit = test(pos, item)
                if hit >= 0:
                    memo[pos] = hit
                    break
                path.append(pos)
                pos = item[0]
            hit = memo[pos]
            for passed in path:
                memo[passed] = hit
            if hit >= 0:
                found.append((start, hit))
                last = hit
        return found

    def valued_attributes(self, starts: Iterable[tuple[int, int, str]]) -> Iterator[tuple[str, int, int]]:
        """(name, value start, value end) of every attribute with a value on the items of `starts`, each once."""
        seen: set[int] = set()
        for _start, boundary, _name in starts:
            pos = boundary
            while pos not in seen:
                seen.add(pos)
                item = self.item(pos)
                if item is _END:
                    break
                if item[1] is not None and item[2] >= 0:
                    yield item[1], item[2], item[3]
                pos = item[0]

    def typed_starts(self, starts: Iterable[tuple[int, int, str]], name: str, value: str) -> list[tuple[int, int, str]]:
        """The `starts` whose items include the attribute `name` with the decoded value `value` (surrounding
        whitespace ignored), such as `<input type=image>`."""
        memo: dict[int, bool] = {}
        kept: list[tuple[int, int, str]] = []
        for start in starts:
            path: list[int] = []
            pos = start[1]
            while pos not in memo:
                item = self.item(pos)
                if item is _END:
                    memo[pos] = False
                    break
                if item[1] == name and item[2] >= 0 and self.values[item[2]:item[3]].strip() == value:
                    memo[pos] = True
                    break
                path.append(pos)
                pos = item[0]
            for passed in path:
                memo[passed] = memo[pos]
            if memo[pos]:
                kept.append(start)
        return kept


def markdown_images(form: str) -> list[tuple[int, int]]:
    """(alt start, closing bracket index) of every Markdown image `![...]` in `form`, in one linear pass.

    Step 5C.2 F-3: a bracket stack replaces the alt-text regex, so a backslash escapes the next character, brackets
    nest to any depth and a blank line (`\\n[ \\t]*\\n`) ends every open bracket, as in a CommonMark paragraph.
    Each `[` and `]` is pushed or popped once, so the cost is linear even for `![` repeated 10,000 times.
    """
    if "![" not in form:
        return []
    found: list[tuple[int, int]] = []
    stack: list[int] = []  # alt start of an image opener, or -1 for a plain `[`
    for token in _MD_TOKEN.finditer(form):
        text = token.group()
        if text == "![":
            stack.append(token.end())
        elif text == "[":
            stack.append(-1)
        elif text == "]":
            if stack and (opener := stack.pop()) >= 0:
                found.append((opener, token.start()))
        elif text[0] == "\n":
            stack.clear()  # a blank line ends the paragraph, so no bracket opened before it can close
        # otherwise a backslash escape: the escaped character is text
    found.sort()
    return found


def _escape_span(text: str) -> str:
    """The neutralize escape of one matched span; never the span unchanged (Step 5C.2 F-5).

    When the escape table leaves the span byte for byte as it was (`1 UNION SELECT`, which holds none of its
    characters), every character is written as a numeric character reference instead, so an intact SQLi or CMDi
    phrase is never released under `Egress-saved` / `masked`. An HTML renderer still shows the same text.
    """
    escaped = text.translate(_ESCAPES)
    if escaped == text:
        escaped = "".join(f"&#{ord(ch)};" for ch in text)
    return escaped


class OutputSanitizer:
    """IP-10 checks on the client-visible content: pattern families (a) xss, (b) image_exfil, (c) sqli_cmdi."""

    def __init__(self, *, mode: str = "block", image_allowlist: Iterable[str] = ()) -> None:
        if mode not in MODES:
            raise ValueError(f"OUTPUT_SANITIZER_MODE must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.image_allowlist = frozenset(_fold(host).strip().rstrip(".") for host in image_allowlist if host.strip())
        self.version = OUTPUT_SANITIZER_VERSION
        # Fingerprint of everything that decides a verdict (logged next to the version label).
        self.fingerprint = canonical_hash(
            {
                "component": "output_sanitizer",
                "version": OUTPUT_SANITIZER_VERSION,
                "mode": mode,
                "allowlist": sorted(self.image_allowlist),
                "patterns": [[p.family, p.name, p.regex.pattern, p.action, p.group, p.scope] for p in PATTERNS],
                "tag_grammar": [_TAG_START.pattern, _TAG_STEP.pattern, _INNER_TAG_START.pattern, _LONGEST_IMAGE_TAG],
                "image_patterns": [r.pattern for r in (_MD_TOKEN, _MD_INLINE_TARGET, _MD_REF_LABEL,
                                                       _MD_REF_DEFINITION, _URL_IN_VALUE)],
                "image_attributes": {tag: sorted(names) for tag, names in IMAGE_URL_ATTRIBUTES.items()},
                "structure_view": {"inert": _INERT, "structural": sorted(_STRUCTURAL), "nfkc_derived": True},
                "unidata_version": unicodedata.unidata_version,
            }
        )

    # Matching

    def _external(self, url: str) -> bool:
        """True unless the URL names an allowlisted host (a URL with no readable host is treated as external)."""
        host = url_host(url)
        return not host or host not in self.image_allowlist

    def _markdown_image_findings(self, form: str, spans: Sequence[tuple[int, int]]) -> list[Finding]:
        found: list[Finding] = []

        def add(start: int, end: int, name: str) -> None:
            found.append(Finding("image_exfil", name, spans[start][0], spans[end - 1][1], REMOVE))

        used = set()
        for alt_start, close in markdown_images(form):
            if form.startswith("(", close + 1):
                target = _MD_INLINE_TARGET.match(form, close + 1)
                if target is not None and self._external(target.group(1)):
                    add(target.start(1), target.end(1), "markdown_image")
                continue  # an inline image names no reference
            reference = _MD_REF_LABEL.match(form, close + 1)
            label = ((reference.group(1) if reference is not None else "") or form[alt_start:close]).strip()
            if label:
                used.add(" ".join(label.split()))
        for match in _MD_REF_DEFINITION.finditer(form):
            if " ".join(match.group(1).strip().split()) in used and self._external(match.group(2)):
                add(match.start(2), match.end(2), "markdown_image_reference")
        return found

    def _tag_findings(self, delimiters: str, form: str, spans: Sequence[tuple[int, int]]) -> list[Finding]:
        """Event handlers, `javascript:` values and HTML image URLs of every tag start, tokenized on `delimiters`."""
        chains = TagChains(delimiters, form)
        if not chains.starts:
            return []
        found: list[Finding] = []
        for pattern in _TAG_PATTERNS:
            regex = pattern.regex

            def test(pos: int, item: _Item, regex: re.Pattern[str] = regex, scope: str = pattern.scope) -> int:
                if scope == ATTRIBUTE:  # a name starting with `on`, tried on the delimiter view
                    hit = item[1] is not None and item[1].startswith("on")
                    match = regex.match(delimiters, pos) if hit else None
                else:  # a value, tried on the decoded form
                    match = regex.match(form, item[2], item[3]) if item[2] >= 0 else None
                return match.end() if match is not None else -1

            for start, end in chains.first_hits(test):
                found.append(Finding(pattern.family, pattern.name, spans[start][0], spans[end - 1][1], pattern.action))
        found.extend(self._html_image_findings(chains, form, spans))
        return found

    def _html_image_findings(self, chains: TagChains, form: str, spans: Sequence[tuple[int, int]]) -> list[Finding]:
        found: list[Finding] = []
        by_finding: dict[str, list[tuple[int, int, str]]] = {}
        for start in chains.starts:
            if start[2] in IMAGE_URL_ATTRIBUTES:
                by_finding.setdefault(_IMAGE_FINDING_NAME[start[2]], []).append(start)
        if "html_input_image" in by_finding:  # only `<input type=image>` fetches its `src`
            by_finding["html_input_image"] = chains.typed_starts(by_finding["html_input_image"], "type", "image")
        for finding_name, starts in by_finding.items():
            if not starts:
                continue
            wanted = IMAGE_URL_ATTRIBUTES[starts[0][2]]  # `img` and `image` share one set
            for attribute, value_start, value_end in chains.valued_attributes(starts):
                if attribute not in wanted or value_end <= value_start:
                    continue
                # The URL parser drops tab, CR and LF (LM-3, F-2); the value is the decoded one. A srcset splits
                # its candidates on them, so it keeps them.
                value = form[value_start:value_end]
                positions = list(range(value_start, value_end))
                if attribute != "srcset":
                    kept = [(ch, pos) for ch, pos in zip(value, positions, strict=True) if ch not in "\t\n\r"]
                    value = "".join(ch for ch, _ in kept)
                    positions = [pos for _, pos in kept]
                for url in _URL_IN_VALUE.finditer(value):
                    if self._external(url.group()):
                        start, end = positions[url.start()], positions[url.end() - 1]
                        found.append(Finding("image_exfil", finding_name, spans[start][0], spans[end][1], REMOVE))
        return found

    def findings(self, content: str) -> tuple[Finding, ...]:
        """Every match of every family in `content`, as source spans."""
        if not content:
            return ()
        form, spans = matching_form(content)
        found: list[Finding] = []
        for pattern in _TEXT_PATTERNS:
            for match in pattern.regex.finditer(form):
                start, end = match.span(pattern.group)
                if end > start:
                    found.append(
                        Finding(pattern.family, pattern.name, spans[start][0], spans[end - 1][1], pattern.action)
                    )
        found.extend(self._markdown_image_findings(form, spans))
        found.extend(self._tag_findings(form, form, spans))
        # Structure-preserving view: derived delimiters are not tag delimiters. Only needed when the decode or NFKC
        # produced one; the merge drops exact duplicates, so matches only add.
        structure = structure_form(form, spans, content)
        if structure != form:
            found.extend(self._tag_findings(structure, form, spans))
        return tuple(sorted(set(found), key=lambda f: (f.start, f.end, f.family, f.name, f.action)))

    # Actions

    @staticmethod
    def neutralize(content: str, findings: Sequence[Finding]) -> str:
        """Escape every escape span and drop every removal span (removal wins where they overlap)."""
        actions = [0] * len(content)  # 0 keep, 1 escape, 2 remove
        for finding in findings:
            level = 2 if finding.action == REMOVE else 1
            for i in range(finding.start, finding.end):
                actions[i] = max(actions[i], level)
        pieces: list[str] = []
        i = 0
        while i < len(content):
            level = actions[i]
            j = i
            while j < len(content) and actions[j] == level:
                j += 1
            if level == 0:
                pieces.append(content[i:j])
            elif level == 1:
                pieces.append(_escape_span(content[i:j]))
            i = j
        return "".join(pieces)

    def check(self, content: str) -> SanitizerResult:
        """Decide `content`. Raises only on a programming error; the pipeline turns any exception into a 503."""
        started = time.perf_counter()
        if not isinstance(content, str):
            raise TypeError("the output sanitizer needs the content string")
        found = self.findings(content)
        families = tuple(sorted({f.family for f in found}))
        if not found:
            verdict, released = "allow", content
        elif self.mode == "block":
            verdict, released = "block", None
        else:
            verdict, released = "neutralized", self.neutralize(content, found)
        return SanitizerResult(verdict, families, released, (time.perf_counter() - started) * 1000.0, found)
