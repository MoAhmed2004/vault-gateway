"""Unit tests for the FastAPI gateway shell (PS-1, CA-1, CA-2, CA-3, TM-R1, TM-R6, IP-1, IP-2, IP-8, §5.2).

The app is driven through `httpx.ASGITransport` with the lifespan run
explicitly (Starlette 0.38's TestClient emits an anyio DeprecationWarning on
import, and the verification gate requires zero warnings). Middleware-level
tests drive the raw ASGI interface to count exactly how many body chunks are
pulled from the client.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.errors import UNIFORM_REFUSAL_BODY, UNIFORM_REFUSAL_DETAIL, UNIFORM_REFUSAL_HEADERS, UNIFORM_REFUSAL_STATUS
from app.inspection import TokenizerLimit
from app.main import create_app
from app.middleware import MAX_REQUEST_BODY_BYTES, BodyCapMiddleware
from contracts import Outcome, TelemetrySink
from contracts.records import UNPARSED_PROMPT_HASH

CAP = 131_072
CHUNK = 64 * 1024
SPEC_BODY = '{"detail": "Service Unavailable. Request blocked by security policy."}'


class FakeSink:
    def __init__(self) -> None:
        self.records: list[Any] = []
        self.shadows: list[Any] = []
        self.flushes = 0
        self.closed = False
        self.healthy = True
        self.fail = False

    def record_request(self, record: Any) -> None:
        if self.fail:
            raise RuntimeError("sink exploded")
        self.records.append(record)

    def record_shadow(self, record: Any) -> None:
        self.shadows.append(record)

    def flush(self, timeout: float = 5.0) -> None:
        self.flushes += 1

    def close(self, timeout: float = 5.0) -> None:  # pragma: no cover - the app must never close an injected sink
        self.closed = True


def make_settings(tmp_path, **overrides) -> Settings:
    values = {
        "_env_file": None,
        "GATEWAY_MODE": "production",
        "UPSTREAM_MODE": "stub",
        "GROQ_API_KEY": "",
        "LOG_DB_PATH": tmp_path / "logs" / "vault.db",
    }
    values.update(overrides)
    return Settings(**values)


def run_client(app, *requests: tuple[str, str, dict[str, Any]]) -> list[httpx.Response]:
    async def go() -> list[httpx.Response]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as client:
                return [await client.request(method, url, **kwargs) for method, url, kwargs in requests]

    return asyncio.run(go())


def padded_json(total_bytes: int, prompt: str = "hello") -> bytes:
    body = json.dumps({"user_prompt": prompt}).encode("utf-8")
    assert len(body) <= total_bytes
    return body + b" " * (total_bytes - len(body))  # trailing JSON whitespace keeps the body valid


def chunked(data: bytes, size: int = CHUNK):
    async def gen():
        for i in range(0, len(data), size):
            yield data[i:i + size]

    return gen()


def assert_uniform(response: httpx.Response) -> None:
    assert response.status_code == UNIFORM_REFUSAL_STATUS == 503
    assert response.content == UNIFORM_REFUSAL_BODY == SPEC_BODY.encode("utf-8")
    assert response.headers.raw == list(UNIFORM_REFUSAL_HEADERS)


@pytest.fixture
def sink() -> FakeSink:
    return FakeSink()


@pytest.fixture
def gateway(tmp_path, sink):
    return create_app(make_settings(tmp_path), sink)


# Uniform refusal constants (TM-R6)


def test_uniform_body_is_the_spec_text_byte_for_byte():
    assert UNIFORM_REFUSAL_BODY.decode("utf-8") == SPEC_BODY
    assert json.loads(UNIFORM_REFUSAL_BODY) == {"detail": UNIFORM_REFUSAL_DETAIL}
    assert dict(UNIFORM_REFUSAL_HEADERS)[b"content-length"] == str(len(UNIFORM_REFUSAL_BODY)).encode()


def test_body_cap_default_is_128_kib():
    assert MAX_REQUEST_BODY_BYTES == CAP == Settings.model_fields["MAX_REQUEST_BODY_BYTES"].default


# TM-R1 body cap through the full app: exact boundary, Content-Length and chunked


@pytest.mark.parametrize("mode", ["content-length", "chunked"])
def test_body_of_exactly_131072_bytes_passes_the_cap(gateway, sink, mode):
    body = padded_json(CAP)
    content = body if mode == "content-length" else chunked(body)
    (response,) = run_client(gateway, ("POST", "/v1/chat", {"content": content, "headers": {"content-type": "application/json"}}))
    assert_uniform(response)  # admitted past every cap, then fails closed: no upstream at Step 4
    assert [(r.outcome, r.blocked_by) for r in sink.records] == [(Outcome.ERROR, "upstream_error")]


@pytest.mark.parametrize("mode", ["content-length", "chunked"])
def test_body_of_131073_bytes_is_refused_by_the_cap(gateway, sink, mode):
    body = padded_json(CAP + 1)
    content = body if mode == "content-length" else chunked(body)
    (response,) = run_client(gateway, ("POST", "/v1/chat", {"content": content, "headers": {"content-type": "application/json"}}))
    assert_uniform(response)
    assert [(r.outcome, r.blocked_by, r.prompt_hash) for r in sink.records] == [
        (Outcome.BLOCKED_INGRESS, "body_cap", UNPARSED_PROMPT_HASH)
    ]


def test_chunked_requests_really_arrive_without_content_length(gateway):
    seen: list[list[tuple[bytes, bytes]]] = []

    async def spy(scope, receive, send):
        if scope["type"] == "http":
            seen.append(scope["headers"])
        await gateway(scope, receive, send)

    async def go():
        async with gateway.router.lifespan_context(gateway):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=spy), base_url="http://t") as client:
                await client.post("/v1/chat", content=chunked(padded_json(CAP)), headers=JSON)

    asyncio.run(go())
    names = {name.lower() for name, _ in seen[0]}
    assert b"content-length" not in names and b"transfer-encoding" in names


# TM-R1 at the ASGI level: count every chunk pulled from the client


class InnerApp:
    def __init__(self) -> None:
        self.called = False
        self.body = b""

    async def __call__(self, scope, receive, send):
        self.called = True
        while True:
            message = await receive()
            self.body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": str(len(self.body)).encode()})


def drive(chunks: Iterator[bytes], headers: list[tuple[bytes, bytes]] | None = None, *, endless: bool = False):
    inner = InnerApp()
    refusals: list[str] = []
    middleware = BodyCapMiddleware(inner, max_body_bytes=CAP, on_refusal=lambda scope, reason: refusals.append(reason))
    sent: list[dict[str, Any]] = []
    pulls = 0
    pending = list(chunks) if not endless else None

    async def receive():
        nonlocal pulls
        pulls += 1
        if endless:
            return {"type": "http.request", "body": b"z" * CHUNK, "more_body": True}
        if not pending:
            return {"type": "http.disconnect"}
        chunk = pending.pop(0)
        return {"type": "http.request", "body": chunk, "more_body": bool(pending)}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/v1/chat", "headers": headers or []}
    asyncio.run(middleware(scope, receive, send))
    return inner, refusals, sent, pulls


def _status(sent):
    return sent[0]["status"] if sent else None


def test_endless_chunked_stream_is_aborted_as_soon_as_the_cap_is_crossed():
    inner, refusals, sent, pulls = drive(iter(()), endless=True)
    # 64 KiB chunks: 2 chunks = 131,072 bytes (allowed), the 3rd crosses the cap. Nothing more is read.
    assert pulls == 3
    assert inner.called is False  # never parsed, never buffered past the cap
    assert refusals == ["body_cap"]
    assert _status(sent) == 503 and sent[1]["body"] == UNIFORM_REFUSAL_BODY


def test_oversized_content_length_is_refused_without_reading_any_body():
    headers = [(b"content-length", str(10 * 1024 * 1024).encode())]
    inner, refusals, sent, pulls = drive(iter(()), headers, endless=True)
    assert pulls == 0 and inner.called is False and refusals == ["body_cap"] and _status(sent) == 503


@pytest.mark.parametrize("declared", [CAP, CAP + 1])
def test_content_length_boundary_at_the_middleware(declared):
    headers = [(b"content-length", str(declared).encode())]
    inner, refusals, sent, pulls = drive(iter([b"a" * declared]), headers)
    if declared == CAP:
        assert inner.called and inner.body == b"a" * CAP and refusals == [] and _status(sent) == 200
    else:
        assert pulls == 0 and not inner.called and refusals == ["body_cap"] and _status(sent) == 503


@pytest.mark.parametrize("chunk_size", [1000, 4096, 65536, 131072])
def test_exact_cap_in_any_chunking_is_accepted_and_replayed_intact(chunk_size):
    data = bytes(i % 251 for i in range(CAP))
    chunks = [data[i:i + chunk_size] for i in range(0, CAP, chunk_size)]
    inner, refusals, sent, _ = drive(iter(chunks))
    assert inner.body == data and refusals == [] and _status(sent) == 200


@pytest.mark.parametrize("chunk_size", [1000, 4096, 65536])
def test_one_byte_over_in_any_chunking_stops_at_the_crossing_chunk(chunk_size):
    data = b"b" * (CAP + 1)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    inner, refusals, sent, pulls = drive(iter(chunks))
    assert pulls == math.ceil((CAP + 1) / chunk_size)
    assert not inner.called and refusals == ["body_cap"] and _status(sent) == 503


def test_lying_content_length_is_caught_while_streaming():
    headers = [(b"content-length", b"10")]
    inner, refusals, sent, pulls = drive(iter([b"c" * CHUNK] * 5), headers)
    assert pulls == 3 and not inner.called and refusals == ["body_cap"] and _status(sent) == 503


@pytest.mark.parametrize(
    "headers",
    [
        [(b"content-length", b"abc")],
        [(b"content-length", b"-1")],
        [(b"content-length", b"10"), (b"content-length", b"20")],
    ],
)
def test_malformed_content_length_is_refused(headers):
    inner, refusals, sent, pulls = drive(iter([b"x"]), headers)
    assert pulls == 0 and not inner.called and refusals == ["body_cap"] and _status(sent) == 503


FIVE_THOUSAND_DIGITS = b"1" * 5000


@pytest.mark.parametrize("value", [b"\xb2", b"1\xb2", "١٢".encode("utf-8"), FIVE_THOUSAND_DIGITS, b"0" * 21])
def test_non_ascii_or_overlong_content_length_is_refused_at_the_middleware(value):
    """F3: superscript digits and huge digit strings are malformed (body_cap), never a ValueError."""
    inner, refusals, sent, pulls = drive(iter([b"x"]), [(b"content-length", value)])
    assert pulls == 0 and not inner.called and refusals == ["body_cap"] and _status(sent) == 503


def test_twenty_digit_content_length_is_still_parsed():
    headers = [(b"content-length", b"0" * 19 + b"5")]
    inner, refusals, sent, _ = drive(iter([b"abcde"]), headers)
    assert inner.called and inner.body == b"abcde" and refusals == [] and _status(sent) == 200


def asgi_post(app, headers: list[tuple[bytes, bytes]], body: bytes = b"") -> list[dict[str, Any]]:
    """Send one raw ASGI POST /v1/chat through the full app (headers exactly as given), inside its lifespan."""
    sent: list[dict[str, Any]] = []
    messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/v1/chat", "raw_path": b"/v1/chat", "root_path": "", "query_string": b"",
        "headers": headers, "client": ("127.0.0.1", 50000), "server": ("gateway.test", 80),
    }

    async def go():
        async with app.router.lifespan_context(app):
            await app(scope, receive, send)

    asyncio.run(go())
    return sent


@pytest.mark.parametrize("value", [b"\xb2", FIVE_THOUSAND_DIGITS], ids=["superscript-two", "5000-digits"])
def test_malformed_content_length_through_the_app_logs_body_cap_not_internal_error(gateway, sink, value):
    """F3: the full app answers with the uniform 503 and logs Blocked-ingress / body_cap."""
    sent = asgi_post(gateway, [(b"content-type", b"application/json"), (b"content-length", value)], b'{"user_prompt":"hi"}')
    start, body = sent
    assert start["status"] == UNIFORM_REFUSAL_STATUS and start["headers"] == list(UNIFORM_REFUSAL_HEADERS)
    assert body["body"] == UNIFORM_REFUSAL_BODY
    assert [(r.outcome, r.blocked_by, r.prompt_hash) for r in sink.records] == [
        (Outcome.BLOCKED_INGRESS, "body_cap", UNPARSED_PROMPT_HASH)
    ]


def test_client_disconnect_mid_body_sends_nothing():
    inner, refusals, sent, _ = drive(iter([b"a" * 100]))  # more_body=False on the only chunk: accepted
    assert inner.called
    inner2 = InnerApp()
    middleware = BodyCapMiddleware(inner2, max_body_bytes=CAP, on_refusal=lambda s, r: None)
    messages = [{"type": "http.request", "body": b"a", "more_body": True}, {"type": "http.disconnect"}]
    out: list[Any] = []

    async def receive():
        return messages.pop(0)

    async def send(message):
        out.append(message)

    asyncio.run(middleware({"type": "http", "headers": []}, receive, send))
    assert out == [] and inner2.called is False


def test_non_http_scopes_pass_through_untouched():
    calls: list[str] = []

    async def inner(scope, receive, send):
        calls.append(scope["type"])

    middleware = BodyCapMiddleware(inner, max_body_bytes=CAP, on_refusal=lambda s, r: None)
    asyncio.run(middleware({"type": "lifespan"}, None, None))
    assert calls == ["lifespan"]
    with pytest.raises(ValueError):
        BodyCapMiddleware(inner, max_body_bytes=0, on_refusal=lambda s, r: None)


# TM-R6: byte-identical refusal across every cause, with the cause logged server-side only


def _boom_app(tmp_path, sink):
    tiny = TokenizerLimit("tiny", 40, lambda text: math.ceil(len(text) / 2) + 2)
    app = create_app(make_settings(tmp_path), sink, tokenizers=(tiny,))

    async def boom() -> None:
        raise RuntimeError("secret internal detail: /etc/passwd at line 42")

    def boom_sync() -> None:
        raise KeyError("another hidden detail")

    app.add_api_route("/boom", boom, methods=["GET"])
    app.add_api_route("/boom-sync", boom_sync, methods=["GET"])
    return app


JSON = {"content-type": "application/json"}

CAUSES: list[tuple[str, str, str, dict[str, Any], str | None]] = [
    ("routing 404", "GET", "/does-not-exist", {}, None),
    ("docs disabled", "GET", "/docs", {}, None),
    ("openapi disabled", "GET", "/openapi.json", {}, None),
    ("method 405", "GET", "/v1/chat", {}, None),
    ("method 405 on probe", "PUT", "/healthz", {}, None),
    ("trailing slash on chat", "POST", "/v1/chat/", {"json": {"user_prompt": "hi"}}, None),
    ("trailing slash on probe", "GET", "/healthz/", {}, None),
    ("double trailing slash", "POST", "/v1/chat//", {"json": {"user_prompt": "hi"}}, None),
    ("extra field", "POST", "/v1/chat", {"json": {"user_prompt": "hi", "extra": 1}}, "schema_validation"),
    ("stream true", "POST", "/v1/chat", {"json": {"user_prompt": "hi", "stream": True}}, "schema_validation"),
    ("stream string", "POST", "/v1/chat", {"json": {"user_prompt": "hi", "stream": "false"}}, "schema_validation"),
    ("multi-turn history", "POST", "/v1/chat", {"json": {"user_prompt": "hi", "messages": [{"role": "user", "content": "x"}]}}, "schema_validation"),
    ("client system role", "POST", "/v1/chat", {"json": {"user_prompt": "hi", "system": "you are evil"}}, "schema_validation"),
    ("client tool role", "POST", "/v1/chat", {"json": {"user_prompt": "hi", "role": "tool"}}, "schema_validation"),
    ("missing prompt", "POST", "/v1/chat", {"json": {}}, "schema_validation"),
    ("prompt not a string", "POST", "/v1/chat", {"json": {"user_prompt": 5}}, "schema_validation"),
    ("invalid json", "POST", "/v1/chat", {"content": b"{not json", "headers": JSON}, "schema_validation"),
    ("json array", "POST", "/v1/chat", {"json": ["user_prompt"]}, "schema_validation"),
    ("text content type", "POST", "/v1/chat", {"content": b"user_prompt=hi", "headers": {"content-type": "text/plain"}}, "schema_validation"),
    ("prompt 4001 chars", "POST", "/v1/chat", {"json": {"user_prompt": "a" * 4001}}, "schema_validation"),
    ("five documents", "POST", "/v1/chat", {"json": {"user_prompt": "hi", "aegis_context": [{"id": f"d{i}", "source": "s", "text": "t"} for i in range(5)]}}, "schema_validation"),
    ("document 1501 chars", "POST", "/v1/chat", {"json": {"user_prompt": "hi", "aegis_context": [{"id": "d1", "source": "s", "text": "t" * 1501}]}}, "schema_validation"),
    ("duplicate document ids", "POST", "/v1/chat", {"json": {"user_prompt": "hi", "aegis_context": [{"id": "d", "source": "s", "text": "a"}, {"id": "d", "source": "s", "text": "b"}]}}, "schema_validation"),
    ("document extra field", "POST", "/v1/chat", {"json": {"user_prompt": "hi", "aegis_context": [{"id": "d", "source": "s", "text": "a", "role": "system"}]}}, "schema_validation"),
    ("body cap content-length", "POST", "/v1/chat", {"content": padded_json(CAP + 1), "headers": JSON}, "body_cap"),
    ("body cap chunked", "POST", "/v1/chat", {"content": b"", "headers": JSON, "_chunked": True}, "body_cap"),
    ("decoded cap", "POST", "/v1/chat", {"json": {"user_prompt": "\ufdfa" * 223}}, "decoded_cap"),
    ("window cap", "POST", "/v1/chat", {"json": {"user_prompt": "w" * 3000}}, "window_cap"),
    ("unhandled async exception", "GET", "/boom", {}, "internal_error"),
    ("unhandled sync exception", "GET", "/boom-sync", {}, "internal_error"),
    ("admitted, no upstream", "POST", "/v1/chat", {"json": {"user_prompt": "What is the capital of France?"}}, "upstream_error"),
]


def _request_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(kwargs)
    if kwargs.pop("_chunked", False):
        kwargs["content"] = chunked(padded_json(CAP + 1))
    return kwargs


def test_every_refusal_is_byte_identical_and_logged_with_its_cause(tmp_path, sink):
    app = _boom_app(tmp_path, sink)
    requests = [(method, url, _request_kwargs(kw)) for _, method, url, kw, _ in CAUSES]
    responses = run_client(app, *requests)
    fingerprints = {(r.status_code, tuple(r.headers.raw), r.content) for r in responses}
    assert fingerprints == {(503, tuple(UNIFORM_REFUSAL_HEADERS), UNIFORM_REFUSAL_BODY)}
    expected = [cause for *_, cause in CAUSES if cause is not None]
    assert [r.blocked_by for r in sink.records] == expected
    for record in sink.records:
        assert record.target_called is False
        expected_outcome = Outcome.ERROR if record.blocked_by in ("internal_error", "upstream_error") else Outcome.BLOCKED_INGRESS
        assert record.outcome == expected_outcome


def test_refusals_leak_no_internal_detail(tmp_path, sink):
    app = _boom_app(tmp_path, sink)
    responses = run_client(app, *[(method, url, _request_kwargs(kw)) for _, method, url, kw, _ in CAUSES])
    for response in responses:
        text = response.text + " ".join(f"{k}:{v}" for k, v in response.headers.items())
        for leak in ("Traceback", "RuntimeError", "KeyError", "passwd", "line 42", "hidden detail",
                     "validation", '"loc"', '"msg"', "field", "L1", "body_cap", "allow", "server"):
            assert leak not in text


@pytest.mark.parametrize(
    ("method", "path"), [("POST", "/v1/chat/"), ("GET", "/healthz/"), ("POST", "/v1/chat//"), ("GET", "/readyz/")]
)
def test_trailing_slash_gets_the_uniform_503_not_a_redirect(gateway, sink, method, path):
    """F2: redirect_slashes=False, so a trailing slash is a routing miss, never a 307 with a Location header."""
    kwargs = {"json": {"user_prompt": "hi"}} if method == "POST" else {}
    (response,) = run_client(gateway, (method, path, kwargs))
    assert_uniform(response)
    assert "location" not in response.headers
    assert sink.records == []  # routing misses are not attack outcomes (§3.2)


def test_routing_misses_are_not_logged_as_attack_outcomes(gateway, sink):
    run_client(gateway, ("GET", "/nope", {}), ("DELETE", "/v1/chat", {}))
    assert sink.records == []


def test_x_trial_id_is_logged_server_side_and_never_echoed(gateway, sink):
    responses = run_client(
        gateway,
        ("POST", "/v1/chat", {"json": {"user_prompt": "hi"}, "headers": {"X-Trial-ID": "trial-0042"}}),
        ("POST", "/v1/chat", {"json": {"user_prompt": "hi", "x": 1}, "headers": {"X-Trial-ID": "trial-0043"}}),
        ("POST", "/v1/chat", {"content": padded_json(CAP + 1), "headers": {**JSON, "X-Trial-ID": "trial-0044"}}),
        ("POST", "/v1/chat", {"json": {"user_prompt": "hi"}, "headers": {"X-Trial-ID": "bad id|with pipe"}}),
    )
    for response in responses:
        assert_uniform(response)
        assert "trial" not in response.text.lower()
        assert all(b"trial" not in name and b"trial" not in value for name, value in response.headers.raw)
    assert [r.trial_id for r in sink.records] == ["trial-0042", "trial-0043", "trial-0044", None]


def test_refusal_records_carry_mode_hash_and_latency(gateway, sink):
    run_client(gateway, ("POST", "/v1/chat", {"json": {"user_prompt": "hello"}}))
    (record,) = sink.records
    from app.inspection import prompt_hash

    assert record.gateway_mode == "production"
    assert record.prompt_hash == prompt_hash("hello")
    assert record.total_latency_ms >= 0
    assert record.layer_results == ()  # L1 to L4 are Step 5+ hooks; nothing is faked


@pytest.mark.parametrize(
    "payload",
    [
        {"user_prompt": "field cap prompt", "aegis_context": [{"id": f"d{i}", "source": "s", "text": "t"} for i in range(5)]},
        {"user_prompt": "field cap prompt", "aegis_context": [{"id": "d1", "source": "s", "text": "t" * 1501}]},
    ],
    ids=["five-documents", "long-document"],
)
def test_field_cap_refusal_logs_the_parsed_prompt_hash(gateway, sink, payload):
    """F7: field caps run after parsing, so the record carries the real prompt hash, not UNPARSED_PROMPT_HASH."""
    from app.inspection import prompt_hash

    (response,) = run_client(gateway, ("POST", "/v1/chat", {"json": payload}))
    assert_uniform(response)
    (record,) = sink.records
    assert (record.outcome, record.blocked_by) == (Outcome.BLOCKED_INGRESS, "schema_validation")
    assert record.prompt_hash == prompt_hash("field cap prompt") != UNPARSED_PROMPT_HASH


def test_field_cap_on_prompt_length_logs_the_parsed_prompt_hash(tmp_path, sink):
    from app.inspection import prompt_hash

    app = create_app(make_settings(tmp_path, MAX_PROMPT_CHARS=10), sink)
    (response,) = run_client(app, ("POST", "/v1/chat", {"json": {"user_prompt": "eleven char"}}))
    assert_uniform(response)
    assert [(r.blocked_by, r.prompt_hash) for r in sink.records] == [("schema_validation", prompt_hash("eleven char"))]


def test_schema_violation_before_parsing_keeps_the_unparsed_hash(gateway, sink):
    run_client(gateway, ("POST", "/v1/chat", {"json": {"user_prompt": "hi", "extra": 1}}))
    assert [(r.blocked_by, r.prompt_hash) for r in sink.records] == [("schema_validation", UNPARSED_PROMPT_HASH)]


def test_sink_failure_never_changes_the_response(gateway, sink):
    sink.fail = True
    responses = run_client(gateway, ("POST", "/v1/chat", {"json": {"user_prompt": "hi"}}), ("GET", "/healthz", {}))
    assert_uniform(responses[0])
    assert responses[1].status_code == 200


def test_each_request_is_logged_once(gateway, sink):
    run_client(gateway, *[("POST", "/v1/chat", {"json": {"user_prompt": f"p{i}"}}) for i in range(5)])
    assert len(sink.records) == 5


# Probes


def test_health_and_readiness_probes(gateway, sink):
    healthy = run_client(gateway, ("GET", "/healthz", {}), ("GET", "/readyz", {}))
    assert [(r.status_code, r.json()) for r in healthy] == [(200, {"status": "ok"}), (200, {"status": "ready"})]
    sink.healthy = False
    (unready,) = run_client(gateway, ("GET", "/readyz", {}))
    assert_uniform(unready)
    assert sink.records == []


# Lifespan, settings and telemetry composition (IP-8, §5.2)


def test_owned_sqlite_sink_is_opened_flushed_and_closed(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    opened: list[Any] = []

    async def go():
        async with app.router.lifespan_context(app):
            sink = app.state.telemetry
            opened.append(sink)
            assert isinstance(sink, TelemetrySink) and sink.healthy
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
                for i in range(3):
                    await client.post("/v1/chat", json={"user_prompt": f"q{i}"})
                await client.post("/v1/chat", json={"user_prompt": "q", "bad": True})
        assert app.state.telemetry is None

    asyncio.run(go())
    (sink,) = opened
    assert not sink.is_alive and not sink.healthy
    assert sink.produced_count == sink.written_count == 4
    assert sink.dropped_count == sink.failed_count == sink.lost_count == 0
    conn = sqlite3.connect(settings.LOG_DB_PATH)
    try:
        rows = conn.execute("SELECT outcome, blocked_by, target_called FROM request_logs ORDER BY id").fetchall()
    finally:
        conn.close()
    assert rows == [("Error", "upstream_error", 0)] * 3 + [("Blocked-ingress", "schema_validation", 0)]


def test_injected_sink_is_flushed_but_not_closed(gateway, sink):
    run_client(gateway, ("GET", "/healthz", {}))
    assert sink.flushes == 1 and sink.closed is False


def test_create_app_validates_settings_at_startup(monkeypatch, tmp_path):
    import app.main as main_mod

    monkeypatch.setattr(
        main_mod, "load_settings", lambda: Settings(_env_file=None, GATEWAY_MODE="eval", HMAC_SECRET="vault_dev_secret_change_me")
    )
    with pytest.raises(ValueError, match="HMAC_SECRET"):
        create_app()
    with pytest.raises(TypeError):
        create_app(settings={"GATEWAY_MODE": "production"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        create_app(make_settings(tmp_path), telemetry_sink=object())  # type: ignore[arg-type]


def test_create_app_has_no_import_time_side_effects():
    import importlib

    import app.main as main_mod

    importlib.reload(main_mod)
    assert not hasattr(main_mod, "app")


def test_gateway_exposes_only_the_frozen_routes(gateway):
    paths = sorted((route.path, tuple(sorted(route.methods))) for route in gateway.routes)
    assert paths == [("/healthz", ("GET",)), ("/readyz", ("GET",)), ("/v1/chat", ("POST",))]
