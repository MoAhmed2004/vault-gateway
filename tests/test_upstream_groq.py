"""Unit tests for the single Groq client module (§2.7, §3.9, TM-R6, IP-6, IP-9; AGENTS.md Rule 2).

Every test drives `httpx.MockTransport`; no test opens a network connection.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.groq_client import GroqClient, ModelCheckError, UpstreamError, parse_provider_json, response_fields, target_parameters
from app.main import build_groq_client
from tests.fakes import ALL_CANDIDATES, TARGET, TEST_API_KEY, FakeGroq

MESSAGES = ({"role": "system", "content": "sys"}, {"role": "user", "content": "hi"})


def run(coro):
    return asyncio.run(coro)


async def _with(client: GroqClient, coro_fn):
    try:
        return await coro_fn(client)
    finally:
        await client.aclose()


# Startup model check (§2.7, §3.9)


def test_model_check_passes_when_target_and_candidates_are_active():
    fake = FakeGroq()
    run(_with(fake.client(), lambda c: c.verify_models([TARGET, *ALL_CANDIDATES])))
    assert fake.model_list_calls == 1
    assert fake.requests[0].headers["authorization"] == f"Bearer {TEST_API_KEY}"


@pytest.mark.parametrize(
    ("models", "inactive", "missing"),
    [
        (("qwen/qwen3.8-27b", "openai/gpt-oss-120b"), (), TARGET),
        (ALL_CANDIDATES, (TARGET,), TARGET),
        ((TARGET, "openai/gpt-oss-120b"), (), "qwen/qwen3.8-27b"),
        (("llama-3.1-8b-instant",), (), TARGET),
    ],
    ids=["target-missing", "target-inactive", "candidate-missing", "only-retired-model"],
)
def test_model_check_fails_fast_on_a_missing_or_inactive_id(models, inactive, missing):
    fake = FakeGroq(models=models, inactive=inactive)
    with pytest.raises(ModelCheckError, match="refusing to start") as info:
        run(_with(fake.client(), lambda c: c.verify_models([TARGET, *ALL_CANDIDATES])))
    assert missing in str(info.value)


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 503])
def test_model_check_fails_fast_when_the_list_cannot_be_read(status):
    fake = FakeGroq(models_status=status)
    with pytest.raises(ModelCheckError, match="cannot read Groq's model list"):
        run(_with(fake.client(), lambda c: c.verify_models([TARGET])))
    assert fake.model_list_calls == 1  # no retry


def test_model_check_fails_fast_when_groq_is_unreachable():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    client = GroqClient(
        api_key=TEST_API_KEY, model=TARGET, base_url="https://api.groq.test/openai/v1", timeout_s=1.0,
        parameters={}, transport=httpx.MockTransport(refuse),
    )
    with pytest.raises(ModelCheckError, match=r"\(transport\)"):
        run(_with(client, lambda c: c.verify_models([TARGET])))


def test_model_list_with_bad_shape_is_refused():
    client = GroqClient(
        api_key=TEST_API_KEY, model=TARGET, base_url="https://api.groq.test/openai/v1", timeout_s=1.0,
        parameters={}, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": "nope"})),
    )
    with pytest.raises(ModelCheckError, match="bad_model_list"):
        run(_with(client, lambda c: c.verify_models([TARGET])))


def test_model_check_error_message_never_carries_the_api_key():
    fake = FakeGroq(models=())
    with pytest.raises(ModelCheckError) as info:
        run(_with(fake.client(), lambda c: c.verify_models([TARGET])))
    assert TEST_API_KEY not in str(info.value)


# Completions: pinned model, parameters, raw JSON (IP-9)


def test_completion_sends_the_pinned_model_and_parameters_and_returns_raw_json():
    fake = FakeGroq(reply=lambda body: ("Paris.", "thinking about geography"))
    raw = run(_with(fake.client(), lambda c: c.complete(MESSAGES)))
    (sent,) = fake.chat_requests
    assert sent == {
        "model": TARGET, "messages": [dict(m) for m in MESSAGES], "stream": False,
        "temperature": 0.0, "seed": 0, "max_tokens": 256,
    }
    assert raw["choices"][0]["message"]["reasoning"] == "thinking about geography"  # IP-9: full raw JSON kept
    assert response_fields(raw) == ("Paris.", "thinking about geography")


def test_failover_is_impossible_a_different_response_model_is_refused():
    """ALLOW_MODEL_FAILOVER=false (§2.7): a provider-side alias or fallback never reaches the client."""
    fake = FakeGroq(response_model="openai/gpt-oss-120b")
    with pytest.raises(UpstreamError) as info:
        run(_with(fake.client(), lambda c: c.complete(MESSAGES)))
    assert info.value.kind == "model_mismatch"
    assert len(fake.chat_requests) == 1  # never retried with another model


@pytest.mark.parametrize(
    ("status", "kind"), [(429, "rate_limited"), (500, "http_500"), (502, "http_502"), (503, "http_503"), (400, "http_400"), (401, "http_401")]
)
def test_http_errors_map_to_upstream_error_without_retry(status, kind):
    fake = FakeGroq(chat_status=status)
    with pytest.raises(UpstreamError) as info:
        run(_with(fake.client(), lambda c: c.complete(MESSAGES)))
    assert info.value.kind == kind
    assert len(fake.chat_requests) == 1  # max_retries=0 (TM-R6)


@pytest.mark.parametrize(
    ("exception", "kind"),
    [
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectTimeout("slow"), "timeout"),
        (httpx.ConnectError("refused"), "transport"),
        (httpx.RemoteProtocolError("bad"), "transport"),
    ],
)
def test_transport_failures_map_to_upstream_error(exception, kind):
    fake = FakeGroq(chat_exception=exception)
    with pytest.raises(UpstreamError) as info:
        run(_with(fake.client(), lambda c: c.complete(MESSAGES)))
    assert info.value.kind == kind


def test_the_whole_call_is_bounded_by_the_timeout():
    fake = FakeGroq(chat_delay_s=2.0)
    started = asyncio.run(_timed(fake.client(timeout_s=0.2)))
    assert started < 1.5


async def _timed(client: GroqClient) -> float:
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    with pytest.raises(UpstreamError) as info:
        await _with(client, lambda c: c.complete(MESSAGES))
    assert info.value.kind == "timeout"
    return loop.time() - t0


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[1, 2]",
        json.dumps({"model": TARGET, "choices": []}).encode(),
        json.dumps({"model": TARGET, "choices": [{"message": "x"}]}).encode(),
        json.dumps({"model": TARGET, "choices": [{"message": {"content": 5}}]}).encode(),
        json.dumps({"model": TARGET, "choices": [{"message": {"content": "a", "reasoning": ["x"]}}]}).encode(),
    ],
    ids=["not-json", "array", "no-choices", "message-not-object", "content-not-string", "reasoning-not-string"],
)
def test_malformed_provider_bodies_are_refused(body):
    fake = FakeGroq(raw_body=body)
    with pytest.raises(UpstreamError) as info:
        run(_with(fake.client(), lambda c: c.complete(MESSAGES)))
    assert info.value.kind in ("bad_json", "bad_response", "model_mismatch")


def test_null_content_is_the_empty_string():
    fake = FakeGroq(reply=lambda body: (None, None))
    raw = run(_with(fake.client(), lambda c: c.complete(MESSAGES)))
    assert response_fields(raw) == ("", None)


# Construction and settings (§2.7, TM-R6)


def test_client_refuses_blank_key_reserved_parameters_and_bad_timeout():
    base = dict(model=TARGET, base_url="https://api.groq.test", timeout_s=1.0, parameters={})
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqClient(api_key="  ", **base)
    for reserved in ("model", "messages", "stream", "n"):
        with pytest.raises(ValueError, match="must not set"):
            GroqClient(api_key=TEST_API_KEY, **{**base, "parameters": {reserved: 1}})
    with pytest.raises(ValueError, match="timeout_s"):
        GroqClient(api_key=TEST_API_KEY, **{**base, "timeout_s": 0})


def test_repr_hides_the_key():
    client = FakeGroq().client()
    try:
        assert TEST_API_KEY not in repr(client) and TARGET in repr(client)
    finally:
        run(client.aclose())


def test_production_client_has_no_retries_and_no_redirects(tmp_path):
    settings = Settings(
        _env_file=None, GROQ_API_KEY=TEST_API_KEY, LOG_DB_PATH=tmp_path / "l.db", CACHE_DB_PATH=tmp_path / "c.db",
        UPSTREAM_TIMEOUT_S=7.5, TARGET_MAX_TOKENS=64, TARGET_REASONING_EFFORT="low",
    )
    client = build_groq_client(settings)
    try:
        transport = client._client._transport
        assert isinstance(transport, httpx.AsyncHTTPTransport)
        assert transport._pool._retries == 0
        assert client._client.follow_redirects is False
        assert client._client.timeout == httpx.Timeout(7.5)
        assert str(client._client.base_url).rstrip("/") == "https://api.groq.com/openai/v1"
        assert client.model == settings.TARGET_MODEL
        assert dict(client.parameters) == {"temperature": 0.0, "seed": 0, "max_tokens": 64, "reasoning_effort": "low"}
    finally:
        run(client.aclose())


def test_target_parameters_omit_reasoning_effort_when_unset():
    assert target_parameters(temperature=0.0, seed=3, max_tokens=64, reasoning_effort=None) == {
        "temperature": 0.0, "seed": 3, "max_tokens": 64,
    }


def test_settings_for_the_upstream(tmp_path):
    base = dict(_env_file=None, LOG_DB_PATH=tmp_path / "l.db", CACHE_DB_PATH=tmp_path / "c.db")
    assert Settings(**base, TARGET_REASONING_EFFORT="  ").TARGET_REASONING_EFFORT is None
    with pytest.raises(ValueError, match="https"):
        Settings(**base, GROQ_BASE_URL="http://api.groq.com/openai/v1")
    with pytest.raises(ValueError, match="ALLOW_MODEL_FAILOVER"):
        Settings(**base, ALLOW_MODEL_FAILOVER=True)
    with pytest.raises(ValueError):
        Settings(**base, UPSTREAM_TIMEOUT_S=0)


# Non-standard JSON constants (TM-R6, RT-V6; Step 5B.1 LM-1, closes Step 5A LM1)

NAN_BODY = (
    b'{"id": "x", "model": "openai/gpt-oss-20b", "usage": {"queue_time": NaN}, '
    b'"choices": [{"message": {"role": "assistant", "content": "Paris."}}]}'
)


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity"])
def test_non_finite_json_constants_are_bad_json(constant):
    fake = FakeGroq(raw_body=NAN_BODY.replace(b"NaN", constant))
    with pytest.raises(UpstreamError) as info:
        run(_with(fake.client(), lambda c: c.complete(MESSAGES)))
    assert info.value.kind == "bad_json"
    assert len(fake.chat_requests) == 1  # refused, never retried


@pytest.mark.parametrize("number", [b"1e999", b"-1e400", b"1.5e308000", b"-0.1e99999"])
def test_numbers_that_overflow_to_infinity_are_bad_json(number):
    """Step 5C F1 (TM-R6, IP-9, RT-V6): valid JSON numbers that float() turns into inf are refused, not parsed."""
    fake = FakeGroq(raw_body=NAN_BODY.replace(b"NaN", number))
    with pytest.raises(UpstreamError) as info:
        run(_with(fake.client(), lambda c: c.complete(MESSAGES)))
    assert info.value.kind == "bad_json"
    assert len(fake.chat_requests) == 1


@pytest.mark.parametrize("number", [b"1e999", b"-1e400"])
def test_parse_provider_json_refuses_overflow_floats_anywhere(number):
    with pytest.raises(ValueError, match="non-finite"):
        parse_provider_json(b'{"a": [1, {"b": ' + number + b"}]}")


def test_large_finite_floats_and_big_integers_are_still_accepted():
    parsed = parse_provider_json(b'{"a": 1.7976931348623157e308, "b": -2.5e-320, "c": 123456789012345678901234567890}')
    assert parsed["a"] == 1.7976931348623157e308 and parsed["b"] == -2.5e-320
    assert parsed["c"] == 123456789012345678901234567890


def test_finite_json_with_the_same_shape_is_accepted():
    fake = FakeGroq(raw_body=NAN_BODY.replace(b"NaN", b"0.25"))
    raw = run(_with(fake.client(), lambda c: c.complete(MESSAGES)))
    assert raw["usage"]["queue_time"] == 0.25 and response_fields(raw) == ("Paris.", None)


@pytest.mark.parametrize("body", [b"\xff\xfe not utf-8", b"[" * 100_000 + b"]" * 100_000], ids=["not-utf8", "deep-nesting"])
def test_undecodable_or_deeply_nested_json_is_bad_json(body):
    fake = FakeGroq(raw_body=body)
    with pytest.raises(UpstreamError) as info:
        run(_with(fake.client(), lambda c: c.complete(MESSAGES)))
    assert info.value.kind == "bad_json"
