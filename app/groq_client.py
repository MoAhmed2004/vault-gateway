"""The single Groq client module (§2.7, §3.9, TM-R6, IP-6, IP-9).

Every Groq call goes through `GroqClient`, so the model pin and the client
settings live in one place.

- Model pinning (§2.7, §3.9): `verify_models()` reads Groq's model list
  (`GET /models`) and raises `ModelCheckError` when `TARGET_MODEL` or any
  `PROBE_CANDIDATES` entry is missing or inactive, or the list cannot be read.
  The lifespan runs it before the gateway accepts traffic, so a bad pin fails
  fast. `ALLOW_MODEL_FAILOVER=false` is enforced twice: no code path ever
  sends another model id, and a completion whose `model` field differs from
  the pinned id is refused (`model_mismatch`), so a provider-side alias or
  fallback can never mix models mid-experiment.
- No retries (TM-R6): every 503 is a refusal, so the client never retries.
  The httpx transport is built with `retries=0`, redirects are not followed,
  and there is no retry loop.
- Strict timeout: httpx enforces `timeout_s` per connect, read, write and pool
  phase, and an outer `asyncio.wait_for` bounds the whole call to `timeout_s`.
  TM-R2 budgets exclude target LLM time, so this timeout is separate from them.
- Failures: a timeout, transport error, non-2xx status (429 rate limit, 5xx,
  4xx), malformed JSON (including the non-standard constants `NaN`,
  `Infinity` and `-Infinity`, which Python's `json` would otherwise accept,
  Step 5B.1 LM-1), or a body without a usable assistant message raises
  `UpstreamError` with a short `kind` label. The gateway maps every kind to
  the uniform 503 logged as `Error` / `upstream_error`. No message carries
  the API key, a request body or a response body.
- Response shape (IP-6, IP-9): `complete()` returns the full raw provider JSON
  (with any `reasoning` field) for the response cache and L4.

Tests inject `transport=httpx.MockTransport(...)`, so no unit test touches the
network.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import httpx

logger = logging.getLogger("vault.upstream")

MODELS_PATH = "/models"
CHAT_PATH = "/chat/completions"


class UpstreamError(Exception):
    """A failed upstream call. `kind` is a short label safe to log; it never holds secrets."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


class ModelCheckError(RuntimeError):
    """Startup model check failed (§2.7, §3.9): the gateway must not start."""


def _reject_non_finite(token: str) -> Any:
    """`json.loads` hook for `NaN`, `Infinity` and `-Infinity`: refuse them (RFC 8259 has no such values)."""
    raise ValueError(f"non-standard JSON constant {token}")


def parse_provider_json(body: bytes) -> Any:
    """Strict JSON parse of a provider body. Raises ValueError on malformed JSON or a non-finite constant."""
    try:
        return json.loads(body, parse_constant=_reject_non_finite)
    except RecursionError:
        raise ValueError("provider JSON nests too deeply") from None


def _message(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("provider JSON has no choices[0]")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ValueError("provider JSON choices[0] has no message object")
    return message


def response_fields(raw: Mapping[str, Any]) -> tuple[str, str | None]:
    """(content, reasoning) of the first choice (IP-6). Missing or null content is the empty string.

    Raises ValueError when the JSON has no assistant message or a field has the wrong type.
    """
    message = _message(raw)
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise ValueError("assistant content must be a string or null")
    reasoning = message.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError("assistant reasoning must be a string or null")
    return content, reasoning


class GroqClient:
    """Async OpenAI-compatible Groq client pinned to one model."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout_s: float,
        parameters: Mapping[str, Any],
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("GROQ_API_KEY is required for the Groq upstream")
        if not model.strip():
            raise ValueError("model must be a non-empty id")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        reserved = {"model", "messages", "stream", "n"} & set(parameters)
        if reserved:
            raise ValueError(f"parameters must not set {sorted(reserved)}")
        self.model = model
        self.timeout_s = timeout_s
        self.parameters: Mapping[str, Any] = MappingProxyType(dict(parameters))
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=httpx.Timeout(timeout_s),
            transport=transport if transport is not None else httpx.AsyncHTTPTransport(retries=0),
            follow_redirects=False,
        )

    def __repr__(self) -> str:  # never print the key
        return f"GroqClient(model={self.model!r})"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await asyncio.wait_for(self._client.request(method, path, **kwargs), timeout=self.timeout_s)
        except (TimeoutError, httpx.TimeoutException):
            raise UpstreamError("timeout") from None
        except httpx.HTTPError:
            raise UpstreamError("transport") from None
        if response.status_code == 429:
            raise UpstreamError("rate_limited")
        if not 200 <= response.status_code < 300:
            raise UpstreamError(f"http_{response.status_code}")
        try:
            body = parse_provider_json(response.content)
        except ValueError:  # JSONDecodeError, UnicodeDecodeError and NaN / Infinity are all ValueError
            raise UpstreamError("bad_json") from None
        if not isinstance(body, dict):
            raise UpstreamError("bad_json")
        return body

    async def list_models(self) -> dict[str, bool]:
        """Model ids from Groq's list, mapped to whether each is active (absent flag means active)."""
        body = await self._request("GET", MODELS_PATH)
        data = body.get("data")
        if not isinstance(data, list):
            raise UpstreamError("bad_model_list")
        models: dict[str, bool] = {}
        for item in data:
            if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                models[item["id"]] = item.get("active", True) is not False
        return models

    async def verify_models(self, required: Sequence[str]) -> None:
        """Fail fast unless every id in `required` is on Groq's active model list (§2.7, §3.9)."""
        try:
            models = await self.list_models()
        except UpstreamError as exc:
            raise ModelCheckError(f"cannot read Groq's model list ({exc.kind}); refusing to start") from None
        missing = [model for model in dict.fromkeys(required) if not models.get(model, False)]
        if missing:
            raise ModelCheckError(f"model ids missing or inactive on Groq: {missing}; refusing to start")
        logger.info("model check passed: %s", ", ".join(dict.fromkeys(required)))

    async def complete(self, messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        """One non-streaming chat completion; returns the raw provider JSON (IP-9)."""
        payload = {"model": self.model, "messages": [dict(m) for m in messages], "stream": False, **self.parameters}
        raw = await self._request("POST", CHAT_PATH, json=payload)
        if raw.get("model") != self.model:
            raise UpstreamError("model_mismatch")  # ALLOW_MODEL_FAILOVER=false (§2.7)
        try:
            response_fields(raw)
        except ValueError:
            raise UpstreamError("bad_response") from None
        return raw


def target_parameters(
    *, temperature: float, seed: int, max_tokens: int, reasoning_effort: str | None
) -> dict[str, Any]:
    """Target parameters (§2.7: temperature 0, fixed seed); part of the IP-5 target-response key."""
    parameters: dict[str, Any] = {"temperature": temperature, "seed": seed, "max_tokens": max_tokens}
    if reasoning_effort is not None:
        parameters["reasoning_effort"] = reasoning_effort
    return parameters
