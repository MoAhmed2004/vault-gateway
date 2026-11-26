"""Test doubles shared by the upstream and gateway tests. No network: every Groq call hits `httpx.MockTransport`."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx

from app.groq_client import GroqClient, target_parameters

TARGET = "openai/gpt-oss-20b"
ALL_CANDIDATES = ("openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b")
TEST_API_KEY = "gsk_unit_test_key_0123456789"


def system_secret(request_json: dict[str, Any], label: str) -> str:
    """The value on the `label: value` line of the system prompt (lets a fake target 'leak' it)."""
    system = request_json["messages"][0]["content"]
    for line in system.splitlines():
        if line.startswith(f"{label}: "):
            return line.split(": ", 1)[1]
    raise KeyError(label)


class FakeGroq:
    """Programmable Groq stand-in. `reply(request_json)` returns (content, reasoning) for each chat call."""

    def __init__(
        self,
        *,
        models: tuple[str, ...] = ALL_CANDIDATES,
        inactive: tuple[str, ...] = (),
        reply: Callable[[dict[str, Any]], tuple[str | None, str | None]] | None = None,
        chat_status: int = 200,
        models_status: int = 200,
        response_model: str | None = None,
        chat_delay_s: float = 0.0,
        chat_exception: Exception | None = None,
        raw_body: bytes | None = None,
    ) -> None:
        self.models = models
        self.inactive = inactive
        self.reply = reply or (lambda body: ("Paris is the capital of France.", None))
        self.chat_status = chat_status
        self.models_status = models_status
        self.response_model = response_model
        self.chat_delay_s = chat_delay_s
        self.chat_exception = chat_exception
        self.raw_body = raw_body
        self.requests: list[httpx.Request] = []

    @property
    def chat_requests(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith("/chat/completions")]

    @property
    def model_list_calls(self) -> int:
        return sum(1 for r in self.requests if r.url.path.endswith("/models"))

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            if self.models_status != 200:
                return httpx.Response(self.models_status, json={"error": {"message": "nope"}})
            data = [{"id": m, "object": "model", "active": m not in self.inactive} for m in self.models]
            return httpx.Response(200, json={"object": "list", "data": data})
        if self.chat_delay_s:
            await asyncio.sleep(self.chat_delay_s)
        if self.chat_exception is not None:
            raise self.chat_exception
        if self.chat_status != 200:
            return httpx.Response(self.chat_status, json={"error": {"message": "upstream says no"}})
        if self.raw_body is not None:
            return httpx.Response(200, content=self.raw_body, headers={"content-type": "application/json"})
        body = json.loads(request.content)
        content, reasoning = self.reply(body)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning is not None:
            message["reasoning"] = reasoning
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": self.response_model or body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    def client(self, *, model: str = TARGET, timeout_s: float = 5.0) -> GroqClient:
        return GroqClient(
            api_key=TEST_API_KEY,
            model=model,
            base_url="https://api.groq.test/openai/v1",
            timeout_s=timeout_s,
            parameters=target_parameters(temperature=0.0, seed=0, max_tokens=256, reasoning_effort=None),
            transport=httpx.MockTransport(self.handler),
        )
