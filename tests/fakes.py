"""Test doubles shared by the upstream, layer and gateway tests.

No network: every Groq call hits `httpx.MockTransport`. `FakeEmbedder` and
`FakeClassifier` are deterministic stand-ins for MiniLM and DeBERTa with the
same interface (`app.onnx_models.Embedder`, `Classifier`), so layer and
pipeline logic is tested without model weights. Tests that need the real
artifacts use `real_models_available()` and skip with a reason when
`scripts/fetch_models.py` has not been run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from app.groq_client import GroqClient, target_parameters
from app.inspection import TokenizerLimit

REPO_ROOT = Path(__file__).resolve().parent.parent
MINILM_DIR = REPO_ROOT / "models" / "minilm"
DEBERTA_DIR = REPO_ROOT / "models" / "deberta"
_WORDS = re.compile(r"\w+")


def real_models_available() -> bool:
    return all((d / f).is_file() for d in (MINILM_DIR, DEBERTA_DIR) for f in ("model.onnx", "tokenizer.json"))


def real_tokenizer_limit(name: str, limit: int) -> TokenizerLimit:
    """The real MiniLM or DeBERTa tokenizer as the gateway loads it (no truncation, special tokens counted,
    MiniLM `max_input_chars_per_word` raised, M-3), without loading the model."""
    from app.onnx_models import MINILM_MAX_INPUT_CHARS_PER_WORD, load_tokenizer

    directory = MINILM_DIR if name == "minilm" else DEBERTA_DIR
    word_limit = MINILM_MAX_INPUT_CHARS_PER_WORD if name == "minilm" else None
    tokenizer = load_tokenizer(directory / "tokenizer.json", max_input_chars_per_word=word_limit)
    return TokenizerLimit(name=name, token_limit=limit, count_tokens=lambda text: len(tokenizer.encode(text).ids))


class FakeEmbedder:
    """Bag-of-words hashing embedder: cosine similarity is word overlap. One token per word plus 2 specials."""

    name = "minilm"
    token_limit = 256

    def __init__(self, *, dimension: int = 4096, delay_s: float = 0.0, version: str = "fake-embedder-v1") -> None:
        self.dimension = dimension
        self.delay_s = delay_s
        self.version = version
        self.calls = 0
        self.on_embed: Callable[[str], None] | None = None
        self._lock = threading.Lock()

    def count_tokens(self, text: str) -> int:
        return len(_WORDS.findall(text)) + 2

    def embed(self, text: str) -> np.ndarray:
        with self._lock:
            self.calls += 1
        if self.on_embed is not None:
            self.on_embed(text)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.count_tokens(text) > self.token_limit:
            raise ValueError("above the fake token limit")
        vector = np.zeros(self.dimension, dtype=np.float64)
        for word in _WORDS.findall(text.casefold()):
            vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = np.linalg.norm(vector)
        if norm == 0:
            vector[0] = 1.0
            norm = 1.0
        return (vector / norm).astype(np.float32)


class FakeClassifier:
    """Malicious probability from trigger words; optional per-call delay for TM-R2 budget tests."""

    name = "deberta"
    token_limit = 512

    def __init__(self, *, triggers: tuple[str, ...] = ("ignore", "jailbreak"), delay_s: float = 0.0,
                 version: str = "fake-classifier-v1", fail: bool = False) -> None:
        self.triggers = triggers
        self.delay_s = delay_s
        self.version = version
        self.fail = fail
        self.calls = 0
        self._lock = threading.Lock()

    def count_tokens(self, text: str) -> int:
        return len(_WORDS.findall(text)) + 2

    def malicious_probability(self, text: str) -> float:
        with self._lock:
            self.calls += 1
        if self.fail:
            raise RuntimeError("classifier exploded")
        if self.delay_s:
            time.sleep(self.delay_s)
        words = set(_WORDS.findall(text.casefold()))
        return 0.99 if words & set(self.triggers) else 0.01

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
