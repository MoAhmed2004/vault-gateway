"""Deterministic content hash for log and cache keys (IP-9, §3.2, IP-5).

Python's built-in `hash()` is salted per process and must never be used for
keys that have to repeat across runs. This module serializes the payload to
canonical JSON and hashes the UTF-8 bytes (lone surrogates kept via surrogatepass; astral characters never collide with surrogate pairs) with SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from enum import Enum
from typing import Any

from pydantic import BaseModel


def _normalize(value: Any) -> Any:
    """Convert a payload into plain JSON types with a single canonical shape."""
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, Enum):
                key = key.value
            if not isinstance(key, str):
                raise TypeError(f"canonical_hash requires string mapping keys, got {type(key).__name__}")
            normalized[key] = _normalize(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        # 0.0, -0.0, and 3.0 hash like 0 and 3. bool is not a float subclass.
        return int(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"canonical_hash cannot serialize type {type(value).__name__}")


def canonical_json(payload: Any) -> str:
    """Return the canonical JSON text that `canonical_hash` digests."""
    return json.dumps(
        _normalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_hash(payload: Any) -> str:
    """Return the lowercase 64-character SHA-256 hex digest of the canonical JSON."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8", errors="surrogatepass")).hexdigest()
