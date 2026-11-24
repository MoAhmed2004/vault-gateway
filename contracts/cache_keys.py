"""IP-5 cache key formats (§5.2 interface freeze). Imports only stdlib, pydantic and `contracts`.

IP-5 defines three caches:

- Layer verdicts by (prompt hash, layer version, bundle version, ChromaDB
  version). The layer name is keyed too, so two layers that share a version
  string never share an entry. The ChromaDB version is keyed only for L2:
  IP-5 says an insertion "invalidates cached L2 verdicts", so it is required
  for L2 and must be None for L1 and L3 (an L2 insertion never evicts them).
- Egress verdicts by (response hash, manifest ID, variant).
- Target responses by (model ID, parameters, full message list including the
  system prompt).

Every key is `canonical_hash` of a payload tagged with the cache name and
`CACHE_KEY_SCHEMA`, so keys of different caches can never collide. No key
contains the gateway configuration (C0 to C3), the attacker seed, or the
gateway mode: under PS-3 the message list, and with it the target-response
key, is identical across seeds and configurations. Python `hash()` is never used.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from contracts.hashing import canonical_hash

CACHE_KEY_SCHEMA = 1
INGRESS_LAYER_NAMES = ("L1", "L2", "L3")
EGRESS_VARIANT_NAMES = ("X0", "X1", "X2", "X3")
MESSAGE_ROLES = ("system", "user", "assistant")
RESERVED_PARAMETER_KEYS = frozenset({"model", "messages"})

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 hex digest")
    return value


def _require_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def raw_response_hash(raw_provider_json: Mapping[str, Any]) -> str:
    """Response hash over the full raw provider JSON, `reasoning` included (IP-5, IP-6, IP-9)."""
    if not isinstance(raw_provider_json, Mapping):
        raise TypeError("raw_provider_json must be a mapping")
    return canonical_hash(raw_provider_json)


def layer_verdict_key(
    *,
    prompt_hash: str,
    layer: str,
    layer_version: str,
    bundle_version: str,
    chroma_version: str | None,
) -> str:
    """Layer-verdict cache key (IP-5)."""
    _require_sha256("prompt_hash", prompt_hash)
    if layer not in INGRESS_LAYER_NAMES:
        raise ValueError(f"layer must be one of {INGRESS_LAYER_NAMES}, got {layer!r}")
    _require_text("layer_version", layer_version)
    _require_text("bundle_version", bundle_version)
    if layer == "L2":
        _require_text("chroma_version", chroma_version)  # type: ignore[arg-type]
    elif chroma_version is not None:
        raise ValueError(f"chroma_version applies only to L2; pass None for {layer}")
    return canonical_hash(
        {
            "cache": "layer_verdict",
            "schema": CACHE_KEY_SCHEMA,
            "prompt_hash": prompt_hash,
            "layer": layer,
            "layer_version": layer_version,
            "bundle_version": bundle_version,
            "chroma_version": chroma_version,
        }
    )


def egress_verdict_key(*, response_hash: str, manifest_id: str, variant: str) -> str:
    """Egress-verdict cache key (IP-5)."""
    _require_sha256("response_hash", response_hash)
    _require_text("manifest_id", manifest_id)
    if variant not in EGRESS_VARIANT_NAMES:
        raise ValueError(f"variant must be one of {EGRESS_VARIANT_NAMES}, got {variant!r}")
    return canonical_hash(
        {
            "cache": "egress_verdict",
            "schema": CACHE_KEY_SCHEMA,
            "response_hash": response_hash,
            "manifest_id": manifest_id,
            "variant": variant,
        }
    )


def _normalize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence) or len(messages) < 2:
        raise ValueError("messages must be a sequence holding at least the system prompt and one more message")
    normalized: list[dict[str, str]] = []
    for position, message in enumerate(messages):
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError(f"message {position} must have exactly the keys 'role' and 'content'")
        role, content = message["role"], message["content"]
        if role not in MESSAGE_ROLES:
            raise ValueError(f"message {position} role must be one of {MESSAGE_ROLES}, got {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"message {position} content must be a string")
        normalized.append({"role": role, "content": content})
    if normalized[0]["role"] != "system":
        raise ValueError("the full message list must start with the gateway-built system prompt (IP-5)")
    if any(m["role"] == "system" for m in normalized[1:]):
        raise ValueError("only the first message may carry the system role")
    return normalized


def target_response_key(
    *,
    model_id: str,
    parameters: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
) -> str:
    """Target-response cache key over (model ID, parameters, full message list) (IP-5, §2.7)."""
    _require_text("model_id", model_id)
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping")
    reserved = RESERVED_PARAMETER_KEYS & set(parameters)
    if reserved:
        raise ValueError(f"parameters must not contain {sorted(reserved)}; they are keyed separately")
    return canonical_hash(
        {
            "cache": "target_response",
            "schema": CACHE_KEY_SCHEMA,
            "model_id": model_id,
            "parameters": dict(parameters),
            "messages": _normalize_messages(messages),
        }
    )
