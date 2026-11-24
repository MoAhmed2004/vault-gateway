"""Canonical gateway modes and secret-derivation mode mapping (PS-3, TM-S3, IP-4, IP-9).

Imports only stdlib so `contracts/`, `app/config.py`, and `app/secret_derivation.py`
share one source of truth without importing `Faker` or `phonenumbers`.
"""

from __future__ import annotations

from typing import Literal

GatewayMode = Literal["production", "demo", "eval", "shadow_ingress_only", "shadow_full_e2e"]
DerivationKind = Literal["deterministic", "random"]

DETERMINISTIC_MODES: tuple[GatewayMode, ...] = ("eval", "shadow_ingress_only", "shadow_full_e2e")
RANDOM_MODES: tuple[GatewayMode, ...] = ("production", "demo")
ALL_GATEWAY_MODES: tuple[GatewayMode, ...] = DETERMINISTIC_MODES + RANDOM_MODES


def resolve_derivation_mode(gateway_mode: str) -> DerivationKind:
    """Map a gateway mode to `deterministic` or `random`; any other value raises ValueError."""
    if gateway_mode in DETERMINISTIC_MODES:
        return "deterministic"
    if gateway_mode in RANDOM_MODES:
        return "random"
    raise ValueError(f"unsupported gateway mode for secret derivation: {gateway_mode!r}")
