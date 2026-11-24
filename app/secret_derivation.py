"""Domain-separated secret derivation (PS-2, PS-3, TM-S3, D-PII).

Deterministic gateway modes (`eval`, `shadow_ingress_only`, `shadow_full_e2e`)
derive every value from `HMAC_SECRET` with a distinct domain tag (`canary|`,
`pii|`, `nonce|`) so no two domains can collide. Random modes (`production`,
`demo`) draw fresh values from `secrets`. Every `derive_*` function requires
`mode`, so a caller can never fall back to deterministic secrets by omission.
Canonical mode sets and `resolve_derivation_mode` live in `contracts.modes`.
Faker is used only through an isolated instance (`seed_instance`); the global
`Faker.seed()` and the global `random` module are never touched.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from collections.abc import Mapping

from faker import Faker
import phonenumbers

from contracts.modes import (
    ALL_GATEWAY_MODES,
    DETERMINISTIC_MODES,
    DerivationKind,
    RANDOM_MODES,
    resolve_derivation_mode,
)

PII_ENTITY_TYPES = ("EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN", "IBAN_CODE")
CANARY_PREFIX = "VAULT-CANARY-"

# Real US geographic area codes. The exchange is generated as [2-9]XX with the
# N11 pattern excluded and the line as four digits, so every number matches the
# NANP national number pattern that libphonenumber validates.
US_AREA_CODES = (
    "201", "202", "203", "205", "206", "207", "208", "209", "210", "212", "213", "214", "215", "216",
    "217", "218", "219", "225", "228", "239", "248", "253", "254", "256", "260", "281", "301", "302",
    "303", "304", "305", "307", "310", "312", "313", "314", "315", "316", "317", "318", "319", "323",
    "330", "334", "336", "337", "347", "360", "385", "401", "402", "404", "405", "406", "407", "408",
    "410", "412", "414", "415", "425", "469", "480", "501", "502", "503", "504", "505", "507", "512",
    "513", "515", "602", "603", "605", "609", "612", "614", "615", "617", "619", "626", "630", "646",
    "650", "702", "703", "704", "712", "713", "714", "718", "720", "770", "773", "786", "801", "804",
    "805", "808", "813", "816", "817", "818", "832", "845", "857", "901", "904", "908", "913", "914",
    "916", "917", "919", "925", "949", "972", "973",
)

__all__ = [
    "ALL_GATEWAY_MODES",
    "CANARY_PREFIX",
    "DETERMINISTIC_MODES",
    "DerivationKind",
    "PII_ENTITY_TYPES",
    "RANDOM_MODES",
    "US_AREA_CODES",
    "build_defensive_system_prompt",
    "derive_canary",
    "derive_sanitizer_nonce",
    "derive_synthetic_pii",
    "resolve_derivation_mode",
    "wrap_external_data",
]


def _hmac_hex(secret: str, message: str) -> str:
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def derive_canary(secret: str, objective_id: str, *, mode: str) -> str:
    """Canary token: `HMAC-SHA256(secret, "canary|{objective_id}")` in deterministic modes."""
    if resolve_derivation_mode(mode) == "deterministic":
        return f"{CANARY_PREFIX}{_hmac_hex(secret, f'canary|{objective_id}')[:24]}"
    return f"{CANARY_PREFIX}{secrets.token_hex(16)}"


def _luhn_valid(number: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(number)):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _generate_phone(fake: Faker) -> str:
    """Return a valid US phone number as `AAA-EEE-LLLL`, drawn only from the seeded instance."""
    for _ in range(200):
        area = fake.random_element(US_AREA_CODES)
        exchange = fake.random_int(min=200, max=999)
        if exchange % 100 == 11:  # N11 exchanges are not assignable
            continue
        line = fake.random_int(min=0, max=9999)
        candidate = f"{area}-{exchange:03d}-{line:04d}"
        try:
            parsed = phonenumbers.parse(candidate, "US")
        except phonenumbers.NumberParseException:
            continue
        if not phonenumbers.is_valid_number(parsed):
            continue
        return candidate
    raise RuntimeError("could not generate a valid US phone number")


_INVALID_SSN_DIGITS = frozenset({"123456789", "078051120"})


def _ssn_valid(ssn: str) -> bool:
    digits = ssn.replace("-", "")
    if len(digits) != 9 or not digits.isdigit():
        return False
    if digits[0] == "9" or digits.startswith(("000", "666", "98765432")):
        return False
    if digits[3:5] == "00" or digits[5:] == "0000":
        return False
    if len(set(digits)) == 1 or digits in _INVALID_SSN_DIGITS:
        return False
    return True


def _generate_pii(fake: Faker, entity_type: str) -> str:
    if entity_type == "EMAIL_ADDRESS":
        return fake.email()
    if entity_type == "PHONE_NUMBER":
        return _generate_phone(fake)
    if entity_type == "CREDIT_CARD":
        for _ in range(100):
            number = fake.credit_card_number(card_type="visa16")
            if _luhn_valid(number):
                return number
        raise RuntimeError("could not generate a Luhn-valid card number")
    if entity_type == "US_SSN":
        for _ in range(100):
            ssn = fake.ssn()
            if _ssn_valid(ssn):
                return ssn
        raise RuntimeError("could not generate a valid US SSN")
    if entity_type == "IBAN_CODE":
        return fake.iban()
    raise ValueError(f"unsupported PII entity type {entity_type!r}")


_LOCAL = threading.local()


def _thread_faker() -> Faker:
    """One isolated Faker per thread (construction is slow); always reseeded before use."""
    fake = getattr(_LOCAL, "faker", None)
    if fake is None:
        fake = Faker("en_US")
        _LOCAL.faker = fake
    return fake


def derive_synthetic_pii(secret: str, objective_id: str, entity_type: str, *, mode: str) -> str:
    """Synthetic PII value for one entity type, from an isolated seeded Faker."""
    kind = resolve_derivation_mode(mode)
    if entity_type not in PII_ENTITY_TYPES:
        raise ValueError(f"entity_type must be one of {PII_ENTITY_TYPES}, got {entity_type!r}")
    if kind == "deterministic":
        seed = int(_hmac_hex(secret, f"pii|{objective_id}|{entity_type}"), 16)
    else:
        seed = secrets.randbits(64)
    fake = _thread_faker()
    fake.seed_instance(seed)
    return _generate_pii(fake, entity_type)


def derive_sanitizer_nonce(secret: str, objective_id: str, doc_id: str, *, mode: str) -> str:
    """Per-document nonce: the full 64-hex HMAC digest (deterministic) or 32 random hex (random)."""
    if resolve_derivation_mode(mode) == "deterministic":
        return _hmac_hex(secret, f"nonce|{objective_id}|{doc_id}")
    return secrets.token_hex(16)


def wrap_external_data(content: str, nonce: str) -> str:
    """Wrap external context in a nonce-tagged delimiter pair (TM-S3)."""
    return f"<EXTERNAL_DATA_{nonce}>\n{content}\n</EXTERNAL_DATA_{nonce}>"


def build_defensive_system_prompt(shared_instructions: str, canary: str, synthetic_pii: Mapping[str, str]) -> str:
    """Shared text first, per-objective secrets last (keeps the shared prefix cacheable)."""
    lines = [shared_instructions.rstrip(), "", f"CANARY: {canary}"]
    lines.extend(f"{entity}: {synthetic_pii[entity]}" for entity in sorted(synthetic_pii))
    return "\n".join(lines)
