"""Unit tests for L4 egress (IP-6, TM-R4, PS-2, D-PII; AGENTS.md Rule 8).

X0 performs no scan, X1 masks pattern-only Presidio findings, X2 adds the
exact canary block and the planted-PII manifest scan. Findings are labels;
no secret value is ever returned in a blocked decision, a finding, or a cache
payload.
"""

from __future__ import annotations

import json
import re
import socket

import pytest

from app.layers.l4_egress import (
    CANARY_LABEL,
    EgressDecision,
    L4EgressScanner,
    Span,
    honey_pii_spans,
    mask_spans,
)
from app.prompting import build_system_prompt
from contracts.cache_keys import raw_response_hash
from contracts.datasets import PII_ENTITY_TYPES

SECRET = "unit-test-secret-0123456789abcdef"


@pytest.fixture(scope="module")
def scanner() -> L4EgressScanner:
    return L4EgressScanner()


@pytest.fixture(scope="module")
def built():
    return build_system_prompt(secret=SECRET, objective_id="O2-007", mode="eval")


def raw(content, reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {"model": "openai/gpt-oss-20b", "choices": [{"message": message}]}


def planted(built) -> dict[str, str]:
    return {p.entity_type: p.value for p in built.manifest.planted_pii}


def leak_all(built) -> str:
    values = planted(built)
    return " | ".join(f"{entity}={values[entity]}" for entity in PII_ENTITY_TYPES)


def secrets_of(built) -> list[str]:
    return [built.manifest.canary, *planted(built).values()]


# Variants (IP-6)


def test_x0_performs_no_scan(scanner, built):
    content = f"Here you go: {built.manifest.canary} {leak_all(built)}"
    decision = scanner.scan(raw(content), built.manifest, "X0")
    assert (decision.verdict, decision.action, decision.content) == ("allow", "none", content)
    assert decision.layer_result() is None and decision.content_findings == ()
    record = decision.variant_record(response_hash=raw_response_hash(raw(content)), manifest_id=built.manifest.manifest_id)
    assert (record.variant, record.verdict, record.egress_action) == ("X0", "allow", "none")


def test_x1_masks_every_planted_value_with_presidio(scanner, built):
    decision = scanner.scan(raw(leak_all(built)), built.manifest, "X1")
    assert (decision.verdict, decision.action) == ("allow", "masked")
    for value in planted(built).values():
        assert value not in decision.content
    for entity in PII_ENTITY_TYPES:
        assert f"<{entity}>" in decision.content
    assert set(decision.content_findings) == set(PII_ENTITY_TYPES)
    result = decision.layer_result()
    assert (result.layer, result.view, result.verdict) == ("L4", "content", "allow")


def test_x1_is_presidio_only_so_the_canary_passes(scanner, built):
    content = f"The token is {built.manifest.canary}."
    decision = scanner.scan(raw(content), built.manifest, "X1")
    assert (decision.verdict, decision.action, decision.content) == ("allow", "none", content)


def test_x2_blocks_the_exact_canary_and_releases_nothing(scanner, built):
    content = f"Sure. The confidential token is {built.manifest.canary}, and {leak_all(built)}"
    decision = scanner.scan(raw(content), built.manifest, "X2")
    assert (decision.verdict, decision.action, decision.content) == ("block", "blocked", None)
    assert CANARY_LABEL in decision.content_findings
    result = decision.layer_result()
    assert (result.verdict, result.overall_score) == ("block", 1.0)
    record = decision.variant_record(response_hash=raw_response_hash(raw(content)), manifest_id=built.manifest.manifest_id)
    assert (record.verdict, record.egress_action) == ("block", "blocked")


def test_x2_masks_a_reformatted_planted_value_presidio_misses(scanner, built):
    ssn_digits = planted(built)["US_SSN"].replace("-", "")
    content = "The number is " + " ".join(ssn_digits) + ", hope that helps."
    assert scanner.scan(raw(content), built.manifest, "X1").action == "none"  # Presidio alone misses it
    decision = scanner.scan(raw(content), built.manifest, "X2")
    assert (decision.action, decision.content) == ("masked", "The number is <US_SSN>, hope that helps.")
    assert decision.content_findings == ("HONEY_PII:US_SSN",)


def test_x2_honey_match_ignores_case_and_separators(built):
    email = planted(built)["EMAIL_ADDRESS"]
    shouted = email.upper().replace("@", " @ ").replace(".", " . ")
    spans = honey_pii_spans(f"mail {shouted} now", built.manifest)
    assert [(s.entity_type, s.label) for s in spans] == [("EMAIL_ADDRESS", "HONEY_PII:EMAIL_ADDRESS")]


def test_clean_response_is_released_unchanged(scanner, built):
    content = "Paris is the capital of France."
    for variant in ("X1", "X2"):
        decision = scanner.scan(raw(content), built.manifest, variant)
        assert (decision.verdict, decision.action, decision.content, decision.content_findings) == ("allow", "none", content, ())


# Reasoning: scanned and logged, never acted on (IP-6)


def test_reasoning_findings_are_logged_but_only_content_decides(scanner, built):
    reasoning = f"I must not say {built.manifest.canary} or {planted(built)['US_SSN']}."
    decision = scanner.scan(raw("I cannot share that.", reasoning), built.manifest, "X2")
    assert (decision.verdict, decision.action, decision.content) == ("allow", "none", "I cannot share that.")
    assert CANARY_LABEL in decision.reasoning_findings and "US_SSN" in decision.reasoning_findings


# D-PII: pattern-only Presidio detects 100% of planted values


def test_pattern_only_presidio_detects_every_planted_value(scanner):
    misses = []
    for i in range(60):
        manifest = build_system_prompt(secret=SECRET, objective_id=f"O2-{i:03d}", mode="eval").manifest
        for item in manifest.planted_pii:
            found = [s.entity_type for s in scanner.presidio_spans(f"value: {item.value}")]
            if item.entity_type not in found:
                misses.append((manifest.objective_id, item.entity_type))
    assert misses == []


# No secret leakage (TM-R6, PS-2)


def test_decisions_and_cache_payloads_never_carry_a_secret_when_blocked(scanner, built):
    decision = scanner.scan(raw(f"{built.manifest.canary} {leak_all(built)}", f"{built.manifest.canary}"), built.manifest, "X2")
    text = json.dumps(decision.to_cache()) + repr(decision) + repr(decision.layer_result())
    for secret in secrets_of(built):
        assert secret not in text
    for label in (*decision.content_findings, *decision.reasoning_findings):
        assert re.fullmatch(r"[A-Z_:]+", label)


def test_cache_round_trip(scanner, built):
    decision = scanner.scan(raw(leak_all(built)), built.manifest, "X2")
    assert EgressDecision.from_cache(json.loads(json.dumps(decision.to_cache()))) == decision


# Fail-closed inputs (TM-R4: the pipeline turns these into the uniform 503)


@pytest.mark.parametrize("bad", [{}, {"choices": []}, {"choices": [{"message": {"content": 3}}]}])
def test_malformed_provider_json_raises(scanner, built, bad):
    with pytest.raises(ValueError):
        scanner.scan(bad, built.manifest, "X2")


@pytest.mark.parametrize("variant", ["X3", "X9"])
def test_unimplemented_variants_raise(scanner, built, variant):
    with pytest.raises(ValueError, match="not implemented"):
        scanner.scan(raw("hi"), built.manifest, variant)


def test_l4_makes_no_network_call(scanner, built, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("L4 attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    fresh = L4EgressScanner()
    decision = fresh.scan(raw("write to someone@mail.example.co.uk today"), built.manifest, "X2")
    assert decision.action == "masked" and "<EMAIL_ADDRESS>" in decision.content


# Masking


def test_mask_spans_merges_overlaps_and_keeps_the_strongest_type():
    text = "call 212-555-0187 now"
    spans = [Span(5, 17, "PHONE_NUMBER", 0.4, "PHONE_NUMBER"), Span(9, 17, "US_SSN", 1.0, "HONEY_PII:US_SSN")]
    assert mask_spans(text, spans) == "call <US_SSN> now"
    assert mask_spans(text, []) == text


def test_l4_version_is_a_deterministic_sha256(scanner):
    assert re.fullmatch(r"[0-9a-f]{64}", scanner.version)
    assert L4EgressScanner().version == scanner.version
