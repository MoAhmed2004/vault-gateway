"""Unit tests for the IP-5 cache key formats (IP-5, PS-3, PS-4, §2.7)."""

from __future__ import annotations

import pytest

from app.prompting import build_messages, build_system_prompt, build_user_content
from contracts.cache_keys import (
    CACHE_KEY_SCHEMA,
    egress_verdict_key,
    layer_verdict_key,
    raw_response_hash,
    target_response_key,
)
from contracts.hashing import canonical_hash

SECRET = "unit-test-hmac-secret-0123456789"
P_HASH = canonical_hash({"raw_view": "hello"})
R_HASH = canonical_hash({"choices": [{"message": {"content": "hi"}}]})
MANIFEST = "e" * 64
PARAMS = {"temperature": 0, "max_tokens": 64, "seed": 7}
MESSAGES = ({"role": "system", "content": "sys"}, {"role": "user", "content": "hi"})


def _layer(**overrides):
    kwargs = {
        "prompt_hash": P_HASH, "layer": "L2", "layer_version": "l2-minilm-v1",
        "bundle_version": "b" * 64, "chroma_version": "chroma-v3",
    }
    kwargs.update(overrides)
    return layer_verdict_key(**kwargs)


def _egress(**overrides):
    kwargs = {"response_hash": R_HASH, "manifest_id": MANIFEST, "variant": "X2"}
    kwargs.update(overrides)
    return egress_verdict_key(**kwargs)


def _target(**overrides):
    kwargs = {"model_id": "openai/gpt-oss-20b", "parameters": PARAMS, "messages": MESSAGES}
    kwargs.update(overrides)
    return target_response_key(**kwargs)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


# Format and determinism


def test_keys_are_sha256_and_deterministic():
    for build in (_layer, _egress, _target):
        key = build()
        assert _is_sha256(key)
        assert build() == key


def test_layer_key_matches_documented_canonical_payload():
    expected = canonical_hash(
        {
            "cache": "layer_verdict", "schema": CACHE_KEY_SCHEMA, "prompt_hash": P_HASH, "layer": "L2",
            "layer_version": "l2-minilm-v1", "bundle_version": "b" * 64, "chroma_version": "chroma-v3",
        }
    )
    assert _layer() == expected


def test_egress_and_target_keys_match_documented_canonical_payload():
    assert _egress() == canonical_hash(
        {"cache": "egress_verdict", "schema": CACHE_KEY_SCHEMA, "response_hash": R_HASH, "manifest_id": MANIFEST, "variant": "X2"}
    )
    assert _target() == canonical_hash(
        {
            "cache": "target_response", "schema": CACHE_KEY_SCHEMA, "model_id": "openai/gpt-oss-20b",
            "parameters": PARAMS, "messages": list(MESSAGES),
        }
    )


def test_caches_never_collide_across_kinds():
    assert len({_layer(), _egress(), _target()}) == 3


def test_parameter_order_and_integral_floats_do_not_change_the_key():
    reordered = {"seed": 7, "max_tokens": 64, "temperature": 0.0}
    assert _target(parameters=reordered) == _target()


def test_raw_response_hash_covers_reasoning():
    with_reasoning = {"choices": [{"message": {"content": "hi", "reasoning": "private"}}]}
    without = {"choices": [{"message": {"content": "hi"}}]}
    assert raw_response_hash(with_reasoning) != raw_response_hash(without)
    assert raw_response_hash(without) == canonical_hash(without)
    with pytest.raises(TypeError):
        raw_response_hash("not a mapping")  # type: ignore[arg-type]


# Sensitivity to every keyed field


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt_hash": canonical_hash({"raw_view": "other"})},
        {"layer_version": "l2-minilm-v2"},
        {"bundle_version": "c" * 64},
        {"chroma_version": "chroma-v4"},  # a ChromaDB insertion invalidates cached L2 verdicts
    ],
)
def test_layer_key_changes_with_every_field(overrides):
    assert _layer(**overrides) != _layer()


def test_layer_key_includes_the_layer_name():
    l1 = layer_verdict_key(prompt_hash=P_HASH, layer="L1", layer_version="v1", bundle_version="b", chroma_version=None)
    l3 = layer_verdict_key(prompt_hash=P_HASH, layer="L3", layer_version="v1", bundle_version="b", chroma_version=None)
    assert l1 != l3


def test_chroma_version_is_keyed_for_l2_only():
    for layer in ("L1", "L3"):
        key = layer_verdict_key(prompt_hash=P_HASH, layer=layer, layer_version="v", bundle_version="b", chroma_version=None)
        assert _is_sha256(key)
        with pytest.raises(ValueError, match="only to L2"):
            layer_verdict_key(prompt_hash=P_HASH, layer=layer, layer_version="v", bundle_version="b", chroma_version="c1")
    with pytest.raises(ValueError, match="chroma_version"):
        _layer(chroma_version=None)


@pytest.mark.parametrize(
    "overrides",
    [{"response_hash": canonical_hash({"other": 1})}, {"manifest_id": "f" * 64}, {"variant": "X3"}],
)
def test_egress_key_changes_with_every_field(overrides):
    assert _egress(**overrides) != _egress()


def test_every_variant_has_a_distinct_egress_key():
    assert len({_egress(variant=v) for v in ("X0", "X1", "X2", "X3")}) == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_id": "openai/gpt-oss-120b"},
        {"parameters": {**PARAMS, "temperature": 0.7}},
        {"parameters": {**PARAMS, "max_tokens": 256}},
        {"parameters": {k: v for k, v in PARAMS.items() if k != "seed"}},
        {"messages": ({"role": "system", "content": "sys2"}, MESSAGES[1])},
        {"messages": (MESSAGES[0], {"role": "user", "content": "hi!"})},
        {"messages": (*MESSAGES, {"role": "assistant", "content": "x"})},
    ],
)
def test_target_key_changes_with_every_field(overrides):
    assert _target(**overrides) != _target()


def test_message_order_is_significant():
    msgs = (MESSAGES[0], {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"})
    swapped = (MESSAGES[0], {"role": "assistant", "content": "b"}, {"role": "user", "content": "a"})
    assert _target(messages=msgs) != _target(messages=swapped)


# Validation


@pytest.mark.parametrize("bad", ["", "ABC", "A" * 64, "g" * 64, "a" * 63])
def test_hash_inputs_must_be_lowercase_sha256(bad):
    with pytest.raises(ValueError):
        _layer(prompt_hash=bad)
    with pytest.raises(ValueError):
        _egress(response_hash=bad)


def test_layer_key_rejects_unknown_layer_and_empty_versions():
    for layer in ("L4", "l1", "L0", ""):
        with pytest.raises(ValueError):
            _layer(layer=layer)
    for field in ("layer_version", "bundle_version"):
        with pytest.raises(ValueError):
            _layer(**{field: "  "})


def test_egress_key_rejects_bad_variant_and_empty_manifest():
    with pytest.raises(ValueError):
        _egress(variant="X4")
    with pytest.raises(ValueError):
        _egress(manifest_id="")


@pytest.mark.parametrize(
    "messages",
    [
        (),
        (MESSAGES[0],),
        ({"role": "user", "content": "hi"}, {"role": "user", "content": "x"}),  # no system prompt first
        (MESSAGES[0], {"role": "system", "content": "second system"}),
        (MESSAGES[0], {"role": "tool", "content": "x"}),
        (MESSAGES[0], {"role": "user", "content": 3}),
        (MESSAGES[0], {"role": "user", "content": "x", "name": "n"}),
        "system user",
    ],
)
def test_target_key_rejects_malformed_message_lists(messages):
    with pytest.raises(ValueError):
        _target(messages=messages)


def test_target_key_rejects_reserved_parameters_and_empty_model():
    with pytest.raises(ValueError, match="keyed separately"):
        _target(parameters={**PARAMS, "model": "x"})
    with pytest.raises(ValueError, match="keyed separately"):
        _target(parameters={**PARAMS, "messages": []})
    with pytest.raises(ValueError):
        _target(model_id="")
    with pytest.raises(TypeError):
        _target(parameters=[("temperature", 0)])


# PS-3 / PS-4: cross-configuration and cross-seed identity of the target-response key


def _key_for(mode: str, objective_id: str = "O3-0001", user_prompt: str = "payload") -> str:
    built = build_system_prompt(secret=SECRET, objective_id=objective_id, mode=mode)
    messages = build_messages(built.system_prompt, build_user_content(user_prompt))
    return target_response_key(model_id="openai/gpt-oss-20b", parameters=PARAMS, messages=messages)


def test_target_key_is_identical_across_deterministic_modes_and_repeated_builds():
    keys = {_key_for(mode) for mode in ("eval", "shadow_ingress_only", "shadow_full_e2e")}
    keys.add(_key_for("eval"))
    assert len(keys) == 1


def test_target_key_has_no_configuration_or_seed_dimension():
    import inspect

    params = set(inspect.signature(target_response_key).parameters)
    assert params == {"model_id", "parameters", "messages"}
    builder_params = set(inspect.signature(build_system_prompt).parameters)
    assert not builder_params & {"config", "configuration", "seed", "tier"}


def test_target_key_differs_per_objective_and_prompt():
    assert _key_for("eval", objective_id="O3-0002") != _key_for("eval")
    assert _key_for("eval", user_prompt="other") != _key_for("eval")


def test_random_modes_never_repeat_the_target_key():
    assert _key_for("production") != _key_for("production")
    assert _key_for("demo") != _key_for("demo")
