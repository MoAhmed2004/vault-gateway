"""Unit tests for the IP-5 verdict and response caches (IP-5, IP-8, IP-9, §2.7).

Keys come only from the frozen builders in `contracts.cache_keys`; these tests
check storage, lookups, hit and miss accounting, the single-writer lifecycle,
the IP-8 pragmas and the fail-fast schema check.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from app.cache import CACHE_SCHEMA_VERSION, VerdictCache
from contracts.cache_keys import egress_verdict_key, layer_verdict_key, raw_response_hash, target_response_key
from contracts.hashing import canonical_hash
from contracts.layer_result import LayerResult

PROMPT_HASH = canonical_hash({"raw_view": "hello"})
BUNDLE_VERSION = "b" * 64
MANIFEST_ID = "c" * 64
MESSAGES = ({"role": "system", "content": "sys"}, {"role": "user", "content": "hello"})
PARAMS = {"temperature": 0.0, "seed": 0, "max_tokens": 256}
RAW = {
    "model": "openai/gpt-oss-20b",
    "choices": [{"message": {"role": "assistant", "content": "Paris.", "reasoning": "step one"}}],
}


def l1_results(verdict: str = "allow") -> tuple[LayerResult, ...]:
    return tuple(
        LayerResult(layer="L1", view=view, per_window_scores=(0.0,), overall_score=0.0, verdict=verdict, latency_ms=0.2)
        for view in ("raw_view", "decoded_view")
    )


def l1_key(layer_version: str = "v1") -> str:
    return layer_verdict_key(
        prompt_hash=PROMPT_HASH, layer="L1", layer_version=layer_version, bundle_version=BUNDLE_VERSION, chroma_version=None
    )


@pytest.fixture
def cache(tmp_path):
    store = VerdictCache(tmp_path / "cache" / "vault_cache.db")
    store.open()
    yield store
    store.close()


def run(coro):
    return asyncio.run(coro)


# Storage and lifecycle (IP-8)


def test_open_creates_the_file_with_wal_normal_busy_timeout_and_schema_version(tmp_path, cache):
    assert (tmp_path / "cache" / "vault_cache.db").exists()
    assert cache.pragmas() == {"journal_mode": "wal", "synchronous": 1, "busy_timeout": 5000, "user_version": CACHE_SCHEMA_VERSION}
    assert cache.is_open


def test_reopen_keeps_rows_and_verifies_schema(tmp_path):
    path = tmp_path / "c.db"
    first = VerdictCache(path)
    first.open()
    first.put_layer(l1_key(), "L1", l1_results())
    first.close()
    second = VerdictCache(path)
    second.open()
    try:
        assert run(second.get_layer(l1_key())) == l1_results()
    finally:
        second.close()


def test_schema_mismatch_fails_fast_and_is_never_migrated(tmp_path):
    path = tmp_path / "c.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE layer_verdicts (cache_key TEXT PRIMARY KEY, payload TEXT)")
    conn.execute(f"PRAGMA user_version = {CACHE_SCHEMA_VERSION}")
    conn.commit()
    conn.close()
    store = VerdictCache(path)
    with pytest.raises(ValueError, match="cache schema column mismatch"):
        store.open()
    assert not store.is_open
    conn = sqlite3.connect(path)
    try:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(layer_verdicts)")]
    finally:
        conn.close()
    assert columns == ["cache_key", "payload"]  # untouched


def test_unversioned_database_with_tables_fails_fast(tmp_path):
    path = tmp_path / "c.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE something_else (x)")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="user_version=0"):
        VerdictCache(path).open()


def test_wrong_schema_version_fails_fast(tmp_path):
    path = tmp_path / "c.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 7")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="user_version=7"):
        VerdictCache(path).open()


def test_use_after_close_and_double_open_are_errors(tmp_path):
    store = VerdictCache(tmp_path / "c.db")
    store.open()
    with pytest.raises(RuntimeError, match="already open"):
        store.open()
    store.close()
    store.close()  # idempotent
    with pytest.raises(RuntimeError, match="not open"):
        run(store.get_layer(l1_key()))


def test_one_thread_owns_every_statement(cache):
    """IP-8: a single writer thread fed by an in-memory queue; reads queue behind earlier writes."""
    import threading

    names: set[str] = set()
    original = cache._submit

    def spy(fn, *args):
        def wrapped(*a):
            names.add(threading.current_thread().name)
            return fn(*a)

        return original(wrapped, *args)

    cache._submit = spy
    cache.put_layer(l1_key(), "L1", l1_results())
    assert run(cache.get_layer(l1_key())) == l1_results()  # sees the write queued just before it
    assert len(names) == 1 and next(iter(names)).startswith("vault-cache")


# Layer verdicts


def test_layer_verdicts_round_trip_and_count_hits_and_misses(cache):
    assert run(cache.get_layer(l1_key())) is None
    cache.put_layer(l1_key(), "L1", l1_results("block"))
    assert run(cache.get_layer(l1_key())) == l1_results("block")
    assert run(cache.get_layer(l1_key("v2"))) is None  # a new layer version is a different key
    assert cache.stats.hits["layer_verdicts"] == 1 and cache.stats.misses["layer_verdicts"] == 2


@pytest.mark.parametrize("verdict", ["timeout", "error"])
def test_timeouts_and_errors_are_never_cached(cache, verdict):
    failed = (LayerResult(layer="L1", view="bundle", per_window_scores=(), overall_score=0.0, verdict=verdict, latency_ms=9.0),)
    cache.put_layer(l1_key(), "L1", failed)
    cache.flush()
    assert run(cache.get_layer(l1_key())) is None


def test_first_write_wins(cache):
    cache.put_layer(l1_key(), "L1", l1_results("allow"))
    cache.put_layer(l1_key(), "L1", l1_results("block"))
    cache.flush()
    assert run(cache.get_layer(l1_key())) == l1_results("allow")


def test_l2_keys_include_the_chroma_version():
    a = layer_verdict_key(prompt_hash=PROMPT_HASH, layer="L2", layer_version="v", bundle_version=BUNDLE_VERSION, chroma_version="c1")
    b = layer_verdict_key(prompt_hash=PROMPT_HASH, layer="L2", layer_version="v", bundle_version=BUNDLE_VERSION, chroma_version="c2")
    assert a != b  # IP-5: a ChromaDB insertion invalidates cached L2 verdicts


# Egress verdicts


def test_egress_rows_are_bound_to_the_l4_version(cache):
    key = egress_verdict_key(response_hash=raw_response_hash(RAW), manifest_id=MANIFEST_ID, variant="X2")
    decision = {"variant": "X2", "verdict": "allow", "action": "none", "content": "Paris."}
    cache.put_egress(key, "l4-v1", decision)
    assert run(cache.get_egress(key, "l4-v1")) == decision
    assert run(cache.get_egress(key, "l4-v2")) is None  # a rule or Presidio change never replays a stale decision
    assert cache.stats.hits["egress_verdicts"] == 1 and cache.stats.misses["egress_verdicts"] == 1


def test_an_l4_version_bump_overwrites_the_stale_egress_row(tmp_path, cache):
    """Step 5A L4: the row written under the old L4 version is replaced, so the key is not a permanent miss."""
    key = egress_verdict_key(response_hash=raw_response_hash(RAW), manifest_id=MANIFEST_ID, variant="X2")
    old = {"variant": "X2", "verdict": "allow", "action": "none", "content": "old"}
    new = {"variant": "X2", "verdict": "allow", "action": "masked", "content": "new"}
    again = {"variant": "X2", "verdict": "allow", "action": "none", "content": "later write, same version"}
    cache.put_egress(key, "l4-v2", old)
    cache.flush()
    cache.put_egress(key, "l4-v3", new)
    cache.flush()
    assert run(cache.get_egress(key, "l4-v3")) == new
    assert run(cache.get_egress(key, "l4-v2")) is None
    cache.put_egress(key, "l4-v3", again)  # same version: the first row stays (deterministic replay)
    cache.flush()
    assert run(cache.get_egress(key, "l4-v3")) == new
    conn = sqlite3.connect(tmp_path / "cache" / "vault_cache.db")
    try:
        rows = conn.execute("SELECT l4_version FROM egress_verdicts WHERE cache_key = ?", (key,)).fetchall()
    finally:
        conn.close()
    assert rows == [("l4-v3",)]


# Target responses


def test_target_responses_store_the_full_raw_json_including_reasoning(cache):
    key = target_response_key(model_id="openai/gpt-oss-20b", parameters=PARAMS, messages=MESSAGES)
    assert run(cache.get_response(key)) is None
    cache.put_response(key, "openai/gpt-oss-20b", RAW)
    assert run(cache.get_response(key)) == RAW
    assert cache.stats.hits["target_responses"] == 1 and cache.stats.misses["target_responses"] == 1


def test_target_key_ignores_configuration_and_seed_but_not_model_or_parameters():
    base = target_response_key(model_id="m", parameters=PARAMS, messages=MESSAGES)
    assert base == target_response_key(model_id="m", parameters=dict(PARAMS), messages=list(MESSAGES))
    assert base != target_response_key(model_id="m2", parameters=PARAMS, messages=MESSAGES)
    assert base != target_response_key(model_id="m", parameters={**PARAMS, "max_tokens": 64}, messages=MESSAGES)


# E4 write bypass (IP-8)


def test_writes_can_be_bypassed(tmp_path):
    store = VerdictCache(tmp_path / "c.db", writes_enabled=False)
    store.open()
    try:
        key = target_response_key(model_id="m", parameters=PARAMS, messages=MESSAGES)
        store.put_response(key, "m", RAW)
        store.put_layer(l1_key(), "L1", l1_results())
        store.flush()
        assert run(store.get_response(key)) is None and run(store.get_layer(l1_key())) is None
        assert sum(store.stats.writes.values()) == 0
    finally:
        store.close()


def test_writes_are_counted_and_flushed_before_close(tmp_path):
    path = tmp_path / "c.db"
    store = VerdictCache(path)
    store.open()
    for i in range(20):
        key = target_response_key(model_id="m", parameters={"seed": i}, messages=MESSAGES)
        store.put_response(key, "m", RAW)
    store.close()
    assert store.stats.writes["target_responses"] == 20 and store.stats.write_errors == 0
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM target_responses").fetchone()[0] == 20
    finally:
        conn.close()
