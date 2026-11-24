"""Unit tests for telemetry/writer.py (IP-7, IP-8, IP-4, RT-V6)."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from contracts import (
    EgressVariantRecord,
    LayerResult,
    Outcome,
    RequestLogRecord,
    ShadowRecord,
    TelemetrySink,
    canonical_hash,
)
from telemetry.writer import SCHEMA_VERSION, SQLiteTelemetrySink


def _layer(layer="L1", verdict="allow"):
    return LayerResult(
        layer=layer, view="raw_view", per_window_scores=[0.1, 0.2], overall_score=0.2, verdict=verdict, latency_ms=1.5
    )


def _request(i=0, **overrides):
    base = {
        "timestamp_utc": "2026-11-23T18:00:00Z",
        "gateway_mode": "production",
        "prompt_hash": canonical_hash({"p": i}),
        "layer_results": (_layer("L1"), _layer("L2")),
        "target_called": True,
        "outcome": Outcome.SUCCESS,
        "total_latency_ms": 12.0,
    }
    base.update(overrides)
    return RequestLogRecord(**base)


def _variants(names=("X0", "X1", "X2", "X3")):
    return tuple(
        EgressVariantRecord(
            response_hash=canonical_hash({"r": 1}),
            manifest_id="M1",
            variant=n,
            verdict="block" if n == "X3" else "allow",
            egress_action="blocked" if n == "X3" else ("masked" if n == "X2" else "none"),
            oracle_result=(n not in ("X2", "X3")),
        )
        for n in names
    )


def _shadow(tier="shadow_full_e2e", **overrides):
    base = {
        "timestamp_utc": "2026-11-23T18:00:00Z",
        "shadow_tier": tier,
        "prompt_hash": canonical_hash({"s": 1}),
        "layer_results": (_layer("L1"), _layer("L2"), _layer("L3", "block")),
        "total_latency_ms": 99.5,
        "target_called": tier == "shadow_full_e2e",
        "cache_hit": False,
        "egress_variants": _variants() if tier == "shadow_full_e2e" else (),
        "layer_version": "l1",
        "bundle_version": "b1",
        "chroma_version": "c1",
    }
    base.update(overrides)
    return ShadowRecord(**base)


@pytest.fixture
def sink(tmp_path):
    s = SQLiteTelemetrySink(tmp_path / "data" / "logs" / "vault.db")
    yield s
    s.close()


def test_satisfies_protocol(sink):
    assert isinstance(sink, TelemetrySink)
    assert sink.healthy is True and sink.is_alive is True
    assert sink.writer_error is None and sink.writer_deaths == 0
    assert sink.schema_version == SCHEMA_VERSION == 1
    assert sink.unbounded is False


def test_creates_parent_directory(tmp_path):
    target = tmp_path / "a" / "b" / "c" / "v.db"
    with SQLiteTelemetrySink(target):
        assert target.parent.is_dir()
    assert target.exists()


def test_pragmas_and_schema_version(sink):
    assert sink.pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "busy_timeout": 5000,
        "user_version": SCHEMA_VERSION,
    }
    conn = sqlite3.connect(sink.db_path)
    try:
        assert conn.execute("PRAGMA journal_mode;").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION
    finally:
        conn.close()


def test_reopening_matching_schema_version_succeeds(tmp_path):
    db = tmp_path / "vault.db"
    with SQLiteTelemetrySink(db) as s1:
        s1.record_request(_request(1))
    with SQLiteTelemetrySink(db) as s2:
        s2.record_request(_request(2))
        s2.flush()
        assert len(s2.fetch_requests()) == 2


def test_schema_version_mismatch_fails_fast(tmp_path):
    db = tmp_path / "wrong_version.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA user_version = 99;")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(RuntimeError, match="schema version mismatch"):
        SQLiteTelemetrySink(db)


def test_unversioned_existing_database_fails_fast(tmp_path):
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE request_logs (id INTEGER PRIMARY KEY);")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(RuntimeError, match="schema version mismatch"):
        SQLiteTelemetrySink(db)


def test_tables_and_columns_exist(sink):
    conn = sqlite3.connect(sink.db_path)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"request_logs", "shadow_records"} <= tables
        cols = {r[1] for r in conn.execute("PRAGMA table_info(request_logs)")}
        assert set(RequestLogRecord.model_fields) <= cols
        cols = {r[1] for r in conn.execute("PRAGMA table_info(shadow_records)")}
        assert set(ShadowRecord.model_fields) <= cols
    finally:
        conn.close()


@pytest.mark.parametrize(
    "subdir",
    ["OneDrive", "Box", "Box Sync", "Mobile Documents"],
)
def test_rejects_synced_folder_db_path(tmp_path, subdir):
    with pytest.raises(ValueError, match="synced"):
        SQLiteTelemetrySink(tmp_path / subdir / "v.db")
    assert not (tmp_path / subdir).exists()


def test_accepts_sandbox_path(tmp_path):
    with SQLiteTelemetrySink(tmp_path / "sandbox" / "v.db") as s:
        assert s.healthy


def test_request_roundtrip(sink):
    rec = _request(
        1, gateway_mode="eval", trial_id="t1", objective_id="O3-01", tier="T2", config="C2", seed=7,
        query_index=3, operator_chain=("base64", "role"), oracle_result=True, judge_score=0.75,
        reasoning_findings=("EMAIL_ADDRESS",), timeout_policy="P-degrade", cache_hit=False, model_id="openai/gpt-oss-20b",
    )
    sink.record_request(rec)
    sink.flush()
    (row,) = sink.fetch_requests()
    assert row["prompt_hash"] == rec.prompt_hash
    assert row["outcome"] == "Success"
    assert row["target_called"] == 1 and row["oracle_result"] == 1 and row["cache_hit"] == 0
    assert row["operator_chain"] == ["base64", "role"]
    assert row["reasoning_findings"] == ["EMAIL_ADDRESS"]
    assert row["layer_results"][0]["layer"] == "L1" and len(row["layer_results"]) == 2
    assert row["judge_score"] == 0.75 and row["seed"] == 7 and row["model_id"] == "openai/gpt-oss-20b"
    assert row["timeout_policy"] == "P-degrade"


def test_request_with_no_eval_fields_stores_nulls(sink):
    sink.record_request(_request())
    sink.flush()
    (row,) = sink.fetch_requests()
    assert row["trial_id"] is None and row["operator_chain"] is None and row["blocked_by"] is None
    assert row["reasoning_findings"] == []


def test_pre_layer_blocked_by_roundtrip(sink):
    rec = _request(
        2, blocked_by="body_cap", outcome=Outcome.BLOCKED_INGRESS, layer_results=(), target_called=False
    )
    sink.record_request(rec)
    sink.flush()
    (row,) = sink.fetch_requests()
    assert row["blocked_by"] == "body_cap" and row["layer_results"] == [] and row["outcome"] == "Blocked-ingress"


def test_egress_saved_roundtrip(sink):
    rec = _request(
        3, outcome=Outcome.EGRESS_SAVED, egress_variant="X2", egress_action="blocked", blocked_by="L4",
        layer_results=(_layer("L1"), _layer("L4", "block")), reasoning_findings=("US_SSN",),
    )
    sink.record_request(rec)
    sink.flush()
    (row,) = sink.fetch_requests()
    assert row["outcome"] == "Egress-saved" and row["egress_variant"] == "X2" and row["blocked_by"] == "L4"
    assert row["egress_action"] == "blocked"
    assert row["layer_results"][1]["verdict"] == "block"


def test_egress_saved_masked_roundtrip(sink):
    rec = _request(
        4,
        outcome=Outcome.EGRESS_SAVED,
        egress_variant="X1",
        egress_action="masked",
        blocked_by=None,
        layer_results=(_layer("L1"), _layer("L4", "allow")),
    )
    sink.record_request(rec)
    sink.flush()
    (row,) = sink.fetch_requests()
    assert row["outcome"] == "Egress-saved"
    assert row["egress_variant"] == "X1"
    assert row["egress_action"] == "masked"
    assert row["blocked_by"] is None


def test_shadow_ingress_only_roundtrip(sink):
    sink.record_shadow(_shadow("shadow_ingress_only", trial_id="t9"))
    sink.flush()
    (row,) = sink.fetch_shadows()
    assert row["shadow_tier"] == "shadow_ingress_only" and row["egress_variants"] == []
    assert row["target_called"] == 0 and row["trial_id"] == "t9"
    assert (row["layer_version"], row["bundle_version"], row["chroma_version"]) == ("l1", "b1", "c1")
    assert len(row["layer_results"]) == 3


def test_shadow_full_e2e_roundtrip_with_four_variants(sink):
    sink.record_shadow(_shadow())
    sink.flush()
    (row,) = sink.fetch_shadows()
    assert row["shadow_tier"] == "shadow_full_e2e" and row["target_called"] == 1 and row["cache_hit"] == 0
    assert [v["variant"] for v in row["egress_variants"]] == ["X0", "X1", "X2", "X3"]
    assert [v["egress_action"] for v in row["egress_variants"]] == ["none", "none", "masked", "blocked"]
    assert all(v["manifest_id"] == "M1" and len(v["response_hash"]) == 64 for v in row["egress_variants"])
    assert row["egress_variants"][3]["verdict"] == "block"


def test_type_validation(sink):
    with pytest.raises(TypeError):
        sink.record_request({"x": 1})
    with pytest.raises(TypeError):
        sink.record_request(_shadow())
    with pytest.raises(TypeError):
        sink.record_shadow(_request())


def test_concurrent_producers_single_writer(tmp_path):
    threads_n, per = 8, 150
    with SQLiteTelemetrySink(tmp_path / "v.db") as s:
        errors: list[BaseException] = []

        def produce(k):
            try:
                for j in range(per):
                    s.record_request(_request(k * 1000 + j))
                    if j % 10 == 0:
                        s.record_shadow(_shadow(trial_id=f"t{k}-{j}"))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        ts = [threading.Thread(target=produce, args=(k,)) for k in range(threads_n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        s.flush(timeout=20)
        assert not errors
        assert s.dropped_count == 0 and s.failed_count == 0
        assert len(s.fetch_requests()) == threads_n * per
        assert len(s.fetch_shadows()) == threads_n * (per // 10)
        assert len({r["prompt_hash"] for r in s.fetch_requests()}) == threads_n * per


def test_readers_while_writing_do_not_lock(tmp_path):
    with SQLiteTelemetrySink(tmp_path / "v.db") as s:
        for i in range(200):
            s.record_request(_request(i))
            if i % 40 == 0:
                s.fetch_requests()
        s.flush()
        assert len(s.fetch_requests()) == 200


def test_flush_makes_rows_visible_and_close_drains(tmp_path):
    s = SQLiteTelemetrySink(tmp_path / "v.db")
    for i in range(300):
        s.record_request(_request(i))
    s.close()
    conn = sqlite3.connect(tmp_path / "v.db")
    try:
        assert conn.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0] == 300
    finally:
        conn.close()
    assert not s.is_alive
    assert not s.healthy


def test_close_is_idempotent_and_context_manager(tmp_path):
    with SQLiteTelemetrySink(tmp_path / "v.db") as s:
        s.record_request(_request())
    s.close()
    s.flush()
    assert not s.is_alive
    assert len(s.fetch_requests()) == 1


def test_record_after_close_counts_as_dropped(tmp_path):
    s = SQLiteTelemetrySink(tmp_path / "v.db")
    s.close()
    s.record_request(_request())
    assert s.dropped_count == 1


def test_full_queue_drops_without_blocking(tmp_path):
    s = SQLiteTelemetrySink(tmp_path / "v.db", queue_maxsize=1)
    gate = threading.Event()
    original = s._write_item

    def slow(conn, item):
        gate.wait(5)
        original(conn, item)

    s._write_item = slow
    try:
        for i in range(50):
            s.record_request(_request(i))
        assert s.dropped_count > 0
    finally:
        gate.set()
        s.close()


@pytest.mark.parametrize("kwargs", [{"unbounded": True}, {"queue_maxsize": 0}])
def test_unbounded_measured_run_mode_never_drops(tmp_path, kwargs):
    s = SQLiteTelemetrySink(tmp_path / "measured.db", **kwargs)
    assert s.unbounded is True
    gate = threading.Event()
    entered = threading.Event()
    original = s._write_item

    def slow(conn, item):
        entered.set()
        gate.wait(5)
        original(conn, item)

    s._write_item = slow
    try:
        s.record_request(_request(0))
        entered.wait(2)
        for i in range(1, 300):
            s.record_request(_request(i))
        assert s.dropped_count == 0
    finally:
        gate.set()
    s.flush()
    assert s.dropped_count == 0 and s.failed_count == 0
    assert len(s.fetch_requests()) == 300
    s.close()


def test_negative_queue_maxsize_rejected(tmp_path):
    with pytest.raises(ValueError, match="queue_maxsize"):
        SQLiteTelemetrySink(tmp_path / "v.db", queue_maxsize=-1)


def test_writer_death_recorded_and_flush_raises(tmp_path):
    s = SQLiteTelemetrySink(tmp_path / "die.db")
    boom = sqlite3.OperationalError("simulated disk I/O error on commit")

    def failing_commit(conn):
        raise boom

    s._commit_tx = failing_commit
    s.record_request(_request(1))
    with pytest.raises(RuntimeError, match="writer thread died") as exc_info:
        s.flush(timeout=5.0)
    assert exc_info.value.__cause__ is boom
    s._thread.join(timeout=2.0)
    assert s.writer_error is boom
    assert s.writer_deaths == 1
    assert s.is_alive is False
    assert s.healthy is False
    # Subsequent flush also surfaces the death rather than returning silently.
    with pytest.raises(RuntimeError, match="writer thread died") as exc_info2:
        s.flush(timeout=5.0)
    assert exc_info2.value.__cause__ is boom
    # Producer calls after death drop cleanly without raising.
    s.record_request(_request(2))
    assert s.dropped_count == 1
    s.close()


def test_close_timeout_does_not_mask_live_thread_and_allows_retry(tmp_path):
    s = SQLiteTelemetrySink(tmp_path / "retry.db")
    gate = threading.Event()
    entered = threading.Event()
    original = s._write_item

    def slow(conn, item):
        entered.set()
        gate.wait(5)
        original(conn, item)

    s._write_item = slow
    try:
        s.record_request(_request(1))
        s.record_request(_request(2))
        assert entered.wait(2)
        with pytest.raises(TimeoutError, match="did not stop in time"):
            s.close(timeout=0.05)
        assert s.is_alive is True
        assert s._closed is False
    finally:
        gate.set()
    # Retrying close() now waits for drain and shuts down cleanly.
    s.close(timeout=5.0)
    assert s.is_alive is False
    assert s._closed is True
    assert len(s.fetch_requests()) == 2


def test_close_timeout_on_full_queue_allows_retry(tmp_path):
    s = SQLiteTelemetrySink(tmp_path / "retry_full.db", queue_maxsize=1)
    gate = threading.Event()
    entered = threading.Event()
    original = s._write_item

    def slow(conn, item):
        entered.set()
        gate.wait(5)
        original(conn, item)

    s._write_item = slow
    try:
        s.record_request(_request(1))
        assert entered.wait(2)
        s.record_request(_request(2))  # fills queue_maxsize=1
        with pytest.raises(TimeoutError, match="queue full"):
            s.close(timeout=0.05)
        assert s.is_alive is True
        assert s._closed is False
    finally:
        gate.set()
    s.close(timeout=5.0)
    assert s.is_alive is False
    assert s._closed is True
    assert len(s.fetch_requests()) == 2


def test_writer_connection_only_used_in_writer_thread(sink):
    sink.record_request(_request())
    sink.flush()
    assert sink._thread.name == "telemetry-writer"
    assert sink._thread is not threading.current_thread()


def test_lone_surrogate_text_does_not_kill_writer(sink):
    sink.record_request(_request(1, trial_id="bad\ud800id"))
    sink.record_request(_request(2))
    sink.flush()
    assert sink.failed_count == 0
    assert len(sink.fetch_requests()) == 2


def test_bad_row_counted_and_writer_survives(sink):
    original = sink._write_item
    calls = {"n": 0}

    def flaky(conn, item):
        calls["n"] += 1
        if calls["n"] == 1:
            with sink._lock:
                sink._failed += 1
            return
        original(conn, item)

    sink._write_item = flaky
    sink.record_request(_request(1))
    sink.record_request(_request(2))
    sink.flush()
    assert sink.failed_count == 1
    assert len(sink.fetch_requests()) == 1


# Step 3.6: Commit atomicity, lost_count accounting, close() race closure, and atomic init


def test_failed_commit_leaves_written_count_zero_and_increments_failed_count(tmp_path):
    s = SQLiteTelemetrySink(tmp_path / "commit_fail.db")
    gate = threading.Event()
    entered_commit = threading.Event()
    boom = sqlite3.OperationalError("simulated disk full on COMMIT")

    def failing_commit(conn):
        entered_commit.set()
        gate.wait(5.0)
        raise boom

    s._commit_tx = failing_commit
    # Pause at COMMIT on first item, then fail; separately test a multi-item batch failing COMMIT.
    s.record_request(_request(1))
    assert entered_commit.wait(2.0)
    gate.set()
    with pytest.raises(RuntimeError, match="writer thread died"):
        s.flush(timeout=5.0)
    s._thread.join(timeout=2.0)
    assert s.written_count == 0
    assert s.failed_count == 1
    assert s.lost_count == 0
    assert s.dropped_count == 0
    assert s.produced_count == 1 == s.written_count + s.dropped_count + s.failed_count + s.lost_count
    assert s.fetch_requests() == []
    s.close()

    # Also prove multi-record batch COMMIT failure leaves written_count == 0 after INSERTs ran.
    s2 = SQLiteTelemetrySink(tmp_path / "batch_commit_fail.db")
    pause = threading.Event()
    first_inserted = threading.Event()
    orig_write = s2._write_item

    def slow_first(conn, item):
        orig_write(conn, item)
        if not first_inserted.is_set():
            first_inserted.set()
            pause.wait(5.0)

    def fail_on_second_commit(conn):
        raise boom

    s2._write_item = slow_first
    s2.record_request(_request(10))
    assert first_inserted.wait(2.0)
    # Queue 4 more records so batch 2 has 4 staged INSERTs before COMMIT fails.
    for k in range(11, 15):
        s2.record_request(_request(k))
    s2._write_item = orig_write
    # Let batch 1 commit normally, then fail batch 2's COMMIT.
    orig_commit = s2._commit_tx
    commit_calls = {"n": 0}

    def fail_second_tx(conn):
        commit_calls["n"] += 1
        if commit_calls["n"] == 1:
            orig_commit(conn)
            return
        fail_on_second_commit(conn)

    s2._commit_tx = fail_second_tx
    pause.set()
    s2._thread.join(timeout=5.0)
    assert s2.written_count == 1
    assert s2.failed_count == 4
    assert s2.lost_count == 0
    assert s2.dropped_count == 0
    assert s2.produced_count == 5 == s2.written_count + s2.dropped_count + s2.failed_count + s2.lost_count
    assert len(s2.fetch_requests()) == 1
    s2.close()


def test_writer_death_accounts_for_all_queued_items_in_lost_count(tmp_path):
    s = SQLiteTelemetrySink(tmp_path / "lost_accounting.db", unbounded=True)
    # First commit 3 records cleanly so written_count > 0.
    for i in range(3):
        s.record_request(_request(i))
    s.flush()
    assert s.written_count == 3

    in_batch = threading.Event()
    release_commit = threading.Event()
    boom = sqlite3.OperationalError("simulated fatal writer error")

    def blocked_failing_commit(conn):
        in_batch.set()
        release_commit.wait(5.0)
        raise boom

    s._commit_tx = blocked_failing_commit
    # Trigger dying batch (1 record in flight).
    s.record_request(_request(100))
    assert in_batch.wait(2.0)

    # Enqueue 17 additional records (requests + shadows) that remain queued in memory.
    for i in range(15):
        s.record_request(_request(200 + i))
    s.record_shadow(_shadow(trial_id="abandoned-1"))
    s.record_shadow(_shadow(trial_id="abandoned-2"))

    release_commit.set()
    s._thread.join(timeout=5.0)
    assert not s.is_alive

    # Submit 4 more records after writer death (counted as dropped).
    for i in range(4):
        s.record_request(_request(400 + i))

    produced = 3 + 1 + 17 + 4
    assert s.produced_count == produced
    assert s.written_count == 3
    assert s.failed_count == 1
    assert s.lost_count == 17
    assert s.dropped_count == 4
    assert produced == s.written_count + s.dropped_count + s.failed_count + s.lost_count
    assert len(s.fetch_requests()) == 3
    assert len(s.fetch_shadows()) == 0
    s.close()


def test_close_immediately_drops_concurrent_submissions_with_zero_silent_loss(tmp_path):
    # Deterministic proof that once close() begins (_stop_enqueued under lock),
    # concurrent submissions while the writer is still draining are counted as dropped.
    s = SQLiteTelemetrySink(tmp_path / "close_gate.db")
    entered = threading.Event()
    gate = threading.Event()
    orig_write = s._write_item

    def slow_write(conn, item):
        entered.set()
        gate.wait(5.0)
        orig_write(conn, item)

    s._write_item = slow_write
    s.record_request(_request(1))
    assert entered.wait(2.0)

    closing_started = threading.Event()

    def run_close():
        # close() sets _stop_enqueued under lock and blocks on join until gate opens.
        closing_started.set()
        s.close(timeout=5.0)

    closer = threading.Thread(target=run_close)
    closer.start()
    assert closing_started.wait(2.0)
    # Wait briefly until _stop_enqueued is observable under lock.
    for _ in range(200):
        with s._lock:
            if s._stop_enqueued:
                break
        threading.Event().wait(0.005)
    assert s._stop_enqueued is True

    # Submit 10 records concurrently while close() is still draining the in-flight batch.
    for i in range(10):
        s.record_request(_request(50 + i))
    assert s.dropped_count == 10

    gate.set()
    closer.join(timeout=5.0)
    assert not closer.is_alive()
    assert s.written_count == 1
    assert s.dropped_count == 10
    assert s.failed_count == 0
    assert s.lost_count == 0
    assert s.produced_count == 11 == s.written_count + s.dropped_count + s.failed_count + s.lost_count
    assert len(s.fetch_requests()) == 1


def test_high_concurrency_close_race_stress_preserves_accounting_invariant(tmp_path):
    for trial in range(5):
        s = SQLiteTelemetrySink(tmp_path / f"race_{trial}.db")
        start = threading.Barrier(7)
        counts = [0] * 6

        def producer(idx: int) -> None:
            start.wait()
            local = 0
            for j in range(120):
                s.record_request(_request(idx * 1000 + j))
                local += 1
            counts[idx] = local

        def closer() -> None:
            start.wait()
            s.close(timeout=10.0)

        threads = [threading.Thread(target=producer, args=(i,)) for i in range(6)]
        close_thread = threading.Thread(target=closer)
        for t in threads:
            t.start()
        close_thread.start()
        for t in threads:
            t.join(timeout=10.0)
        close_thread.join(timeout=10.0)

        produced = sum(counts)
        assert s.produced_count == produced
        assert s.failed_count == 0
        assert s.lost_count == 0
        assert produced == s.written_count + s.dropped_count + s.failed_count + s.lost_count
        assert len(s.fetch_requests()) == s.written_count


def test_atomic_first_db_initialization_rolls_back_on_mid_init_failure(monkeypatch, tmp_path):
    import telemetry.writer as writer_mod

    db = tmp_path / "atomic_init.db"
    real_set_version = writer_mod._set_user_version

    def crash_before_user_version(conn, version):
        # Verify tables were already created inside the uncommitted transaction before crashing.
        tables_in_tx = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert {"request_logs", "shadow_records"} <= tables_in_tx
        raise sqlite3.OperationalError("simulated crash before PRAGMA user_version")

    monkeypatch.setattr(writer_mod, "_set_user_version", crash_before_user_version)
    with pytest.raises(RuntimeError, match="simulated crash before PRAGMA user_version"):
        SQLiteTelemetrySink(db)

    # Transaction rolled back atomically: zero tables and user_version == 0 on disk.
    conn = sqlite3.connect(db)
    try:
        tables_after = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        user_ver = conn.execute("PRAGMA user_version;").fetchone()[0]
        assert tables_after == []
        assert user_ver == 0
    finally:
        conn.close()

    # Subsequent startup on the same DB file succeeds cleanly without schema mismatch.
    monkeypatch.setattr(writer_mod, "_set_user_version", real_set_version)
    with SQLiteTelemetrySink(db) as recovered:
        recovered.record_request(_request(1))
        recovered.flush()
        assert recovered.written_count == 1
        assert len(recovered.fetch_requests()) == 1


# Step 4 (Part A1): column-level schema verification on existing databases (IP-8)


def _build_db(db, statements, user_version):
    conn = sqlite3.connect(db)
    try:
        for statement in statements:
            conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {user_version};")
        conn.commit()
    finally:
        conn.close()


def _current_statements():
    import telemetry.writer as writer_mod

    return list(writer_mod._SCHEMA_STATEMENTS)


def _row_count(db, table):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def test_outdated_v1_database_missing_egress_action_fails_fast_before_any_insert(monkeypatch, tmp_path):
    import re

    import telemetry.writer as writer_mod

    db = tmp_path / "outdated_v1.db"
    statements = _current_statements()
    outdated_request_ddl = re.sub(r"\s*egress_action TEXT NOT NULL,", "", statements[0])
    assert "egress_action" not in outdated_request_ddl
    _build_db(db, [outdated_request_ddl, *statements[1:]], user_version=SCHEMA_VERSION)

    inserts = []
    real_request_row = writer_mod._request_row
    monkeypatch.setattr(writer_mod, "_request_row", lambda record: inserts.append(record) or real_request_row(record))

    with pytest.raises(RuntimeError, match=r"column mismatch.*request_logs: missing columns \['egress_action'\]"):
        SQLiteTelemetrySink(db)
    assert inserts == []
    assert _row_count(db, "request_logs") == 0
    conn = sqlite3.connect(db)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(request_logs)")}
    finally:
        conn.close()
    assert "egress_action" not in cols  # the check never migrates or alters the table


def test_existing_database_with_mismatched_column_type_fails_fast(tmp_path):
    db = tmp_path / "wrong_type.db"
    statements = _current_statements()
    statements[0] = statements[0].replace("total_latency_ms REAL NOT NULL", "total_latency_ms TEXT NOT NULL")
    _build_db(db, statements, user_version=SCHEMA_VERSION)
    with pytest.raises(RuntimeError, match=r"mismatched column definitions \['total_latency_ms'\]"):
        SQLiteTelemetrySink(db)


def test_existing_database_with_dropped_not_null_fails_fast(tmp_path):
    db = tmp_path / "nullable.db"
    statements = _current_statements()
    statements[0] = statements[0].replace("egress_action TEXT NOT NULL", "egress_action TEXT")
    _build_db(db, statements, user_version=SCHEMA_VERSION)
    with pytest.raises(RuntimeError, match=r"mismatched column definitions \['egress_action'\]"):
        SQLiteTelemetrySink(db)


def test_existing_database_with_unexpected_column_fails_fast(tmp_path):
    db = tmp_path / "extra_col.db"
    statements = _current_statements()
    statements[3] = statements[3].replace("objective_id TEXT\n", "objective_id TEXT,\n        legacy_flag INTEGER\n")
    _build_db(db, statements, user_version=SCHEMA_VERSION)
    with pytest.raises(RuntimeError, match=r"shadow_records: unexpected columns \['legacy_flag'\]"):
        SQLiteTelemetrySink(db)


def test_existing_versioned_database_missing_table_fails_fast_instead_of_recreating(tmp_path):
    db = tmp_path / "missing_table.db"
    statements = [s for s in _current_statements() if "shadow_records" not in s]
    _build_db(db, statements, user_version=SCHEMA_VERSION)
    with pytest.raises(RuntimeError, match="shadow_records: table missing"):
        SQLiteTelemetrySink(db)
    conn = sqlite3.connect(db)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert "shadow_records" not in tables  # the failed startup rolled back; nothing was created


def test_expected_columns_cover_every_record_field():
    import telemetry.writer as writer_mod

    request_cols = {spec[0] for spec in writer_mod.EXPECTED_TABLE_COLUMNS["request_logs"]}
    shadow_cols = {spec[0] for spec in writer_mod.EXPECTED_TABLE_COLUMNS["shadow_records"]}
    assert request_cols == {"id"} | set(RequestLogRecord.model_fields)
    assert shadow_cols == {"id"} | set(ShadowRecord.model_fields)