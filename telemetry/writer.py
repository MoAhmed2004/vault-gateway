"""SQLite single-writer telemetry sink (IP-7, IP-8, IP-4, RT-V6, TM-R6).

One dedicated thread owns the only SQLite connection. Producers validate the
record type and enqueue it on an in-memory queue (bounded by default for
production, or unbounded for measured/E4 runs); they never block on disk I/O
and never raise on a full queue (the drop is counted instead).

Accounting invariant: for every submitted record,
`produced_count == written_count + dropped_count + failed_count + lost_count`.
`written_count` is incremented only after `COMMIT` succeeds; failed commits
increment `failed_count`, and queued records abandoned on an unexpected writer
thread death are counted in `lost_count`.

Imports only stdlib and `contracts`.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
from pathlib import Path
from typing import Any

from contracts import RequestLogRecord, ShadowRecord
from contracts.paths import synced_folder_component

SCHEMA_VERSION = 1
DEFAULT_QUEUE_MAXSIZE = 10000
_BATCH_LIMIT = 200

REQUEST_COLUMNS = (
    "timestamp_utc", "gateway_mode", "prompt_hash", "layer_results", "target_called", "outcome",
    "total_latency_ms", "reasoning_findings", "blocked_by", "egress_variant", "egress_action",
    "timeout_policy", "cache_hit", "model_id", "trial_id", "objective_id", "tier", "config",
    "seed", "query_index", "operator_chain", "oracle_result", "judge_score",
)
SHADOW_COLUMNS = (
    "timestamp_utc", "shadow_tier", "prompt_hash", "layer_results", "total_latency_ms", "target_called",
    "cache_hit", "egress_variants", "layer_version", "bundle_version", "chroma_version", "trial_id",
    "objective_id",
)
_REQUEST_JSON = ("layer_results", "operator_chain", "reasoning_findings")
_SHADOW_JSON = ("layer_results", "egress_variants")

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS request_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_utc TEXT NOT NULL,
        gateway_mode TEXT NOT NULL,
        prompt_hash TEXT NOT NULL,
        layer_results TEXT NOT NULL,
        target_called INTEGER NOT NULL,
        outcome TEXT NOT NULL,
        total_latency_ms REAL NOT NULL,
        reasoning_findings TEXT NOT NULL,
        blocked_by TEXT,
        egress_variant TEXT,
        egress_action TEXT NOT NULL,
        timeout_policy TEXT,
        cache_hit INTEGER,
        model_id TEXT,
        trial_id TEXT,
        objective_id TEXT,
        tier TEXT,
        config TEXT,
        seed INTEGER,
        query_index INTEGER,
        operator_chain TEXT,
        oracle_result INTEGER,
        judge_score REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_request_logs_trial ON request_logs (trial_id)",
    "CREATE INDEX IF NOT EXISTS idx_request_logs_prompt ON request_logs (prompt_hash)",
    """
    CREATE TABLE IF NOT EXISTS shadow_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_utc TEXT NOT NULL,
        shadow_tier TEXT NOT NULL,
        prompt_hash TEXT NOT NULL,
        layer_results TEXT NOT NULL,
        total_latency_ms REAL NOT NULL,
        target_called INTEGER NOT NULL,
        cache_hit INTEGER,
        egress_variants TEXT NOT NULL,
        layer_version TEXT NOT NULL,
        bundle_version TEXT NOT NULL,
        chroma_version TEXT NOT NULL,
        trial_id TEXT,
        objective_id TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_shadow_records_trial ON shadow_records (trial_id)",
    "CREATE INDEX IF NOT EXISTS idx_shadow_records_prompt ON shadow_records (prompt_hash)",
)


def _dump_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _text(value: str | None) -> str | None:
    """Make a string bindable by SQLite (lone surrogates become backslash escapes)."""
    if value is None:
        return None
    return value.encode("utf-8", errors="backslashreplace").decode("utf-8")


def _request_row(record: RequestLogRecord) -> tuple[Any, ...]:
    data = record.model_dump(mode="json")
    values: list[Any] = []
    for column in REQUEST_COLUMNS:
        value = data[column]
        if column in _REQUEST_JSON:
            if column == "operator_chain" and value is None:
                values.append(None)
            else:
                values.append(_dump_json(value))
        elif isinstance(value, bool):
            values.append(int(value))
        elif isinstance(value, str):
            values.append(_text(value))
        else:
            values.append(value)
    return tuple(values)


def _shadow_row(record: ShadowRecord) -> tuple[Any, ...]:
    data = record.model_dump(mode="json")
    values: list[Any] = []
    for column in SHADOW_COLUMNS:
        value = data[column]
        if column in _SHADOW_JSON:
            values.append(_dump_json(value))
        elif isinstance(value, bool):
            values.append(int(value))
        elif isinstance(value, str):
            values.append(_text(value))
        else:
            values.append(value)
    return tuple(values)


def _insert_sql(table: str, columns: tuple[str, ...]) -> str:
    marks = ", ".join("?" for _ in columns)
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({marks})"


_REQUEST_SQL = _insert_sql("request_logs", REQUEST_COLUMNS)
_SHADOW_SQL = _insert_sql("shadow_records", SHADOW_COLUMNS)


class _Flush:
    def __init__(self) -> None:
        self.event = threading.Event()


def _is_record_item(item: Any) -> bool:
    return item is not None and not isinstance(item, _Flush)


MANAGED_TABLES: tuple[str, ...] = ("request_logs", "shadow_records")

# One `PRAGMA table_info` row reduced to the fields that define a column:
# (name, declared type, NOT NULL flag, primary-key position).
ColumnSpec = tuple[str, str, int, int]


def _apply_schema_statements(conn: sqlite3.Connection) -> None:
    for statement in _SCHEMA_STATEMENTS:
        conn.execute(statement)


def _table_columns(conn: sqlite3.Connection, table: str) -> tuple[ColumnSpec, ...]:
    return tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in conn.execute(f"PRAGMA table_info({table})")
    )


def _expected_table_columns() -> dict[str, tuple[ColumnSpec, ...]]:
    """Column layout produced by `_SCHEMA_STATEMENTS`, read back from a scratch in-memory database."""
    scratch = sqlite3.connect(":memory:")
    try:
        _apply_schema_statements(scratch)
        return {table: _table_columns(scratch, table) for table in MANAGED_TABLES}
    finally:
        scratch.close()


EXPECTED_TABLE_COLUMNS = _expected_table_columns()


def _verify_table_columns(conn: sqlite3.Connection) -> None:
    """Fail fast when any managed table is missing or its columns differ from the current schema (IP-8).

    `CREATE TABLE IF NOT EXISTS` never alters an existing table, so a database
    whose `user_version` matches but whose tables predate a column (for example
    `egress_action`) would otherwise fail only at the first insert.
    """
    problems: list[str] = []
    for table, expected in EXPECTED_TABLE_COLUMNS.items():
        actual = _table_columns(conn, table)
        if not actual:
            problems.append(f"{table}: table missing")
            continue
        expected_by_name = {spec[0]: spec for spec in expected}
        actual_by_name = {spec[0]: spec for spec in actual}
        missing = [name for name in expected_by_name if name not in actual_by_name]
        unexpected = [name for name in actual_by_name if name not in expected_by_name]
        mismatched = [
            name
            for name, spec in expected_by_name.items()
            if name in actual_by_name and actual_by_name[name] != spec
        ]
        if missing:
            problems.append(f"{table}: missing columns {missing}")
        if unexpected:
            problems.append(f"{table}: unexpected columns {unexpected}")
        if mismatched:
            problems.append(f"{table}: mismatched column definitions {mismatched}")
    if problems:
        raise ValueError(
            f"telemetry schema column mismatch (user_version={SCHEMA_VERSION}): " + "; ".join(problems)
        )


def _set_user_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(f"PRAGMA user_version = {version};")


def _init_and_check_schema(conn: sqlite3.Connection) -> None:
    """Atomically initialize schema and PRAGMA user_version, or verify an existing DB."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        user_version = int(conn.execute("PRAGMA user_version;").fetchone()[0])
        existing_tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        if user_version == 0:
            if existing_tables:
                raise ValueError(
                    f"telemetry schema version mismatch: existing database has user_version=0, "
                    f"expected {SCHEMA_VERSION}"
                )
            _apply_schema_statements(conn)
            _set_user_version(conn, SCHEMA_VERSION)
        elif user_version != SCHEMA_VERSION:
            raise ValueError(
                f"telemetry schema version mismatch: database has user_version={user_version}, "
                f"expected {SCHEMA_VERSION}"
            )
        else:
            # Verify before the idempotent DDL so a missing table is reported, not silently recreated.
            _verify_table_columns(conn)
            _apply_schema_statements(conn)
        _verify_table_columns(conn)
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


class SQLiteTelemetrySink:
    """`TelemetrySink` backed by a single writer thread and a WAL-mode SQLite file."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        unbounded: bool = False,
    ) -> None:
        if queue_maxsize < 0:
            raise ValueError("queue_maxsize must be >= 0")
        path = Path(db_path)
        offender = synced_folder_component(path.expanduser().resolve())
        if offender is not None:
            raise ValueError(f"telemetry db must be on local disk, not a synced folder (IP-8): {offender!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        effective_maxsize = 0 if unbounded else queue_maxsize
        self._db_path = path
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=effective_maxsize)
        self._lock = threading.Lock()
        self._produced = 0
        self._dropped = 0
        self._failed = 0
        self._written = 0
        self._lost = 0
        self._batch_staged = 0
        self._batch_failed = 0
        self._writer_deaths = 0
        self._writer_error: BaseException | None = None
        self._closed = False
        self._stop_enqueued = False
        self._startup_error: BaseException | None = None
        self._ready = threading.Event()
        self.pragmas: dict[str, Any] = {}
        self._thread = threading.Thread(target=self._run, name="telemetry-writer", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            self._thread.join()
            raise RuntimeError(f"telemetry writer failed to start: {self._startup_error}") from self._startup_error

    # Writer thread

    def _run(self) -> None:
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0, isolation_level=None)
        except BaseException as exc:  # noqa: BLE001
            self._startup_error = exc
            self._ready.set()
            return
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            _init_and_check_schema(conn)
            self.pragmas = {
                "journal_mode": str(conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower(),
                "synchronous": int(conn.execute("PRAGMA synchronous;").fetchone()[0]),
                "busy_timeout": int(conn.execute("PRAGMA busy_timeout;").fetchone()[0]),
                "user_version": int(conn.execute("PRAGMA user_version;").fetchone()[0]),
            }
        except BaseException as exc:  # noqa: BLE001
            self._startup_error = exc
            conn.close()
            self._ready.set()
            return
        self._ready.set()
        try:
            self._drain_loop(conn)
        except BaseException as exc:  # noqa: BLE001
            self._record_writer_death(exc)
            try:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
            except BaseException:  # noqa: BLE001
                pass
        finally:
            for waiter in self._drain_abandoned_queue():
                waiter.event.set()
            conn.close()

    def _record_writer_death(self, exc: BaseException) -> None:
        with self._lock:
            if self._writer_error is None:
                self._writer_error = exc
                self._writer_deaths += 1

    def _drain_abandoned_queue(self) -> list[_Flush]:
        """Drain remaining queue items on writer exit, counting abandoned records as `lost`."""
        waiters: list[_Flush] = []
        abandoned = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, _Flush):
                waiters.append(item)
            elif item is not None:
                abandoned += 1
        if abandoned:
            with self._lock:
                self._lost += abandoned
        return waiters

    def _begin_tx(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN")

    def _commit_tx(self, conn: sqlite3.Connection) -> None:
        conn.execute("COMMIT")

    def _drain_loop(self, conn: sqlite3.Connection) -> None:
        stop = False
        while not stop:
            items = [self._queue.get()]
            while len(items) < _BATCH_LIMIT:
                try:
                    items.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            total_records = sum(1 for item in items if _is_record_item(item))
            waiters: list[_Flush] = []
            self._batch_staged = 0
            self._batch_failed = 0
            attempted_records = 0
            try:
                self._begin_tx(conn)
                for item in items:
                    if item is None:
                        stop = True
                    elif isinstance(item, _Flush):
                        waiters.append(item)
                    else:
                        attempted_records += 1
                        self._write_item(conn, item)
                self._commit_tx(conn)
            except BaseException as exc:
                # Seal _writer_error under lock first so concurrent producers drop
                # rather than enqueueing behind _drain_abandoned_queue().
                self._record_writer_death(exc)
                with self._lock:
                    self._failed += attempted_records
                    self._lost += total_records - attempted_records
                try:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                except BaseException:  # noqa: BLE001
                    pass
                queued_waiters = self._drain_abandoned_queue()
                for item in items:
                    if isinstance(item, _Flush):
                        item.event.set()
                for waiter in queued_waiters:
                    waiter.event.set()
                raise
            with self._lock:
                self._written += self._batch_staged
                self._failed += self._batch_failed
            for waiter in waiters:
                waiter.event.set()

    def _write_item(self, conn: sqlite3.Connection, item: tuple[str, Any]) -> None:
        kind, record = item
        try:
            if kind == "request":
                conn.execute(_REQUEST_SQL, _request_row(record))
            else:
                conn.execute(_SHADOW_SQL, _shadow_row(record))
            self._batch_staged += 1
        except Exception:  # noqa: BLE001  a bad row must not kill the writer
            self._batch_failed += 1

    # Producer API

    def _enqueue(self, item: tuple[str, Any]) -> None:
        with self._lock:
            self._produced += 1
            if self._closed or self._stop_enqueued or self._writer_error is not None:
                self._dropped += 1
                return
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self._dropped += 1

    def record_request(self, record: RequestLogRecord) -> None:
        if not isinstance(record, RequestLogRecord):
            raise TypeError("record_request requires a RequestLogRecord")
        self._enqueue(("request", record))

    def record_shadow(self, record: ShadowRecord) -> None:
        if not isinstance(record, ShadowRecord):
            raise TypeError("record_shadow requires a ShadowRecord")
        self._enqueue(("shadow", record))

    # Counters and health

    @property
    def produced_count(self) -> int:
        with self._lock:
            return self._produced

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped

    @property
    def failed_count(self) -> int:
        with self._lock:
            return self._failed

    @property
    def written_count(self) -> int:
        with self._lock:
            return self._written

    @property
    def lost_count(self) -> int:
        with self._lock:
            return self._lost

    @property
    def writer_deaths(self) -> int:
        with self._lock:
            return self._writer_deaths

    @property
    def writer_error(self) -> BaseException | None:
        with self._lock:
            return self._writer_error

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def healthy(self) -> bool:
        with self._lock:
            return (
                self._thread.is_alive()
                and self._writer_error is None
                and not self._closed
                and not self._stop_enqueued
            )

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    @property
    def unbounded(self) -> bool:
        return self._queue.maxsize == 0

    @property
    def db_path(self) -> Path:
        return self._db_path

    # Lifecycle

    def _raise_if_writer_died(self) -> None:
        err = self.writer_error
        if err is not None:
            raise RuntimeError(f"telemetry writer thread died: {err}") from err

    def flush(self, timeout: float = 5.0) -> None:
        """Block until every record queued before this call is committed."""
        self._raise_if_writer_died()
        if not self._thread.is_alive():
            if self._closed:
                return
            raise RuntimeError("telemetry writer thread is not running")
        marker = _Flush()
        try:
            self._queue.put(marker, timeout=timeout)
        except queue.Full as exc:
            self._raise_if_writer_died()
            raise TimeoutError("telemetry queue full; flush timed out") from exc
        self._raise_if_writer_died()
        if not marker.event.wait(timeout):
            self._raise_if_writer_died()
            raise TimeoutError("telemetry flush timed out")
        self._raise_if_writer_died()

    def close(self, timeout: float = 5.0) -> None:
        """Drain queued writes, stop the writer thread, and close its connection.

        Once `close()` begins, `_stop_enqueued` is set under lock so concurrent
        submissions are counted as `dropped` instead of racing behind the stop
        sentinel. If `timeout` expires while the writer thread is still alive,
        `_closed` remains `False` so a subsequent `close()` call can retry and
        finish draining cleanly.
        """
        with self._lock:
            if self._closed:
                return
            if not self._thread.is_alive():
                self._closed = True
                return
            need_sentinel = not self._stop_enqueued
            self._stop_enqueued = True
        if need_sentinel:
            try:
                self._queue.put(None, timeout=timeout)
            except queue.Full as exc:
                with self._lock:
                    self._stop_enqueued = False
                if not self._thread.is_alive():
                    with self._lock:
                        self._closed = True
                    return
                raise TimeoutError("telemetry queue full; close timed out") from exc
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("telemetry writer did not stop in time")
        with self._lock:
            self._closed = True

    def __enter__(self) -> SQLiteTelemetrySink:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # Read helpers (separate short-lived read-only connection)

    def _read(self, table: str, json_columns: tuple[str, ...], limit: int | None) -> list[dict[str, Any]]:
        uri = f"{self._db_path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        try:
            conn.row_factory = sqlite3.Row
            sql = f"SELECT * FROM {table} ORDER BY id"
            params: tuple[Any, ...] = ()
            if limit is not None:
                sql += " LIMIT ?"
                params = (limit,)
            rows = []
            for row in conn.execute(sql, params):
                item = dict(row)
                for column in json_columns:
                    if item.get(column) is not None:
                        item[column] = json.loads(item[column])
                rows.append(item)
            return rows
        finally:
            conn.close()

    def fetch_requests(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return stored request_logs rows (JSON columns decoded), oldest first."""
        return self._read("request_logs", _REQUEST_JSON, limit)

    def fetch_shadows(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return stored shadow_records rows (JSON columns decoded), oldest first."""
        return self._read("shadow_records", _SHADOW_JSON, limit)
