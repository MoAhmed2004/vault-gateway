"""IP-5 verdict and response caches in one local SQLite file (IP-5, IP-8, IP-9, §2.7).

Three caches, keyed only by the frozen builders in `contracts.cache_keys`:

- `layer_verdicts`: the `LayerResult` tuple an ingress layer returned for one
  bundle (`layer_verdict_key`: prompt hash, layer, layer version, bundle
  version, ChromaDB version for L2 only).
- `egress_verdicts`: one L4 decision per (response hash, manifest ID, variant)
  (`egress_verdict_key`). The frozen key carries no L4 version, so each row
  also stores the `l4_version` that produced it and a lookup under a
  different version is a miss [Project Decision, extends IP-5: a Presidio or
  rule change must never replay a stale egress decision].
- `target_responses`: the full raw provider JSON, `reasoning` included
  (`target_response_key`: model ID, parameters, full message list).

Storage (IP-8): the file must be on local disk (checked by `app.config`),
with `journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`. Every
statement runs on one dedicated thread that owns the only connection, fed by
the executor's in-memory FIFO queue, so there is exactly one writer and a read
submitted after a write always sees it. Writes are not awaited by the request
path; `flush()` waits for every queued write. Rows are written with
`INSERT OR IGNORE`, so the first value stored under a key is kept and replays
are deterministic. `writes_enabled=False` bypasses every write (E4, IP-8).

Schema setup mirrors `telemetry/writer.py` without importing it (package
boundaries, §5.2): one `BEGIN IMMEDIATE ... COMMIT` transaction creates the
tables and sets `PRAGMA user_version`, and every open compares `PRAGMA
table_info` of each table with the expected layout and fails fast on any
difference. Nothing is migrated in place.

Only allow and block decisions are cached; a timeout or error is never stored.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from contracts.layer_result import LayerResult

logger = logging.getLogger("vault.cache")

CACHE_SCHEMA_VERSION = 1
CACHE_NAMES = ("layer_verdicts", "egress_verdicts", "target_responses")

_SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS layer_verdicts (
        cache_key TEXT PRIMARY KEY NOT NULL,
        layer TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_utc TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS egress_verdicts (
        cache_key TEXT PRIMARY KEY NOT NULL,
        l4_version TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_utc TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS target_responses (
        cache_key TEXT PRIMARY KEY NOT NULL,
        model_id TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_utc TEXT NOT NULL
    )""",
)

ColumnSpec = tuple[str, str, int, int]


def _table_columns(conn: sqlite3.Connection, table: str) -> tuple[ColumnSpec, ...]:
    return tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in conn.execute(f"PRAGMA table_info({table})")
    )


def _expected_columns() -> dict[str, tuple[ColumnSpec, ...]]:
    scratch = sqlite3.connect(":memory:")
    try:
        for statement in _SCHEMA_STATEMENTS:
            scratch.execute(statement)
        return {table: _table_columns(scratch, table) for table in CACHE_NAMES}
    finally:
        scratch.close()


EXPECTED_CACHE_COLUMNS = _expected_columns()


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class CacheStats:
    """Hit and miss counters per cache, for telemetry and tests."""

    hits: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CACHE_NAMES, 0))
    misses: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CACHE_NAMES, 0))
    writes: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CACHE_NAMES, 0))
    write_errors: int = 0


class VerdictCache:
    """Single-thread SQLite store for the three IP-5 caches."""

    def __init__(self, path: Path | str, *, writes_enabled: bool = True) -> None:
        self.path = Path(path)
        self.writes_enabled = writes_enabled
        self.stats = CacheStats()
        self._executor: ThreadPoolExecutor | None = None
        self._conn: sqlite3.Connection | None = None
        self._pending: set[Future[Any]] = set()
        self._lock = threading.Lock()

    # Lifecycle

    def open(self) -> None:
        """Create the directory, connect on the cache thread, set pragmas and verify the schema (fail fast)."""
        if self._executor is not None:
            raise RuntimeError("cache is already open")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vault-cache")
        try:
            self._executor.submit(self._open_on_thread).result()
        except BaseException:
            self._executor.shutdown(wait=True)
            self._executor = None
            raise

    def _open_on_thread(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            self._init_schema(conn)
        except BaseException:
            conn.close()
            raise
        self._conn = conn

    @staticmethod
    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        try:
            version = int(conn.execute("PRAGMA user_version;").fetchone()[0])
            tables = [
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            ]
            if version == 0:
                if tables:
                    raise ValueError("cache schema mismatch: existing database has user_version=0")
                for statement in _SCHEMA_STATEMENTS:
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {CACHE_SCHEMA_VERSION};")
            elif version != CACHE_SCHEMA_VERSION:
                raise ValueError(f"cache schema mismatch: user_version={version}, expected {CACHE_SCHEMA_VERSION}")
            VerdictCache._verify_columns(conn)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    @staticmethod
    def _verify_columns(conn: sqlite3.Connection) -> None:
        problems = []
        for table, expected in EXPECTED_CACHE_COLUMNS.items():
            actual = _table_columns(conn, table)
            if not actual:
                problems.append(f"{table}: table missing")
            elif actual != expected:
                problems.append(f"{table}: columns {actual} differ from {expected}")
        if problems:
            raise ValueError("cache schema column mismatch: " + "; ".join(problems))

    def pragmas(self) -> dict[str, Any]:
        """Current connection pragmas (tests and readiness checks)."""

        def read() -> dict[str, Any]:
            conn = self._require_conn()
            return {
                "journal_mode": str(conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower(),
                "synchronous": int(conn.execute("PRAGMA synchronous;").fetchone()[0]),
                "busy_timeout": int(conn.execute("PRAGMA busy_timeout;").fetchone()[0]),
                "user_version": int(conn.execute("PRAGMA user_version;").fetchone()[0]),
            }

        return self._submit(read).result()

    def flush(self, timeout: float | None = 5.0) -> None:
        """Wait until every queued write has committed."""
        with self._lock:
            pending = list(self._pending)
        for future in pending:
            future.result(timeout=timeout)

    def close(self, timeout: float | None = 5.0) -> None:
        if self._executor is None:
            return
        try:
            self.flush(timeout=timeout)
        except Exception:  # noqa: BLE001  close the connection even when a write failed
            logger.exception("cache flush failed during close")
        executor = self._executor

        def shut() -> None:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

        executor.submit(shut).result(timeout=timeout)
        executor.shutdown(wait=True)
        self._executor = None

    @property
    def is_open(self) -> bool:
        return self._executor is not None and self._conn is not None

    # Internals

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("cache is not open")
        return self._conn

    def _submit(self, fn: Any, *args: Any) -> Future[Any]:
        if self._executor is None:
            raise RuntimeError("cache is not open")
        return self._executor.submit(fn, *args)

    async def _read(self, sql: str, params: tuple[Any, ...]) -> tuple[Any, ...] | None:
        def run() -> tuple[Any, ...] | None:
            row = self._require_conn().execute(sql, params).fetchone()
            return tuple(row) if row is not None else None

        return await asyncio.wrap_future(self._submit(run))

    def _write(self, cache: str, sql: str, params: tuple[Any, ...]) -> None:
        if not self.writes_enabled:
            return

        def run() -> None:
            self._require_conn().execute(sql, params)

        future = self._submit(run)
        with self._lock:
            self._pending.add(future)

        def done(f: Future[Any]) -> None:
            error = f.exception()
            with self._lock:
                self._pending.discard(f)
                if error is not None:
                    self.stats.write_errors += 1
                else:
                    self.stats.writes[cache] += 1
            if error is not None:
                logger.error("cache write to %s failed: %s", cache, type(error).__name__)

        future.add_done_callback(done)

    def _count(self, cache: str, hit: bool) -> None:
        (self.stats.hits if hit else self.stats.misses)[cache] += 1

    # Layer verdicts

    async def get_layer(self, key: str) -> tuple[LayerResult, ...] | None:
        row = await self._read("SELECT payload FROM layer_verdicts WHERE cache_key = ?", (key,))
        self._count("layer_verdicts", row is not None)
        if row is None:
            return None
        return tuple(LayerResult.model_validate(item) for item in json.loads(row[0]))

    def put_layer(self, key: str, layer: str, results: tuple[LayerResult, ...]) -> None:
        if not results or any(r.verdict not in ("allow", "block") for r in results):
            return  # timeouts and errors are never cached
        payload = _dumps([r.model_dump(mode="json") for r in results])
        self._write(
            "layer_verdicts",
            "INSERT OR IGNORE INTO layer_verdicts (cache_key, layer, payload, created_utc) VALUES (?, ?, ?, ?)",
            (key, layer, payload, _now()),
        )

    # Egress verdicts

    async def get_egress(self, key: str, l4_version: str) -> dict[str, Any] | None:
        row = await self._read("SELECT l4_version, payload FROM egress_verdicts WHERE cache_key = ?", (key,))
        hit = row is not None and row[0] == l4_version
        self._count("egress_verdicts", hit)
        return json.loads(row[1]) if hit else None

    def put_egress(self, key: str, l4_version: str, payload: Mapping[str, Any]) -> None:
        self._write(
            "egress_verdicts",
            "INSERT OR IGNORE INTO egress_verdicts (cache_key, l4_version, payload, created_utc) VALUES (?, ?, ?, ?)",
            (key, l4_version, _dumps(dict(payload)), _now()),
        )

    # Target responses

    async def get_response(self, key: str) -> dict[str, Any] | None:
        row = await self._read("SELECT payload FROM target_responses WHERE cache_key = ?", (key,))
        self._count("target_responses", row is not None)
        return json.loads(row[0]) if row is not None else None

    def put_response(self, key: str, model_id: str, raw_provider_json: Mapping[str, Any]) -> None:
        self._write(
            "target_responses",
            "INSERT OR IGNORE INTO target_responses (cache_key, model_id, payload, created_utc) VALUES (?, ?, ?, ?)",
            (key, model_id, _dumps(dict(raw_provider_json)), _now()),
        )
