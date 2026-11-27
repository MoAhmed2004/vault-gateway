"""Bounded layer threadpool with load shedding (IP-7, TM-R2, TM-R6).

Every layer call (bundle build, L1 regex, L2 ChromaDB and MiniLM, L3 ONNX,
L4 Presidio) runs on this pool, never on the event loop. The pool is bounded:

- `workers` threads (default: one per physical core, IP-7).
- At most `workers + queue_limit` calls outstanding (running or queued). A
  call that would exceed that raises `LoadShed`, which the pipeline maps to
  `Error` / `load_shed` with the uniform 503 (TM-R6). The check and the
  increment happen under one lock, so it acts as a non-blocking semaphore.
- A call's slot is released only when the call actually finishes. An awaiting
  request that stops waiting (TM-R2 timeout) does not free the slot: a
  timed-out thread keeps running and keeps occupying the pool until it
  returns (IP-7), which `PoolStats` exposes so E4 can report saturation.

`run(fn)` is a coroutine: the event loop only awaits the future, so it never
blocks on layer work.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger("vault.executor")

T = TypeVar("T")
DEFAULT_QUEUE_LIMIT = 32


class LoadShed(RuntimeError):
    """The pool is full; the request is refused with the uniform 503 (IP-7)."""


def physical_cores() -> int:
    """Physical core count (IP-7 default worker count); falls back to the logical count, then 1."""
    try:
        import psutil

        count = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001  sizing must never stop startup
        count = None
    return max(1, int(count or os.cpu_count() or 1))


@dataclass(frozen=True)
class PoolStats:
    """Snapshot of the pool counters (E4 saturation reporting)."""

    workers: int
    capacity: int
    outstanding: int
    peak_outstanding: int
    submitted: int
    completed: int
    shed: int


class BoundedLayerPool:
    """Thread pool with a hard bound on running plus queued calls."""

    def __init__(self, workers: int | None = None, queue_limit: int = DEFAULT_QUEUE_LIMIT) -> None:
        resolved = physical_cores() if workers is None or workers <= 0 else workers
        if queue_limit < 0:
            raise ValueError("queue_limit must be >= 0")
        self.workers = resolved
        self.queue_limit = queue_limit
        self.capacity = resolved + queue_limit
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=resolved, thread_name_prefix="vault-layer"
        )
        self._lock = threading.Lock()
        self._outstanding = 0
        self._peak = 0
        self._submitted = 0
        self._completed = 0
        self._shed = 0

    def _release(self, _future: Future[object]) -> None:
        with self._lock:
            self._outstanding -= 1
            self._completed += 1

    def submit(self, fn: Callable[[], T]) -> Future[T]:
        """Queue `fn` or raise `LoadShed` when `capacity` calls are already outstanding."""
        with self._lock:
            if self._executor is None:
                raise RuntimeError("layer pool is closed")
            if self._outstanding >= self.capacity:
                self._shed += 1
                raise LoadShed(f"layer pool full ({self._outstanding}/{self.capacity} outstanding)")
            self._outstanding += 1
            self._submitted += 1
            self._peak = max(self._peak, self._outstanding)
            executor = self._executor
        try:
            future = executor.submit(fn)
        except BaseException:
            with self._lock:
                self._outstanding -= 1
            raise
        future.add_done_callback(self._release)
        return future

    async def run(self, fn: Callable[[], T]) -> T:
        """Run `fn` on the pool and await it without blocking the event loop."""
        return await asyncio.wrap_future(self.submit(fn))

    def stats(self) -> PoolStats:
        with self._lock:
            return PoolStats(
                workers=self.workers,
                capacity=self.capacity,
                outstanding=self._outstanding,
                peak_outstanding=self._peak,
                submitted=self._submitted,
                completed=self._completed,
                shed=self._shed,
            )

    def close(self, wait: bool = True) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=False)
