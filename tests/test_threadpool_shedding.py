"""Unit tests for the bounded layer pool and load shedding (IP-7, TM-R2, TM-R6)."""

from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest

from app.executor import BoundedLayerPool, LoadShed, physical_cores
from app.layers.l4_egress import L4EgressScanner
from app.main import create_app
from app.pipeline import run_budgeted
from contracts import Outcome
from tests.fakes import TEST_API_KEY, FakeGroq
from tests.test_gateway_shell import FakeSink, assert_uniform, make_settings


def test_default_worker_count_is_one_per_physical_core():
    import psutil

    pool = BoundedLayerPool(workers=0, queue_limit=4)
    try:
        assert pool.workers == physical_cores() == max(1, psutil.cpu_count(logical=False) or 1)
        assert pool.capacity == pool.workers + 4
    finally:
        pool.close()


def test_pool_sheds_once_running_plus_queued_reaches_capacity():
    pool = BoundedLayerPool(workers=1, queue_limit=1)
    release = threading.Event()
    try:
        first = pool.submit(release.wait)
        second = pool.submit(release.wait)  # queued behind the first
        with pytest.raises(LoadShed):
            pool.submit(release.wait)
        assert pool.stats().shed == 1 and pool.stats().outstanding == 2
        release.set()
        first.result(timeout=2)
        second.result(timeout=2)
        deadline = time.monotonic() + 2
        while pool.stats().outstanding and time.monotonic() < deadline:
            time.sleep(0.005)
        assert pool.submit(lambda: 7).result(timeout=2) == 7  # capacity is back
    finally:
        release.set()
        pool.close()


def test_a_timed_out_call_keeps_its_slot_until_it_finishes():
    """IP-7: a timeout does not stop a running thread; the slot frees only when the call returns."""
    pool = BoundedLayerPool(workers=1, queue_limit=0)

    async def go():
        started = time.perf_counter()
        outcome = await run_budgeted(lambda: time.sleep(0.3), 10, pool)
        waited = time.perf_counter() - started
        held = pool.stats().outstanding
        with pytest.raises(LoadShed):
            pool.submit(lambda: None)
        await asyncio.sleep(0.4)
        return outcome, waited, held, pool.stats().outstanding

    try:
        outcome, waited, held, after = asyncio.run(go())
        assert outcome.status == "timeout" and waited < 0.25
        assert held == 1 and after == 0
    finally:
        pool.close()


def test_layer_work_never_blocks_the_event_loop():
    pool = BoundedLayerPool(workers=1, queue_limit=0)

    async def go():
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        await pool.run(lambda: time.sleep(0.25))  # blocking work, on a pool thread
        beat.cancel()
        return ticks

    try:
        assert asyncio.run(go()) >= 8  # the loop kept running while the layer slept
    finally:
        pool.close()


def test_shed_call_is_reported_as_shed_by_run_budgeted():
    pool = BoundedLayerPool(workers=1, queue_limit=0)
    release = threading.Event()
    try:
        pool.submit(release.wait)
        outcome = asyncio.run(run_budgeted(lambda: 1, 100, pool))
        assert outcome.status == "shed"
    finally:
        release.set()
        pool.close()


class _HoggingL1:
    """L1 stand-in whose first real call holds the only pool thread for 300 ms (warm-up is call 1)."""

    name, version, enabled = "L1", "hogging-l1", True
    chroma_version = None

    def __init__(self) -> None:
        self.calls = 0

    def inspect(self, bundle, *, degraded=False):
        from app.layers.l1_regex import L1RegexLayer

        self.calls += 1
        if self.calls == 2:
            time.sleep(0.3)
        return L1RegexLayer().inspect(bundle)


def test_gateway_sheds_load_with_the_uniform_503(tmp_path):
    sink = FakeSink()
    fake = FakeGroq()
    settings = make_settings(tmp_path, UPSTREAM_MODE="groq", GROQ_API_KEY=TEST_API_KEY, TIMEOUT_POLICY="P-closed")
    app = create_app(
        settings, sink, upstream=fake.client(), l4=L4EgressScanner(), extra_layers=(_HoggingL1(),),
        pool=BoundedLayerPool(workers=1, queue_limit=0),
    )

    async def go():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
                first = await client.post("/v1/chat", json={"user_prompt": "hello"})  # L1 times out, thread keeps running
                second = await client.post("/v1/chat", json={"user_prompt": "hello again"})  # pool still full
                await asyncio.sleep(0.4)
                third = await client.post("/v1/chat", json={"user_prompt": "What is the capital of France?"})
                return first, second, third

    first, second, third = asyncio.run(go())
    assert_uniform(first)
    assert_uniform(second)
    assert third.status_code == 200
    assert [(r.outcome, r.blocked_by) for r in sink.records] == [
        (Outcome.ERROR, "p_closed"), (Outcome.ERROR, "load_shed"), (Outcome.SUCCESS, None),
    ]
    assert sink.records[1].target_called is False and len(fake.chat_requests) == 1
