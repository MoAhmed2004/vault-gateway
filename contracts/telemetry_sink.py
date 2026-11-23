"""Neutral logging interface (§3.2, AGENTS.md Rule 5).

`app/` layers, middleware, and request handlers depend on this Protocol, not
on `telemetry/`. The concrete SQLite writer in `telemetry/` satisfies it and
is injected at startup composition.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contracts.records import RequestLogRecord, ShadowRecord


@runtime_checkable
class TelemetrySink(Protocol):
    """Minimal sink for typed server-side log records."""

    def record_request(self, record: RequestLogRecord) -> None:
        """Record one gateway request (RT-V6)."""
        ...

    def record_shadow(self, record: ShadowRecord) -> None:
        """Record one shadow-mode (evaluate-all) record (IP-4)."""
        ...
