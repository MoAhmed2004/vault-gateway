"""Suite-wide test configuration: zero real network calls (§5.5, AGENTS.md Rule 2).

`_no_network` is autouse, so it wraps every test. It refuses, with
`NetworkBlocked`, any socket connection to a non-loopback address and any
name lookup of a non-loopback host, on every path a library can take:

- `socket.socket.connect` and `connect_ex` (blocking sockets, `socketpair`);
- `socket.getaddrinfo` (DNS resolution, also used by asyncio and httpx);
- `sock_connect` of the asyncio selector and proactor event loops (the
  Windows proactor connects through `ConnectEx` and never calls
  `socket.connect`).

Loopback (`127.0.0.0/8`, `::1`, `localhost`) stays allowed, because asyncio's
self-pipe on Windows is a loopback socket pair; Unix-domain sockets are
allowed too. Unit tests reach Groq only through `httpx.MockTransport`.
"""

from __future__ import annotations

import asyncio.proactor_events
import asyncio.selector_events
import ipaddress
import socket
from typing import Any

import pytest

LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})


class NetworkBlocked(RuntimeError):
    """A test tried to reach a non-loopback host (§5.5: zero real network calls during pytest)."""


def is_loopback_host(host: Any) -> bool:
    if host is None or host == "":
        return True  # getaddrinfo(None, port): the local wildcard, never a remote host
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    text = str(host).strip("[]").split("%", 1)[0].lower()
    if text in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def is_loopback_address(address: Any) -> bool:
    if isinstance(address, (str, bytes)):  # AF_UNIX path
        return True
    if isinstance(address, tuple) and address:
        return is_loopback_host(address[0])
    return False


def _refuse(target: Any) -> NetworkBlocked:
    return NetworkBlocked(f"network access to {target!r} is blocked during tests (§5.5)")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def connect(self: socket.socket, address: Any) -> Any:
        if not is_loopback_address(address):
            raise _refuse(address)
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> Any:
        if not is_loopback_address(address):
            raise _refuse(address)
        return real_connect_ex(self, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_loopback_host(host):
            raise _refuse(host)
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)

    for loop_class in (asyncio.selector_events.BaseSelectorEventLoop, asyncio.proactor_events.BaseProactorEventLoop):
        real_sock_connect = loop_class.sock_connect

        def make_guard(real: Any) -> Any:
            async def sock_connect(self: Any, sock: socket.socket, address: Any) -> Any:
                if not is_loopback_address(address):
                    raise _refuse(address)
                return await real(self, sock, address)

            return sock_connect

        monkeypatch.setattr(loop_class, "sock_connect", make_guard(real_sock_connect))
