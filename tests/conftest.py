"""Suite-wide test configuration: zero real network calls (§5.5, AGENTS.md Rule 2).

`_no_network` is autouse, so it wraps every test. It refuses, with
`NetworkBlocked`, any socket connection to a non-loopback address and any
name lookup of a non-loopback host, on every path a library can take:

- `socket.socket.connect` and `connect_ex` (blocking sockets, `socketpair`);
- `socket.socket.sendto` and, where the platform has it, `sendmsg` with a
  destination address (connectionless UDP datagrams, Step 5C F4);
- `socket.getaddrinfo`, `gethostbyname`, `gethostbyname_ex` and
  `gethostbyaddr` (DNS resolution and reverse lookups, also used by asyncio
  and httpx; Step 5C F4 adds the three legacy resolver calls);
- `socket.getnameinfo` and the asyncio `loop.getnameinfo` (reverse lookups
  by socket address, Step 5C.1 L-2);
- `sock_connect` and, where present, `sock_sendto` of the asyncio selector
  and proactor event loops (the Windows proactor connects through
  `ConnectEx` and never calls `socket.connect`).

Loopback (`127.0.0.0/8`, `::1`, `localhost`) stays allowed, because asyncio's
self-pipe on Windows is a loopback socket pair; Unix-domain sockets are
allowed too. Unit tests reach Groq only through `httpx.MockTransport`.
"""

from __future__ import annotations

import asyncio.base_events
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

    # Legacy resolver calls (Step 5C F4): forward lookups by name, reverse lookups by address.
    for name in ("gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        real_lookup = getattr(socket, name, None)
        if real_lookup is None:
            continue

        def make_lookup(real: Any) -> Any:
            def lookup(host: Any, *args: Any, **kwargs: Any) -> Any:
                if not is_loopback_host(host):
                    raise _refuse(host)
                return real(host, *args, **kwargs)

            return lookup

        monkeypatch.setattr(socket, name, make_lookup(real_lookup))

    # Reverse lookups by socket address (Step 5C.1 L-2): `getnameinfo((host, port), flags)`.
    real_getnameinfo = socket.getnameinfo

    def getnameinfo(sockaddr: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_loopback_address(sockaddr):
            raise _refuse(sockaddr)
        return real_getnameinfo(sockaddr, *args, **kwargs)

    monkeypatch.setattr(socket, "getnameinfo", getnameinfo)
    real_loop_getnameinfo = asyncio.base_events.BaseEventLoop.getnameinfo

    async def loop_getnameinfo(self: Any, sockaddr: Any, flags: int = 0) -> Any:
        if not is_loopback_address(sockaddr):
            raise _refuse(sockaddr)
        return await real_loop_getnameinfo(self, sockaddr, flags)

    monkeypatch.setattr(asyncio.base_events.BaseEventLoop, "getnameinfo", loop_getnameinfo)

    # Connectionless sends (Step 5C F4): the destination is the last positional argument of
    # `sendto(data[, flags], address)` and the fourth of `sendmsg(buffers[, ancdata[, flags[, address]]])`.
    real_sendto = socket.socket.sendto

    def sendto(self: socket.socket, *args: Any) -> Any:
        if args and not is_loopback_address(args[-1]):
            raise _refuse(args[-1])
        return real_sendto(self, *args)

    monkeypatch.setattr(socket.socket, "sendto", sendto)
    real_sendmsg = getattr(socket.socket, "sendmsg", None)
    if real_sendmsg is not None:  # not available on Windows

        def sendmsg(self: socket.socket, *args: Any) -> Any:
            if len(args) >= 4 and args[3] is not None and not is_loopback_address(args[3]):
                raise _refuse(args[3])
            return real_sendmsg(self, *args)

        monkeypatch.setattr(socket.socket, "sendmsg", sendmsg)

    for loop_class in (asyncio.selector_events.BaseSelectorEventLoop, asyncio.proactor_events.BaseProactorEventLoop):
        real_sock_connect = loop_class.sock_connect

        def make_guard(real: Any) -> Any:
            async def sock_connect(self: Any, sock: socket.socket, address: Any) -> Any:
                if not is_loopback_address(address):
                    raise _refuse(address)
                return await real(self, sock, address)

            return sock_connect

        monkeypatch.setattr(loop_class, "sock_connect", make_guard(real_sock_connect))
        real_sock_sendto = getattr(loop_class, "sock_sendto", None)
        if real_sock_sendto is not None:

            def make_sendto_guard(real: Any) -> Any:
                async def sock_sendto(self: Any, sock: socket.socket, data: Any, address: Any) -> Any:
                    if not is_loopback_address(address):
                        raise _refuse(address)
                    return await real(self, sock, data, address)

                return sock_sendto

            monkeypatch.setattr(loop_class, "sock_sendto", make_sendto_guard(real_sock_sendto))
