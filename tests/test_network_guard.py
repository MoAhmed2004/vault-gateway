"""Unit tests for the suite-wide network guard in `tests/conftest.py` (§5.5: zero real network calls).

The refused addresses are documentation ranges (TEST-NET-3, RFC 5737; and
RFC 3849 for IPv6): the guard raises before any packet is sent.
"""

from __future__ import annotations

import asyncio
import socket
import traceback

import httpx
import pytest

from tests.conftest import NetworkBlocked, is_loopback_address, is_loopback_host

REMOTE_V4 = ("203.0.113.7", 443)
REMOTE_V6 = ("2001:db8::7", 443, 0, 0)


@pytest.mark.parametrize("host", ["127.0.0.1", "127.8.9.10", "::1", "[::1]", "localhost", "LOCALHOST", None, ""])
def test_loopback_hosts_are_allowed(host):
    assert is_loopback_host(host)


@pytest.mark.parametrize("host", ["203.0.113.7", "2001:db8::7", "api.groq.com", "example.org", "0.0.0.1"])
def test_remote_hosts_are_not_loopback(host):
    assert not is_loopback_host(host)


def test_unix_socket_paths_are_allowed():
    assert is_loopback_address("/tmp/sock") and is_loopback_address(b"/tmp/sock")


@pytest.mark.parametrize("address", [REMOTE_V4])
def test_blocking_connect_to_a_remote_address_is_refused(address):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(NetworkBlocked):
            sock.connect(address)
        with pytest.raises(NetworkBlocked):
            sock.connect_ex(address)


def test_create_connection_and_dns_lookups_are_refused():
    with pytest.raises(NetworkBlocked):
        socket.create_connection(REMOTE_V4, timeout=0.5)
    with pytest.raises(NetworkBlocked):
        socket.getaddrinfo("api.groq.com", 443)


def test_loopback_connections_still_work():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=2.0) as client:
            accepted, _ = server.accept()
            with accepted:
                client.sendall(b"ping")
                assert accepted.recv(4) == b"ping"


def test_async_connections_are_refused_on_the_event_loop():
    async def go() -> None:
        loop = asyncio.get_running_loop()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setblocking(False)
            with pytest.raises(NetworkBlocked):
                await loop.sock_connect(sock, REMOTE_V4)
        with pytest.raises(NetworkBlocked):
            await asyncio.open_connection(*REMOTE_V4)

    asyncio.run(go())


def test_a_real_httpx_transport_cannot_reach_the_network():
    """httpx (through anyio) may wrap the refusal in a ConnectError or an exception group; the cause is the guard."""

    async def go() -> None:
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(retries=0), timeout=2.0) as client:
            with pytest.raises((NetworkBlocked, httpx.ConnectError, BaseExceptionGroup)) as info:
                await client.get("https://203.0.113.7/v1/models")
            assert "blocked during tests" in "".join(traceback.format_exception(info.value))

    asyncio.run(go())
