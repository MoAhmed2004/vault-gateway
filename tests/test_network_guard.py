"""Unit tests for the suite-wide network guard in `tests/conftest.py` (§5.5: zero real network calls).

The refused addresses are documentation ranges (TEST-NET-3, RFC 5737; and
RFC 3849 for IPv6): the guard raises before any packet is sent.
"""

from __future__ import annotations

import asyncio
import ipaddress
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


@pytest.mark.parametrize("name", ["gethostbyname", "gethostbyname_ex"])
@pytest.mark.parametrize("host", ["api.groq.com", "203.0.113.7", "example.org"])
def test_legacy_forward_lookups_of_remote_hosts_are_refused(name, host):
    """Step 5C F4: the legacy resolver calls bypass getaddrinfo on some platforms, so they are guarded too."""
    with pytest.raises(NetworkBlocked):
        getattr(socket, name)(host)


def test_reverse_lookup_of_a_remote_address_is_refused():
    with pytest.raises(NetworkBlocked):
        socket.gethostbyaddr("203.0.113.7")


@pytest.mark.parametrize("sockaddr", [("203.0.113.7", 443), ("2001:db8::7", 443, 0, 0), ("api.groq.com", 443)])
def test_getnameinfo_of_a_remote_address_is_refused(sockaddr):
    """Step 5C.1 L-2: a reverse lookup by socket address is a DNS query too."""
    with pytest.raises(NetworkBlocked):
        socket.getnameinfo(sockaddr, socket.NI_NUMERICSERV)


def test_getnameinfo_of_loopback_still_works():
    host, port = socket.getnameinfo(("127.0.0.1", 8080), socket.NI_NUMERICHOST | socket.NI_NUMERICSERV)
    assert (host, port) == ("127.0.0.1", "8080")


def test_async_getnameinfo_is_refused_on_the_event_loop():
    async def go() -> None:
        loop = asyncio.get_running_loop()
        with pytest.raises(NetworkBlocked):
            await loop.getnameinfo(("203.0.113.7", 443))
        host, _ = await loop.getnameinfo(("127.0.0.1", 80), socket.NI_NUMERICHOST | socket.NI_NUMERICSERV)
        assert host == "127.0.0.1"

    asyncio.run(go())


@pytest.mark.parametrize("name", ["gethostbyname", "gethostbyname_ex"])
def test_legacy_forward_lookups_of_loopback_still_work(name):
    result = getattr(socket, name)("localhost")
    addresses = [result] if isinstance(result, str) else result[2]
    assert addresses and all(ipaddress.ip_address(a).is_loopback for a in addresses)


def test_udp_sendto_a_remote_address_is_refused():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(NetworkBlocked):
            sock.sendto(b"\x00" * 12, ("203.0.113.7", 53))
        with pytest.raises(NetworkBlocked):
            sock.sendto(b"\x00" * 12, 0, ("203.0.113.7", 53))  # the (data, flags, address) form


def test_udp_sendto_loopback_still_works():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        server.bind(("127.0.0.1", 0))
        server.settimeout(2.0)
        assert client.sendto(b"ping", server.getsockname()) == 4
        assert server.recvfrom(16)[0] == b"ping"


@pytest.mark.skipif(not hasattr(socket.socket, "sendmsg"), reason="socket.sendmsg is not available on this platform")
def test_udp_sendmsg_to_a_remote_address_is_refused():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(NetworkBlocked):
            sock.sendmsg([b"\x00" * 12], [], 0, ("203.0.113.7", 53))


def test_async_udp_sendto_is_refused_on_the_event_loop():
    async def go() -> None:
        loop = asyncio.get_running_loop()
        if not hasattr(loop, "sock_sendto"):
            pytest.skip("this event loop has no sock_sendto")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setblocking(False)
            with pytest.raises(NetworkBlocked):
                await loop.sock_sendto(sock, b"\x00" * 12, ("203.0.113.7", 53))

    asyncio.run(go())


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
