import socket

import pytest

from slime.utils.port_allocator import reserve_ports


def _can_bind(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def test_reserve_port_blocks_until_release():
    lease = reserve_ports("127.0.0.1", role="test")
    try:
        assert not _can_bind("127.0.0.1", lease.port)
    finally:
        lease.release()

    assert _can_bind("127.0.0.1", lease.port)


def test_reserve_consecutive_ports_and_idempotent_release():
    lease = reserve_ports("127.0.0.1", count=3, role="test-range")
    try:
        assert lease.ports == tuple(range(lease.port, lease.port + 3))
        assert all(not _can_bind("127.0.0.1", port) for port in lease.ports)
    finally:
        lease.release()
        lease.release()

    assert all(_can_bind("127.0.0.1", port) for port in lease.ports)


def test_reserve_ports_rejects_invalid_count():
    with pytest.raises(ValueError):
        reserve_ports("127.0.0.1", count=0)
