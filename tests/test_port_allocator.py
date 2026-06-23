import socket

import pytest

from slime.utils import port_allocator
from slime.utils.port_allocator import reserve_ports

NUM_GPUS = 0


class _FakeSocket:
    def __init__(self, port: int):
        self._port = port
        self.closed = False

    def getsockname(self):
        return "127.0.0.1", self._port

    def close(self):
        self.closed = True


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


def test_reserve_ports_retries_ports_above_max(monkeypatch):
    created = []
    start_ports = iter([63677, 55535])

    def fake_bind_socket(_host: str, port: int):
        sock = _FakeSocket(next(start_ports) if port == 0 else port)
        created.append(sock)
        return sock

    monkeypatch.setattr(port_allocator, "_bind_socket", fake_bind_socket)

    lease = reserve_ports(
        "127.0.0.1",
        role="sglang",
        max_port=55535,
        max_attempts=2,
    )
    try:
        assert lease.port == 55535
        assert lease.ports == (55535,)
    finally:
        lease.release()

    assert created[0].closed
    assert created[-1].closed


def test_reserve_ports_rejects_invalid_max_port():
    with pytest.raises(ValueError):
        reserve_ports("127.0.0.1", max_port=65536)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
