"""Utilities for reserving local TCP ports for short-lived services.

The allocator keeps sockets open until the caller is ready to start the target
service.  This narrows the usual "probe then bind" race window while still
letting the OS choose ports instead of relying on fixed ranges.
"""

from __future__ import annotations

import contextlib
import ipaddress
import socket
from dataclasses import dataclass, field
from typing import Iterable


def _normalize_host(host: str | None) -> str:
    if not host:
        return ""
    host = host.strip()
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


def _socket_family(host: str) -> socket.AddressFamily:
    try:
        if ipaddress.ip_address(host).version == 6:
            return socket.AF_INET6
    except ValueError:
        pass
    return socket.AF_INET


def _bind_socket(host: str, port: int) -> socket.socket:
    family = _socket_family(host)
    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
    except Exception:
        sock.close()
        raise
    return sock


@dataclass
class PortLease:
    """A reserved TCP port or consecutive port range.

    The lease owns bound sockets.  Call :meth:`release` immediately before the
    target service binds the same port(s), or during cleanup if startup fails.
    """

    host: str
    ports: tuple[int, ...]
    role: str = ""
    sockets: list[socket.socket] = field(default_factory=list, repr=False)
    released: bool = False

    @property
    def port(self) -> int:
        return self.ports[0]

    def release(self) -> None:
        if self.released:
            return
        for sock in self.sockets:
            with contextlib.suppress(Exception):
                sock.close()
        self.sockets.clear()
        self.released = True

    def __enter__(self) -> "PortLease":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()


def reserve_ports(host: str | None, *, count: int = 1, role: str = "", max_attempts: int = 256) -> PortLease:
    """Reserve one TCP port or a consecutive TCP port range on *host*.

    ``count=1`` binds to port 0 and lets the OS choose an available port.
    ``count>1`` first asks the OS for a start port, then tries to reserve the
    following ports consecutively.  This keeps the caller off fixed ranges while
    still supporting components that need adjacent helper ports.
    """

    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")

    bind_host = _normalize_host(host)
    for _ in range(max_attempts):
        sockets: list[socket.socket] = []
        try:
            first = _bind_socket(bind_host, 0)
            sockets.append(first)
            start_port = first.getsockname()[1]
            ports = [start_port]

            for offset in range(1, count):
                port = start_port + offset
                if port > 65535:
                    raise OSError(f"consecutive port range exceeds 65535: start={start_port}, count={count}")
                sockets.append(_bind_socket(bind_host, port))
                ports.append(port)

            return PortLease(host=bind_host, ports=tuple(ports), role=role, sockets=sockets)
        except OSError:
            for sock in sockets:
                with contextlib.suppress(Exception):
                    sock.close()
            continue

    raise RuntimeError(f"Could not reserve {count} consecutive port(s) on {bind_host!r} for role={role!r}")


def release_port_leases(leases: Iterable[PortLease]) -> None:
    for lease in leases:
        lease.release()
