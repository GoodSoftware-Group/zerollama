"""Loopback checks for internal-only HTTP routes."""

from __future__ import annotations

LOOPBACK_HOSTS = frozenset({
    "127.0.0.1",
    "::1",
    "localhost",
    "testclient",
    "testserver",
})


def is_loopback_host(host: str) -> bool:
    return host.lower() in LOOPBACK_HOSTS
