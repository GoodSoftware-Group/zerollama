"""Loopback-safe base URL for in-process runtime → Go /internal/* calls."""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse, urlunparse


def connectable_go_base_url() -> str:
    """Base URL for runtime clients calling Go (cross-queue-seq, render-chat, …).

    Prefer ``ZEROLLAMA_GO_URL``. Otherwise map ``OLLAMA_HOST`` bind addresses
    (0.0.0.0 / ::) to loopback so ``internalLoopbackOnly`` accepts the request.
    Mirrors ``envconfig.ConnectableHost`` in Go.
    """
    explicit = os.environ.get("ZEROLLAMA_GO_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")

    raw = os.environ.get("OLLAMA_HOST", "").strip() or "127.0.0.1:11434"
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80 if scheme == "http" else 11434

    try:
        ip = ipaddress.ip_address(host)
        if ip.is_unspecified:
            host = "127.0.0.1" if ip.version == 4 else "::1"
    except ValueError:
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1" if host == "0.0.0.0" else "::1"

    netloc = f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"
    return urlunparse((scheme, netloc, "", "", "", "")).rstrip("/")
