"""L3-R10 — HTTP peer pull for content-addressed LMCache slot blobs.

When ``materialize_blob`` misses locally, try ``ZEROLLAMA_LMCACHE_BLOB_PEERS``
(comma-separated Go ``:11434`` or runtime ``:8081`` bases). Peers serve
``GET /api/kv/blob/{digest}`` (Go proxy) or ``GET /kv/blob/{digest}`` (sidecar).

Non-goals: NIXL RDMA, Mooncake page arena — same digest contract, HTTP transport.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_log = logging.getLogger(__name__)

_peer_pull_total = 0
_peer_pull_bytes_total = 0
_peer_pull_fail_total = 0


def reset_lmcache_blob_http_stats_for_tests() -> None:
    global _peer_pull_total, _peer_pull_bytes_total, _peer_pull_fail_total
    _peer_pull_total = 0
    _peer_pull_bytes_total = 0
    _peer_pull_fail_total = 0


def lmcache_blob_http_enabled() -> bool:
    """Serve/pull blobs over HTTP — default on when LMCache blobs are on."""
    from runtime.env import env_bool, env_tri_state
    from runtime.kv.lmcache_blob import lmcache_blobs_enabled

    explicit = env_tri_state("ZEROLLAMA_LMCACHE_BLOB_HTTP")
    if explicit is not None:
        return explicit
    return lmcache_blobs_enabled() and env_bool("ZEROLLAMA_LMCACHE_BLOB_HTTP", default=True)


def lmcache_blob_http_token() -> str:
    return (os.environ.get("ZEROLLAMA_LMCACHE_BLOB_HTTP_TOKEN") or "").strip()


def lmcache_blob_peers() -> list[str]:
    """Peer bases for L3-R10 pull.

    Order: explicit ``ZEROLLAMA_LMCACHE_BLOB_PEERS``, else Go coordination
    ``lmcache_blob_peers``, else ``ZEROLLAMA_FLEET_PEERS`` (same LAN URLs).
    """
    raw = (os.environ.get("ZEROLLAMA_LMCACHE_BLOB_PEERS") or "").strip()
    if not raw:
        from runtime.go_coordination import go_lmcache_blob_peers

        coord = go_lmcache_blob_peers()
        if coord:
            return list(coord)
        raw = (os.environ.get("ZEROLLAMA_FLEET_PEERS") or "").strip()
    if not raw:
        return []
    out: list[str] = []
    for part in raw.split(","):
        p = part.strip().rstrip("/")
        if p:
            out.append(p)
    return out

def validate_blob_digest(digest: str) -> str | None:
    d = (digest or "").strip().lower()
    if len(d) != 64 or any(c not in "0123456789abcdef" for c in d):
        return None
    return d


def local_blob_path(digest: str) -> Path | None:
    from runtime.kv.lmcache_blob import resolve_blob_root

    d = validate_blob_digest(digest)
    if d is None:
        return None
    root = resolve_blob_root()
    if root is None:
        return None
    path = root / d[:2] / f"{d}.bin"
    if path.is_file():
        return path
    return None


def _auth_headers() -> dict[str, str]:
    token = lmcache_blob_http_token()
    if not token:
        return {}
    return {
        "Authorization": f"Bearer {token}",
        "X-Zerollama-Blob-Token": token,
    }


def _peer_urls_for_digest(base: str, digest: str) -> list[str]:
    """Try Go public API first, then runtime sidecar path."""
    d = digest.lower()
    return [
        f"{base}/api/kv/blob/{d}",
        f"{base}/kv/blob/{d}",
    ]


def ingest_blob_bytes(digest: str, data: bytes) -> bool:
    """Write validated bytes into the local blob tree."""
    from runtime.kv.lmcache_blob import resolve_blob_root, sha256_file

    d = validate_blob_digest(digest)
    if d is None or not data:
        return False
    root = resolve_blob_root()
    if root is None:
        return False
    # Verify content matches digest before publishing.
    import hashlib

    if hashlib.sha256(data).hexdigest() != d:
        _log.warning("lmcache blob peer pull digest mismatch want=%s", d[:12])
        return False
    dest = root / d[:2] / f"{d}.bin"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and dest.stat().st_size == len(data):
            if sha256_file(dest) == d:
                return True
        tmp = dest.with_suffix(".bin.tmp")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return True
    except OSError as exc:
        _log.warning("lmcache blob ingest failed digest=%s err=%s", d[:12], exc)
        return False


def fetch_blob_from_peers(digest: str, *, timeout_sec: float = 30.0) -> bool:
    """Pull ``digest`` from configured peers into the local blob root."""
    global _peer_pull_total, _peer_pull_bytes_total, _peer_pull_fail_total

    if not lmcache_blob_http_enabled():
        return False
    d = validate_blob_digest(digest)
    if d is None:
        return False
    if local_blob_path(d) is not None:
        return True
    peers = lmcache_blob_peers()
    if not peers:
        return False

    headers = _auth_headers()
    for base in peers:
        # Skip obviously same-host self pulls when peer is loopback with our path.
        parsed = urlparse(base if "://" in base else f"http://{base}")
        if not parsed.scheme:
            base = f"http://{base}"
        for url in _peer_urls_for_digest(base.rstrip("/"), d):
            req = urllib.request.Request(url, headers=headers, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    if getattr(resp, "status", 200) >= 300:
                        continue
                    data = resp.read()
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
                _log.debug("lmcache blob peer miss url=%s err=%s", url, exc)
                continue
            if ingest_blob_bytes(d, data):
                nbytes = len(data)
                _peer_pull_total += 1
                _peer_pull_bytes_total += nbytes
                _log.info(
                    "lmcache blob peer pull ok digest=%s bytes=%d url=%s",
                    d[:12],
                    nbytes,
                    url,
                )
                return True
            _peer_pull_fail_total += 1
    _peer_pull_fail_total += 1
    return False


def blob_http_health() -> dict[str, Any]:
    return {
        "enabled": lmcache_blob_http_enabled(),
        "peers": lmcache_blob_peers(),
        "token_required": bool(lmcache_blob_http_token()),
        "peer_pull_total": _peer_pull_total,
        "peer_pull_bytes_total": _peer_pull_bytes_total,
        "peer_pull_fail_total": _peer_pull_fail_total,
        "note": (
            "GET /kv/blob/{digest} (runtime) or /api/kv/blob/{digest} (Go); "
            "set ZEROLLAMA_LMCACHE_BLOB_PEERS for cold-node pull"
            if lmcache_blob_http_enabled()
            else "off — ZEROLLAMA_LMCACHE_BLOB_HTTP=0"
        ),
    }
