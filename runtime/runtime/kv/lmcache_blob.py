"""L3-R7 — content-addressed LMCache slot-blob federation (shared FS / file root).

WHY: L3-R4 Redis shares *metadata* only; ``blob_path`` pointed at a donor node's
local ``slot_*.bin`` and was never pulled. Cold nodes could hydrate hashes but
still full-prefill. This module copies slot blobs into a content-addressed tree
keyed by SHA-256 digest so any node with the blob root (NFS, shared ``file://``
LMCache root, or ``ZEROLLAMA_LMCACHE_BLOB_ROOT``) can materialize and restore.

Non-goals: NIXL RDMA, Mooncake page arena, S3 — those plug in behind the same
digest contract later. L3-R10 adds HTTP peer pull as a LAN transport.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_log = logging.getLogger(__name__)

_publish_total = 0
_materialize_total = 0
_publish_bytes_total = 0
_LOCK = threading.Lock()


def reset_lmcache_blob_stats_for_tests() -> None:
    global _publish_total, _materialize_total, _publish_bytes_total
    with _LOCK:
        _publish_total = 0
        _materialize_total = 0
        _publish_bytes_total = 0


def lmcache_blobs_enabled() -> bool:
    """Blob publish/pull — default on when LMCache tier is configured."""
    from runtime.env import env_bool, env_tri_state, lmcache_tier_enabled

    explicit = env_tri_state("ZEROLLAMA_LMCACHE_BLOBS")
    if explicit is not None:
        return explicit
    if not lmcache_tier_enabled():
        return False
    return env_bool("ZEROLLAMA_LMCACHE_BLOBS", default=True)


def resolve_blob_root() -> Path | None:
    """Shared blob tree root, or None when blobs cannot be stored."""
    override = os.environ.get("ZEROLLAMA_LMCACHE_BLOB_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    from runtime.env import lmcache_uri

    uri = (lmcache_uri() or "").strip()
    if not uri:
        return None
    parsed = urlparse(uri)
    scheme = parsed.scheme or "file"
    if scheme in ("", "file"):
        path = parsed.path if parsed.scheme == "file" else uri
        if parsed.scheme == "file" and not path and parsed.netloc:
            path = parsed.netloc
        return Path(path).expanduser() / "blobs"
    # redis:// — need explicit blob root (shared FS); metadata alone is not enough.
    return None


def _digest_path(root: Path, digest: str) -> Path:
    d = digest.lower().strip()
    if len(d) < 4 or any(c not in "0123456789abcdef" for c in d):
        raise ValueError(f"invalid blob digest: {digest!r}")
    return root / d[:2] / f"{d}.bin"


def sha256_file(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def publish_slot_blob(src: Path | str) -> str | None:
    """Copy ``src`` into the blob tree; return content digest or None."""
    global _publish_total, _publish_bytes_total
    if not lmcache_blobs_enabled():
        return None
    root = resolve_blob_root()
    if root is None:
        return None
    src_path = Path(src)
    if not src_path.is_file():
        return None
    try:
        digest = sha256_file(src_path)
        dest = _digest_path(root, digest)
        if dest.is_file() and dest.stat().st_size == src_path.stat().st_size:
            return digest
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".bin.tmp")
        shutil.copy2(src_path, tmp)
        tmp.replace(dest)
        size = dest.stat().st_size
        with _LOCK:
            _publish_total += 1
            _publish_bytes_total += size
        return digest
    except OSError as exc:
        _log.warning("lmcache blob publish failed path=%s err=%s", src_path, exc)
        return None


def materialize_blob(digest: str, dest: Path | str) -> bool:
    """Copy blob ``digest`` to ``dest`` (slot path). Returns True on success.

    On local miss, attempts L3-R10 HTTP peer pull when peers are configured.
    """
    global _materialize_total
    if not lmcache_blobs_enabled():
        return False
    root = resolve_blob_root()
    if root is None:
        return False
    try:
        src = _digest_path(root, digest)
    except ValueError:
        return False
    if not src.is_file():
        from runtime.kv.lmcache_blob_http import fetch_blob_from_peers

        if not fetch_blob_from_peers(digest):
            return False
        if not src.is_file():
            return False
    dest_path = Path(dest)
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.is_file() and dest_path.stat().st_size == src.stat().st_size:
            if sha256_file(dest_path) == digest.lower():
                return True
        tmp = dest_path.with_suffix(dest_path.suffix + ".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dest_path)
        with _LOCK:
            _materialize_total += 1
        return True
    except OSError as exc:
        _log.warning(
            "lmcache blob materialize failed digest=%s dest=%s err=%s",
            digest[:12],
            dest_path,
            exc,
        )
        return False


def blob_store_health() -> dict[str, Any]:
    root = resolve_blob_root()
    enabled = lmcache_blobs_enabled()
    count = 0
    if root is not None and root.is_dir():
        try:
            count = sum(1 for _ in root.rglob("*.bin") if _.is_file())
        except OSError:
            count = -1
    from runtime.kv.lmcache_blob_http import blob_http_health

    return {
        "enabled": enabled,
        "root": str(root) if root else None,
        "blob_count": count,
        "publish_total": _publish_total,
        "materialize_total": _materialize_total,
        "publish_bytes_total": _publish_bytes_total,
        "http": blob_http_health(),
        "note": (
            "content-addressed slot blobs; shared FS and/or L3-R10 HTTP peer pull "
            "(ZEROLLAMA_LMCACHE_BLOB_PEERS); set ZEROLLAMA_LMCACHE_BLOB_ROOT for redis://"
            if enabled
            else "off — set ZEROLLAMA_LMCACHE_URI (+ BLOB_ROOT for redis)"
        ),
    }
