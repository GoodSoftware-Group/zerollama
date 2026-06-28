"""Optional LMCache-style object tier for prefix block metadata (env-gated).

WHY: vLLM multi-tier KV offloads block metadata and blobs to object storage.
Zerollama L3 keeps llama-server slot blobs as the warm tier; this module adds
an optional cold tier for hash-chained block index persistence across restarts
and fleet nodes (metadata first — blob paths point at existing slot files).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from runtime.env import lmcache_tier_enabled, lmcache_uri


@dataclass(frozen=True)
class LMCacheBlockRecord:
    block_hash: str
    parent_hash: str
    block_index: int
    token_end: int
    model_scope: str
    session_key: str | None
    slot_id: int | None
    blob_path: str | None = None
    updated_at_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LMCacheBlockRecord:
        return cls(
            block_hash=str(raw["block_hash"]),
            parent_hash=str(raw["parent_hash"]),
            block_index=int(raw["block_index"]),
            token_end=int(raw["token_end"]),
            model_scope=str(raw["model_scope"]),
            session_key=raw.get("session_key"),
            slot_id=int(raw["slot_id"]) if raw.get("slot_id") is not None else None,
            blob_path=raw.get("blob_path"),
            updated_at_ms=int(raw.get("updated_at_ms") or 0),
        )


class LMCacheTierStore:
    """Filesystem-backed LMCache tier (``file://`` URI)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, model_scope: str, block_hash: str) -> Path:
        scope_dir = self.root / _safe_dir_name(model_scope)
        scope_dir.mkdir(parents=True, exist_ok=True)
        return scope_dir / f"{block_hash}.json"

    def put(self, record: LMCacheBlockRecord) -> None:
        path = self._meta_path(record.model_scope, record.block_hash)
        tmp = path.with_suffix(".json.tmp")
        payload = record.to_dict()
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(path)

    def get(self, *, model_scope: str, block_hash: str) -> LMCacheBlockRecord | None:
        path = self._meta_path(model_scope, block_hash)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        try:
            rec = LMCacheBlockRecord.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            return None
        if rec.model_scope != model_scope or rec.block_hash != block_hash:
            return None
        return rec

    def delete(self, *, model_scope: str, block_hash: str) -> bool:
        path = self._meta_path(model_scope, block_hash)
        try:
            if path.is_file():
                path.unlink()
                return True
        except OSError:
            pass
        return False

    def health(self) -> dict[str, Any]:
        count = 0
        try:
            for scope_dir in self.root.iterdir():
                if scope_dir.is_dir():
                    count += sum(1 for p in scope_dir.glob("*.json") if p.is_file())
        except OSError:
            pass
        return {
            "enabled": True,
            "backend": "filesystem",
            "root": str(self.root),
            "record_count": count,
            "uri": lmcache_uri(),
        }


class NoOpLMCacheTier:
    def put(self, record: LMCacheBlockRecord) -> None:
        return None

    def get(self, *, model_scope: str, block_hash: str) -> LMCacheBlockRecord | None:
        return None

    def delete(self, *, model_scope: str, block_hash: str) -> bool:
        return False

    def health(self) -> dict[str, Any]:
        return {"enabled": False, "backend": "none", "uri": lmcache_uri()}


_TIER_LOCK = threading.Lock()
_TIER: LMCacheTierStore | NoOpLMCacheTier | None = None


def _safe_dir_name(scope: str) -> str:
    import hashlib

    return hashlib.sha256(scope.encode()).hexdigest()[:32]


def _resolve_file_root(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        path = parsed.path if parsed.scheme == "file" else uri
        if parsed.scheme == "file" and not path and parsed.netloc:
            path = parsed.netloc
        return Path(path).expanduser()
    raise ValueError(f"unsupported LMCache URI scheme: {parsed.scheme}")


def lmcache_tier() -> LMCacheTierStore | NoOpLMCacheTier:
    global _TIER
    if _TIER is not None:
        return _TIER
    with _TIER_LOCK:
        if _TIER is not None:
            return _TIER
        if not lmcache_tier_enabled():
            _TIER = NoOpLMCacheTier()
        else:
            _TIER = LMCacheTierStore(_resolve_file_root(lmcache_uri()))
        return _TIER


def reset_lmcache_tier_for_tests() -> None:
    global _TIER
    with _TIER_LOCK:
        _TIER = None
