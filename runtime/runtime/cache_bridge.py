"""Prompt cache key → llama-server slot bridge (ROADMAP borrowings L3).

WHY this module exists
----------------------
Phase 15 assigns dynamic ``id_slot`` values and releases them when a request
finishes, so llama-server drops prefix KV every turn. Agent workloads resend the
same system prompt repeatedly — effective latency is prefill-bound, not decode
tok/s. L1/L2 do not fix that.

WHY not gpu_profiles.py
-----------------------
GPU JSON is per-hardware (-b, -np, cache types). Cache keys are per-session and
arrive on every request ``options`` blob. Mixing them would reload profile logic
on each chat turn.

WHY subprocess-first
--------------------
``--slot-save-path`` and ``cache_prompt`` are llama-server HTTP features.
In-process backends get pinned slots for in-RAM reuse; disk parity is future work.

Ports eliza-v3 ``cache-bridge.ts``: stable cache keys hash to ``id_slot`` in
``[0, parallel)`` so repeat prefixes reuse in-RAM KV. Optional ``--slot-save-path``
persists slot state on disk with TTL eviction by mtime.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Literal

SlotCacheTtlClass = Literal["short", "long", "extended"]

DEFAULT_CACHE_TTLS: dict[str, int] = {
    "short": 5 * 60 * 1000,
    "long": 60 * 60 * 1000,
    "extended": 24 * 60 * 60 * 1000,
}


def llama_cache_enabled() -> bool:
    raw = os.environ.get("ZEROLLAMA_LLAMA_CACHE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def llama_cache_root() -> Path:
    override = os.environ.get("ZEROLLAMA_LLAMA_CACHE_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        return Path(xdg) / "zerollama" / "llama-cache"
    return Path.home() / ".cache" / "zerollama" / "llama-cache"


def cache_root(model_hash: str) -> Path:
    if not model_hash:
        raise ValueError("cache_root requires non-empty model_hash")
    return llama_cache_root() / model_hash


def slot_save_path(model_hash: str) -> Path:
    return cache_root(model_hash)


def build_model_hash(
    *,
    target_model_path: str | Path,
    drafter_model_path: str | Path | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    extra: str | None = None,
) -> str:
    h = hashlib.sha256()
    h.update(str(target_model_path).encode())
    h.update(b"\x01")
    h.update(str(drafter_model_path or "").encode())
    h.update(b"\x01")
    h.update(str(cache_type_k or "").encode())
    h.update(b"\x01")
    h.update(str(cache_type_v or "").encode())
    h.update(b"\x01")
    h.update(str(extra or "").encode())
    return h.hexdigest()[:16]


def derive_slot_id(prompt_cache_key: str, parallel: int) -> int:
    """Map cache key → llama-server slot in [0, parallel); -1 when disabled.

    WHY hash mod parallel: O(1), no registry, matches eliza-v3. Same key always
    lands on the same slot so llama-server can reuse prefix KV across turns.
    """
    if not llama_cache_enabled():
        return -1
    try:
        n = int(parallel)
    except (TypeError, ValueError):
        return -1
    if n <= 0 or not prompt_cache_key:
        return -1
    if n == 1:
        return 0
    digest = hashlib.sha256(prompt_cache_key.encode()).digest()
    value = int.from_bytes(digest[:4], "big")
    return value % n


def default_slot_ttl_ms() -> int:
    """Default disk slot TTL (llama-server names files without a class suffix).

    WHY env override: llama-server writes ``slot_<id>_<seq>.bin`` — we cannot
    embed short/long class in those names. One operator-tunable horizon is enough
    for idle session cleanup; eliza-style ``*.short.bin`` names still use class TTLs.
    """
    raw = os.environ.get("ZEROLLAMA_LLAMA_CACHE_TTL_MS", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return DEFAULT_CACHE_TTLS["long"]


def ttl_ms_for_key(
    ttl: SlotCacheTtlClass | None,
    ttls: dict[str, int] | None = None,
) -> int:
    table = ttls or DEFAULT_CACHE_TTLS
    if ttl == "long":
        return table["long"]
    if ttl == "extended":
        return table.get("extended", table["long"])
    return table["short"]


def slot_cache_file_name(base: str, ttl: SlotCacheTtlClass) -> str:
    """Optional eliza-style filename; llama-server writes ``slot_<id>_<seq>.bin`` instead."""
    return f"{base}.{ttl}.bin"


def parse_slot_cache_ttl_class(file_name: str) -> SlotCacheTtlClass | None:
    """Parse ``*.short.bin`` / ``*.long.bin`` suffix; absent for llama-server slot files."""
    without_bin = file_name[:-4] if file_name.endswith(".bin") else file_name
    last_dot = without_bin.rfind(".")
    if last_dot < 0:
        return None
    candidate = without_bin[last_dot + 1 :]
    if candidate in ("short", "long", "extended"):
        return candidate  # type: ignore[return-value]
    return None


def evict_ttl_ms_for_file(
    file_name: str,
    *,
    ttls: dict[str, int] | None = None,
) -> int:
    ttl_class = parse_slot_cache_ttl_class(file_name)
    if ttl_class is not None:
        return ttl_ms_for_key(ttl_class, ttls)
    return default_slot_ttl_ms()


def evict_expired(
    root_dir: Path | str,
    *,
    ttls: dict[str, int] | None = None,
    now_ms: float | None = None,
) -> int:
    """Delete slot files older than TTL (mtime).

    llama-server writes ``slot_<id>_<seq>.bin`` without a class suffix — those use
    ``ZEROLLAMA_LLAMA_CACHE_TTL_MS`` (default 1h). Optional ``*.short.bin`` names
    keep eliza-style per-class horizons.
    """
    root = Path(root_dir)
    if not root.is_dir():
        return 0
    now = time.time() * 1000 if now_ms is None else now_ms
    deleted = 0
    for entry in root.iterdir():
        if not entry.is_file():
            continue
        horizon = evict_ttl_ms_for_file(entry.name, ttls=ttls)
        try:
            mtime_ms = entry.stat().st_mtime * 1000
        except OSError:
            continue
        if now - mtime_ms > horizon:
            try:
                entry.unlink()
                deleted += 1
            except OSError:
                pass
    return deleted


def read_cache_stats(root_dir: Path | str, *, now_ms: float | None = None) -> list[dict[str, Any]]:
    root = Path(root_dir)
    if not root.is_dir():
        return []
    now = time.time() * 1000 if now_ms is None else now_ms
    out: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_file():
            continue
        try:
            st = entry.stat()
        except OSError:
            continue
        out.append(
            {
                "file": entry.name,
                "size_bytes": st.st_size,
                "mtime_ms": st.st_mtime * 1000,
                "age_ms": max(0, now - st.st_mtime * 1000),
            }
        )
    return out


def extract_prompt_cache_key(provider_options: Any) -> str | None:
    if not isinstance(provider_options, dict):
        return None
    eliza = provider_options.get("eliza")
    if isinstance(eliza, dict):
        raw = eliza.get("promptCacheKey")
        if isinstance(raw, str) and raw:
            return raw
    raw = provider_options.get("prompt_cache_key")
    if isinstance(raw, str) and raw:
        return raw
    return None


def extract_prefix_hash(provider_options: Any) -> str | None:
    if not isinstance(provider_options, dict):
        return None
    eliza = provider_options.get("eliza")
    if isinstance(eliza, dict):
        raw = eliza.get("prefixHash")
        if isinstance(raw, str) and raw:
            return raw
    return None


def extract_conversation_id(provider_options: Any) -> str | None:
    if not isinstance(provider_options, dict):
        return None
    eliza = provider_options.get("eliza")
    if isinstance(eliza, dict):
        raw = eliza.get("conversationId")
        if isinstance(raw, str) and raw:
            return raw
    raw = provider_options.get("conversation_id")
    if isinstance(raw, str) and raw:
        return raw
    return None


def hash_stable_prefix(segments: list[dict[str, Any]]) -> str | None:
    if not segments:
        return None
    h = hashlib.sha256()
    consumed = 0
    for seg in segments:
        if not seg.get("stable"):
            break
        content = seg.get("content")
        if not isinstance(content, str):
            break
        h.update(content.encode())
        h.update(b"\x01")
        consumed += 1
    if consumed == 0:
        return None
    return h.hexdigest()[:16]


def extract_annotated_segments(provider_options: Any) -> list[dict[str, Any]] | None:
    if not isinstance(provider_options, dict):
        return None
    eliza = provider_options.get("eliza")
    if not isinstance(eliza, dict):
        return None
    raw = eliza.get("promptSegments")
    if not isinstance(raw, list):
        return None
    out: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            return None
        content = entry.get("content")
        stable = entry.get("stable")
        if not isinstance(content, str) or not isinstance(stable, bool):
            return None
        out.append({"content": content, "stable": stable})
    return out


def resolve_local_cache_key(provider_options: Any) -> str | None:
    """Strongest cache key from options (conversation → prefix → promptCacheKey).

    WHY precedence: conversation id is the natural agent thread key; segment hash
    avoids re-sending stable prefix bytes; explicit promptCacheKey is last resort.
    """
    if not llama_cache_enabled():
        return None
    conv = extract_conversation_id(provider_options)
    if conv:
        return f"conv:{conv}"
    segments = extract_annotated_segments(provider_options)
    if segments:
        hashed = hash_stable_prefix(segments)
        if hashed:
            return f"seg:{hashed}"
    prefix = extract_prefix_hash(provider_options)
    if prefix:
        return f"pfx:{prefix}"
    return extract_prompt_cache_key(provider_options)


def cache_prompt_for_request(prompt_cache_key: str | None) -> bool:
    return bool(prompt_cache_key) and llama_cache_enabled()


def resolve_cache_key_from_options(options: dict[str, Any] | None) -> str | None:
    return resolve_local_cache_key(options) if options else None


def cache_type_from_llama_argv(args: list[str]) -> tuple[str | None, str | None]:
    ck = cv = None
    for i, a in enumerate(args):
        if a == "--cache-type-k" and i + 1 < len(args):
            ck = args[i + 1]
        if a == "--cache-type-v" and i + 1 < len(args):
            cv = args[i + 1]
    return ck, cv


def llama_server_cache_argv(
    model_path: Path,
    llama_server_args: list[str],
    *,
    draft_model: Path | None = None,
) -> list[str]:
    """``--slot-save-path`` argv when disk cache is enabled.

    WHY evict on start: crash leaves stale slot blobs; directory is small (≈ np files).
    Sync scan is acceptable on cold llama-server boot — not on hot request path.
    """
    if not llama_cache_enabled():
        return []
    ck, cv = cache_type_from_llama_argv(llama_server_args)
    model_hash = build_model_hash(
        target_model_path=model_path,
        drafter_model_path=draft_model,
        cache_type_k=ck,
        cache_type_v=cv,
    )
    save_dir = slot_save_path(model_hash)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Sync scan before llama-server start; small dir expected (one entry per slot).
    evict_expired(save_dir)
    return ["--slot-save-path", str(save_dir)]


def cache_health(
    model_path: Path | None,
    llama_server_args: list[str],
    *,
    draft_model: Path | None = None,
) -> dict[str, Any]:
    """``/health.llama_cache`` snapshot.

    WHY hash before file exists: operators set LLAMA_MODEL before pull; slot dir
    layout should be stable for pre-warm and monitoring.
    """
    enabled = llama_cache_enabled()
    out: dict[str, Any] = {
        "enabled": enabled,
        "root": str(llama_cache_root()),
        "default_ttl_ms": default_slot_ttl_ms(),
    }
    if not enabled or model_path is None:
        return out
    out["model_path"] = str(model_path)
    out["model_loaded"] = model_path.is_file()
    ck, cv = cache_type_from_llama_argv(llama_server_args)
    model_hash = build_model_hash(
        target_model_path=model_path,
        drafter_model_path=draft_model,
        cache_type_k=ck,
        cache_type_v=cv,
    )
    save_dir = slot_save_path(model_hash)
    out["model_hash"] = model_hash
    out["slot_save_path"] = str(save_dir)
    out["files"] = read_cache_stats(save_dir)
    out["file_count"] = len(out["files"])
    return out
