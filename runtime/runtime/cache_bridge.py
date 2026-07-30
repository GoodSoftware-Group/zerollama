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
In-process backends use pinned slots for in-RAM reuse (Phase 15 v17) and optional
``llama_state_seq_{save,load}_file`` disk blobs under the same ``model_hash`` dirs.

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
    from runtime.env import llama_cache_enabled as _enabled

    return _enabled()


def inprocess_disk_cache_enabled(*, backend: str | None = None) -> bool:
    """In-process slot save/load under ``llama_cache_root/<modelHash>/``.

    WHY separate from ``llama_cache_enabled``: operators can keep RAM prefix reuse
    (Phase 15 v17) while disabling disk I/O on latency-sensitive Metal paths.
    When env is unset, default follows platform + backend (see ``runtime.env``).
    """
    from runtime.env import llama_cache_disk_enabled

    return llama_cache_disk_enabled(backend=backend)


def slot_cache_filename(slot_id: int, seq: int = 0) -> str:
    """llama-server / in-process slot blob name under ``slot_save_path``."""
    return f"slot_{int(slot_id)}_{int(seq)}.bin"


def slot_cache_file_path(model_hash: str, slot_id: int, seq: int = 0) -> Path:
    return slot_save_path(model_hash) / slot_cache_filename(slot_id, seq)


def prepare_slot_cache_dir(
    model_hash: str,
    *,
    evict: bool = False,
    slot_bin_ttl_ms: int | None = None,
) -> Path:
    """Ensure save dir exists.

    ``evict=True`` sweeps expired blobs — pass this only on session start, not on
    every save call, to avoid per-turn directory scans on latency-sensitive Metal paths.
    """
    save_dir = slot_save_path(model_hash)
    save_dir.mkdir(parents=True, exist_ok=True)
    if evict:
        evict_expired(save_dir, slot_bin_ttl_ms=slot_bin_ttl_ms)
    return save_dir


def llama_cache_root() -> Path:
    from runtime.env import llama_cache_root as _root

    return _root()


def cache_root(model_hash: str) -> Path:
    if not model_hash:
        raise ValueError("cache_root requires non-empty model_hash")
    return llama_cache_root() / model_hash


def slot_save_path(model_hash: str) -> Path:
    return cache_root(model_hash)


def _canonical_model_path(path: str | Path) -> str:
    """Stable path for ``build_model_hash`` (symlink target, expanduser).

    WHY: same GGUF via symlink vs absolute path must share one slot-save dir.
    """
    p = Path(path).expanduser()
    try:
        if p.is_file():
            return str(p.resolve())
    except OSError:
        pass
    return str(p)


def build_model_hash(
    *,
    target_model_path: str | Path,
    drafter_model_path: str | Path | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    extra: str | None = None,
) -> str:
    """Fingerprint for ``--slot-save-path`` subdirectory.

    WHY mix path + draft + cache types: L2 fork profiles change KV layout;
    reusing one directory would load incompatible slot blobs after a switch.
    WHY canonical path: symlinks must not fragment cache across duplicate dirs.
    """
    h = hashlib.sha256()
    h.update(_canonical_model_path(target_model_path).encode())
    h.update(b"\x01")
    draft = drafter_model_path
    h.update(_canonical_model_path(draft).encode() if draft else b"")
    h.update(b"\x01")
    h.update(str(cache_type_k or "").encode())
    h.update(b"\x01")
    h.update(str(cache_type_v or "").encode())
    h.update(b"\x01")
    h.update(str(extra or "").encode())
    return h.hexdigest()[:16]


def derive_slot_id(
    prompt_cache_key: str,
    parallel: int,
    *,
    cache_salt: str | None = None,
) -> int:
    """Map cache key → llama-server slot in [0, parallel); -1 when disabled.

    WHY hash mod parallel: O(1), no registry, matches eliza-v3. Same key always
    lands on the same slot so llama-server can reuse prefix KV across turns.
    ``cache_salt`` prevents cross-tenant slot collisions when keys collide.
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
    material = prefix_cache_hash_material(prompt_cache_key, cache_salt)
    digest = hashlib.sha256(material.encode()).digest()
    value = int.from_bytes(digest[:4], "big")
    return value % n


def default_slot_ttl_ms() -> int:
    """Default disk slot TTL (llama-server names files without a class suffix).

    WHY env override: llama-server writes ``slot_<id>_<seq>.bin`` — we cannot
    embed short/long class in those names. One operator-tunable horizon is enough
    for idle session cleanup; eliza-style ``*.short.bin`` names still use class TTLs.
    """
    from runtime.env import default_slot_ttl_ms as _ttl

    return _ttl(default_ms=DEFAULT_CACHE_TTLS["long"])


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
    slot_bin_ttl_ms: int | None = None,
) -> int:
    ttl_class = parse_slot_cache_ttl_class(file_name)
    if ttl_class is not None:
        return ttl_ms_for_key(ttl_class, ttls)
    if slot_bin_ttl_ms is not None:
        return slot_bin_ttl_ms
    return default_slot_ttl_ms()


def evict_expired(
    root_dir: Path | str,
    *,
    ttls: dict[str, int] | None = None,
    now_ms: float | None = None,
    slot_bin_ttl_ms: int | None = None,
) -> int:
    """Delete slot files older than TTL (mtime).

    llama-server writes ``slot_<id>_<seq>.bin`` without a class suffix — those use
    ``ZEROLLAMA_LLAMA_CACHE_TTL_MS`` (default 1h). Optional ``*.short.bin`` names
    keep eliza-style per-class horizons.

    Active ``/api/cache/pin`` leases (via ``/internal/cache/pin``) extend TTL for
    matching slot ids — see ``runtime.cache_pins``.
    """
    from runtime.cache_pins import pin_ttl_ms_for_file

    root = Path(root_dir)
    if not root.is_dir():
        return 0
    now = time.time() * 1000 if now_ms is None else now_ms
    deleted = 0
    for entry in root.iterdir():
        if not entry.is_file():
            continue
        horizon = evict_ttl_ms_for_file(
            entry.name, ttls=ttls, slot_bin_ttl_ms=slot_bin_ttl_ms
        )
        horizon = pin_ttl_ms_for_file(entry.name, horizon)
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


def evict_orphaned_cache_dirs(
    *,
    keep_model_hash: str | None = None,
    ttls: dict[str, int] | None = None,
    now_ms: float | None = None,
) -> int:
    """Remove empty sibling model-hash dirs under ``llama_cache_root``.

    WHY: switching fork/stock or cache types creates new hashes; expired slot
    files are evicted per-dir and empty directories are deleted on startup.
    """
    root = llama_cache_root()
    if not root.is_dir():
        return 0
    removed = 0
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if keep_model_hash and entry.name == keep_model_hash:
            continue
        evict_expired(entry, ttls=ttls, now_ms=now_ms)
        try:
            if not any(entry.iterdir()):
                entry.rmdir()
                removed += 1
        except OSError:
            pass
    return removed


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


def extract_cache_salt(provider_options: Any) -> str | None:
    """Tenant isolation salt for L3 slot derivation (vLLM ``cache_salt`` analog)."""
    from runtime.env import default_cache_salt

    env = default_cache_salt()
    if not isinstance(provider_options, dict):
        return env or None
    eliza = provider_options.get("eliza")
    if isinstance(eliza, dict):
        raw = eliza.get("cacheSalt")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    raw = provider_options.get("cache_salt")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return env or None


def prefix_cache_hash_material(
    prompt_cache_key: str,
    cache_salt: str | None = None,
) -> str:
    """Hash input for ``derive_slot_id`` — isolates tenants sharing the same thread key."""
    if cache_salt:
        return f"{cache_salt}\0{prompt_cache_key}"
    return prompt_cache_key


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


def cache_prompt_for_request(
    prompt_cache_key: str | None,
    policy: Any | None = None,
    *,
    seq_pos: int | None = None,
    prompt_tokens: int | None = None,
) -> bool:
    """Whether to pass ``cache_prompt: true`` to llama-server / resume pinned slots."""
    from runtime.prefix_cache_policy import (
        PrefixCachePolicy,
        cache_prompt_for_request as policy_cache_prompt,
    )

    pol: PrefixCachePolicy | None = policy if isinstance(policy, PrefixCachePolicy) else None
    return policy_cache_prompt(
        prompt_cache_key,
        pol,
        seq_pos=seq_pos,
        prompt_tokens=prompt_tokens,
    )


def resolve_cache_key_from_options(options: dict[str, Any] | None) -> str | None:
    return resolve_local_cache_key(options) if options else None


def resolve_cache_key_for_batch(
    options: dict[str, Any] | None,
    index: int,
) -> str | None:
    """Per-prompt cache key for batch generate (`options.prompt_cache_keys[i]`).

    When ``prompt_cache_keys`` is present, out-of-range indices return ``None``
    (no flat-key fallback) so unrelated batch rows do not share one slot.

    WHY no fallback: callers that pass a partial list intend per-row isolation;
    falling back to ``prompt_cache_key`` would pin extra rows to one session slot.
    Omit ``prompt_cache_keys`` to apply one flat key to every batch row.
    """
    if not options:
        return None
    keys = options.get("prompt_cache_keys")
    if isinstance(keys, list):
        if 0 <= index < len(keys):
            raw = keys[index]
            if isinstance(raw, str) and raw:
                return raw
        return None
    return resolve_cache_key_from_options(options)


def cache_pin_from_options(
    options: dict[str, Any] | None,
    *,
    parallel: int,
    batch_index: int | None = None,
) -> tuple[str | None, int | None, bool, str | None]:
    """Return ``(prompt_cache_key, kv_slot, slot_pinned, cache_salt)`` for admission.

    WHY at admit (not at HTTP forward): llama-server ``id_slot`` must be stable
    before ``/completion``; scheduler ``try_acquire`` serializes same-slot traffic.
    """
    if batch_index is not None:
        key = resolve_cache_key_for_batch(options, batch_index)
    else:
        key = resolve_cache_key_from_options(options)
    cache_salt = extract_cache_salt(options or {})
    kv_slot: int | None = None
    slot_pinned = False
    if key:
        derived = derive_slot_id(key, parallel, cache_salt=cache_salt)
        if derived >= 0:
            kv_slot = derived
            slot_pinned = True
    return key, kv_slot, slot_pinned, cache_salt


def zerollama_block_from_options(options: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(options, dict):
        return {}
    z = options.get("zerollama")
    return z if isinstance(z, dict) else {}


def extract_session_parent(options: dict[str, Any] | None) -> str | None:
    """Parent thread prompt_cache_key for Radix prefer (Go gate does wait_parent).

    WHY passed through to admission: equal-length donor ties break toward parent
    without overriding hash verification.
    """
    z = zerollama_block_from_options(options)
    for key in ("session_parent",):
        raw = z.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    raw = (options or {}).get("mlx_session_parent") if isinstance(options, dict) else None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def extract_session_group(options: dict[str, Any] | None) -> str | None:
    z = zerollama_block_from_options(options)
    for key in ("session_group", "harness"):
        raw = z.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def extract_cache_reset(options: dict[str, Any] | None) -> bool:
    """Force miss under the same prompt_cache_key this turn.

    WHY same key (not a cold: prefix): harnesses already own a stable interactive
    key; resetting validity is orthogonal to identity. Engine skips L3 resume and
    Radix seed when True — see docs/agent-qos-and-project-tracking.md.
    """
    z = zerollama_block_from_options(options)
    raw = z.get("cache_reset")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return False


_CACHE_LEVEL_ALIASES = {
    "auto": "auto",
    "gpu": "gpu",
    "vram": "gpu",
    "dram": "dram",
    "ram": "dram",
    "memory": "dram",
    "disk": "disk",
    "ssd": "disk",
    "persist": "disk",
}


def extract_cache_level(options: dict[str, Any] | None) -> str:
    """KV retention tier.

    WHY auto default: Tier-1 clients must not surprise-flip disk.
    WHY gpu≈dram: both forbid disk persist until a real VRAM↔host spill path exists.
    """
    z = zerollama_block_from_options(options)
    raw = z.get("cache_level")
    if raw is None:
        raw = z.get("cache_tier")
    if not isinstance(raw, str) or not raw.strip():
        return "auto"
    return _CACHE_LEVEL_ALIASES.get(raw.strip().lower(), "auto")


def extract_kv_load_tiers(options: dict[str, Any] | None) -> Any:
    """Per-request secondary-tier load filter (vLLM #48123 ``kv_load_tiers``).

    Prefer ``options.zerollama.kv_load_tiers``; fall back to top-level
    ``kv_load_tiers`` / ``kvLoadTiers``. Returns raw JSON list or None.
    """
    z = zerollama_block_from_options(options)
    raw = z.get("kv_load_tiers")
    if raw is None:
        raw = z.get("kvLoadTiers")
    if raw is None and isinstance(options, dict):
        raw = options.get("kv_load_tiers")
        if raw is None:
            raw = options.get("kvLoadTiers")
    return raw


def apply_cache_level_to_policy(policy: Any, cache_level: str | None) -> Any:
    """Override disk persist from options.zerollama.cache_level.

    ``auto`` / unset leaves policy unchanged. ``gpu``/``dram`` forbid disk.
    ``disk`` allows disk when hard denies (draft-spec) do not apply.
    """
    from dataclasses import replace

    level = (cache_level or "auto").strip().lower() or "auto"
    if level == "auto":
        return policy
    if level in ("gpu", "dram", "vram", "ram", "memory"):
        return replace(policy, allow_disk_persist=False)
    if level in ("disk", "ssd", "persist"):
        if getattr(policy, "speculative_draft", False):
            return policy
        return replace(policy, allow_disk_persist=True)
    return policy


def slot_resume_owner_key(kv_bind_req: Any | None) -> str | None:
    """Stable owner id for in-process KV resume (Phase 15 v17).

    WHY not ``request_id`` alone: L3 pinned sessions allocate a fresh
    ``request_id`` every HTTP turn but reuse ``prompt_cache_key`` and ``kv_slot``.
    v16b keyed only on ``request_id``, so multi-turn agent chat always cleared
    good prefix KV and re-prefilled from scratch.

    Pinned → ``cache:{salt}:{prompt_cache_key}`` when salt set, else ``cache:{key}``;
    otherwise → ``request_id`` string.

    If ``slot_pinned`` is set but ``prompt_cache_key`` is missing (should not
    happen via ``cache_pin_from_options``, which only pins when key is truthy),
    falls back to ``request_id`` so resume degrades to v16b behaviour.
    """
    if kv_bind_req is None:
        return None
    if getattr(kv_bind_req, "slot_pinned", False):
        cache_key = getattr(kv_bind_req, "prompt_cache_key", None)
        if cache_key:
            salt = getattr(kv_bind_req, "cache_salt", None)
            if isinstance(salt, str) and salt.strip():
                return f"cache:{salt.strip()}:{cache_key}"
            return f"cache:{cache_key}"
    rid = getattr(kv_bind_req, "request_id", None)
    return str(rid) if rid else None


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
    spec_method: str = "none",
    num_ctx: int | None = None,
) -> list[str]:
    """``--slot-save-path`` argv when disk cache is enabled.

    WHY evict on start: crash leaves stale slot blobs; directory is small (≈ np files).
    Sync scan is acceptable on cold llama-server boot — not on hot request path.
    """
    if not llama_cache_enabled():
        return []
    from runtime.prefix_cache_policy import (
        effective_disk_cache_enabled,
        resolve_prefix_cache_policy,
    )

    policy = resolve_prefix_cache_policy(
        gguf=model_path,
        num_ctx=num_ctx,
        spec_method=spec_method,
    )
    if not effective_disk_cache_enabled(policy):
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
    # WHY sweep siblings first: profile/fork switches create new hashes; stale dirs
    # should not accumulate expired slot blobs under ~/.cache/zerollama/llama-cache/.
    evict_orphaned_cache_dirs(keep_model_hash=model_hash)
    # Sync scan before llama-server start; small dir expected (one entry per slot).
    evict_expired(save_dir, slot_bin_ttl_ms=policy.disk_ttl_ms)
    return ["--slot-save-path", str(save_dir)]


def cache_health(
    model_path: Path | None,
    llama_server_args: list[str],
    *,
    draft_model: Path | None = None,
    num_ctx: int | None = None,
    spec_method: str = "none",
) -> dict[str, Any]:
    """``/health.llama_cache`` snapshot.

    WHY hash before file exists: operators set LLAMA_MODEL before pull; slot dir
    layout should be stable for pre-warm and monitoring.
    """
    enabled = llama_cache_enabled()
    from runtime.prefix_cache_policy import (
        effective_disk_cache_enabled,
        policy_to_health,
        resolve_prefix_cache_policy,
    )

    model_loaded = model_path is not None and model_path.is_file()
    policy = resolve_prefix_cache_policy(
        gguf=model_path if model_loaded else None,
        num_ctx=num_ctx,
        spec_method=spec_method,
    )
    inprocess_disk = (
        enabled
        and model_loaded
        and effective_disk_cache_enabled(policy)
    )
    out: dict[str, Any] = {
        "enabled": enabled,
        "inprocess_disk_cache": inprocess_disk,
        "root": str(llama_cache_root()),
        "default_ttl_ms": default_slot_ttl_ms(),
        "policy": policy_to_health(policy),
    }
    from runtime.kv.spec_bind import spec_bind_health
    from runtime.kv_cache_spec import resolve_kv_cache_spec
    from runtime.decode_graph_cache import decode_graph_cache

    spec = resolve_kv_cache_spec(
        gguf=model_path if model_loaded else None,
        num_ctx=num_ctx,
        spec_method=spec_method,
    )
    out["kv_cache_spec"] = spec.to_health()
    out["spec_bind"] = spec_bind_health(spec)
    out["decode_graph"] = decode_graph_cache().health()
    from runtime.kv.prefix_block_pool import build_model_scope, prefix_block_pool_health

    scope = None
    if enabled and model_path is not None and model_loaded:
        ck, cv = cache_type_from_llama_argv(llama_server_args)
        mh = build_model_hash(
            target_model_path=model_path,
            drafter_model_path=draft_model,
            cache_type_k=ck,
            cache_type_v=cv,
        )
        scope = build_model_scope(model_hash=mh)
    out["prefix_block_pool"] = prefix_block_pool_health(model_scope=scope)
    from runtime.kv.radix_prefix_share import radix_share_health

    out["prefix_block_pool"]["radix_share"] = radix_share_health(model_scope=scope)
    from runtime.env import runtime_env_health

    out["runtime_env"] = runtime_env_health()
    if not enabled or model_path is None:
        return out
    out["model_path"] = str(model_path)
    out["model_loaded"] = model_loaded
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
