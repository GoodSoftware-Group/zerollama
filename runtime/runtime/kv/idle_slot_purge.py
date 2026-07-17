"""Phase 15 v57 — in-process idle-slot purge under unified KV.

WHY: with ``kv_unified`` / ``n_stream=1``, idle sequences keep cells in the shared
pool and inflate attention / starve new work. llama-server already calls
``try_clear_idle_slots()`` when KV alloc fails; in-process ctypes decode had no
equivalent (v52–v56 non-goal).

Behavior mirrors vendor: on ``llama_decode`` failure, clear **one** idle seq
(not the active ``keep_seq``) that still holds tokens, then retry once.
Kill-switch: ``ZEROLLAMA_KV_UNIFIED_IDLE_PURGE=0``.
"""

from __future__ import annotations

from typing import Any, Callable

_purge_total = 0
_purge_last_seq: int | None = None


def reset_idle_purge_stats_for_tests() -> None:
    global _purge_total, _purge_last_seq
    _purge_total = 0
    _purge_last_seq = None


def idle_slot_purge_enabled(*, kv_unified: bool) -> bool:
    """On when unified and kill-switch not off (default on under unified)."""
    if not kv_unified:
        return False
    from runtime.env import env_bool

    return env_bool("ZEROLLAMA_KV_UNIFIED_IDLE_PURGE", default=True)


def idle_slot_purge_health() -> dict[str, Any]:
    return {
        "purged_total": _purge_total,
        "last_purged_seq": _purge_last_seq,
        "note": (
            "in-process: clear one idle seq on llama_decode fail under kv_unified "
            "(vendor try_clear_idle_slots parity); kill-switch "
            "ZEROLLAMA_KV_UNIFIED_IDLE_PURGE=0"
        ),
    }


def try_clear_idle_slot(
    lib: Any,
    ctx: Any,
    *,
    keep_seq: int,
    n_seq_max: int,
    clear_fn: Callable[[Any, Any, int], None] | None = None,
) -> int | None:
    """Purge one idle sequence with live KV; return cleared seq id or None.

    Picks the idle seq with the largest ``token_cells`` (most cells freed). Never
    clears ``keep_seq``.
    """
    global _purge_total, _purge_last_seq

    import ctypes

    from runtime.kv.physical import usage_from_libllama

    n_max = max(1, int(n_seq_max))
    keep = int(keep_seq)
    best_sid: int | None = None
    best_cells = 0
    for sid in range(n_max):
        if sid == keep:
            continue
        usage = usage_from_libllama(lib, ctx, sid)
        cells = int(usage.token_cells) if usage is not None else 0
        if cells <= 0:
            continue
        if cells > best_cells:
            best_cells = cells
            best_sid = sid
    if best_sid is None:
        return None

    if clear_fn is not None:
        clear_fn(lib, ctx, best_sid)
    else:
        mem = lib.llama_get_memory(ctx)
        if not mem:
            return None
        lib.llama_memory_seq_rm(
            mem,
            ctypes.c_int32(best_sid),
            ctypes.c_int32(-1),
            ctypes.c_int32(-1),
        )

    _purge_total += 1
    _purge_last_seq = best_sid
    return best_sid


def llama_decode_with_idle_purge(
    lib: Any,
    ctx: Any,
    batch: Any,
    *,
    keep_seq: int,
    n_seq_max: int,
    kv_unified: bool,
    on_purge: Callable[[int], None] | None = None,
    clear_fn: Callable[[Any, Any, int], None] | None = None,
) -> int:
    """``llama_decode``; on failure under unified, purge one idle seq and retry once."""
    rc = int(lib.llama_decode(ctx, batch))
    if rc == 0:
        return rc
    if not idle_slot_purge_enabled(kv_unified=kv_unified):
        return rc
    cleared = try_clear_idle_slot(
        lib,
        ctx,
        keep_seq=keep_seq,
        n_seq_max=n_seq_max,
        clear_fn=clear_fn,
    )
    if cleared is None:
        return rc
    if on_purge is not None:
        on_purge(cleared)
    return int(lib.llama_decode(ctx, batch))
