"""Breakable decode graph epoch (vLLM CUDA-graph invalidation inspired).

WHY scaffold: when prefix cache is cleared (SWA block, owner change, draft spec),
any captured CUDA decode graph for that slot is stale. Epoch bumps call
``llama_context_cuda_graph_invalidate`` when a live context pointer is available.

Enable tracing: ``ZEROLLAMA_DECODE_GRAPH_TRACE=1``.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

_log = logging.getLogger("zerollama-runtime")

_LOCK = threading.Lock()
_EPOCH_BY_SLOT: dict[int, int] = {}
_GLOBAL_EPOCH = 0


def decode_graph_trace_enabled() -> bool:
    return os.environ.get("ZEROLLAMA_DECODE_GRAPH_TRACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def decode_graph_epoch(slot_id: int = -1) -> int:
    """Current epoch for ``slot_id`` (0 when never bumped; global when slot < 0)."""
    with _LOCK:
        if slot_id >= 0:
            return _EPOCH_BY_SLOT.get(slot_id, 0)
        return _GLOBAL_EPOCH


def bump_decode_graph_epoch(
    slot_id: int,
    *,
    reason: str,
    ctx_ptr: int | None = None,
) -> int:
    """Invalidate captured decode graphs for ``slot_id`` (+ global counter).

    WHY two steps: (1) epoch for future ``DecodeGraphCache.lookup`` keys and trace
    replay; (2) ``invalidate_cuda_graphs(ctx_ptr)`` clears ggml's actual CUDA graph
    map today — ggml does not read our epoch counter.
    """
    global _GLOBAL_EPOCH
    with _LOCK:
        _GLOBAL_EPOCH += 1
        prev = _EPOCH_BY_SLOT.get(slot_id, 0)
        new = prev + 1
        _EPOCH_BY_SLOT[slot_id] = new
        if decode_graph_trace_enabled():
            _log.info(
                "decode_graph_epoch slot=%s epoch=%s global=%s reason=%s",
                slot_id,
                new,
                _GLOBAL_EPOCH,
                reason,
            )
    if ctx_ptr is not None:
        from runtime.kv.cuda_graph_invalidate import invalidate_cuda_graphs

        invalidate_cuda_graphs(ctx_ptr, reason=reason)
    return new


def bump_all_decode_graph_epochs(*, reason: str, ctx_ptr: int | None = None) -> int:
    """Model swap / teardown — invalidate every slot.

    WHY bump every known slot + global: session close and model swap must poison
    capture keys for slots that were never individually cleared this session.
    """
    global _GLOBAL_EPOCH
    with _LOCK:
        _GLOBAL_EPOCH += 1
        for sid in list(_EPOCH_BY_SLOT):
            _EPOCH_BY_SLOT[sid] = _EPOCH_BY_SLOT[sid] + 1
        g = _GLOBAL_EPOCH
        if decode_graph_trace_enabled():
            _log.info(
                "decode_graph_epoch global=%s reason=%s",
                g,
                reason,
            )
    if ctx_ptr is not None:
        from runtime.kv.cuda_graph_invalidate import invalidate_cuda_graphs

        invalidate_cuda_graphs(ctx_ptr, reason=reason)
    return g


def decode_graph_health() -> dict[str, Any]:
    with _LOCK:
        return {
            "global_epoch": _GLOBAL_EPOCH,
            "slot_epochs": {str(k): v for k, v in sorted(_EPOCH_BY_SLOT.items())},
            "capture_ready": False,
            "capture_key_format": "slot_id:slot_epoch:global_epoch",
            "note": (
                "epoch + ggml invalidation — ``llama_context_cuda_graph_invalidate`` "
                "clears ggml CUDA graph cache on prefix cache clear when ctx is wired"
            ),
        }


def graph_capture_key(slot_id: int) -> str:
    """Future CUDA graph cache lookup key (vLLM breakable-graph pattern).

    Includes global epoch so ``bump_all_decode_graph_epochs`` (model swap /
    session close) invalidates graphs for slots that were never individually bumped.
    """
    sid = int(slot_id)
    return f"{sid}:{decode_graph_epoch(sid)}:{decode_graph_epoch(-1)}"


def reset_decode_graph_epochs() -> None:
    """Test helper."""
    global _GLOBAL_EPOCH
    with _LOCK:
        _GLOBAL_EPOCH = 0
        _EPOCH_BY_SLOT.clear()
