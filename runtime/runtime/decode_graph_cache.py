"""Decode CUDA graph cache stub (vLLM breakable-graph pattern).

WHY: ``decode_graph_policy`` tracks invalidation epochs; this module is the
future lookup surface for llama.cpp graph capture. Until capture is linked,
``lookup`` always returns ``None`` and ``capture_ready`` stays false.
"""

from __future__ import annotations

from typing import Any

from runtime.decode_graph_policy import (
    bump_decode_graph_epoch,
    decode_graph_health,
    graph_capture_key,
)


class DecodeGraphCache:
    """Per-process decode graph cache (scaffold — no native capture yet)."""

    def lookup_key(self, slot_id: int) -> str:
        return graph_capture_key(slot_id)

    def lookup(self, slot_id: int) -> Any | None:
        """Return a captured graph handle, or ``None`` when not ready."""
        return None

    def invalidate_slot(self, slot_id: int, *, reason: str, ctx_ptr: int | None = None) -> int:
        """Bump epoch (+ ggml invalidate when ``ctx_ptr`` wired).

        WHY ctx_ptr optional: subprocess llama-server owns its own context; in-process
        Phase 15 passes the live ``llama_context`` pointer from ``libllama_ctypes``.
        """
        from runtime.decode_graph_policy import bump_decode_graph_epoch

        return bump_decode_graph_epoch(slot_id, reason=reason, ctx_ptr=ctx_ptr)

    def is_capture_ready(self) -> bool:
        return False

    def health(self) -> dict[str, Any]:
        from runtime.llama_cpp_probe import probe_llama_cpp

        out = decode_graph_health()
        out["stub"] = True
        out["lookup"] = "epoch_plus_ggml_invalidate"
        out["llama_cpp"] = probe_llama_cpp()
        return out


_CACHE = DecodeGraphCache()


def decode_graph_cache() -> DecodeGraphCache:
    return _CACHE
