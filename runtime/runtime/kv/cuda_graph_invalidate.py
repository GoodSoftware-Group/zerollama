"""ggml CUDA graph invalidation (vLLM breakable-graph bind).

WHY: prefix cache clear changes KV contents while ggml may reuse a captured
CUDA graph keyed by cgraph node[0]. ``llama_context_cuda_graph_invalidate``
clears ggml's per-backend graph cache; epoch bumps in ``decode_graph_policy``
call this when a live ``llama_context`` pointer is available.
"""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Any

_log = logging.getLogger("zerollama-runtime")

_ctypes_invalidate_fn: Any | None = None


def cuda_graph_invalidate_enabled() -> bool:
    return os.environ.get("ZEROLLAMA_DECODE_GRAPH_INVALIDATE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _ctypes_invalidate(ctx_ptr: int) -> dict[str, Any]:
    global _ctypes_invalidate_fn
    if ctx_ptr <= 0:
        return {"ok": False, "backends_cleared": 0, "path": "none", "reason": "no_ctx"}
    try:
        from runtime.worker.libllama_ctypes import get_lib

        lib = get_lib()
    except Exception as exc:
        return {"ok": False, "backends_cleared": 0, "path": "ctypes", "reason": str(exc)}
    if _ctypes_invalidate_fn is None:
        if not hasattr(lib, "llama_context_cuda_graph_invalidate"):
            return {
                "ok": False,
                "backends_cleared": 0,
                "path": "ctypes",
                "reason": "symbol_missing_rebuild_libllama",
            }
        fn = lib.llama_context_cuda_graph_invalidate
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_int32
        _ctypes_invalidate_fn = fn
    cleared = int(_ctypes_invalidate_fn(ctypes.c_void_p(ctx_ptr)))
    return {"ok": True, "backends_cleared": cleared, "path": "ctypes"}


def invalidate_cuda_graphs(ctx_ptr: int | None, *, reason: str = "") -> dict[str, Any]:
    """Invalidate captured CUDA graphs for ``ctx_ptr``; no-op when disabled or missing.

    WHY native before ctypes: Phase 15 linked build avoids Python ctypes overhead on
    the hot invalidation path; ctypes remains fallback when extension lacks symbol.
    """
    if not cuda_graph_invalidate_enabled():
        return {"ok": False, "backends_cleared": 0, "path": "disabled", "reason": reason}
    if ctx_ptr is None or int(ctx_ptr) <= 0:
        return {"ok": False, "backends_cleared": 0, "path": "none", "reason": reason or "no_ctx"}

    ptr = int(ctx_ptr)
    try:
        from runtime.kv._kv_native import invalidate_cuda_graphs as _native_inv

        raw = _native_inv(ptr)
        if isinstance(raw, dict):
            out = {
                "ok": bool(raw.get("ok")),
                "backends_cleared": int(raw.get("backends_cleared") or 0),
                "path": "native",
                "reason": reason,
            }
            if out["ok"] or out["backends_cleared"] > 0:
                return out
    except ImportError:
        pass
    except Exception as exc:
        _log.debug("native cuda graph invalidate failed: %s", exc)

    out = _ctypes_invalidate(ptr)
    out["reason"] = reason
    if out.get("ok") and reason:
        _log.debug(
            "cuda_graph_invalidate reason=%s cleared=%s path=%s",
            reason,
            out.get("backends_cleared"),
            out.get("path"),
        )
    return out
