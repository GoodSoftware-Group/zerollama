"""Phase 15 v48 — CPU-only donor-buffer overlay bind (opt-in, zero-copy KV alloc).

WHY this module exists: per-page ``tensor->data`` rebase is architecturally
invalid — each KV layer is ONE ggml_tensor covering the entire ``kv_size``, so
``llama_memory_kv_page_map``'s ``k_data``/``v_data`` are pointer arithmetic into
that single tensor, not separate allocations (see ``llama-kv-cache.cpp``'s
per-buft allocation loop). Mutating a page's byte range post-hoc would corrupt
stride math for every other page sharing the tensor.

The only real zero-copy design: register PA-pool-owned host memory as a
*donor* BEFORE the ``llama_context``/model is constructed, so
``llama_kv_cache``'s CPU-buft allocation loop uses
``ggml_backend_cpu_buffer_from_ptr`` on that memory instead of ggml's own
allocator. See ``llama_kv_ext_register_donor_buffer`` in ``llama-kv-ext.h``.

Gated entirely behind ``ZEROLLAMA_KV_OVERLAY_BIND=1`` — without it, none of
this module's bind functions are reachable from the runtime's model-load path,
and behavior is byte-for-byte identical to today (no registration call is ever
made). Metal/CUDA KV tensors are never touched by this module: the vendor hook
only consults the donor registry for CPU (host) ``ggml_backend_buffer_type``
groups.

Required load order (two-step, see ``docs/phase15-native-kv.md`` v48 section):
  1. Query the exact byte size ggml will request for a given model's KV cache
     (best done via a throwaway/dry-run construction, or an operator-supplied
     known-good size) — getting this wrong silently causes the donor to be
     rejected (undersized) or, if oversized, wastes memory but is otherwise
     safe (rejection falls through to normal allocation, never a partial bind).
  2. Allocate a correctly-sized, page-aligned host buffer and register it
     BEFORE constructing the context/model that will use it.
"""

from __future__ import annotations

import os
from typing import Any


def overlay_bind_enabled() -> bool:
    """True when the operator explicitly opted in via ZEROLLAMA_KV_OVERLAY_BIND=1.

    WHY a dedicated getter rather than checking inline at each call site: every
    bind attempt in this module must go through this single gate — mirrors how
    ``ZEROLLAMA_RUNTIME_KV_NATIVE`` gates the native block pool in
    ``runtime/kv/backend.py``.
    """
    return os.environ.get("ZEROLLAMA_KV_OVERLAY_BIND", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def donor_buffer_available() -> bool:
    try:
        from runtime.kv._kv_native import register_donor_buffer  # noqa: F401

        return True
    except ImportError:
        return False
    except Exception:
        return False


def register_donor_buffer(ptr: int, size: int) -> int:
    """Register an external CPU host buffer as a KV-cache allocation donor.

    Must be called BEFORE the context/model that will consume it is
    constructed. Raises when overlay bind is not opted in — this is the
    "never touch tensors unless explicitly enabled" guarantee; callers must
    not bypass ``overlay_bind_enabled()`` by calling the native ext directly.
    """
    if not overlay_bind_enabled():
        raise RuntimeError(
            "ZEROLLAMA_KV_OVERLAY_BIND is not set — donor buffer registration refused"
        )
    if not donor_buffer_available():
        raise RuntimeError("native ext with llama-kv-ext donor-buffer link required")
    if ptr <= 0 or size <= 0:
        raise ValueError("ptr and size must be positive")

    from runtime.kv._kv_native import register_donor_buffer as _register

    return int(_register(int(ptr), int(size)))


def unregister_donor_buffer(donor_id: int) -> None:
    """Unregister a donor buffer. Safe/idempotent when the id is unknown.

    Callers MUST ensure no llama_context still uses the memory (i.e. call only
    after the model/context has been fully unloaded) — freeing or reusing the
    memory while a context is alive is undefined behavior, identical to the
    contract of any externally-owned ggml buffer.
    """
    if not donor_buffer_available():
        return
    from runtime.kv._kv_native import unregister_donor_buffer as _unregister

    _unregister(int(donor_id))


def donor_buffer_status(donor_id: int) -> dict[str, Any] | None:
    """Query whether a registered donor was actually consumed by a KV cache
    construction. Returns ``None`` when the native ext is not built or the
    donor id is unknown (e.g. never registered, or already unregistered).

    WHY this must exist: registration can silently fail to be consumed (wrong
    buft — e.g. the model offloaded to Metal/CUDA, undersized donor, or no
    cache built yet). Operators must be able to tell whether the zero-copy
    path actually happened rather than assuming success from registration
    alone.
    """
    if not donor_buffer_available():
        return None
    from runtime.kv._kv_native import donor_buffer_status as _status

    try:
        return dict(_status(int(donor_id)))
    except KeyError:
        return None
