"""Phase 15 v48/v49/v50 — donor-buffer overlay bind (opt-in, zero-copy KV alloc).

WHY this module exists: per-page ``tensor->data`` rebase is architecturally
invalid — each KV layer is ONE ggml_tensor covering the entire ``kv_size``, so
``llama_memory_kv_page_map``'s ``k_data``/``v_data`` are pointer arithmetic into
that single tensor, not separate allocations (see ``llama-kv-cache.cpp``'s
per-buft allocation loop). Mutating a page's byte range post-hoc would corrupt
stride math for every other page sharing the tensor.

The only real zero-copy design: register PA-pool-owned host memory as a
*donor* BEFORE the ``llama_context``/model is constructed, so
``llama_kv_cache``'s allocation loop uses that memory instead of ggml's own
allocator. See ``llama_kv_ext_register_donor_buffer`` in ``llama-kv-ext.h``.

v48 (CPU-only): ``ggml_backend_buft_is_host(buft)`` KV buft groups use
``ggml_backend_cpu_buffer_from_ptr`` on the donor.

v49 (Metal): device buft groups (Metal's KV tensors report ``is_host()==false``
even though the backing memory is Apple Silicon unified memory) are tried next
via ``ggml_backend_dev_buffer_from_host_ptr`` when
``ggml_backend_dev_caps.buffer_from_host_ptr`` is true — this uses Metal's
``newBufferWithBytesNoCopy`` + ``MTLResourceStorageModeShared``, the identical
zero-copy-mmap mechanism llama-model.cpp's weight loader already uses in
production. CUDA does not implement this device capability (discrete VRAM, no
unified memory) so CUDA KV tensors are never consumed by either v48 or v49 —
they always fall through to normal ggml allocation. From this module's (and
the C ABI's) point of view v48/v49 are the SAME API: register/unregister/
status calls are unchanged: which path (host vs. device) actually consumed a
given donor is an internal vendor-hook decision driven by where the model
placed that layer's KV buft (CPU vs. GPU offload), not something the caller
selects explicitly.

v50 (auto-wire): when ``ZEROLLAMA_KV_OVERLAY_BIND=1`` (and auto not disabled),
in-process load allocates a page-aligned host donor from a GGUF KV-size
estimate (or ``ZEROLLAMA_KV_OVERLAY_DONOR_BYTES``) and registers it before
context construction. This is a **prerequisite** for physical shared KV pages
(L3-R6) — runtime owns the KV byte region — not yet multi-seq cell sharing
or COW. Undersized estimates silently fall through to ggml allocation
(``overlay_bind_bound=false``); oversized wastes RAM but is safe.

Gated entirely behind ``ZEROLLAMA_KV_OVERLAY_BIND=1`` — without it, none of
this module's bind functions are reachable from the runtime's model-load path,
and behavior is byte-for-byte identical to today (no registration call is ever
made).

Required load order (two-step, see ``docs/phase15-native-kv.md`` v48/v49
sections):
  1. Query the exact byte size ggml will request for a given model's KV cache
     (best done via a throwaway/dry-run construction, or an operator-supplied
     known-good size) — getting this wrong silently causes the donor to be
     rejected (undersized) or, if oversized, wastes memory but is otherwise
     safe (rejection falls through to normal allocation, never a partial bind).
  2. Allocate a correctly-sized, page-aligned host buffer and register it
     BEFORE constructing the context/model that will use it.

v50 step 1 may use a padded GGUF estimate instead of a dry-run when no exact
size ABI is available; operators can force exact sizing via
``ZEROLLAMA_KV_OVERLAY_DONOR_BYTES``.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PAGE = 4096


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


def overlay_bind_auto_enabled() -> bool:
    """True when v50 should auto-allocate/register a donor on in-process load.

    Requires ``ZEROLLAMA_KV_OVERLAY_BIND=1``. Disable auto with
    ``ZEROLLAMA_KV_OVERLAY_AUTO=0`` while keeping manual
    ``register_kv_overlay_donor`` / ``register_donor_buffer``.
    """
    if not overlay_bind_enabled():
        return False
    raw = os.environ.get("ZEROLLAMA_KV_OVERLAY_AUTO", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def page_align_bytes(n: int) -> int:
    if n <= 0:
        return _PAGE
    return (int(n) + _PAGE - 1) & ~(_PAGE - 1)


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


@dataclass
class AutoDonorHandle:
    """Owns a page-aligned host donor until unregister/teardown.

    WHY keepalive: ctypes buffers are freed when the Python object is GC'd;
    the donor ptr must remain valid for the lifetime of any llama_context that
    consumed it.
    """

    donor_id: int
    ptr: int
    size: int
    source: str  # "env" | "estimate"
    _keepalive: Any = field(repr=False)


def allocate_aligned_host_buffer(size: int) -> tuple[Any, int, int]:
    """Allocate a page-aligned host buffer of at least ``size`` usable bytes.

    Returns ``(keepalive, aligned_ptr, usable_size)``. Keepalive must be held
    until after unregister + context free.
    """
    usable = page_align_bytes(size)
    # Extra page so we can align the start without shrinking usable size.
    raw = ctypes.create_string_buffer(usable + _PAGE)
    raw_addr = ctypes.addressof(raw)
    aligned = (raw_addr + _PAGE - 1) & ~(_PAGE - 1)
    return raw, int(aligned), usable


def estimate_overlay_donor_bytes(
    gguf: Path | None,
    *,
    num_ctx: int,
    n_seq_max: int = 1,
    n_gpu_layers: int | None = None,
    kv_unified: bool | None = None,
) -> tuple[int, str]:
    """Return ``(page_aligned_bytes, source)`` for an auto-wired donor.

    Prefer ``ZEROLLAMA_KV_OVERLAY_DONOR_BYTES`` when set (exact operator size).
    Otherwise pad ``estimate_kv_cache_bytes`` × streams × 2 + 32 MiB —
    oversize is safe; undersize silently skips bind.

    WHY streams=1 when unified (v52): ``kv_unified`` packs all seqs into one
    ``n_stream`` tensor; multiplying by ``n_seq_max`` oversizes the donor.
    """
    raw_env = os.environ.get("ZEROLLAMA_KV_OVERLAY_DONOR_BYTES", "").strip()
    if raw_env:
        return page_align_bytes(int(raw_env, 0)), "env"

    if kv_unified is None:
        try:
            from runtime.env import kv_unified_enabled

            kv_unified = kv_unified_enabled()
        except Exception:
            kv_unified = False

    base: int | None = None
    if gguf is not None:
        try:
            from runtime.gguf_estimate import estimate_kv_cache_bytes, gguf_arch_hints

            hints = gguf_arch_hints(Path(gguf))
            ctx = max(1, int(num_ctx) if num_ctx and num_ctx > 0 else 4096)
            base = estimate_kv_cache_bytes(
                hints, ctx, n_gpu_layers=n_gpu_layers
            )
        except Exception:
            base = None
    if base is None or base <= 0:
        base = 64 << 20
    streams = 1 if kv_unified else max(1, int(n_seq_max))
    # WHY 2× + 32 MiB: GGUF estimate omits ggml per-buft alignment/padding and
    # multi-buft grouping; Metal mapped windows also need headroom.
    sized = int(base * streams * 2.0) + (32 << 20)
    return page_align_bytes(max(sized, 64 << 20)), "estimate"


def prepare_auto_donor(
    gguf: Path | None,
    *,
    num_ctx: int,
    n_seq_max: int = 1,
    n_gpu_layers: int | None = None,
    kv_unified: bool | None = None,
) -> AutoDonorHandle | None:
    """Allocate + register a host donor when v50 auto-wire is enabled.

    Returns ``None`` when auto is off, native donor API missing, or sizing
    fails. Does not raise on missing native ext — callers treat that as
    "overlay unavailable; continue with normal ggml alloc".
    """
    if not overlay_bind_auto_enabled():
        return None
    if not donor_buffer_available():
        return None
    size, source = estimate_overlay_donor_bytes(
        gguf,
        num_ctx=num_ctx,
        n_seq_max=n_seq_max,
        n_gpu_layers=n_gpu_layers,
        kv_unified=kv_unified,
    )
    keepalive, ptr, usable = allocate_aligned_host_buffer(size)
    donor_id = register_donor_buffer(ptr, usable)
    return AutoDonorHandle(
        donor_id=donor_id,
        ptr=ptr,
        size=usable,
        source=source,
        _keepalive=keepalive,
    )


def release_auto_donor(handle: AutoDonorHandle | None) -> None:
    """Unregister then drop keepalive. Call only after llama_free of consumers."""
    if handle is None:
        return
    unregister_donor_buffer(handle.donor_id)
