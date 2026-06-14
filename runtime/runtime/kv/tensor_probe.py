"""Phase 15 v19 — llama memory probe + PA page accounting (tensor bind scaffold)."""

from __future__ import annotations

from typing import Any


def tensor_probe_available() -> bool:
    try:
        from runtime.kv._kv_native import page_bind_tensor_probe  # noqa: F401

        return True
    except ImportError:
        return False
    except Exception:
        return False


def writable_bind_probe() -> dict[str, Any]:
    """Static probe: is writable PA→tensor page bind API linked in libllama?

    WHY no ctx: upstream page-handle availability is a build-time capability,
    not per-request. Operators watch ``writable_bind_available`` on /health
    until llama.cpp ships a stable writable page-map API.
    """
    try:
        from runtime.kv._kv_native import page_bind_writable_probe
    except Exception:
        return {
            "writable_bind_available": False,
            "writable_bind_api": "none",
            "writable_bind_blocker": "native_ext_not_built",
        }
    return dict(page_bind_writable_probe())


def run_tensor_probe(ctx_ptr: int, seq_id: int, kv_slot: int) -> dict[str, Any] | None:
    """Probe llama memory vs native page_bind table for one sequence.

    WHY: upstream llama.cpp has no public KV tensor page handles yet; this
    reports ``llama_get_memory`` / seq position data and whether PA pages
    cover live llama cells (accounting bind).  Returns ``None`` when the
    linked ext is not built.
    """
    if not tensor_probe_available() or ctx_ptr == 0:
        return None
    from runtime.kv._kv_native import page_bind_tensor_probe

    raw = page_bind_tensor_probe(int(ctx_ptr), int(seq_id), int(kv_slot))
    return dict(raw) if raw is not None else None


def export_page_table(kv_slot: int) -> list[dict[str, int]]:
    """Export registered PA page rows for ``kv_slot`` (native C registry)."""
    try:
        from runtime.kv._kv_native import page_bind_table
    except ImportError:
        return []
    rows = page_bind_table(int(kv_slot))
    return [dict(r) for r in rows]


def page_table_native_parity(
    logical_pages: list[dict[str, int]],
    native_pages: list[dict[str, int]],
) -> bool:
    """True when native C registry rows match the logical ``pages[]`` export."""
    if len(logical_pages) != len(native_pages):
        return False
    keys = ("page", "block_id", "token_start", "token_end")
    for logical, native in zip(logical_pages, native_pages):
        for key in keys:
            if int(logical.get(key, -1)) != int(native.get(key, -1)):
                return False
    return True
