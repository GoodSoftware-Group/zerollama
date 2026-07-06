"""Phase 15 v19/v47 — llama memory probe + PA page accounting + alias validate.

WHY this module: tensor page bind and external-buffer alias need llama-kv-ext symbols
without pulling llama types into every Python caller. Static probes (writable bind,
external alias) need no live ctx; live probes need ``ctx_ptr`` from the in-process engine.
"""

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
    """Static probe: is writable PA→tensor page-map API linked in libllama?

    WHY no ctx: page-map availability is a build-time capability on the forked
    llama.cpp pin; operators watch ``writable_bind_available`` on /health.
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


def external_alias_probe() -> dict[str, Any]:
    """Static probe: is external buffer alias validate API linked in libllama?

    WHY no ctx: alias validate availability is a build-time capability on the forked
    llama.cpp pin (patch 0019 + ``LLAMA_KV_EXT_EXTERNAL_ALIAS``); operators watch
    ``external_alias_available`` on /health before wiring migration bind (v48+).
    """
    try:
        from runtime.kv._kv_native import page_bind_external_alias_probe
    except Exception:
        return {
            "external_alias_available": False,
            "external_alias_validate_api": False,
            "external_alias_api": "none",
            "external_alias_blocker": "native_ext_not_built",
        }
    out = dict(page_bind_external_alias_probe())
    out["external_alias_blocker"] = (
        "" if out.get("external_alias_available") else "external_alias_api_not_linked"
    )
    return out


def alias_validate(
    ctx_ptr: int,
    seq_id: int,
    kv_slot: int,
    page_index: int,
    *,
    kv_layer: int = 0,
    ext_k_data: int = 0,
    ext_k_span_bytes: int = 0,
    ext_v_data: int = 0,
    ext_v_span_bytes: int = 0,
) -> dict[str, Any]:
    """Validate external K/V pointers against llama page_map geometry (no mutation).

    WHY: v38 copy descriptors need to know if migration can skip memcpy (SAME_POINTER)
    or must copy/scatter (BLOCKED_V_TRANS, BLOCKED_DEVICE on Metal). Returns plan dict
    with ``alias_mode``, ``blocker``, ``alias_ready`` — see ``llama_kv_ext_alias_mode``.
    """
    if not tensor_probe_available() or ctx_ptr == 0:
        raise RuntimeError("native ext with llama-kv-ext link required")
    from runtime.kv._kv_native import page_bind_alias_validate

    return dict(
        page_bind_alias_validate(
            int(ctx_ptr),
            int(seq_id),
            int(kv_slot),
            int(page_index),
            int(kv_layer),
            int(ext_k_data),
            int(ext_k_span_bytes),
            int(ext_v_data),
            int(ext_v_span_bytes),
        )
    )


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


def last_tensor_probe_entries() -> list[dict[str, Any]]:
    """Last successful tensor probes per kv_slot (survives page_bind_clear)."""
    if not tensor_probe_available():
        return []
    try:
        from runtime.kv._kv_native import page_bind_last_tensor_probe
    except AttributeError:
        return []
    rows = page_bind_last_tensor_probe()
    return [{"kv_slot": int(r["kv_slot"]), "probe": dict(r["probe"])} for r in rows]


def map_page(
    ctx_ptr: int,
    seq_id: int,
    kv_slot: int,
    page_index: int,
    *,
    kv_layer: int = 0,
) -> dict[str, Any]:
    """Resolve writable K/V tensor spans for one registered PA page on one layer."""
    if not tensor_probe_available() or ctx_ptr == 0:
        raise RuntimeError("native ext with llama-kv-ext link required")
    from runtime.kv._kv_native import page_bind_map_page

    return dict(
        page_bind_map_page(
            int(ctx_ptr), int(seq_id), int(kv_slot), int(page_index), int(kv_layer)
        )
    )


def map_page_all_layers(
    ctx_ptr: int,
    seq_id: int,
    kv_slot: int,
    page_index: int,
    *,
    n_layers: int,
) -> list[dict[str, Any]]:
    """Fan-out ``map_page`` across ``0..n_layers-1`` (mirrors v34 materialize loop)."""
    if n_layers <= 0:
        return []
    return [
        map_page(ctx_ptr, seq_id, kv_slot, page_index, kv_layer=layer)
        for layer in range(n_layers)
    ]


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
