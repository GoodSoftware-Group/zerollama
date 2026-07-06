"""Phase 15 v38 — external-buffer copy descriptors for writable page-map spans.

WHY: ``llama_memory_kv_page_map`` returns raw K/V pointers + byte spans, but
external migration code cannot ``memcpy`` V data when ``v_transposed=1`` — embedding
values are scattered across rows at stride ``kv_size``. v38 adds structured copy
descriptors so operators and future migration routines know the correct access
pattern without reading llama.cpp internals.

NOTE: ``external_buffer_alias_ready`` is ``True`` only when an optional
``alias_plan`` from ``llama_memory_kv_page_alias_validate`` reports
``alias_ready`` (SAME_POINTER zero-copy). Without a plan, it stays ``False``.
"""

from __future__ import annotations

from typing import Any, Literal

CopyMode = Literal["contiguous", "row_stride", "absent"]


def page_copy_descriptor(
    page_map: dict[str, Any],
    *,
    kv_cache_kv_size: int | None = None,
    element_size: int = 2,
    alias_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a copy plan for one ``page_bind_map_page`` result.

    Parameters
    ----------
    page_map:
        Dict from ``page_bind_map_page`` / ``map_page`` (includes ``k_data``,
        ``v_data``, ``v_transposed``, ``n_cells``, ``kv_layer``).
    kv_cache_kv_size:
        From ``llama_memory_kv_cache_layout`` / probe ``kv_cache_kv_size``.
        Required for accurate ``row_stride_elements`` when ``v_transposed=1``.
    element_size:
        Bytes per element (2 for fp16/bf16, 4 for fp32). Used for operator docs
        only — callers still use byte spans from the page map.
    alias_plan:
        Optional dict from ``alias_validate()`` / ``page_bind_alias_validate``. When
        ``alias_ready`` is true (SAME_POINTER), ``external_buffer_alias_ready`` is set.
        WHY: v47 validate must gate the bool — v38 alone cannot infer zero-copy alias.
    """
    v_trans = bool(page_map.get("v_transposed"))
    n_cells = int(page_map.get("n_cells") or 0)
    k_data = int(page_map.get("k_data") or 0)
    v_data = int(page_map.get("v_data") or 0)
    k_span = int(page_map.get("k_span_bytes") or 0)
    v_span = int(page_map.get("v_span_bytes") or 0)

    k_copy: dict[str, Any] = {
        "mode": "contiguous",
        "src_ptr": k_data,
        "byte_length": k_span,
        "n_cells": n_cells,
        "element_size": element_size,
    }

    if v_data == 0 or v_span == 0:
        v_copy: dict[str, Any] = {
            "mode": "absent",
            "src_ptr": 0,
            "byte_length": 0,
            "note": "MLA or model without V cache",
        }
    elif v_trans:
        stride = int(kv_cache_kv_size or 0)
        v_copy = {
            "mode": "row_stride",
            "src_ptr": v_data,
            "byte_length": v_span,
            "n_cells": n_cells,
            "element_size": element_size,
            "row_stride_elements": stride,
            "warning": (
                "V embedding values are interleaved across rows; "
                "memcpy(v_span) produces wrong data — use scatter/gather at "
                "stride kv_size elements per row"
            ),
        }
        if stride <= 0:
            v_copy["warning"] += "; pass kv_cache_kv_size from probe for stride"
    else:
        v_copy = {
            "mode": "contiguous",
            "src_ptr": v_data,
            "byte_length": v_span,
            "n_cells": n_cells,
            "element_size": element_size,
            "note": "Flash-attention layout — flat per-cell buffer",
        }

    migration_ready = k_data != 0 and k_span > 0 and (
        v_copy["mode"] == "absent" or (v_data != 0 and v_span > 0)
    )

    external_buffer_alias_ready = False
    if alias_plan is not None:
        external_buffer_alias_ready = bool(alias_plan.get("alias_ready"))

    return {
        "page": int(page_map.get("page", -1)),
        "block_id": int(page_map.get("block_id", -1)),
        "kv_layer": int(page_map.get("kv_layer", 0)),
        "pos_start": int(page_map.get("pos_start", 0)),
        "pos_end": int(page_map.get("pos_end", 0)),
        "v_transposed": v_trans,
        "k_copy": k_copy,
        "v_copy": v_copy,
        "migration_ready": migration_ready,
        "external_buffer_alias_ready": external_buffer_alias_ready,
    }


def page_copy_descriptors_for_layers(
    page_maps: list[dict[str, Any]],
    *,
    kv_cache_kv_size: int | None = None,
    element_size: int = 2,
    alias_plans: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build copy descriptors for a fan-out of ``map_page(..., kv_layer=N)`` results."""
    plans = alias_plans or []
    return [
        page_copy_descriptor(
            pm,
            kv_cache_kv_size=kv_cache_kv_size,
            element_size=element_size,
            alias_plan=plans[i] if i < len(plans) else None,
        )
        for i, pm in enumerate(page_maps)
    ]
