"""Phase 15 v51 — overlay donor page-offset catalog (read-only).

WHY: v50 makes the runtime own the KV byte region via an overlay donor. L3-R6
(physical shared pages) still cannot share cells until we prove PA pages are
addressable ranges *inside* that donor. This module asserts containment and
publishes ``block_id`` → donor byte offsets — geometry for future share, no
allocator rewrite and no ``seq_cp`` behavior change.

Hard constraint (unchanged): one ggml tensor per layer covers the whole
``kv_size``; ``page_map`` pointers are offsets into that tensor (already inside
the donor when overlay bind consumed it). We never rebase ``tensor->data``.
"""

from __future__ import annotations

from typing import Any

# Cap live map_page fan-out on /health so polling stays cheap.
_HEALTH_PAGE_CAP = 8


def span_in_donor(
    donor_base: int,
    donor_size: int,
    ptr: int,
    span_bytes: int,
) -> bool:
    """True when ``[ptr, ptr+span)`` lies entirely inside the donor buffer.

    Empty / absent spans (ptr=0 or span=0, e.g. MLA null-V) are treated as
    in-donor — they do not contradict overlay binding.
    """
    if donor_base <= 0 or donor_size <= 0:
        return False
    if ptr == 0 or span_bytes <= 0:
        return True
    if ptr < donor_base:
        return False
    end = ptr + int(span_bytes)
    return end <= donor_base + int(donor_size)


def page_donor_offsets(
    donor_base: int,
    donor_size: int,
    page_map: dict[str, Any],
    *,
    block_id: int | None = None,
    page_index: int | None = None,
) -> dict[str, Any]:
    """Map one ``page_map`` row to donor-relative offsets + containment."""
    k_data = int(page_map.get("k_data") or 0)
    v_data = int(page_map.get("v_data") or 0)
    k_span = int(page_map.get("k_span_bytes") or 0)
    v_span = int(page_map.get("v_span_bytes") or 0)
    k_in = span_in_donor(donor_base, donor_size, k_data, k_span)
    v_in = span_in_donor(donor_base, donor_size, v_data, v_span)
    k_off = (k_data - donor_base) if k_data and k_in else None
    v_off = (v_data - donor_base) if v_data and v_in else None
    page = page_index
    if page is None and page_map.get("page") is not None:
        page = int(page_map["page"])
    bid = block_id
    if bid is None and page_map.get("block_id") is not None:
        bid = int(page_map["block_id"])
    return {
        "page": page,
        "block_id": bid,
        "kv_layer": int(page_map.get("kv_layer") or 0),
        "k_offset": k_off,
        "v_offset": v_off,
        "k_span": k_span,
        "v_span": v_span,
        "in_donor": bool(k_in and v_in),
    }


def build_overlay_page_catalog(
    *,
    donor_base: int,
    donor_size: int,
    ctx_ptr: int,
    seq_id: int,
    kv_slot: int,
    block_size: int,
    probe: dict[str, Any] | None = None,
    max_pages: int | None = None,
    kv_layer: int = 0,
    include_pages: bool = True,
) -> dict[str, Any] | None:
    """Build donor containment catalog from live ``map_page`` (layer ``kv_layer``).

    Returns ``None`` when donor geometry is unknown or tensor probe/map is
    unavailable. Does not mutate llama tensors.
    """
    if donor_base <= 0 or donor_size <= 0 or ctx_ptr == 0:
        return None
    try:
        from runtime.kv.tensor_probe import (
            export_page_table,
            map_page,
            tensor_probe_available,
        )
    except Exception:
        return None
    if not tensor_probe_available():
        return None

    registered = export_page_table(kv_slot)
    pages_live = len(registered)
    if pages_live <= 0 and probe:
        cells = int(probe.get("llama_token_cells") or 0)
        if cells > 0 and block_size > 0:
            pages_live = (cells + block_size - 1) // block_size
    if pages_live <= 0:
        return {
            "donor_base": int(donor_base),
            "donor_bytes": int(donor_size),
            "pages_checked": 0,
            "pages_in_donor": 0,
            "all_in_donor": True,
            "kv_slot": int(kv_slot),
            "kv_layer": int(kv_layer),
            "note": "no registered PA pages yet",
            "pages": [] if include_pages else None,
        }

    limit = pages_live
    if max_pages is not None and max_pages > 0:
        limit = min(limit, int(max_pages))

    rows: list[dict[str, Any]] = []
    in_donor = 0
    for page_index in range(limit):
        block_id = None
        if page_index < len(registered):
            block_id = int(registered[page_index].get("block_id", -1))
        try:
            pm = map_page(
                ctx_ptr, seq_id, kv_slot, page_index, kv_layer=kv_layer
            )
        except (KeyError, RuntimeError, OSError):
            rows.append(
                {
                    "page": page_index,
                    "block_id": block_id,
                    "kv_layer": kv_layer,
                    "k_offset": None,
                    "v_offset": None,
                    "k_span": 0,
                    "v_span": 0,
                    "in_donor": False,
                    "map_failed": True,
                }
            )
            continue
        row = page_donor_offsets(
            donor_base,
            donor_size,
            pm,
            block_id=block_id,
            page_index=page_index,
        )
        if row["in_donor"]:
            in_donor += 1
        rows.append(row)

    checked = len(rows)
    out: dict[str, Any] = {
        "donor_base": int(donor_base),
        "donor_bytes": int(donor_size),
        "pages_checked": checked,
        "pages_in_donor": in_donor,
        "all_in_donor": checked > 0 and in_donor == checked,
        "pages_live": pages_live,
        "kv_slot": int(kv_slot),
        "kv_layer": int(kv_layer),
        "truncated": limit < pages_live,
    }
    if include_pages:
        out["pages"] = rows
    return out


def overlay_page_catalog_summary(catalog: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip per-page rows for lightweight ``/health`` polling."""
    if not catalog:
        return None
    return {
        "donor_bytes": catalog.get("donor_bytes"),
        "pages_checked": catalog.get("pages_checked"),
        "pages_in_donor": catalog.get("pages_in_donor"),
        "all_in_donor": catalog.get("all_in_donor"),
        "pages_live": catalog.get("pages_live"),
        "kv_slot": catalog.get("kv_slot"),
        "kv_layer": catalog.get("kv_layer"),
        "truncated": catalog.get("truncated"),
        "full_plan_endpoint": "/internal/kv-snapshot",
        "note": catalog.get("note"),
    }


def health_page_cap() -> int:
    return _HEALTH_PAGE_CAP
