"""Phase 15 v39–v40 — page migration plan export for /internal/kv-snapshot.

WHY: v38 ``page_copy_descriptor`` is the per-page copy contract, but operators
still had to call ``map_page`` manually from scripts. v39 builds a full
``{pages[], layers[]}`` migration plan when tensor/physical bind succeeded and
attaches it to ``/internal/kv-snapshot`` for loopback debug.

v40 adds lightweight ``page_migration_summary`` on ``kv_forward_plans`` (no raw
pointers on /health) and redacts ``src_ptr`` in snapshot plans by default —
set ``ZEROLLAMA_KV_MIGRATION_INCLUDE_PTRS=1`` for full pointer export.

v42 surfaces the same summary on ``/health.kv_page_bind`` and as
``migration_summary`` on ``kv_page_migration`` snapshot branches (including
last-probe idle paths).
"""

from __future__ import annotations

import copy
from typing import Any

from runtime.kv.page_descriptor import page_copy_descriptor
from runtime.kv.tensor_probe import export_page_table, map_page, tensor_probe_available


def migration_plan_summary(
    probe: dict[str, Any],
    *,
    block_size: int,
    kv_slot: int,
    tensor_layers_expected: int | None = None,
) -> dict[str, Any] | None:
    """Lightweight migration status for ``/health.kv_forward_plans`` (no map_page).

    WHY separate from full plan: forward plans are polled frequently on /health;
    building pages×layers descriptors on every poll is expensive and exposes
    raw pointers. Summary gives operators bind progress; full plan stays on
    ``GET /internal/kv-snapshot``.
    """
    if not probe.get("tensor_pages_bound") and not probe.get("physical_pages_bound"):
        return None

    n_layers = int(probe.get("kv_n_layers") or probe.get("tensor_layers_verified") or 0)
    if n_layers <= 0:
        return None

    cells = int(probe.get("llama_token_cells") or 0)
    if cells <= 0 or block_size <= 0:
        return None

    pages_live = (cells + block_size - 1) // block_size
    registered = export_page_table(kv_slot)
    pages_registered = len(registered)
    if registered:
        pages_live = min(pages_live, pages_registered)

    verified = probe.get("tensor_layers_verified")
    expected = tensor_layers_expected
    if expected is None and probe.get("kv_n_layers") is not None:
        expected = int(probe["kv_n_layers"])

    bind_complete: bool | None = None
    if verified is not None and expected is not None:
        bind_complete = int(verified) == int(expected)

    return {
        "pages_live": pages_live,
        "pages_registered": pages_registered,
        "n_layers": n_layers,
        "kv_v_transposed": bool(probe.get("kv_v_transposed")),
        "physical_pages_bound": bool(probe.get("physical_pages_bound")),
        "tensor_layers_verified": int(verified) if verified is not None else None,
        "tensor_layers_expected": int(expected) if expected is not None else None,
        "tensor_layers_bind_complete": bind_complete,
        "full_plan_endpoint": "/internal/kv-snapshot",
    }


def redact_migration_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``plan`` with ``src_ptr`` fields removed from copy descriptors."""
    out = copy.deepcopy(plan)
    for page in out.get("pages") or []:
        for layer in page.get("layers") or []:
            for key in ("k_copy", "v_copy"):
                block = layer.get(key)
                if isinstance(block, dict):
                    block.pop("src_ptr", None)
    return out


def migration_include_pointers() -> bool:
    from runtime.env import kv_migration_include_pointers

    return kv_migration_include_pointers()


def prepare_migration_plan_for_export(plan: dict[str, Any]) -> dict[str, Any]:
    """Apply pointer redaction unless ``ZEROLLAMA_KV_MIGRATION_INCLUDE_PTRS=1``."""
    if migration_include_pointers():
        return plan
    return redact_migration_plan(plan)


def build_page_migration_plan(
    ctx_ptr: int,
    seq_id: int,
    kv_slot: int,
    *,
    block_size: int,
    probe: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a multi-page, multi-layer migration plan from a successful probe.

    Returns ``None`` when bind is not far enough along to map writable spans
    (needs ``tensor_pages_bound`` or ``physical_pages_bound`` on the probe).
    """
    if ctx_ptr == 0 or not tensor_probe_available():
        return None
    if not probe.get("tensor_pages_bound") and not probe.get("physical_pages_bound"):
        return None

    n_layers = int(probe.get("kv_n_layers") or probe.get("tensor_layers_verified") or 0)
    if n_layers <= 0:
        return None

    kv_cache_kv_size = probe.get("kv_cache_kv_size")
    kv_size = int(kv_cache_kv_size) if kv_cache_kv_size is not None else None
    cells = int(probe.get("llama_token_cells") or 0)
    if cells <= 0 or block_size <= 0:
        return None

    pages_live = (cells + block_size - 1) // block_size
    registered = export_page_table(kv_slot)
    if registered:
        pages_live = min(pages_live, len(registered))

    page_rows: list[dict[str, Any]] = []
    complete_pages = 0
    for page_index in range(pages_live):
        layer_descs: list[dict[str, Any]] = []
        for layer in range(n_layers):
            try:
                pm = map_page(ctx_ptr, seq_id, kv_slot, page_index, kv_layer=layer)
            except (KeyError, RuntimeError, OSError):
                break
            layer_descs.append(
                page_copy_descriptor(pm, kv_cache_kv_size=kv_size)
            )
        if len(layer_descs) == n_layers:
            complete_pages += 1
        page_rows.append(
            {
                "page_index": page_index,
                "layers_mapped": len(layer_descs),
                "layers": layer_descs,
            }
        )

    return {
        "kv_slot": kv_slot,
        "seq_id": seq_id,
        "n_layers": n_layers,
        "pages_live": pages_live,
        "pages_complete": complete_pages,
        "migration_pages_complete": complete_pages == pages_live and pages_live > 0,
        "kv_cache_kv_size": kv_size,
        "kv_v_transposed": bool(probe.get("kv_v_transposed")),
        "pages": page_rows,
        "external_buffer_alias_ready": False,
    }


def migration_plan_from_last_probe(
    ctx_ptr: int,
    seq_id: int,
    kv_slot: int,
    *,
    block_size: int,
    probe: dict[str, Any],
) -> dict[str, Any] | None:
    """Best-effort wrapper — same as ``build_page_migration_plan``."""
    return build_page_migration_plan(
        ctx_ptr,
        seq_id,
        kv_slot,
        block_size=block_size,
        probe=probe,
    )
