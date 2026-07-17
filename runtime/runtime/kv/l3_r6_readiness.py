"""Phase 15 — L3-R6 metadata-path readiness + L3-R6b COW probe."""

from __future__ import annotations

from typing import Any


def l3_r6_metadata_readiness(
    *,
    n_ctx: int | None = None,
    n_parallel: int = 1,
    backend: str | None = None,
) -> dict[str, Any]:
    """Operator probe: L3-R6 metadata complete + L3-R6b COW status."""
    from runtime.env import (
        kv_cow_enabled,
        kv_cow_pages_enabled,
        kv_cow_pages_source,
        kv_cow_source,
        kv_cow_tensors_enabled,
        kv_cow_tensors_source,
        kv_unified_enabled,
        kv_unified_source,
        kv_unified_sizing_status,
        kv_unified_strict_sizing_enabled,
        radix_prefix_share_enabled,
    )
    from runtime.kv.idle_slot_purge import idle_slot_purge_enabled
    from runtime.kv.radix_seq_copy import seq_cp_mode

    unified = kv_unified_enabled()
    source = kv_unified_source()
    radix = radix_prefix_share_enabled()
    mode = seq_cp_mode()
    cow = kv_cow_enabled()
    cow_src = kv_cow_source()
    cow_t = kv_cow_tensors_enabled()
    cow_t_src = kv_cow_tensors_source()
    cow_p = kv_cow_pages_enabled()
    cow_p_src = kv_cow_pages_source()
    bk = (backend or "").strip().lower()
    if not unified:
        idle_ready = False
    elif bk in ("subprocess", "llama-server", ""):
        idle_ready = True
    elif bk in ("inprocess", "ctypes", "libllama"):
        idle_ready = idle_slot_purge_enabled(kv_unified=True)
    else:
        idle_ready = True

    sizing = kv_unified_sizing_status(n_ctx=n_ctx, n_parallel=n_parallel)
    sizing_ok = True
    if sizing is not None and sizing.get("ok") is False and kv_unified_strict_sizing_enabled():
        sizing_ok = False

    checks = {
        "kv_unified": unified,
        "kv_unified_source": source,
        "seq_cp_mode_metadata": (not radix) or mode == "metadata",
        "idle_purge_ready": idle_ready,
        "sizing_strict_ok": sizing_ok,
        "kv_cow_env": cow,
        "kv_cow_source": cow_src,
        "kv_cow_tensors": cow_t,
        "kv_cow_tensors_source": cow_t_src,
        "kv_cow_pages": cow_p,
        "kv_cow_pages_source": cow_p_src,
    }
    complete = all(
        (
            checks["kv_unified"],
            checks["seq_cp_mode_metadata"],
            checks["idle_purge_ready"],
            checks["sizing_strict_ok"],
        )
    )
    deferred = [
        "non_agent_global_default_on",
        "nixl_rdma_blobs",
    ]
    if not cow:
        deferred.insert(0, "true_cow_metadata_cells_env_off")
    if cow and not cow_t:
        deferred.insert(0, "tensor_deep_copy_cow_env_off")
    if cow_t and not cow_p:
        deferred.insert(0, "page_granular_cow_optimization")

    if cow and cow_t and cow_p:
        l3_r6b = "done"
    elif cow and cow_t:
        l3_r6b = "done_full_tensor"
    elif cow:
        l3_r6b = "partial_cells"
    else:
        l3_r6b = "opt_in"

    note_parts = []
    if complete:
        note_parts.append(
            "L3-R6 metadata path (v50–v58): unified metadata seq_cp + idle purge + sizing."
        )
    else:
        note_parts.append(
            "L3-R6 metadata incomplete — enable kv_unified (YAML/env/radix couple) "
            "and check failed checks."
        )
    if cow and cow_t and cow_p:
        note_parts.append(
            f"L3-R6b Done: cells ({cow_src}) + tensors ({cow_t_src}) + used-cell pages "
            f"({cow_p_src}); VRAM still full-size at fork."
        )
    elif cow and cow_t:
        note_parts.append(
            f"L3-R6b Done (full-tensor copy): cells ({cow_src}) + tensors ({cow_t_src}); "
            "enable kv_cow_pages for used-cell-range copy."
        )
    elif cow:
        note_parts.append(
            f"L3-R6b Partial: cell COW on ({cow_src}); enable kv_cow_tensors."
        )
    else:
        note_parts.append(
            "L3-R6b: enable agent YAML l3.kv_cow[+kv_cow_tensors][+kv_cow_pages] "
            "or ZEROLLAMA_KV_COW[_TENSORS|_PAGES]=1."
        )
    return {
        "complete": complete,
        "checks": checks,
        "seq_cp_mode": mode,
        "radix_share": radix,
        "kv_cow": cow,
        "kv_cow_source": cow_src,
        "kv_cow_tensors": cow_t,
        "kv_cow_tensors_source": cow_t_src,
        "kv_cow_pages": cow_p,
        "kv_cow_pages_source": cow_p_src,
        "l3_r6b": l3_r6b,
        "deferred": deferred,
        "note": " ".join(note_parts),
    }
