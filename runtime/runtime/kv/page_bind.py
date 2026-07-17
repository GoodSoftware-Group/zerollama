"""Phase 15 v8–v47 — PA block pool → llama KV page bind (seq-position → tensor → alias validate).

v0–v7 export logical page tables (``kv_forward_plans``) and bind llama sequence/slot ids.
v8 registers PA ``block_ids`` per ``kv_slot`` in native C for seq-position validation
before decode. v20+ with linked ``llama-kv-ext`` reports ``status=bound`` +
``bind_level=tensor`` after K/V tensor verify — GPU smokes accept that as success.
v33 adds ``llama_memory_kv_page_map`` in the forked llama.cpp tree: writable K/V spans
per PA page for external KV migration (``physical_pages_bound`` on /health).
v34 fans out the writable page_map across all KV layers; ``tensor_layers_verified``
tracks how many layers were successfully backed.
v35 adds ``kv_v_transposed`` / ``kv_cache_kv_size`` / ``kv_cache_n_stream`` from
``llama_memory_kv_cache_layout``; ``page_bind_last_tensor_probe`` persists the last
decode probe so /health can show layout data after ``page_bind_clear``.
v36 adds GGUF-derived layer-group enrichment: ``kv_full_layers`` / ``kv_swa_layers`` /
``tensor_layers_expected`` on /health.kv_page_bind so operators can distinguish an
expected SWA-layer gap from a real bind failure on hybrid models (Gemma 3/4, etc.).
v47 adds external-buffer alias probe + validate on /health (patch 0019) — classifies
whether external PA pointers can zero-copy alias ``page_map`` spans without mutating
ggml tensors (``external_alias_*`` fields; true bind is v48+).

WHY the last-probe fallback: ``page_bind_clear`` fires on generation complete. At that
point /health has no running request to probe against. Without the snapshot, operators
would see ``status=partial`` (no live probe) even for a model that just completed a
fully-bound decode. ``last_tensor_probe=True`` in the health dict signals that the data
comes from the most recent completed generation, not a live request.

WHY v36 layer enrichment: the tensor probe reports ``kv_n_layers`` (from llama's attn
cache) and ``tensor_layers_verified`` (how many layers passed the bind attempt). For
hybrid models the llama attn cache only holds full-attention layers; SWA layers have a
separate windowed cache that is NOT the PA bind target. Without the GGUF layer-group
context, operators see ``tensor_layers_verified < kv_n_layers`` and cannot tell if the
gap is from SWA layers (expected) or a real bind failure on full-attention layers. With
``tensor_layers_expected`` set to the full-attention layer count, operators can compare:
``tensor_layers_verified == tensor_layers_expected`` → full bind, not a failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from runtime.worker.llama_server import LlamaServerError

if TYPE_CHECKING:
    from runtime.kv.hybrid_kv_coordinator import HybridKVCacheCoordinator
    from runtime.scheduler.scheduler import Request


def _native_page_bind_available() -> bool:
    try:
        from runtime.kv._kv_native import page_bind_set  # noqa: F401

        return True
    except ImportError:
        return False
    except Exception:
        return False


def register_request_bind(req: Request, *, block_size: int) -> None:
    """Register admitted request page table for native seq-position bind.

    Why on admit (not at decode time): the PA allocator already chose block_ids;
    native C registry lets decode validate token positions against that table before
    each llama_batch — fail fast instead of silent llama KV overrun.
    """
    if req.kv_slot is None or req.kv_slot < 0:
        return
    from runtime.kv.forward_plan import kv_forward_plan

    plan = kv_forward_plan(req, block_size=block_size)
    pages = plan.get("pages") or []
    if not pages:
        return
    block_ids = [int(p["block_id"]) for p in pages]
    if _native_page_bind_available():
        from runtime.kv._kv_native import page_bind_set

        page_bind_set(int(req.kv_slot), int(block_size), block_ids)


def unregister_request_bind(kv_slot: int | None) -> None:
    if kv_slot is None or kv_slot < 0:
        return
    if _native_page_bind_available():
        from runtime.kv._kv_native import page_bind_clear

        page_bind_clear(int(kv_slot))


def validate_token_positions(
    kv_slot: int, token_start: int, n_tokens: int
) -> None:
    """Fail fast when decode positions fall outside registered page table.

    Why check endpoints only: batch positions are contiguous; if first and last
    token index resolve, every index in between lies in the same page table.
    Raises LlamaServerError (not raw C exceptions) so generate/stream paths surface
    a clean operator error.
    """
    if not _native_page_bind_available() or n_tokens <= 0:
        return
    from runtime.kv._kv_native import page_bind_resolve

    end = token_start + n_tokens
    for pos in (token_start, end - 1):
        try:
            page_bind_resolve(int(kv_slot), int(pos))
        except (KeyError, ValueError) as e:
            raise LlamaServerError(
                f"KV page bind: token position {pos} out of range for kv_slot {kv_slot}"
            ) from e


def page_bind_stats() -> dict[str, Any]:
    if _native_page_bind_available():
        from runtime.kv._kv_native import page_bind_slots, page_bind_stats as native_stats

        raw = dict(native_stats())
        raw["tensor_pages_bound"] = bool(raw.get("tensor_pages_bound"))
        raw["physical_pages_bound"] = bool(raw.get("physical_pages_bound"))
        raw["slots"] = [
            {
                **dict(s),
                "cell_pages_bound": bool(s.get("cell_pages_bound")),
                "tensor_pages_bound": bool(s.get("tensor_pages_bound")),
                "physical_pages_bound": bool(s.get("physical_pages_bound")),
            }
            for s in page_bind_slots()
        ]
        return raw
    return {
        "active_binds": 0,
        "total_registers": 0,
        "tensor_pages_bound": False,
        "physical_pages_bound": False,
        "slots": [],
    }


def page_bind_last_probe_row_for_health() -> dict[str, Any] | None:
    """Return ``{kv_slot, probe}`` for the best last decode probe snapshot.

    WHY separate from ``page_bind_last_tensor_probe_for_health``: migration summary
    needs the kv_slot that produced the probe (export_page_table is slot-scoped).
    """
    if not _native_page_bind_available():
        return None
    try:
        from runtime.kv._kv_native import page_bind_last_tensor_probe
    except (ImportError, AttributeError):
        return None
    rows = page_bind_last_tensor_probe()
    if not rows:
        return None
    best_row: dict[str, Any] | None = None
    for row in rows:
        probe = dict(row.get("probe") or {})
        entry = {"kv_slot": int(row.get("kv_slot", -1)), "probe": probe}
        if probe.get("tensor_pages_bound"):
            return entry
        if best_row is None:
            best_row = entry
    return best_row


def page_bind_last_tensor_probe_for_health() -> dict[str, Any] | None:
    """Return the best last decode probe when no running request is active.

    WHY needed: after ``page_bind_clear`` fires on generation complete there is no
    running request to probe, so the live probe path returns None. This function
    falls back to the last-probe snapshot stored by ``kv_tensor_probe_last_save``
    during the most recent decode that reached ``tensor_pages_bound=True``.

    Preference order: first probe with ``tensor_pages_bound=True`` wins; any probe
    returned if none reached tensor bound (gives layout data even on partial bind).
    Returns None when native ext is not built or no snapshot exists yet.
    """
    row = page_bind_last_probe_row_for_health()
    return row["probe"] if row else None


def page_bind_tensor_probe_for_ctx(
    lib: Any,
    ctx: Any,
    *,
    seq_id: int,
    kv_slot: int,
) -> dict[str, Any] | None:
    """Run v19 tensor probe when in-process shared ctx is loaded."""
    from runtime.kv.tensor_probe import run_tensor_probe

    if ctx is None:
        return None
    try:
        ctx_ptr = int(ctx) if isinstance(ctx, int) else int(getattr(ctx, "value", 0) or 0)
    except (TypeError, ValueError):
        return None
    _ = lib
    return run_tensor_probe(ctx_ptr, seq_id, kv_slot)


def page_bind_health(
    *,
    native_ext_available: bool,
    tensor_probe: dict[str, Any] | None = None,
    writable_probe: dict[str, Any] | None = None,
    external_alias_probe: dict[str, Any] | None = None,
    overlay_bind_donor_id: int | None = None,
    overlay_donor_base: int | None = None,
    overlay_donor_size: int | None = None,
    overlay_catalog_ctx: tuple[int, int, int] | None = None,
    kv_coordinator: "HybridKVCacheCoordinator | None" = None,
    kv_slot: int | None = None,
    block_size: int | None = None,
) -> dict[str, Any]:
    """Operator-facing status for tensor/page bind readiness.

    WHY status escalates partial → bound: seq-position bind proves PA accounting;
    cell_index adds llama cell map; tensor adds K/V backing verify via linked kv-ext.
    Smokes treat bound+tensor as the linked-build success path (see runtime_smoke_lib).

    WHY kv_coordinator (v36): hybrid models (Gemma 3/4, etc.) have both full-attention
    and SWA layers. The llama attn cache only backs full-attention layers — SWA layers
    use a separate windowed cache that is NOT the PA bind target. Without the coordinator,
    operators see ``tensor_layers_verified < kv_n_layers`` and cannot tell if the gap is
    an expected SWA-layer exclusion or a real bind failure on full-attention layers.
    When provided, ``kv_full_layers`` / ``kv_swa_layers`` / ``tensor_layers_expected``
    are added to the output so a correct comparison is possible.

    WHY overlay_bind_donor_id (v48 CPU / v49 Metal): donor-buffer registration
    happens at model-load time, outside any single decode/probe call — the
    engine passes the donor id it registered (if any) so operators can see
    whether the zero-copy KV allocation actually happened
    (``overlay_bind_bound``) without needing a separate endpoint. Whether the
    consuming buft was CPU-host (v48) or a device buft like Metal (v49) is an
    internal vendor-hook decision; this status reflects the outcome either
    way, not which path was taken.

    WHY overlay_donor_base/size + overlay_catalog_ctx (v51): when the donor is
    bound, publish a read-only page→donor-offset summary so L3-R6 can prove PA
    pages are addressable ranges inside the owned buffer (no allocator rewrite).
    """
    from runtime.kv.overlay_bind import (
        donor_buffer_status,
        overlay_bind_auto_enabled,
        overlay_bind_enabled,
    )
    from runtime.kv.tensor_probe import (
        external_alias_probe as default_external_alias_probe,
        writable_bind_probe as default_writable_probe,
    )

    stats = page_bind_stats()
    active = int(stats.get("active_binds") or 0)
    base_probe = tensor_probe or {}
    last_probe_fallback = False
    if not tensor_probe:
        last = page_bind_last_tensor_probe_for_health()
        if last:
            base_probe = last
            last_probe_fallback = True
    writable = writable_probe if writable_probe is not None else default_writable_probe()
    # WHY external alias on health: v47 validate is build-time + per-page; operators need
    # the same static visibility as writable_bind_* before migration bind ships (v48).
    ext_alias = (
        external_alias_probe
        if external_alias_probe is not None
        else default_external_alias_probe()
    )

    # v48/v49/v50: donor-buffer overlay bind status (opt-in; see overlay_bind.py).
    # v48 consumes CPU-host buft groups, v49 additionally consumes
    # buffer_from_host_ptr-capable device buft groups (Metal); both paths are
    # reported identically here since the caller only registers/queries a
    # donor id and does not choose which buft consumes it.
    # v50 adds overlay_bind_auto (in-process auto-wire on load).
    overlay_enabled = overlay_bind_enabled()
    overlay_auto = overlay_bind_auto_enabled()
    overlay_bound = False
    overlay_bytes = None
    if overlay_enabled and overlay_bind_donor_id is not None:
        donor_status = donor_buffer_status(int(overlay_bind_donor_id))
        if donor_status is not None:
            overlay_bound = bool(donor_status.get("bound"))
            overlay_bytes = int(donor_status.get("bytes_used") or 0)

    # v51: read-only donor page-offset summary (L3-R6 geometry).
    overlay_catalog_summary: dict[str, Any] | None = None
    if (
        overlay_bound
        and overlay_donor_base
        and overlay_donor_size
        and overlay_catalog_ctx is not None
        and block_size
    ):
        try:
            from runtime.kv.overlay_page_catalog import (
                build_overlay_page_catalog,
                health_page_cap,
                overlay_page_catalog_summary,
            )

            ctx_ptr, seq_id, cat_slot = overlay_catalog_ctx
            full = build_overlay_page_catalog(
                donor_base=int(overlay_donor_base),
                donor_size=int(overlay_donor_size),
                ctx_ptr=int(ctx_ptr),
                seq_id=int(seq_id),
                kv_slot=int(cat_slot),
                block_size=int(block_size),
                probe=base_probe or None,
                max_pages=health_page_cap(),
                include_pages=False,
            )
            overlay_catalog_summary = overlay_page_catalog_summary(full)
        except Exception:
            overlay_catalog_summary = None

    # Normalise int 0/1 from C to bool; None means no probe was run.
    def _bool_probe(key: str) -> bool | None:
        v = base_probe.get(key)
        return bool(v) if v is not None else None

    accounting_ok = _bool_probe("aligned") if base_probe else None
    cell_bound = bool(base_probe.get("cell_pages_bound")) if base_probe else False
    tensor_bound = bool(base_probe.get("tensor_pages_bound")) if base_probe else False
    physical_bound = bool(base_probe.get("physical_pages_bound")) if base_probe else False

    if native_ext_available and _native_page_bind_available():
        if tensor_bound:
            status = "bound"
            bind_level = "tensor"
            reason = (
                "PA block_ids mapped to llama KV cells; K/V tensor backing verified "
                "(zerollama llama-kv-ext staging API)."
            )
            if physical_bound:
                bind_level = "physical"
                reason = (
                    "Writable K/V tensor spans resolved for live PA pages "
                    "(llama_memory_kv_page_map)."
                )
        elif cell_bound:
            status = "partial"
            bind_level = "cell_index"
            reason = (
                "PA pages resolved to llama KV cell indices; K/V tensors not yet "
                "materialized or unsupported memory type."
            )
        else:
            status = "partial"
            bind_level = "seq_position"
            reason = (
                "PA block_ids registered per kv_slot in native C; "
                "llama cell/tensor bind pending decode or llama-kv-ext link."
            )

        if base_probe:
            # Override reason with more specific message once memory is available.
            if base_probe.get("memory_non_null") and not tensor_bound:
                reason = (
                    "llama_get_memory available; PA page table vs seq positions "
                    "verified (accounting bind). Run decode + linked llama-kv-ext for tensor bind."
                )
            # Misaligned only fires when tensor bind has NOT succeeded — otherwise
            # it would contradict tensor_bind_ready=True.
            if not tensor_bound and accounting_ok is not None and not accounting_ok:
                status = "misaligned"
                bind_level = "seq_position"
                reason = (
                    "llama token cells exceed PA page reserve for kv_slot; "
                    "check kv_forward_plans and admission"
                )

        # WHY blocker logic: if the probe ran, use its blocker string (most accurate).
        # cell_bound (but not tensor_bound) → blocker from probe explains why tensor failed.
        # no probe at all → indicate that linking llama-kv-ext is the next step.
        if tensor_bound:
            blocker = ""
        elif base_probe:
            blocker = base_probe.get("blocker") or ""
        else:
            blocker = "llama_kv_ext_not_linked_or_no_decode"

        out: dict[str, Any] = {
            "available": True,
            "status": status,
            "bind_level": bind_level,
            "tensor_pages_bound": tensor_bound,
            "tensor_bind_ready": tensor_bound,
            "physical_pages_bound": physical_bound,
            "writable_bind_ready": physical_bound,
            "writable_bind_available": bool(writable.get("writable_bind_available")),
            "writable_bind_api": writable.get("writable_bind_api") or "none",
            "writable_bind_blocker": writable.get("writable_bind_blocker") or "",
            "external_alias_available": bool(ext_alias.get("external_alias_available")),
            "external_alias_api": ext_alias.get("external_alias_api") or "none",
            "external_alias_blocker": ext_alias.get("external_alias_blocker") or "",
            "overlay_bind_enabled": overlay_enabled,
            "overlay_bind_auto": overlay_auto,
            "overlay_bind_bound": overlay_bound,
            "overlay_bind_bytes": overlay_bytes,
            "overlay_page_catalog": overlay_catalog_summary,
            "reason": reason,
            "native_ext_available": True,
            "active_binds": active,
            "total_registers": int(stats.get("total_registers") or 0),
            "blocker": blocker,
            "slots": stats.get("slots") or [],
        }
        if base_probe:
            out["tensor_probe"] = base_probe
            out["accounting_aligned"] = bool(accounting_ok) if accounting_ok is not None else None
            out["cell_pages_bound"] = cell_bound
            if base_probe.get("kv_n_layers") is not None:
                out["kv_n_layers"] = int(base_probe["kv_n_layers"])
            if base_probe.get("tensor_layers_verified") is not None:
                out["tensor_layers_verified"] = int(base_probe["tensor_layers_verified"])
            if base_probe.get("kv_v_transposed") is not None:
                out["kv_v_transposed"] = bool(base_probe["kv_v_transposed"])
            if base_probe.get("kv_cache_kv_size") is not None:
                out["kv_cache_kv_size"] = int(base_probe["kv_cache_kv_size"])
            if base_probe.get("kv_cache_n_stream") is not None:
                out["kv_cache_n_stream"] = int(base_probe["kv_cache_n_stream"])
            if last_probe_fallback:
                out["last_tensor_probe"] = True

        # v36: layer-group enrichment from GGUF coordinator.
        # WHY: for hybrid models (full-attn + SWA), tensor_layers_verified < kv_n_layers
        # is EXPECTED because the llama attn cache only holds full-attention layers.
        # Emitting kv_full_layers / kv_swa_layers / tensor_layers_expected lets operators
        # distinguish an expected SWA gap from a real bind failure.
        if kv_coordinator is not None:
            full = kv_coordinator.full_layer_count
            swa = kv_coordinator.swa_layer_count
            kind = kv_coordinator.kind
            out["kv_coordinator_kind"] = kind
            if kind in ("hybrid", "sliding_window"):
                out["kv_full_layers"] = full
                out["kv_swa_layers"] = swa
                # For a hybrid model, the bind target is the full-attention cache only.
                # tensor_layers_expected == full gives operators the right comparison.
                out["tensor_layers_expected"] = full
            else:
                # Standard model: all layers are full-attention; expected == n_layers.
                n_layers = out.get("kv_n_layers")
                if n_layers is not None:
                    out["tensor_layers_expected"] = int(n_layers)

        # v38: single boolean for bind-success (uses v36 expected count when present).
        verified = out.get("tensor_layers_verified")
        expected = out.get("tensor_layers_expected")
        if verified is not None and expected is not None:
            out["tensor_layers_bind_complete"] = int(verified) == int(expected)
        elif verified is not None and out.get("kv_n_layers") is not None:
            out["tensor_layers_bind_complete"] = int(verified) == int(out["kv_n_layers"])

        # v42: lightweight migration summary on /health when bind progressed far
        # enough for operators to see page/layer progress without kv-snapshot fan-out.
        if block_size and base_probe and (tensor_bound or physical_bound):
            summary_slot = kv_slot
            if summary_slot is None:
                row = page_bind_last_probe_row_for_health()
                if row is not None:
                    summary_slot = int(row["kv_slot"])
            if summary_slot is not None and summary_slot >= 0:
                from runtime.kv.page_migration_plan import migration_plan_summary

                summary = migration_plan_summary(
                    base_probe,
                    block_size=int(block_size),
                    kv_slot=int(summary_slot),
                    tensor_layers_expected=out.get("tensor_layers_expected"),
                )
                if summary:
                    out["page_migration_summary"] = summary

        return out

    return {
        "available": False,
        "status": "not_implemented",
        "bind_level": None,
        "tensor_pages_bound": False,
        "tensor_bind_ready": False,
        "writable_bind_available": False,
        "writable_bind_api": "none",
        "writable_bind_blocker": "native_ext_not_built",
        "external_alias_available": False,
        "external_alias_api": "none",
        "external_alias_blocker": "native_ext_not_built",
        "overlay_bind_enabled": overlay_enabled,
        "overlay_bind_auto": overlay_auto,
        "overlay_bind_bound": False,
        "overlay_bind_bytes": None,
        "overlay_page_catalog": None,
        "reason": (
            "build native ext (cd runtime && python3 setup.py build_ext --inplace); "
            "use kv_forward_plans for logical page tables"
        ),
        "native_ext_available": native_ext_available,
        "active_binds": active,
        "blocker": "native_ext_not_built",
        "slots": [],
    }
