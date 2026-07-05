"""Phase 15 v8 — PA block pool → llama KV page bind (seq-position → tensor).

v0–v7 export logical page tables (``kv_forward_plans``) and bind llama sequence/slot ids.
v8 registers PA ``block_ids`` per ``kv_slot`` in native C for seq-position validation
before decode. v20+ with linked ``llama-kv-ext`` reports ``status=bound`` +
``bind_level=tensor`` after K/V tensor verify — GPU smokes accept that as success.
v33 adds ``llama_memory_kv_page_map`` in the forked llama.cpp tree: writable K/V spans
per PA page for external KV migration (``physical_pages_bound`` on /health).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from runtime.worker.llama_server import LlamaServerError

if TYPE_CHECKING:
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


def page_bind_last_tensor_probe_for_health() -> dict[str, Any] | None:
    """Return the best last decode probe when no running request is active."""
    if not _native_page_bind_available():
        return None
    try:
        from runtime.kv._kv_native import page_bind_last_tensor_probe
    except (ImportError, AttributeError):
        return None
    rows = page_bind_last_tensor_probe()
    if not rows:
        return None
    best: dict[str, Any] | None = None
    for row in rows:
        probe = dict(row.get("probe") or {})
        if probe.get("tensor_pages_bound"):
            return probe
        if best is None:
            best = probe
    return best


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
) -> dict[str, Any]:
    """Operator-facing status for tensor/page bind readiness.

    WHY status escalates partial → bound: seq-position bind proves PA accounting;
    cell_index adds llama cell map; tensor adds K/V backing verify via linked kv-ext.
    Smokes treat bound+tensor as the linked-build success path (see runtime_smoke_lib).
    """
    from runtime.kv.tensor_probe import writable_bind_probe as default_writable_probe

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
        "reason": (
            "build native ext (cd runtime && python3 setup.py build_ext --inplace); "
            "use kv_forward_plans for logical page tables"
        ),
        "native_ext_available": native_ext_available,
        "active_binds": active,
        "blocker": "native_ext_not_built",
        "slots": [],
    }
