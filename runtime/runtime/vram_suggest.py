"""VRAM-aware num_ctx suggestions and optional clamp (Phase 13).

Why this module exists: Phase 11 admission uses coarse free-VRAM gates; this module
answers "what is the largest num_ctx that fits?" using the same estimate path as
``check_gguf_vram_budget``. Clamp is opt-in (default off) so operators are not surprised
by silent context reduction — see ``vram_num_ctx_policy_health`` and API ``vram_num_ctx``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from runtime.gpu.priority import InferencePriority, priority_from_options

_log = logging.getLogger(__name__)


def _suggest_ctx_max_cap() -> int:
    raw = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_SUGGEST_CTX_MAX", "131072").strip()
    try:
        return max(512, int(raw))
    except ValueError:
        return 131072


def build_suggest_profile(
    vram_estimate: dict[str, Any] | None = None,
    *,
    tensor_parallel: int = 1,
    llama_args: list[str] | None = None,
    options: dict | None = None,
    parallel_slots_default: int = 1,
    n_gpu_layers_default: int = -1,
    draft_gguf: Path | None = None,
    draft_n_gpu_layers: int = -1,
) -> dict[str, Any]:
    """Kwargs for ``suggest_max_num_ctx`` aligned with ``describe_vram_estimate`` / load."""
    kw: dict[str, Any] = {
        "tensor_parallel": tensor_parallel,
        "llama_args": llama_args,
        "options": options,
        "parallel_slots_default": parallel_slots_default,
        "n_gpu_layers_default": n_gpu_layers_default,
        "draft_gguf": draft_gguf,
        "draft_n_gpu_layers": draft_n_gpu_layers,
    }
    if not vram_estimate:
        return kw
    n_gl = vram_estimate.get("n_gpu_layers")
    if isinstance(n_gl, int):
        kw["n_gpu_layers_default"] = n_gl
    slots = vram_estimate.get("parallel_slots")
    if isinstance(slots, int) and slots > 0:
        kw["parallel_slots_default"] = slots
    draft = vram_estimate.get("draft_model")
    if isinstance(draft, str) and draft.strip():
        p = Path(draft)
        if p.is_file():
            kw["draft_gguf"] = p
    return kw


def _resolve_min_free(min_free_bytes: int | None) -> int | None:
    if min_free_bytes is not None:
        return min_free_bytes if min_free_bytes > 0 else None
    from runtime.gpu.admission import (
        admission_vram_gate_enabled,
        min_free_vram_for_admission,
    )

    if admission_vram_gate_enabled():
        return min_free_vram_for_admission()
    return None


def _required_for_load(
    estimate_bytes: int,
    *,
    margin: float,
    min_free: int | None,
    priority: InferencePriority,
) -> int:
    """Same floor as ``check_gguf_vram_budget`` (max(estimate×margin, min_free))."""
    req = int(estimate_bytes * margin)
    if min_free is None:
        return req
    from runtime.gpu.admission import (
        admission_vram_gate_enabled,
        effective_min_free_for_priority,
        vram_gate_bypassed,
    )

    if not admission_vram_gate_enabled() or vram_gate_bypassed(priority):
        return req
    return max(req, effective_min_free_for_priority(min_free, priority))


def suggest_max_num_ctx(
    gguf: Path,
    effective_free_bytes: int,
    *,
    margin: float | None = None,
    min_free_bytes: int | None = None,
    priority: InferencePriority | None = None,
    tensor_parallel: int = 1,
    options: dict | None = None,
    llama_args: list[str] | None = None,
    parallel_slots_default: int = 1,
    n_gpu_layers_default: int = -1,
    draft_gguf: Path | None = None,
    draft_n_gpu_layers: int = -1,
) -> int | None:
    """Largest num_ctx whose load requirement fits in effective_free_bytes.

    Uses the same estimate path and admission floor as load pre-check.
    """
    if effective_free_bytes <= 0:
        return None
    if margin is None:
        margin = float(os.environ.get("ZEROLLAMA_RUNTIME_VRAM_MARGIN", "1.0"))
    margin = max(1.0, margin)
    if priority is None:
        priority = priority_from_options(options)

    from runtime.gguf_estimate import gguf_arch_hints
    from runtime.gpu_vram import estimate_gguf_vram_bytes

    try:
        resolved = gguf.resolve()
    except OSError:
        resolved = gguf
    if not resolved.is_file():
        return None

    min_free = _resolve_min_free(min_free_bytes)

    arch = gguf_arch_hints(resolved)
    meta_ctx = arch.scalar.get("context_length") or 0
    hi = min(_suggest_ctx_max_cap(), meta_ctx if meta_ctx > 0 else _suggest_ctx_max_cap())
    lo = 512

    kw: dict[str, Any] = {
        "tensor_parallel": tensor_parallel,
        "options": options,
        "llama_args": llama_args,
        "parallel_slots_default": parallel_slots_default,
        "n_gpu_layers_default": n_gpu_layers_default,
        "draft_gguf": draft_gguf,
        "draft_n_gpu_layers": draft_n_gpu_layers,
    }

    def _estimate_bytes(ctx: int) -> int:
        return estimate_gguf_vram_bytes(resolved, num_ctx=ctx, **kw)

    def _fits(ctx: int) -> bool:
        return _required_for_load(
            _estimate_bytes(ctx),
            margin=margin,
            min_free=min_free,
            priority=priority,
        ) <= effective_free_bytes

    if not _fits(lo):
        return None
    if _fits(hi):
        return hi

    best = lo
    left, right = lo, hi
    while left <= right:
        mid = (left + right) // 2
        if _fits(mid):
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return best


def format_suggest_num_ctx_hint(
    gguf: Path,
    effective_free_bytes: int,
    *,
    margin: float | None = None,
    min_free_bytes: int | None = None,
    priority: InferencePriority | None = None,
    num_ctx: int | None = None,
    options: dict | None = None,
    **profile: Any,
) -> str:
    """Actionable suffix for VRAM reject errors (best-effort; empty on failure)."""
    try:
        suggested = suggest_max_num_ctx(
            gguf,
            effective_free_bytes,
            margin=margin,
            min_free_bytes=min_free_bytes,
            priority=priority,
            options=options,
            **profile,
        )
    except OSError:
        return ""
    if suggested is None:
        return ""
    if num_ctx is None and options:
        from runtime.gpu_vram import resolve_vram_num_ctx

        num_ctx = resolve_vram_num_ctx(
            options,
            gguf,
            llama_args=profile.get("llama_args"),
        )
    if isinstance(num_ctx, int) and num_ctx > suggested:
        return f"; try num_ctx<={suggested} (requested {num_ctx})"
    return f"; suggested_max_num_ctx={suggested}"


def api_vram_num_ctx_meta(
    meta: dict[str, Any],
    effective_ctx: int | None,
) -> dict[str, Any] | None:
    """Client-visible slice when num_ctx was clamped for VRAM.

    Why expose in API: clamp only logged a warning; agents and operators need the
    effective context in the response/stream, not only in logs or /health.
    """
    if not meta.get("num_ctx_clamped"):
        return None
    out: dict[str, Any] = {"num_ctx_clamped": True}
    if isinstance(meta.get("num_ctx_clamped_from"), int):
        out["num_ctx_clamped_from"] = meta["num_ctx_clamped_from"]
    if effective_ctx is not None:
        out["num_ctx"] = effective_ctx
    if isinstance(meta.get("suggested_max_num_ctx"), int):
        out["suggested_max_num_ctx"] = meta["suggested_max_num_ctx"]
    return out


def vram_num_ctx_clamp_enabled() -> bool:
    """Whether to lower request num_ctx to suggested_max when it exceeds VRAM budget.

    Default ``0`` (off). ``auto`` follows ``CHECK_GPU_VRAM`` — why: silent clamp broke
    operator trust; single-GPU smoke can opt in with ``auto`` or ``1``.
    """
    v = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX", "0").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    from runtime.gpu_vram import gpu_vram_check_enabled

    return gpu_vram_check_enabled()


def vram_num_ctx_policy_health() -> dict[str, Any]:
    raw = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX", "0").strip()
    return {
        "clamp_enabled": vram_num_ctx_clamp_enabled(),
        "env": raw or "0",
        "note": (
            "Default off (0). Set 1 or auto (with CHECK_GPU_VRAM=1) to lower num_ctx "
            "above suggested_max before enqueue; responses include vram_num_ctx when clamped."
        ),
    }


def cap_num_ctx_for_vram(
    gguf: Path,
    num_ctx: int | None,
    effective_free_bytes: int | None,
    *,
    options: dict | None = None,
    priority: InferencePriority | None = None,
    **profile: Any,
) -> tuple[int | None, dict[str, Any]]:
    """Optionally clamp num_ctx to VRAM suggestion; return (ctx, meta)."""
    meta: dict[str, Any] = {}
    if num_ctx is None or num_ctx <= 0 or effective_free_bytes is None or effective_free_bytes <= 0:
        return num_ctx, meta
    if not vram_num_ctx_clamp_enabled():
        return num_ctx, meta
    try:
        suggested = suggest_max_num_ctx(
            gguf,
            effective_free_bytes,
            options=options,
            priority=priority,
            **profile,
        )
    except OSError:
        return num_ctx, meta
    if suggested is None:
        return num_ctx, meta
    meta["suggested_max_num_ctx"] = suggested
    if num_ctx <= suggested:
        return num_ctx, meta
    meta["num_ctx_clamped_from"] = num_ctx
    meta["num_ctx_clamped"] = True
    _log.warning(
        "num_ctx clamped %s -> %s for VRAM (gguf=%s)",
        num_ctx,
        suggested,
        gguf,
    )
    return suggested, meta
