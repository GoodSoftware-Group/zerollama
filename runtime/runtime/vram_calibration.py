"""Probe-backed VRAM calibration after llama-server load (Phase 13).

Records estimated vs observed GPU memory delta so operators can tune
ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR without a unified autotuner loop.
Persisted via ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST when enabled.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_last_sample: dict[str, Any] | None = None


def vram_probe_calibrate_enabled() -> bool:
    """auto: on when GPU VRAM check is enabled."""
    v = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE", "auto").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    from runtime.gpu_vram import gpu_vram_check_enabled

    return gpu_vram_check_enabled()


def record_vram_load_sample(
    *,
    model_path: Path,
    device_index: int,
    estimated_raw_bytes: int,
    estimated_effective_bytes: int,
    free_before: int | None,
    free_after: int | None,
    probe: str | None,
    tensor_parallel: int = 1,
) -> None:
    """Store last load observation for /health (best-effort; not used for admission)."""
    global _last_sample
    observed: int | None = None
    if (
        free_before is not None
        and free_after is not None
        and free_before > free_after
    ):
        observed = free_before - free_after
    suggested: float | None = None
    if observed is not None and estimated_raw_bytes > 0:
        suggested = max(0.1, min(3.0, observed / estimated_raw_bytes))
    if suggested is not None and estimated_raw_bytes > 0:
        from runtime.gpu_vram import (
            set_session_vram_estimate_factor,
            vram_estimate_autotune_enabled,
        )

        if vram_estimate_autotune_enabled():
            set_session_vram_estimate_factor(
                suggested, model=str(model_path)
            )

    effective_bytes = estimated_effective_bytes
    if suggested is not None and estimated_raw_bytes > 0:
        effective_bytes = int(round(estimated_raw_bytes * suggested))
    precheck_bytes: int | None = None
    if effective_bytes != estimated_effective_bytes:
        precheck_bytes = estimated_effective_bytes

    with _lock:
        _last_sample = {
            "model": str(model_path),
            "device": device_index,
            "tensor_parallel": max(1, tensor_parallel),
            "estimated_raw_bytes": estimated_raw_bytes,
            "estimated_effective_bytes": effective_bytes,
            "observed_bytes": observed,
            "free_before": free_before,
            "free_after": free_after,
            "suggested_estimate_factor": suggested,
            "suggested_factor_note": (
                "Set ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR to suggested_estimate_factor "
                "(replaces raw estimate multiplier; not multiplied on top of current factor)."
            ),
            "active_estimate_factor": _active_estimate_factor(model_path),
            "autotune_active": _autotune_active(),
            "autotune_pending_first_load": _autotune_pending_first_load(),
            "probe_calibrate_required_for_autotune": (
                _autotune_requires_probe_calibrate()
            ),
            "probe": probe,
            "age_s": 0.0,
            "recorded_at": time.monotonic(),
        }
        if precheck_bytes is not None:
            _last_sample["estimated_precheck_bytes"] = precheck_bytes
        if tensor_parallel > 1:
            _last_sample["scope_warning"] = (
                "observed_bytes is free-V RAM delta on main_gpu only; "
                "estimated_* is per-GPU after tensor_parallel split"
            )
    if suggested is not None:
        from runtime.vram_factor_export import (
            export_factor_catalog,
            export_last_calibration,
        )

        export_last_calibration(
            suggested,
            model=model_path,
            observed_bytes=observed,
            estimated_raw_bytes=estimated_raw_bytes,
        )
        export_factor_catalog()


def _active_estimate_factor(model_path: Path) -> float:
    from runtime.gpu_vram import effective_vram_estimate_factor

    return effective_vram_estimate_factor(gguf=model_path)


def _autotune_active() -> bool:
    from runtime.gpu_vram import (
        session_vram_estimate_factor,
        vram_estimate_autotune_enabled,
    )

    return vram_estimate_autotune_enabled() and session_vram_estimate_factor() is not None


def _autotune_pending_first_load() -> bool:
    from runtime.gpu_vram import (
        session_vram_estimate_factor,
        vram_estimate_autotune_enabled,
    )

    return (
        vram_estimate_autotune_enabled()
        and session_vram_estimate_factor() is None
    )


def _autotune_requires_probe_calibrate() -> bool:
    from runtime.gpu_vram import vram_estimate_autotune_enabled

    return vram_estimate_autotune_enabled() and not vram_probe_calibrate_enabled()


def vram_calibration_health() -> dict[str, Any] | None:
    """Last load sample for /health (age_s updated on read)."""
    with _lock:
        if _last_sample is None:
            return None
        out = dict(_last_sample)
        recorded = out.pop("recorded_at", None)
        if isinstance(recorded, (int, float)):
            out["age_s"] = round(time.monotonic() - recorded, 3)
        return out


def maybe_record_vram_after_load(
    *,
    model_path: Path,
    device_index: int,
    free_before: int | None,
    estimated_raw_bytes: int,
    estimated_effective_bytes: int,
    tensor_parallel: int = 1,
    probe: str | None = None,
) -> None:
    """Probe free VRAM after start (fresh read) and compare to pre-load estimates."""
    if not vram_probe_calibrate_enabled():
        return
    from runtime.gpu_vram import active_vram_probe, nvidia_free_vram_bytes

    free_after = nvidia_free_vram_bytes(device_index, fresh=True)
    if probe is None:
        probe = active_vram_probe()
    record_vram_load_sample(
        model_path=model_path,
        device_index=device_index,
        estimated_raw_bytes=estimated_raw_bytes,
        estimated_effective_bytes=estimated_effective_bytes,
        free_before=free_before,
        free_after=free_after,
        probe=probe,
        tensor_parallel=tensor_parallel,
    )
