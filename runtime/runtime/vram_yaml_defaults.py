"""Apply optional ``vram:`` block from runtime YAML when env is unset (Phase 13).

WHY: ``autoconfig`` already picks ``single_gpu.yaml`` on one GPU. Operators should not have
to duplicate min-free, training-reserve, and autotune env in every systemd unit when the
same defaults belong in-repo. Env always wins when set — production overrides stay explicit.
Applied before ``apply_exported_vram_env`` so exported factor files remain opt-in.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_APPLIED = False
_APPLY_RESULT: dict[str, Any] | None = None

# YAML key → process env (only set when env empty).
_VRAM_ENV_MAP: tuple[tuple[str, str], ...] = (
    ("min_free", "ZEROLLAMA_RUNTIME_VRAM_MIN_FREE"),
    ("training_reserve", "ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE"),
    ("estimate_factor", "ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR"),
    ("estimate_factor_autotune", "ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE"),
    ("probe_calibrate", "ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE"),
    ("clamp_num_ctx", "ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX"),
    ("margin", "ZEROLLAMA_RUNTIME_VRAM_MARGIN"),
    ("apply_exported_env", "ZEROLLAMA_RUNTIME_VRAM_APPLY_EXPORTED_ENV"),
    ("check_gpu_vram", "ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM"),
    ("inference_policy", "ZEROLLAMA_RUNTIME_INFERENCE_POLICY"),
)


def _load_vram_block(path: Path) -> dict[str, Any]:
    from runtime.config import _load_yaml

    raw = _load_yaml(path).get("vram")
    return raw if isinstance(raw, dict) else {}


def apply_vram_defaults_from_config(
    config_path: Path | None = None, *, force: bool = False
) -> dict[str, Any]:
    """Set ``ZEROLLAMA_RUNTIME_*`` from YAML ``vram:`` when not already in env."""
    global _APPLIED, _APPLY_RESULT
    if _APPLIED and not force and _APPLY_RESULT is not None:
        return dict(_APPLY_RESULT)

    result: dict[str, Any] = {"applied": [], "skipped": []}
    if config_path is None:
        from runtime.autoconfig import resolved_config_path

        config_path = resolved_config_path()

    result["config_path"] = str(config_path)
    if not config_path.is_file():
        result["reason"] = "no_config_file"
        _APPLIED = True
        _APPLY_RESULT = result
        return dict(result)

    block = _load_vram_block(config_path)
    if not block:
        result["reason"] = "no_vram_block"
        _APPLIED = True
        _APPLY_RESULT = result
        return dict(result)

    for yaml_key, env_key in _VRAM_ENV_MAP:
        if yaml_key not in block:
            continue
        if os.environ.get(env_key, "").strip():
            result["skipped"].append(env_key)
            continue
        value = block[yaml_key]
        if value is None:
            continue
        os.environ[env_key] = str(value).strip()
        result["applied"].append(env_key)

    _APPLIED = True
    _APPLY_RESULT = result
    return dict(result)


def apply_status() -> dict[str, Any]:
    """Snapshot for tests / diagnostics."""
    out: dict[str, Any] = {}
    if _APPLY_RESULT is not None:
        out.update(_APPLY_RESULT)
    else:
        out["reason"] = "not_run"
    return out
