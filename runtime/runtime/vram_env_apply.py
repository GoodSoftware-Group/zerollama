"""Apply exported VRAM_ESTIMATE_FACTOR on startup (Phase 13, explicit opt-in)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_APPLIED = False
_APPLY_RESULT: dict[str, Any] | None = None


def vram_apply_exported_env_enabled() -> bool:
    """Only when ZEROLLAMA_RUNTIME_VRAM_APPLY_EXPORTED_ENV=1 (not auto)."""
    from runtime.env import vram_apply_exported_env_enabled as _enabled

    return _enabled()


def apply_export_path() -> Path:
    from runtime.env import vram_apply_exported_env_path

    override = vram_apply_exported_env_path()
    if override:
        return Path(override).expanduser()
    from runtime.vram_factor_export import export_last_factor_path

    return export_last_factor_path()


def _parse_env_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            out[key] = value
    return out


def apply_exported_vram_env(*, force: bool = False) -> dict[str, Any]:
    """Load ``vram_estimate_factor.env`` into the process environment once.

    Does not override an already-set ``ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR``.
    Per-model autotune still wins at pre-check time when enabled.
    """
    global _APPLIED, _APPLY_RESULT
    if _APPLIED and not force and _APPLY_RESULT is not None:
        return dict(_APPLY_RESULT)

    result: dict[str, Any] = {"applied": False}
    if not vram_apply_exported_env_enabled():
        result["reason"] = "disabled"
        _APPLIED = True
        _APPLY_RESULT = result
        return dict(result)

    if (
        os.environ.get("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR", "").strip()
        and not force
    ):
        result["reason"] = "env_already_set"
        result["existing"] = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR")
        _APPLIED = True
        _APPLY_RESULT = result
        return dict(result)

    path = apply_export_path()
    result["path"] = str(path)
    if not path.is_file():
        result["reason"] = "no_export_file"
        _APPLIED = True
        _APPLY_RESULT = result
        return dict(result)

    try:
        parsed = _parse_env_lines(path.read_text(encoding="utf-8"))
    except OSError as e:
        result["reason"] = "read_error"
        result["error"] = str(e)
        _APPLIED = True
        _APPLY_RESULT = result
        return dict(result)

    raw = parsed.get("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR", "").strip()
    if not raw:
        result["reason"] = "no_factor_in_file"
        _APPLIED = True
        _APPLY_RESULT = result
        return dict(result)

    try:
        factor = max(0.1, min(3.0, float(raw)))
    except ValueError:
        result["reason"] = "invalid_factor"
        result["raw"] = raw
        _APPLIED = True
        _APPLY_RESULT = result
        return dict(result)

    os.environ["ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR"] = f"{factor:g}"
    result["applied"] = True
    result["factor"] = factor
    _APPLIED = True
    _APPLY_RESULT = result
    return dict(result)


def apply_status() -> dict[str, Any]:
    """Snapshot for /health (includes last apply attempt)."""
    out: dict[str, Any] = {
        "enabled": vram_apply_exported_env_enabled(),
        "path": str(apply_export_path()),
    }
    if _APPLY_RESULT is not None:
        out.update(_APPLY_RESULT)
    else:
        out["applied"] = False
        out["reason"] = "not_run"
    out["note"] = (
        "Set VRAM_APPLY_EXPORTED_ENV=1 to load vram_estimate_factor.env at startup; "
        "autotune persist still overrides per GGUF when enabled."
    )
    return out
