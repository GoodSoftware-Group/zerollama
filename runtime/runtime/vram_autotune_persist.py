"""Persist VRAM estimate autotune factor across runtime restarts (Phase 13)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_STATE_VERSION = 2
_state_cache: dict[str, Any] | None = None
_state_cache_mtime: float | None = None


def vram_autotune_persist_enabled() -> bool:
    """On with autotune; disable only via VRAM_AUTOTUNE_PERSIST=0."""
    v = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    from runtime.gpu_vram import vram_estimate_autotune_enabled

    return vram_estimate_autotune_enabled()


def autotune_state_dir() -> Path:
    raw = os.environ.get("ZEROLLAMA_RUNTIME_STATE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache" / "zerollama"


def autotune_state_path() -> Path:
    override = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_STATE", "").strip()
    if override:
        return Path(override).expanduser()
    return autotune_state_dir() / "vram_autotune.json"


def model_autotune_key(model: str | Path) -> str:
    """Stable key for per-model factors (resolved absolute path)."""
    return str(Path(model).expanduser().resolve())


def try_model_autotune_key(model: str | Path) -> str | None:
    """Like ``model_autotune_key`` but returns None when the path cannot be resolved."""
    try:
        return model_autotune_key(model)
    except (OSError, ValueError):
        return None


def _invalidate_cache() -> None:
    global _state_cache, _state_cache_mtime
    _state_cache = None
    _state_cache_mtime = None


def _clamp_factor(raw: Any) -> float | None:
    try:
        return max(0.1, min(3.0, float(raw)))
    except (TypeError, ValueError):
        return None


def _migrate_v1(data: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, Any] = dict(data.get("models") or {})
    factor = _clamp_factor(data.get("estimate_factor"))
    model = data.get("model")
    if factor is not None and isinstance(model, str) and model.strip():
        key = model_autotune_key(model)
        models[key] = {
            "estimate_factor": factor,
            "updated_at": data.get("updated_at", time.time()),
        }
        data = {
            "version": _STATE_VERSION,
            "models": models,
            "last_model": key,
        }
    return data


def _parse_state(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    version = data.get("version", 1)
    if isinstance(version, int) and version > _STATE_VERSION:
        return None
    if version == 1 or "models" not in data:
        data = _migrate_v1(data)
    if data.get("version", 1) != _STATE_VERSION:
        data["version"] = _STATE_VERSION
    models = data.get("models")
    if not isinstance(models, dict):
        data["models"] = {}
    return data


def _read_state(*, force: bool = False) -> dict[str, Any] | None:
    global _state_cache, _state_cache_mtime
    if not vram_autotune_persist_enabled():
        return None
    path = autotune_state_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _state_cache = {"version": _STATE_VERSION, "models": {}}
        _state_cache_mtime = None
        return _state_cache
    if not force and _state_cache is not None and _state_cache_mtime == mtime:
        return _state_cache
    try:
        parsed = _parse_state(path.read_text(encoding="utf-8"))
    except OSError:
        parsed = None
    if parsed is None:
        parsed = {"version": _STATE_VERSION, "models": {}}
    _state_cache = parsed
    _state_cache_mtime = mtime
    return parsed


def _factor_from_entry(entry: Any) -> float | None:
    if not isinstance(entry, dict):
        return None
    return _clamp_factor(entry.get("estimate_factor"))


def load_persisted_autotune(model: str | Path | None = None) -> float | None:
    """Per-model factor when model is set; else last calibrated model only when model is None."""
    state = _read_state()
    if state is None:
        return None
    models: dict[str, Any] = state.get("models") or {}
    if model is not None:
        key = try_model_autotune_key(model)
        if key is None:
            return None
        return _factor_from_entry(models.get(key))
    last = state.get("last_model")
    if isinstance(last, str):
        return _factor_from_entry(models.get(last))
    return None


def _atomic_write(path: Path, text: str) -> bool:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if tmp.is_file():
            tmp.unlink()
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        return True
    except OSError:
        try:
            if tmp.is_file():
                tmp.unlink()
        except OSError:
            pass
        return False


def save_persisted_autotune(
    factor: float,
    *,
    model: str | Path | None = None,
) -> bool:
    """Atomic write; per-model entry when model path is provided."""
    if not vram_autotune_persist_enabled():
        return False
    clamped = max(0.1, min(3.0, float(factor)))
    state = _read_state() or {"version": _STATE_VERSION, "models": {}}
    models: dict[str, Any] = dict(state.get("models") or {})
    now = time.time()
    if model is not None:
        key = model_autotune_key(model)
        models[key] = {"estimate_factor": clamped, "updated_at": now}
        state["last_model"] = key
    else:
        state["legacy_factor"] = clamped
    state["version"] = _STATE_VERSION
    state["models"] = models
    state["updated_at"] = now
    path = autotune_state_path()
    ok = _atomic_write(path, json.dumps(state, indent=2) + "\n")
    if ok:
        _invalidate_cache()
        _read_state(force=True)
        from runtime.vram_factor_export import export_factor_catalog

        export_factor_catalog()
    return ok


def clear_persisted_autotune(model: str | Path | None = None) -> None:
    path = autotune_state_path()
    if model is None:
        _invalidate_cache()
        try:
            if path.is_file():
                path.unlink()
            tmp = path.with_suffix(path.suffix + ".tmp")
            if tmp.is_file():
                tmp.unlink()
        except OSError:
            pass
        from runtime.vram_factor_export import clear_export_files

        clear_export_files()
        return
    state = _read_state()
    if state is None:
        return
    key = model_autotune_key(model)
    models: dict[str, Any] = dict(state.get("models") or {})
    if key not in models:
        return
    del models[key]
    state["models"] = models
    if state.get("last_model") == key:
        state.pop("last_model", None)
    if not models:
        clear_persisted_autotune()
        return
    _atomic_write(path, json.dumps(state, indent=2) + "\n")
    _invalidate_cache()
    from runtime.vram_factor_export import export_factor_catalog

    export_factor_catalog()


def model_in_persist_catalog(model: str | Path) -> bool:
    """True when ``model`` has a persisted autotune entry."""
    if not vram_autotune_persist_enabled():
        return False
    key = try_model_autotune_key(model)
    if key is None:
        return False
    state = _read_state()
    if state is None:
        return False
    models: dict[str, Any] = state.get("models") or {}
    if not isinstance(models, dict):
        return False
    return _factor_from_entry(models.get(key)) is not None


def persist_catalog(*, max_entries: int = 64) -> tuple[list[dict[str, Any]], bool]:
    """Persisted per-GGUF factors for /health (loopback ops).

    Returns ``(rows, truncated)``.
    """
    if not vram_autotune_persist_enabled():
        return [], False
    state = _read_state()
    if state is None:
        return [], False
    models: dict[str, Any] = state.get("models") or {}
    if not isinstance(models, dict) or not models:
        return [], False
    last = state.get("last_model")
    rows: list[dict[str, Any]] = []
    for key in sorted(models.keys()):
        raw = models.get(key)
        factor = _factor_from_entry(raw)
        if factor is None:
            continue
        row: dict[str, Any] = {
            "model": key,
            "basename": Path(key).name,
            "estimate_factor": factor,
            "last": isinstance(last, str) and key == last,
        }
        if isinstance(raw, dict) and isinstance(raw.get("updated_at"), (int, float)):
            row["updated_at"] = raw["updated_at"]
        rows.append(row)
    cap = max(1, int(max_entries))
    if len(rows) > cap:
        return rows[:cap], True
    return rows, False


def persist_status(
    *,
    session_factor: float | None = None,
    session_model: str | None = None,
) -> dict[str, Any]:
    path = autotune_state_path()
    state = _read_state() if vram_autotune_persist_enabled() else None
    models = (state or {}).get("models") or {}
    last = (state or {}).get("last_model")
    persisted = session_factor
    if persisted is None and isinstance(last, str):
        persisted = _factor_from_entry(models.get(last))
    catalog, catalog_truncated = persist_catalog()
    model_count = len(models) if isinstance(models, dict) else 0
    return {
        "enabled": vram_autotune_persist_enabled(),
        "path": str(path),
        "file_exists": path.is_file(),
        "model_count": model_count,
        "last_model": last,
        "session_model": session_model,
        "session_factor": session_factor,
        "persisted_factor": persisted,
        "catalog": catalog,
        "catalog_truncated": catalog_truncated or model_count > len(catalog),
    }
