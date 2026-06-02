"""Pick runtime YAML from visible GPU topology (Phase 13)."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

_PROBE_TTL_S = 5.0
_probe_lock = threading.Lock()
_gpu_count_cache: tuple[float, int | None] | None = None
_gpu_total_cache: dict[int, tuple[float, int | None]] = {}


def clear_autoconfig_probe_cache() -> None:
    """Test helper: drop cached nvidia-smi probe results."""
    global _gpu_count_cache
    with _probe_lock:
        _gpu_count_cache = None
        _gpu_total_cache.clear()


def _configs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def detect_visible_gpu_count() -> int | None:
    """Return GPU count from nvidia-smi, or None if unavailable."""
    global _gpu_count_cache
    now = time.monotonic()
    with _probe_lock:
        if _gpu_count_cache is not None:
            ts, val = _gpu_count_cache
            if now - ts < _PROBE_TTL_S:
                return val
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        val = None
    else:
        val = (
            sum(1 for line in proc.stdout.splitlines() if line.strip().startswith("GPU "))
            if proc.returncode == 0
            else None
        )
    with _probe_lock:
        _gpu_count_cache = (now, val)
    return val


def auto_config_enabled() -> bool:
    v = os.environ.get("ZEROLLAMA_AUTO_CONFIG", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def detect_gpu_total_vram_bytes(device_index: int = 0) -> int | None:
    """Total VRAM for a GPU from nvidia-smi (MiB → bytes), or None."""
    global _gpu_total_cache
    now = time.monotonic()
    with _probe_lock:
        cached = _gpu_total_cache.get(device_index)
        if cached is not None and now - cached[0] < _PROBE_TTL_S:
            return cached[1]
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                str(device_index),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        val = None
    else:
        if proc.returncode != 0:
            val = None
        else:
            line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
            try:
                val = int(float(line.split()[0]) * 1024 * 1024)
            except (IndexError, ValueError):
                val = None
    with _probe_lock:
        _gpu_total_cache[device_index] = (now, val)
    return val


def resolved_config_path() -> Path:
    """Config file in use: explicit env or autoconfig default."""
    cfg = os.environ.get("ZEROLLAMA_RUNTIME_CONFIG", "").strip()
    if cfg:
        return Path(cfg).expanduser()
    return resolve_default_config_path()


def autoconfig_health(*, main_gpu: int = 0) -> dict[str, object]:
    """Snapshot for /health: how runtime YAML was chosen."""
    path = resolved_config_path()
    n = detect_visible_gpu_count()
    pick = "custom"
    name = path.name
    if name == "single_gpu.yaml":
        pick = "single_gpu"
    elif name == "dual_4090.yaml":
        pick = "dual_4090"
    total = detect_gpu_total_vram_bytes(main_gpu)
    out: dict[str, object] = {
        "enabled": auto_config_enabled(),
        "config_path": str(path),
        "pick": pick,
        "visible_gpu_count": n,
    }
    if total is not None:
        from runtime.host_memory import format_bytes

        out["gpu_total_vram_bytes"] = total
        out["gpu_total_vram"] = format_bytes(total)
    return out


def resolve_default_config_path() -> Path:
    """Default YAML when ZEROLLAMA_RUNTIME_CONFIG is unset."""
    configs = _configs_dir()
    single = configs / "single_gpu.yaml"
    dual = configs / "dual_4090.yaml"

    if auto_config_enabled():
        n = detect_visible_gpu_count()
        if n is not None and n <= 1 and single.is_file():
            return single
        if n is not None and n >= 2 and dual.is_file():
            return dual

    if dual.is_file():
        return dual
    if single.is_file():
        return single
    return dual
