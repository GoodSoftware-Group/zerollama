"""Pick runtime YAML from visible GPU topology (Phase 13)."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

_PROBE_TTL_S = 5.0
_probe_lock = threading.Lock()
_gpu_count_cache: tuple[float, int | None] | None = None


def clear_autoconfig_probe_cache() -> None:
    """Test helper: drop cached nvidia-smi probe results."""
    global _gpu_count_cache
    from runtime.nvidia_probe import clear_nvidia_probe_cache

    with _probe_lock:
        _gpu_count_cache = None
    clear_nvidia_probe_cache()


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
    from runtime.nvidia_probe import detect_gpu_total_vram_bytes as _probe_vram

    return _probe_vram(device_index)


def resolved_config_path() -> Path:
    """Config file in use: explicit env, L3 profile preset, or autoconfig default."""
    cfg = os.environ.get("ZEROLLAMA_RUNTIME_CONFIG", "").strip()
    if cfg:
        return Path(cfg).expanduser()
    from runtime.env import resolve_l3_profile_config_path

    profile_path = resolve_l3_profile_config_path()
    if profile_path is not None:
        return profile_path
    return resolve_default_config_path()


def autoconfig_health(*, main_gpu: int = 0) -> dict[str, object]:
    """Snapshot for /health: how runtime YAML was chosen."""
    path = resolved_config_path()
    n = detect_visible_gpu_count()
    pick = "custom"
    name = path.name
    if name == "l3_agent_subprocess.yaml":
        pick = "l3_agent"
    elif name == "single_gpu.yaml":
        pick = "single_gpu"
    elif name == "apple_silicon.yaml":
        pick = "apple_silicon"
    elif name == "dual_4090.yaml":
        pick = "dual_4090"
    total = detect_gpu_total_vram_bytes(main_gpu)
    if total is None and sys.platform == "darwin":
        from runtime.host_memory import darwin_total_memory_bytes

        total = darwin_total_memory_bytes()
    out: dict[str, object] = {
        "enabled": auto_config_enabled(),
        "config_path": str(path),
        "pick": pick,
        "visible_gpu_count": n,
    }
    from runtime.env import l3_profile_name, resolve_l3_profile_config_path

    if l3_profile_name():
        out["l3_profile"] = l3_profile_name()
        prof = resolve_l3_profile_config_path()
        if prof is not None:
            out["l3_profile_config"] = str(prof)
    if total is not None:
        from runtime.host_memory import format_bytes

        out["gpu_total_vram_bytes"] = total
        out["gpu_total_vram"] = format_bytes(total)
    return out


def resolve_default_config_path() -> Path:
    """Default YAML when ZEROLLAMA_RUNTIME_CONFIG is unset."""
    configs = _configs_dir()
    apple = configs / "apple_silicon.yaml"
    single = configs / "single_gpu.yaml"
    dual = configs / "dual_4090.yaml"

    if auto_config_enabled():
        # Why darwin first: no nvidia-smi; single_gpu.yaml targets discrete 16GB CUDA hosts.
        if sys.platform == "darwin" and apple.is_file():
            return apple
        n = detect_visible_gpu_count()
        if n is not None and n <= 1 and single.is_file():
            return single
        if n is not None and n >= 2 and dual.is_file():
            return dual

    if apple.is_file() and sys.platform == "darwin":
        return apple
    # Prefer single_gpu when the probe failed or autoconfig is off — dual tensor-split
    # topologies break on one card (RTX 5080). Operators on dual-4090 set
    # ZEROLLAMA_RUNTIME_CONFIG explicitly (see scripts/dual_4090_env.sh).
    if single.is_file():
        return single
    if dual.is_file():
        return dual
    return dual
