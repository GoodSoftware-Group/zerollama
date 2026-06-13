"""Shared nvidia-smi probes for autoconfig and L1 GPU profiles.

WHY a separate module: ``autoconfig`` and ``gpu_profiles`` both need GPU name
and VRAM without importing each other (config load would cycle
autoconfig → config → gpu_profiles → autoconfig). Cached probes avoid duplicate
subprocess spam at startup.
"""

from __future__ import annotations

import subprocess
import threading
import time

_PROBE_TTL_S = 5.0
_probe_lock = threading.Lock()
_gpu_name_cache: dict[int, tuple[float, str | None]] = {}
_gpu_total_cache: dict[int, tuple[float, int | None]] = {}


def clear_nvidia_probe_cache() -> None:
    """Test helper: drop cached nvidia-smi probe results."""
    with _probe_lock:
        _gpu_name_cache.clear()
        _gpu_total_cache.clear()


def detect_nvidia_gpu_name(device_index: int = 0) -> str | None:
    """GPU marketing name from nvidia-smi, or None if unavailable."""
    now = time.monotonic()
    with _probe_lock:
        cached = _gpu_name_cache.get(device_index)
        if cached is not None and now - cached[0] < _PROBE_TTL_S:
            return cached[1]
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
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
            line = proc.stdout.strip().splitlines()
            val = line[0].strip() or None if line else None
    with _probe_lock:
        _gpu_name_cache[device_index] = (now, val)
    return val


def detect_gpu_total_vram_bytes(device_index: int = 0) -> int | None:
    """Total VRAM for a GPU from nvidia-smi (MiB → bytes), or None."""
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


def detect_gpu_total_vram_gb(device_index: int = 0) -> float | None:
    total = detect_gpu_total_vram_bytes(device_index)
    if total is None:
        return None
    return total / (1024**3)
