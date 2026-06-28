"""Host RAM availability for pre-load checks (Linux /proc/meminfo, macOS vm_stat).

Why macOS: Apple Silicon uses unified memory — there is no NVML framebuffer. Runtime
admission and host budget checks must read available RAM the same way ggml treats Metal.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HostMemory:
    available_bytes: int
    swap_free_bytes: int

    @property
    def load_budget_bytes(self) -> int:
        """RAM + swap Linux uses for overcommit decisions (matches ggml verify on Linux)."""
        return self.available_bytes + self.swap_free_bytes


def read_linux_host_memory() -> HostMemory | None:
    if sys.platform != "linux":
        return None
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    avail_kb = 0
    swap_kb = 0
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            avail_kb = int(line.split()[1])
        elif line.startswith("SwapFree:"):
            swap_kb = int(line.split()[1])
    return HostMemory(
        available_bytes=avail_kb * 1024,
        swap_free_bytes=swap_kb * 1024,
    )


_VM_STAT_RE = re.compile(r"^Pages\s+([^:]+):\s+([\d.]+)")


def read_darwin_host_memory() -> HostMemory | None:
    """Approximate free+reclaimable memory from vm_stat (Metal unified-memory hosts)."""
    if sys.platform != "darwin":
        return None
    try:
        pagesize = int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.pagesize"], text=True, timeout=5
            ).strip()
        )
        vm_text = subprocess.check_output(["vm_stat"], text=True, timeout=5)
    except (subprocess.SubprocessError, ValueError, OSError):
        return None
    counts: dict[str, int] = {}
    for line in vm_text.splitlines():
        m = _VM_STAT_RE.match(line.strip())
        if not m:
            continue
        key = m.group(1).strip().lower().replace(" ", "_")
        raw = m.group(2).replace(".", "")
        try:
            counts[key] = int(raw)
        except ValueError:
            continue
    # free + inactive + speculative ≈ pressure-available (same heuristic as many macOS tools)
    pages = (
        counts.get("free", 0)
        + counts.get("inactive", 0)
        + counts.get("speculative", 0)
    )
    if pages <= 0:
        return None
    swap_free = 0
    try:
        # macOS format: "vm.swapusage: total = 2048.00M  used = 512.00M  free = 1536.00M"
        # Use sysctl without -n to get the key-value line, then parse "free = <value><unit>"
        swap_text = subprocess.check_output(["sysctl", "vm.swapusage"], text=True, timeout=5)
        m_swap = re.search(r"free\s*=\s*([\d.]+)([KMGT]?)i?B?", swap_text, re.IGNORECASE)
        if m_swap:
            num = float(m_swap.group(1))
            unit = m_swap.group(2).upper()
            mult = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}.get(unit, 1)
            swap_free = int(num * mult)
    except (subprocess.SubprocessError, ValueError, OSError, AttributeError):
        swap_free = 0
    return HostMemory(available_bytes=pages * pagesize, swap_free_bytes=swap_free)


def read_host_memory() -> HostMemory | None:
    """Platform host memory snapshot, or None when unavailable."""
    if sys.platform == "linux":
        return read_linux_host_memory()
    if sys.platform == "darwin":
        return read_darwin_host_memory()
    return None


def darwin_total_memory_bytes() -> int | None:
    """Unified memory pool size (hw.memsize) for autoconfig /health on Apple Silicon."""
    if sys.platform != "darwin":
        return None
    try:
        return int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=5
            ).strip()
        )
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def format_bytes(n: int) -> str:
    f = float(n)
    for i, unit in enumerate(("B", "KiB", "MiB", "GiB", "TiB")):
        if f < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(f)} B"
            return f"{f:.1f} {unit}"
        f /= 1024.0
    return f"{int(n)} B"


def estimate_gguf_ram_bytes(gguf: Path) -> int:
    """Best-effort RAM needed for weights (tensor payload, not full file size)."""
    from runtime.env import vram_ram_overhead
    from runtime.gguf_estimate import gguf_weight_bytes

    weights = gguf_weight_bytes(gguf)
    base = weights if weights is not None else gguf.stat().st_size
    return int(base * vram_ram_overhead())


def host_ram_margin() -> float:
    from runtime.env import vram_ram_margin

    return vram_ram_margin()


def host_ram_budget_snapshot(
    gguf: Path, *, margin: float | None = None
) -> dict[str, int | bool | str | float] | None:
    """Host RAM fit for mmap/weights (for /health and /internal/vram-estimate)."""
    if margin is None:
        margin = host_ram_margin()
    mem = read_host_memory()
    if mem is None:
        return None
    required = int(estimate_gguf_ram_bytes(gguf) * margin)
    budget = mem.load_budget_bytes
    return {
        "required_bytes": required,
        "load_budget_bytes": budget,
        "fits": required <= budget,
        "margin": margin,
        "required": format_bytes(required),
        "load_budget": format_bytes(budget),
    }


def check_gguf_host_budget(gguf: Path, *, margin: float | None = None) -> None:
    """Raise LlamaServerError if GGUF file likely exceeds available host memory."""
    from runtime.worker.llama_server import LlamaServerError

    if margin is None:
        margin = host_ram_margin()
    mem = read_host_memory()
    if mem is None:
        return
    required = int(estimate_gguf_ram_bytes(gguf) * margin)
    budget = mem.load_budget_bytes
    if required > budget:
        weights = estimate_gguf_ram_bytes(gguf)
        raise LlamaServerError(
            f"model requires about {format_bytes(required)} host memory "
            f"(weights ~{format_bytes(weights)}) "
            f"but only {format_bytes(budget)} is available "
            f"({format_bytes(mem.available_bytes)} RAM + "
            f"{format_bytes(mem.swap_free_bytes)} swap free)"
        )
