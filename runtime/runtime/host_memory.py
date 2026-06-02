"""Host RAM availability (Linux /proc/meminfo) for pre-load checks."""

from __future__ import annotations

import os
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
    from runtime.gguf_estimate import gguf_weight_bytes

    weights = gguf_weight_bytes(gguf)
    base = weights if weights is not None else gguf.stat().st_size
    overhead = float(os.environ.get("ZEROLLAMA_RUNTIME_RAM_OVERHEAD", "1.12"))
    return int(base * overhead)


def host_ram_margin() -> float:
    return float(os.environ.get("ZEROLLAMA_RUNTIME_RAM_MARGIN", "1.0"))


def host_ram_budget_snapshot(
    gguf: Path, *, margin: float | None = None
) -> dict[str, int | bool | str | float] | None:
    """Host RAM fit for mmap/weights (for /health and /internal/vram-estimate)."""
    if margin is None:
        margin = host_ram_margin()
    mem = read_linux_host_memory()
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
    mem = read_linux_host_memory()
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
