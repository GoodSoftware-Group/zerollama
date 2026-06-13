"""elizaOS/llama.cpp fork detection (ROADMAP borrowings L2).

WHY separate from gpu_profiles: fork gating is about the *binary* (does this
llama-server understand QJL/Polar/TurboQuant and --ctx-checkpoints?), not which
GPU JSON we selected. Operators may point LLAMA_SERVER_BIN at a fork build while
keeping stock vendor ggml in the Go binary during evaluation.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

# CLI names accepted by elizaOS/llama.cpp (--cache-type-k/v).
ELIZA_FORK_CACHE_TYPES = frozenset(
    {
        "qjl1_256",
        "q4_polar",
        "tbq3_0",
        "tbq4_0",
        "tbq3_tcq",
        # L1 profile aliases (eliza uses tbq* in ggml enum)
        "turbo3_0",
        "turbo4_0",
        "turbo3_tcq",
    }
)

# Normalize L1/eliza alias strings to fork CLI names.
CACHE_TYPE_ALIASES: dict[str, str] = {
    "turbo3_0": "tbq3_0",
    "turbo4_0": "tbq4_0",
    "turbo3_tcq": "tbq3_tcq",
}

# Pin documented for L2 spike builds (eliza-v3 submodule HEAD, Jun 2026).
ELIZA_LLAMA_CPP_DEFAULT_REF = "96dd1a8466c84bdd419faf3866425260623fb6b0"
ELIZA_LLAMA_CPP_REPO = "https://github.com/elizaOS/llama.cpp.git"


def normalize_cache_type(value: str) -> str:
    key = value.strip()
    return CACHE_TYPE_ALIASES.get(key, key)


def llama_fork_env_override() -> bool | None:
    """Tri-state env: True=force on, False=force off, None=auto/probe."""
    raw = os.environ.get("ZEROLLAMA_LLAMA_FORK", "").strip().lower()
    if raw in ("0", "false", "no", "off", "stock"):
        return False
    if raw in ("1", "true", "yes", "on", "eliza", "fork"):
        return True
    return None


def clear_fork_probe_cache() -> None:
    """Test helper: drop cached --help probe results."""
    probe_fork_llama_server.cache_clear()


@lru_cache(maxsize=8)
def probe_fork_llama_server(binary: str) -> bool:
    """True when *binary* advertises fork KV types or --ctx-checkpoints in --help."""
    path = Path(binary)
    if not path.is_file():
        return False
    try:
        proc = subprocess.run(
            [str(path), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    text = (proc.stdout or "") + (proc.stderr or "")
    # Fork advertises QJL/Polar/TBQ in --cache-type-k help and/or --ctx-checkpoints.
    return (
        "ctx-checkpoints" in text
        or "qjl1_256" in text
        or "q4_polar" in text
    )


def llama_fork_enabled(*, llama_server_bin: Path | str | None = None) -> bool:
    """Whether to emit fork KV types and --ctx-checkpoints from GPU profiles."""
    override = llama_fork_env_override()
    if override is not None:
        return override
    if llama_server_bin is not None:
        return probe_fork_llama_server(str(llama_server_bin))
    return False


def fork_detection_source(*, llama_server_bin: Path | str | None = None) -> str:
    override = llama_fork_env_override()
    if override is True:
        return "env"
    if override is False:
        return "env_off"
    if llama_server_bin is not None and probe_fork_llama_server(str(llama_server_bin)):
        return "probe"
    return "stock"


def fork_health(*, llama_server_bin: Path | str | None = None) -> dict[str, object]:
    enabled = llama_fork_enabled(llama_server_bin=llama_server_bin)
    out: dict[str, object] = {
        "enabled": enabled,
        "source": fork_detection_source(llama_server_bin=llama_server_bin),
        "pin_ref": ELIZA_LLAMA_CPP_DEFAULT_REF,
        "repo": ELIZA_LLAMA_CPP_REPO,
    }
    if llama_server_bin is not None:
        out["binary"] = str(llama_server_bin)
    return out
