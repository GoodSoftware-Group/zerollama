"""elizaOS/llama.cpp fork detection (ROADMAP borrowings L2).

WHY separate from gpu_profiles: fork gating is about the *binary* (does this
llama-server understand QJL/Polar/TurboQuant and --ctx-checkpoints?), not which
GPU JSON we selected. Operators may point LLAMA_SERVER_BIN at a fork build while
keeping stock vendor ggml in the Go binary during evaluation.

L2 Done (Jul 2026): defaults stay L1 (tok/s). VRAM opt-in via ``ZEROLLAMA_LLAMA_FORK=1``
or ``ZEROLLAMA_LLAMA_FORK_AUTO_VRAM=1`` when configured ctx ≥ threshold.
"""

from __future__ import annotations

import os
import subprocess
import sys
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

# Pin documented for unified runtime builds (ggml-org + llama/patches, Jul 2026).
ELIZA_LLAMA_CPP_DEFAULT_REF = "8f114a9b573b69035299f9b924047f53c1e22c7e"
ELIZA_LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"

_CUDA_FORK_REQUIRED_SYMBOLS = (
    "ggml_cuda_op_set_rows",
    "qjl_project_q_cuda",
    "fused_attn_qjl_polar_cuda",
)


def normalize_cache_type(value: str) -> str:
    key = value.strip()
    return CACHE_TYPE_ALIASES.get(key, key)


def auto_fork_kv_supported() -> bool:
    """Whether auto-probe may enable fork KV profiles on this host.

    Linux CUDA needs SET_ROWS + fused QJL attn in libggml-cuda (checked per
    binary via cuda_fork_backend_capable). macOS Metal uses a separate path.
    """
    if sys.platform == "darwin":
        return True
    return sys.platform.startswith("linux")


def _resolve_ggml_cuda_lib(binary: Path) -> Path | None:
    bindir = binary.parent
    names = (
        "libggml-cuda.so.0.12.0",
        "libggml-cuda.so.0",
        "libggml-cuda.so",
    )
    search_dirs = [bindir, bindir / "cuda_v12"]
    backend = os.environ.get("GGML_BACKEND_PATH", "").strip()
    if backend:
        search_dirs.insert(0, Path(backend).resolve().parent)
    for base in search_dirs:
        for name in names:
            cand = base / name
            if cand.is_file() or cand.is_symlink():
                return cand.resolve()
    return None


@lru_cache(maxsize=8)
def cuda_fork_backend_capable(binary: str) -> bool:
    """True when libggml-cuda beside *binary* exports fork KV + fused attn symbols."""
    if not sys.platform.startswith("linux"):
        return True
    path = Path(binary)
    if not path.is_file():
        return False
    cuda_lib = _resolve_ggml_cuda_lib(path)
    if cuda_lib is None:
        return False
    try:
        proc = subprocess.run(
            ["nm", "-D", "--demangle", str(cuda_lib)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        try:
            proc = subprocess.run(
                ["nm", "-D", str(cuda_lib)],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return False
    sym = (proc.stdout or "") + (proc.stderr or "")
    return all(marker in sym for marker in _CUDA_FORK_REQUIRED_SYMBOLS)


def llama_fork_env_override() -> bool | None:
    """Tri-state env: True=force on, False=force off, None=auto/probe."""
    raw = os.environ.get("ZEROLLAMA_LLAMA_FORK", "").strip().lower()
    if raw in ("0", "false", "no", "off", "stock"):
        return False
    if raw in ("1", "true", "yes", "on", "eliza", "fork"):
        return True
    return None


# WHY separate from ZEROLLAMA_LLAMA_FORK default: short-ctx wants stock q8_0 tok/s;
# long-ctx wants TBQ VRAM without forcing FORK=1 globally. Explicit FORK=0/1 wins.
DEFAULT_AUTO_VRAM_CTX_THRESHOLD = 32768


def llama_fork_auto_vram_enabled() -> bool:
    """``ZEROLLAMA_LLAMA_FORK_AUTO_VRAM=1`` — enable TBQ fork when configured ctx is large."""
    from runtime.env import env_bool

    return env_bool("ZEROLLAMA_LLAMA_FORK_AUTO_VRAM", default=False)


def auto_vram_ctx_threshold() -> int:
    """Min configured ``num_ctx`` to auto-enable fork (default 32768)."""
    raw = os.environ.get("ZEROLLAMA_LLAMA_FORK_AUTO_VRAM_CTX", "").strip()
    if not raw:
        return DEFAULT_AUTO_VRAM_CTX_THRESHOLD
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_AUTO_VRAM_CTX_THRESHOLD


def configured_num_ctx_hint() -> int | None:
    """Serve/default context for AUTO_VRAM gating (not per-request).

    Precedence: ``ZEROLLAMA_RUNTIME_VRAM_NUM_CTX`` → ``ZEROLLAMA_LLAMA_CTX`` →
    ``LLAMA_ARG_CTX_SIZE`` / ``LLAMA_N_CTX``. Missing → None (AUTO_VRAM stays off).
    """
    from runtime.env import vram_num_ctx_override

    override = vram_num_ctx_override()
    if override is not None:
        return override
    for key in ("ZEROLLAMA_LLAMA_CTX", "LLAMA_ARG_CTX_SIZE", "LLAMA_N_CTX"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            continue
    return None


def auto_vram_ctx_threshold_met() -> bool:
    """True when AUTO_VRAM is on and configured ctx ≥ threshold."""
    if not llama_fork_auto_vram_enabled():
        return False
    ctx = configured_num_ctx_hint()
    if ctx is None:
        return False
    return ctx >= auto_vram_ctx_threshold()


def clear_fork_probe_cache() -> None:
    """Test helper: drop cached --help probe results."""
    probe_fork_llama_server.cache_clear()
    cuda_fork_backend_capable.cache_clear()


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
    # Fork advertises QJL/Polar/TBQ in --cache-type-k help. Stock zerollama also
    # ships --ctx-checkpoints without fork KV types; do not treat that as fork.
    return any(marker in text for marker in ("qjl1_256", "q4_polar", "tbq3_0", "tbq4_0"))


def _binary_fork_capable(llama_server_bin: Path | str) -> bool:
    if not probe_fork_llama_server(str(llama_server_bin)):
        return False
    if not auto_fork_kv_supported():
        return False
    if sys.platform.startswith("linux") and not cuda_fork_backend_capable(
        str(llama_server_bin)
    ):
        return False
    return True


def llama_fork_enabled(*, llama_server_bin: Path | str | None = None) -> bool:
    """Whether to emit fork KV types and --ctx-checkpoints from GPU profiles."""
    override = llama_fork_env_override()
    if override is not None:
        return override
    # AUTO_VRAM: long configured ctx → TBQ profile; short ctx stays stock.
    if llama_fork_auto_vram_enabled():
        if not auto_vram_ctx_threshold_met():
            return False
        if llama_server_bin is not None:
            return _binary_fork_capable(llama_server_bin)
        return True
    if llama_server_bin is not None:
        return _binary_fork_capable(llama_server_bin)
    return False


def fork_detection_source(*, llama_server_bin: Path | str | None = None) -> str:
    override = llama_fork_env_override()
    if override is True:
        return "env"
    if override is False:
        return "env_off"
    if llama_fork_auto_vram_enabled():
        if auto_vram_ctx_threshold_met():
            if llama_server_bin is not None and not _binary_fork_capable(llama_server_bin):
                if not probe_fork_llama_server(str(llama_server_bin)):
                    return "auto_vram_binary_stock"
                if not auto_fork_kv_supported():
                    return "probe_disabled_platform"
                if sys.platform.startswith("linux") and not cuda_fork_backend_capable(
                    str(llama_server_bin)
                ):
                    return "probe_disabled_cuda_backend"
                return "auto_vram_binary_stock"
            return "auto_vram"
        return "auto_vram_below_ctx"
    if llama_server_bin is not None and probe_fork_llama_server(str(llama_server_bin)):
        if not auto_fork_kv_supported():
            return "probe_disabled_platform"
        if sys.platform.startswith("linux") and not cuda_fork_backend_capable(
            str(llama_server_bin)
        ):
            return "probe_disabled_cuda_backend"
        return "probe"
    return "stock"


def fork_health(*, llama_server_bin: Path | str | None = None) -> dict[str, object]:
    enabled = llama_fork_enabled(llama_server_bin=llama_server_bin)
    out: dict[str, object] = {
        "enabled": enabled,
        "source": fork_detection_source(llama_server_bin=llama_server_bin),
        "pin_ref": ELIZA_LLAMA_CPP_DEFAULT_REF,
        "repo": ELIZA_LLAMA_CPP_REPO,
        "auto_vram": llama_fork_auto_vram_enabled(),
        "auto_vram_ctx_threshold": auto_vram_ctx_threshold(),
        "configured_num_ctx_hint": configured_num_ctx_hint(),
    }
    if llama_server_bin is not None:
        bin_path = Path(llama_server_bin)
        out["binary"] = str(bin_path)
        if sys.platform.startswith("linux"):
            cuda_lib = _resolve_ggml_cuda_lib(bin_path)
            out["cuda_backend_capable"] = cuda_fork_backend_capable(str(bin_path))
            if cuda_lib is not None:
                out["cuda_lib"] = str(cuda_lib)
    return out
