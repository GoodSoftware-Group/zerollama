"""Probe sibling llama.cpp checkout (default ``~/Sites/inference/llama.cpp``).

WHY: zerollama resolves ``LLAMA_CPP_ROOT`` to ``${repo}/../llama.cpp``. CUDA graph
capture lives inside ggml (`GGML_CUDA_USE_GRAPHS`); there is no per-slot llama.h
API yet. This probe surfaces build flags + pin drift for operator health and
future epoch → ggml graph bind work.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

_PIN_FILE = "LLAMA_CPP_VERSION"


def default_llama_cpp_root() -> Path:
    raw = os.environ.get("LLAMA_CPP_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    return (repo.parent / "llama.cpp").resolve()


def pinned_llama_cpp_version(repo_root: Path | None = None) -> str | None:
    root = repo_root or Path(__file__).resolve().parents[2]
    pin = root / _PIN_FILE
    if not pin.is_file():
        return None
    return pin.read_text(encoding="utf-8").strip() or None


def _read_cmake_cache_bool(cache: Path, key: str) -> bool | None:
    if not cache.is_file():
        return None
    pat = re.compile(rf"^{re.escape(key)}:BOOL=(ON|OFF)$", re.MULTILINE)
    m = pat.search(cache.read_text(encoding="utf-8", errors="replace"))
    if not m:
        return None
    return m.group(1) == "ON"


def _git_describe(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "describe", "--tags", "--always"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = (out.stdout or "").strip()
    return line or None


def probe_llama_cpp(
    *,
    cpp_root: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Return operator hints for the sibling llama.cpp tree."""
    root = (cpp_root or default_llama_cpp_root()).expanduser().resolve()
    pin = pinned_llama_cpp_version(repo_root)
    describe = _git_describe(root)
    cache = root / "build" / "CMakeCache.txt"
    cuda = _read_cmake_cache_bool(cache, "GGML_CUDA")
    cuda_graphs = _read_cmake_cache_bool(cache, "GGML_CUDA_GRAPHS")
    graphs_disabled_env = os.environ.get("GGML_CUDA_DISABLE_GRAPHS", "").strip() != ""

    lib_candidates = list((root / "build" / "bin").glob("libllama.*"))
    lib_candidates += list((root / "build" / "lib").glob("libllama.*"))
    libllama = next((p for p in lib_candidates if p.is_file()), None)

    pin_matches = None
    if pin and describe:
        pin_matches = pin in describe or describe.startswith(pin)

    graphs_compile_ready = cuda is True and cuda_graphs is True
    graphs_runtime_ready = graphs_compile_ready and not graphs_disabled_env

    return {
        "root": str(root),
        "present": root.is_dir() and (root / "CMakeLists.txt").is_file(),
        "pin_file": pin,
        "git_describe": describe,
        "pin_matches_checkout": pin_matches,
        "libllama": str(libllama) if libllama else None,
        "cmake_cache": str(cache) if cache.is_file() else None,
        "ggml_cuda": cuda,
        "ggml_cuda_graphs": cuda_graphs,
        "ggml_cuda_graphs_disabled_env": graphs_disabled_env,
        "ggml_graph_key": "cgraph.nodes[0] (internal ggml-cuda; not per-slot)",
        "graphs_compile_ready": graphs_compile_ready,
        "graphs_runtime_ready": graphs_runtime_ready,
        "epoch_bind_status": (
            "wired — bump_decode_graph_epoch calls llama_context_cuda_graph_invalidate "
            "when ctx_ptr is available (rebuild libllama after pull)"
        ),
    }
