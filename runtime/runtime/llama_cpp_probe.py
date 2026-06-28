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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sibling_llama_cpp_root(repo: Path | None = None) -> Path:
    return ((repo or _repo_root()).parent / "llama.cpp").resolve()


def _vendor_llama_cpp_root(repo: Path | None = None) -> Path | None:
    pin = pinned_llama_cpp_version(repo) or ""
    if not pin:
        return None
    vendor = (repo or _repo_root()) / "vendor" / f"llama-cpp-{pin}"
    if (vendor / "CMakeLists.txt").is_file():
        return vendor.resolve()
    return None


def default_llama_cpp_root() -> Path:
    raw = os.environ.get("LLAMA_CPP_ROOT", "").strip()
    repo = _repo_root()
    vendor = _vendor_llama_cpp_root(repo)
    if raw:
        p = Path(raw).expanduser().resolve()
        # WHY: stale shell ``LLAMA_CPP_ROOT=../llama.cpp`` must not beat patched vendor.
        if vendor is not None and (
            p == _sibling_llama_cpp_root(repo) or not (p / "CMakeLists.txt").is_file()
        ):
            return vendor
        return p
    if vendor is not None:
        return vendor
    return _sibling_llama_cpp_root(repo)


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
            "wired — in-process: llama_context_cuda_graph_invalidate when ctx_ptr set; "
            "subprocess: POST /cuda-graph/invalidate on llama-server (rebuild after pull)"
        ),
    }
