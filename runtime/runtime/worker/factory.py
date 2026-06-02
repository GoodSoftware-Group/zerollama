"""Construct llama forward workers (Phase 14).

Why a factory: InferenceEngine should not branch on subprocess vs ctypes vs wheel
everywhere. One protocol (start/stop/completion/stream) keeps Phase 11 admission and
Phase 13 VRAM estimates unchanged while forward moves in-process.

Why env wins over config: operators set ZEROLLAMA_RUNTIME_LLAMA_BACKEND on the serve
process at startup; YAML is fallback for packaged defaults.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Union

from runtime.worker.llama_cpp_python import LlamaCppPythonWorker
from runtime.worker.llama_inprocess import LlamaInprocessWorker
from runtime.worker.llama_server import LlamaServerError, LlamaServerProcess

# Avoid circular import: config only used as TYPE_CHECKING optional
def _config_backend_raw(config: object | None) -> str:
    if config is None:
        return ""
    return str(getattr(config, "llama_backend", "") or "").strip().lower()

LlamaForwardWorker = Union[LlamaServerProcess, LlamaInprocessWorker, LlamaCppPythonWorker]


class LlamaBackendKind(str, Enum):
    SUBPROCESS = "subprocess"
    INPROCESS = "inprocess"
    LLAMA_CPP_PYTHON = "llama-cpp-python"


def _parse_backend_name(raw: str) -> LlamaBackendKind:
    if raw in ("inprocess", "in-process", "libllama", "embed"):
        return LlamaBackendKind.INPROCESS
    if raw in (
        "llama-cpp-python",
        "llama_cpp_python",
        "cpp-python",
        "pypi",
        "wheel",
    ):
        return LlamaBackendKind.LLAMA_CPP_PYTHON
    if raw in ("subprocess", "server", "llama-server"):
        return LlamaBackendKind.SUBPROCESS
    raise LlamaServerError(
        f"unknown llama backend {raw!r} "
        "(use subprocess, inprocess, or llama-cpp-python)"
    )


def llama_backend_from_env() -> LlamaBackendKind:
    raw = os.environ.get("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "").strip().lower()
    if raw:
        return _parse_backend_name(raw)
    return LlamaBackendKind.SUBPROCESS


def resolve_llama_backend(config: object | None = None) -> LlamaBackendKind:
    """Env wins; else ``RuntimeConfig.llama_backend``; default subprocess."""
    raw = os.environ.get("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "").strip().lower()
    if raw:
        return _parse_backend_name(raw)
    cfg_raw = _config_backend_raw(config)
    if cfg_raw:
        return _parse_backend_name(cfg_raw)
    return LlamaBackendKind.SUBPROCESS


def create_llama_worker(
    *,
    kind: LlamaBackendKind | None,
    binary: Path | None,
    model: Path,
    host: str,
    port: int,
    n_gpu_layers: int = -1,
    lib_path: Path | None = None,
    cpp_root: Path | None = None,
    main_gpu: int = 0,
    config: object | None = None,
) -> LlamaForwardWorker:
    backend = kind or resolve_llama_backend(config)
    if backend == LlamaBackendKind.INPROCESS:
        from runtime.llama_args import resolve_parallel_slots

        llama_args: list[str] = []
        default_slots = 1
        if config is not None:
            default_slots = max(1, int(getattr(config, "llama_parallel_slots", 1) or 1))
            llama_args = list(getattr(config, "llama_server_args", lambda: [])())
        parallel_slots = resolve_parallel_slots(llama_args, default=default_slots)
        kv_pool_token_cap: int | None = None
        if config is not None:
            nb = int(getattr(config, "num_blocks", 0) or 0)
            bs = int(getattr(config, "block_size", 0) or 0)
            if nb > 0 and bs > 0:
                kv_pool_token_cap = nb * bs
        return LlamaInprocessWorker(
            model=model,
            n_gpu_layers=n_gpu_layers,
            lib_path=lib_path,
            cpp_root=cpp_root,
            host=host,
            port=port,
            main_gpu=main_gpu,
            parallel_slots=parallel_slots,
            kv_pool_token_cap=kv_pool_token_cap,
        )
    if backend == LlamaBackendKind.LLAMA_CPP_PYTHON:
        return LlamaCppPythonWorker(
            model=model,
            n_gpu_layers=n_gpu_layers,
            host=host,
            port=port,
            main_gpu=main_gpu,
        )
    if binary is None:
        raise LlamaServerError(
            "llama-server binary required for subprocess backend "
            "(set LLAMA_SERVER_BIN or ZEROLLAMA_RUNTIME_LLAMA_BACKEND="
            "inprocess|llama-cpp-python)"
        )
    return LlamaServerProcess(
        binary=binary,
        model=model,
        host=host,
        port=port,
        n_gpu_layers=n_gpu_layers,
    )
