"""llama-cpp-python wheel backend (Phase 14 optional)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from runtime.worker.factory import LlamaBackendKind, create_llama_worker, resolve_llama_backend
from runtime.worker.llama_cpp_python import (
    LlamaCppPythonWorker,
    llama_cpp_n_gpu_layers,
    llama_cpp_wheel_health,
)


def test_llama_cpp_n_gpu_layers_defaults_cpu():
    assert llama_cpp_n_gpu_layers(-1, None) == 0
    assert llama_cpp_n_gpu_layers(8, None) == 8
    assert llama_cpp_n_gpu_layers(-1, 16) == 16


def test_llama_cpp_n_gpu_layers_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS", "24")
    assert llama_cpp_n_gpu_layers(-1, None) == 24


def test_llama_cpp_n_gpu_layers_env_negative_uses_cpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS", "-1")
    assert llama_cpp_n_gpu_layers(-1, None) == 0


def test_llama_cpp_wheel_health_cpu_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS", raising=False)
    h = llama_cpp_wheel_health(None)
    assert h["gpu_mode"] == "cpu"
    assert h["n_gpu_layers"] == 0
    assert h["loaded"] is False
    assert h["env_n_gpu_layers"] is None


def test_llama_cpp_wheel_health_env_and_loaded(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS", "24")
    worker = LlamaCppPythonWorker(model=Path("/tmp/x.gguf"))
    worker._loaded_n_gpu_layers = 24
    h = llama_cpp_wheel_health(worker)
    assert h["gpu_mode"] == "gpu"
    assert h["n_gpu_layers"] == 24
    assert h["loaded"] is True
    assert h["env_n_gpu_layers"] == "24"


def test_llama_cpp_python_backend_parse(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "pypi")
    assert resolve_llama_backend() == LlamaBackendKind.LLAMA_CPP_PYTHON


def test_llama_cpp_python_import():
    from runtime.worker.llama_cpp_python import _import_llama

    Llama = _import_llama()
    assert Llama is not None


@pytest.mark.skipif(
    not os.environ.get("RUN_E2E_LLAMA_CPP_PYTHON"),
    reason="set RUN_E2E_LLAMA_CPP_PYTHON=1 and LLAMA_MODEL",
)
def test_llama_cpp_python_generate():
    gguf = os.environ.get("LLAMA_MODEL", "").strip()
    if not gguf:
        pytest.skip("LLAMA_MODEL not set")
    path = Path(gguf)
    if not path.is_file():
        pytest.skip(f"GGUF missing: {path}")
    worker = create_llama_worker(
        kind=LlamaBackendKind.LLAMA_CPP_PYTHON,
        binary=None,
        model=path,
        host="127.0.0.1",
        port=19983,
    )
    assert isinstance(worker, LlamaCppPythonWorker)
    extra = ["-c", "4096"]
    ngl = os.environ.get("ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS", "").strip()
    if ngl:
        extra.extend(["-ngl", ngl])
    worker.start(extra_args=extra)
    try:
        out = worker.completion("1+1=", n_predict=16)
        assert out.get("content") or out.get("response")
        toks = worker.tokenize_text("hi")
        assert len(toks) >= 1
    finally:
        worker.stop()
