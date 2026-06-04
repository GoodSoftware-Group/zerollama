"""Phase 14 in-process libllama forward (optional GPU)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from runtime.config import RuntimeConfig
from runtime.worker.factory import (
    LlamaBackendKind,
    canonical_llama_backend,
    create_llama_worker,
    llama_backend_from_env,
    llama_backend_source,
    resolve_llama_backend,
)
from runtime.worker.libllama_ctypes import (
    LlamaContextParams,
    LlamaModelParams,
    resolve_libllama_path,
)


def test_llama_backend_env_parse(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", raising=False)
    assert llama_backend_from_env() == LlamaBackendKind.SUBPROCESS
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "inprocess")
    assert llama_backend_from_env() == LlamaBackendKind.INPROCESS
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "subprocess")
    assert llama_backend_from_env() == LlamaBackendKind.SUBPROCESS
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "llama-cpp-python")
    assert llama_backend_from_env() == LlamaBackendKind.LLAMA_CPP_PYTHON


def test_resolve_llama_backend_config_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", raising=False)
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=Path("/tmp"),
        llama_server_bin=None,
        llama_model=None,
        num_blocks=8,
        block_size=16,
        device_count=1,
        llama_backend="inprocess",
    )
    assert resolve_llama_backend(cfg) == LlamaBackendKind.INPROCESS


def test_llama_backend_source_env_vs_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", raising=False)
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=Path("/tmp"),
        llama_server_bin=None,
        llama_model=None,
        num_blocks=8,
        block_size=16,
        device_count=1,
        llama_backend="inprocess",
        llama_backend_from_file=True,
    )
    assert llama_backend_source(cfg) == "config"
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "subprocess")
    assert llama_backend_source(cfg) == "env"


def test_llama_backend_source_default_without_yaml_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", raising=False)
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=Path("/tmp"),
        llama_server_bin=None,
        llama_model=None,
        num_blocks=8,
        block_size=16,
        device_count=1,
    )
    assert llama_backend_source(cfg) == "default"


def test_canonical_llama_backend_aliases():
    assert canonical_llama_backend("libllama") == "inprocess"
    assert canonical_llama_backend("llama-server") == "subprocess"
    assert canonical_llama_backend(None) == "subprocess"


def test_canonical_llama_backend_rejects_unknown():
    with pytest.raises(ValueError, match="unknown llama backend"):
        canonical_llama_backend("bogus")


def test_batch_from_tokens_sets_pos():
    from runtime.worker.libllama_ctypes import _batch_from_tokens, get_lib

    try:
        lib = get_lib(cpp_root=Path("/root/llama.cpp"))
    except Exception:
        pytest.skip("libllama unavailable")
    batch = _batch_from_tokens(
        lib,
        [10, 20, 30],
        seq_id=0,
        n_seq_max=1,
        logits_last=True,
        pos_start=7,
    )
    try:
        assert batch.n_tokens == 3
        assert int(batch.pos[0]) == 7
        assert int(batch.pos[1]) == 8
        assert int(batch.pos[2]) == 9
        assert int(batch.token[0]) == 10
    finally:
        lib.llama_batch_free(batch)


def test_ctypes_struct_sizes():
    import ctypes

    assert ctypes.sizeof(LlamaModelParams) == 72
    assert ctypes.sizeof(LlamaContextParams) == 144


def test_resolve_libllama_path_sibling_checkout():
    try:
        p = resolve_libllama_path(cpp_root=Path("/root/llama.cpp"))
        assert p.name.startswith("libllama")
    except Exception:
        pytest.skip("libllama.so not built on this host")


@pytest.mark.skipif(
    not os.environ.get("LLAMA_MODEL"),
    reason="set LLAMA_MODEL to a local GGUF",
)
def test_vocab_only_tokenize():
    from runtime.worker.libllama_ctypes import LlamaVocabSession

    gguf = os.environ.get("LLAMA_MODEL", "").strip()
    if not gguf:
        pytest.skip("LLAMA_MODEL not set")
    path = Path(gguf)
    if not path.is_file():
        pytest.skip(f"GGUF missing: {path}")
    session = LlamaVocabSession(path)
    try:
        tokens = session.tokenize_text("Hello")
        assert len(tokens) >= 1
    finally:
        session.close()


def test_inprocess_rejects_speculative(tmp_path: Path):
    from runtime.worker.llama_inprocess import LlamaInprocessWorker

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    worker = LlamaInprocessWorker(model=gguf, lib_path=tmp_path / "nolib")
    with pytest.raises(Exception, match="speculative"):
        worker.start(extra_args=["--model-draft", "/other.gguf"])


@pytest.mark.skipif(
    not os.environ.get("RUN_E2E_INPROCESS"),
    reason="set RUN_E2E_INPROCESS=1 and LLAMA_MODEL to a small GGUF",
)
def test_inprocess_worker_stream():
    gguf = os.environ.get("LLAMA_MODEL", "").strip()
    if not gguf:
        pytest.skip("LLAMA_MODEL not set")
    path = Path(gguf)
    if not path.is_file():
        pytest.skip(f"GGUF missing: {path}")
    worker = create_llama_worker(
        kind=LlamaBackendKind.INPROCESS,
        binary=None,
        model=path,
        host="127.0.0.1",
        port=19982,
    )
    worker.start(extra_args=["-c", "4096", "-ngl", "99"])
    try:
        chunks = list(worker.completion_stream("Count: 1,", n_predict=12))
        text = "".join(c.get("content") or "" for c in chunks)
        assert chunks[-1].get("stop") is True
        assert text
    finally:
        worker.stop()


@pytest.mark.skipif(
    not os.environ.get("RUN_E2E_INPROCESS"),
    reason="set RUN_E2E_INPROCESS=1 and LLAMA_MODEL to a small GGUF",
)
def test_inprocess_worker_generate():
    gguf = os.environ.get("LLAMA_MODEL", "").strip()
    if not gguf:
        pytest.skip("LLAMA_MODEL not set")
    path = Path(gguf)
    if not path.is_file():
        pytest.skip(f"GGUF missing: {path}")
    worker = create_llama_worker(
        kind=LlamaBackendKind.INPROCESS,
        binary=None,
        model=path,
        host="127.0.0.1",
        port=19981,
    )
    worker.start(extra_args=["-c", "4096", "-ngl", "99"])
    try:
        out = worker.completion("1+1=", n_predict=16)
        assert out.get("content") or out.get("response")
    finally:
        worker.stop()
