"""Vocab session selection for render tokenize (Phase 14)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.worker.factory import LlamaBackendKind


@pytest.fixture
def engine(tmp_path: Path):
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=18080,
        llama_cpp_root=tmp_path,
        llama_server_bin=tmp_path / "llama-server",
        llama_model=None,
        num_blocks=8,
        block_size=16,
        device_count=1,
        main_gpu=0,
        llama_backend="subprocess",
    )
    return InferenceEngine(cfg)


def test_new_vocab_session_uses_wheel_when_backend_is_cpp_python(
    engine: InferenceEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x" * 64)
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "llama-cpp-python")
    with patch(
        "runtime.worker.llama_cpp_python.LlamaCppVocabSession",
        autospec=True,
    ) as mock_cls:
        mock_cls.return_value = MagicMock()
        engine._new_vocab_session(gguf)
        mock_cls.assert_called_once()


def test_build_llama_cpp_load_kwargs_split_mode():
    from runtime.llama_args import LlamaServerArgHints, split_mode_to_llama_cpp_int
    import llama_cpp

    assert split_mode_to_llama_cpp_int("none") == llama_cpp.LLAMA_SPLIT_MODE_NONE
    assert split_mode_to_llama_cpp_int("row") == llama_cpp.LLAMA_SPLIT_MODE_ROW

    from runtime.worker.llama_cpp_python import build_llama_cpp_load_kwargs

    hints = LlamaServerArgHints(split_mode="row", n_gpu_layers=8, num_ctx=2048)
    kw = build_llama_cpp_load_kwargs(
        Path("/tmp/m.gguf"),
        hints,
        n_gpu_layers=-1,
        main_gpu=0,
    )
    assert kw["split_mode"] == llama_cpp.LLAMA_SPLIT_MODE_ROW
    assert kw["n_ctx"] == 2048
    assert kw["n_gpu_layers"] == 8
