from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.server.openai_v1 import prepare_v1_chat, v1_request_options


def test_v1_request_options_promotes_max_tokens_to_num_predict():
    opts = v1_request_options({"max_tokens": 48, "messages": []})
    assert opts["num_predict"] == 48


def test_v1_request_options_merges_body_and_extra():
    body = {
        "options": {"num_ctx": 4096, "gguf": "/tmp/m.gguf"},
        "extra_body": {"options": {"num_predict": 64}},
        "num_ctx": 8192,
    }
    opts = v1_request_options(body)
    assert opts["num_ctx"] == 4096
    assert opts["gguf"] == "/tmp/m.gguf"
    assert opts["num_predict"] == 64


def test_prepare_v1_chat_max_tokens_float(monkeypatch, cfg_root, tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=gguf,
        num_blocks=64,
        block_size=16,
        device_count=1,
    )
    eng = InferenceEngine(cfg)
    with patch.object(eng, "resolve_num_ctx_for_request", return_value=(4096, {})):
        prep = prepare_v1_chat(
            eng,
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 42.0,
            },
        )
    assert prep.n_predict == 42


def test_prepare_v1_chat_uses_resolve_num_ctx(monkeypatch, cfg_root, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX", "1")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=gguf,
        num_blocks=64,
        block_size=16,
        device_count=1,
    )
    eng = InferenceEngine(cfg)
    with patch.object(eng, "_effective_vram_free_for_suggest", return_value=16 * 1024**3):
        with patch(
            "runtime.vram_suggest.suggest_max_num_ctx",
            return_value=4096,
        ):
            with patch(
                "runtime.server.openai_v1.resolve_tools_chat_prompt",
                return_value=("prompt", "{", {}),
            ):
                prep = prepare_v1_chat(
                    eng,
                    {
                        "model": "m",
                        "messages": [{"role": "user", "content": "hi"}],
                        "options": {"num_ctx": 32768},
                    },
                )
    assert prep.num_ctx == 4096
    assert prep.options.get("num_ctx") == 32768
