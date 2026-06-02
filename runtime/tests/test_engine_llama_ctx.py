from pathlib import Path

import pytest

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine


@pytest.fixture
def engine(cfg_root, tmp_path: Path):
    bin_path = tmp_path / "llama-server"
    bin_path.write_text("#!/bin/sh\ntrue\n")
    bin_path.chmod(0o755)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=bin_path,
        llama_model=gguf,
        num_blocks=64,
        block_size=16,
        device_count=1,
    )
    return InferenceEngine(cfg)


def test_llama_server_start_args_injects_num_ctx(engine):
    args = engine._llama_server_start_args(
        engine.config.llama_model,
        num_ctx=8192,
        options={"num_ctx": 8192},
    )
    from runtime.llama_args import parse_llama_server_args

    assert parse_llama_server_args(args).num_ctx == 8192
