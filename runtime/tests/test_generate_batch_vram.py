from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.gpu.inference_policy import InferencePriority
from runtime.worker.llama_server import LlamaServerError


@pytest.fixture
def engine(cfg_root, tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    bin_path = tmp_path / "llama-server"
    bin_path.write_text("#!/bin/sh\ntrue\n")
    bin_path.chmod(0o755)
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=bin_path,
        llama_model=gguf,
        num_blocks=520,
        block_size=16,
        device_count=1,
    )
    return InferenceEngine(cfg)


def test_generate_batch_load_passes_num_ctx(engine, monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX", "0")
    admitted = []

    def fake_tick(max_admit=4, *, vram_check=None):
        while engine.scheduler.waiting:
            req = engine.scheduler.waiting.popleft()
            if req.block_table is not None:
                req.block_table.ensure_capacity(engine.loop._tokens_to_reserve(req))
            engine.scheduler.running.append(req)
            admitted.append(req)
            if len(admitted) >= max_admit:
                break
        return admitted

    mock_srv = MagicMock()
    mock_srv.completions_parallel.return_value = [{"content": "ok"}]

    with patch.object(engine, "_check_admit_policy", return_value=InferencePriority.LOW):
        with patch.object(engine, "_vram_precheck_enqueue"):
            with patch.object(engine.loop, "tick", side_effect=fake_tick):
                with patch.object(
                    engine, "_ensure_gguf_loaded_unlocked", return_value=mock_srv
                ) as load:
                    engine.generate_batch(
                        ["a"],
                        n_predict=4,
                        options={"num_ctx": 8192},
                    )
    load.assert_called_once()
    _, kwargs = load.call_args
    assert kwargs.get("num_ctx") == 8192
    assert kwargs.get("options", {}).get("num_ctx") == 8192
