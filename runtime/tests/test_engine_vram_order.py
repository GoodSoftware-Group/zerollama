from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from runtime.engine import InferenceEngine

@pytest.fixture
def engine(tmp_path: Path):
    from runtime.config import RuntimeConfig

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
    )
    cfg.llama_server_bin.write_text("#!/bin/sh\ntrue\n")
    cfg.llama_server_bin.chmod(0o755)
    return InferenceEngine(cfg)


def test_vram_check_runs_after_stop_server(engine, tmp_path: Path):
    a = tmp_path / "a.gguf"
    b = tmp_path / "b.gguf"
    a.write_bytes(b"x" * 64)
    b.write_bytes(b"y" * 64)

    order: list[str] = []

    existing = MagicMock()
    existing.model.resolve.return_value = a.resolve()
    existing.is_running.return_value = True
    engine._server = existing

    replacement = MagicMock()
    replacement.is_running.return_value = False

    def stop_tracked():
        order.append("stop")
        engine._server = None

    def check_tracked(gguf, **kwargs):
        order.append("check")

    with patch.object(engine, "_stop_server", side_effect=stop_tracked):
        with patch("runtime.engine.check_gguf_vram_budget", side_effect=check_tracked):
            with patch("runtime.engine.check_gguf_host_budget"):
                with patch.object(
                    engine,
                    "_create_llama_worker",
                    return_value=replacement,
                ):
                    engine._ensure_gguf_loaded_unlocked(b)

    assert order == ["stop", "check"]
