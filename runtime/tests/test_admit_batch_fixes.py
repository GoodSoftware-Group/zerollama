from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.scheduler.scheduler import Request
from runtime.worker.llama_server import LlamaServerError


@pytest.fixture
def engine(cfg_root):
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=None,
        num_blocks=64,
        block_size=16,
        device_count=1,
    )
    return InferenceEngine(cfg)


def test_vram_precheck_runs_when_num_ctx_grows(engine, tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    engine._server = MagicMock()
    engine._server._proc = MagicMock()
    engine._server._proc.poll.return_value = None
    engine._server.model.resolve.return_value = gguf.resolve()
    engine._loaded_vram_num_ctx = 2048
    req = Request("r1", [1], 8, gguf=gguf, num_ctx=8192)
    with patch("runtime.engine.gpu_vram_check_enabled", return_value=True):
        with patch("runtime.engine.check_gguf_vram_budget") as chk:
            engine._vram_precheck_load(req)
            chk.assert_called_once()


def test_tick_rollback_running_on_vram_error(engine):
    engine.scheduler.add_request(Request("a", [1], 8))
    engine.scheduler.add_request(Request("b", [1], 8))
    calls = 0

    def vram_fail(_req: Request) -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise LlamaServerError("GPU memory")

    with pytest.raises(LlamaServerError, match="GPU memory"):
        engine.loop.tick(max_admit=2, vram_check=vram_fail)
    assert len(engine.scheduler.running) == 0
    assert len(engine.scheduler.waiting) == 0


def test_generate_batch_checks_admission_once(engine):
    seen: list[int] = []

    def capture(*, waiting: int, **kwargs):
        seen.append(waiting)

    with patch.object(engine.coordinator, "check_admit", side_effect=capture):
        with patch.object(engine.loop, "tick", return_value=[]):
            with pytest.raises(LlamaServerError, match="could not admit batch"):
                engine.generate_batch(["a", "b", "c"], max_admit=3)
    assert seen == [3]
