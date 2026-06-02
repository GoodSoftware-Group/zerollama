from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.scheduler.scheduler import Request
from runtime.worker.llama_server import LlamaServerError


@pytest.fixture
def engine(cfg_root, tmp_path: Path):
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
    return InferenceEngine(cfg)


def test_vram_precheck_skips_when_same_model_loaded(engine, tmp_path: Path):
    gguf = engine.config.llama_model
    assert gguf is not None
    engine._server = MagicMock()
    engine._server._proc = MagicMock()
    engine._server._proc.poll.return_value = None
    engine._server.model.resolve.return_value = gguf.resolve()
    engine._loaded_vram_num_ctx = 4096
    req = Request("r1", [1], 8, gguf=gguf, num_ctx=2048)
    with patch("runtime.engine.check_gguf_vram_budget") as chk:
        engine._vram_precheck_load(req)
        chk.assert_not_called()


def test_vram_precheck_runs_host_before_gpu(engine, tmp_path: Path, monkeypatch):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    calls: list[str] = []

    monkeypatch.setattr(
        "runtime.engine.check_gguf_host_budget",
        lambda _p: calls.append("host"),
    )
    monkeypatch.setattr("runtime.engine.gpu_vram_check_enabled", lambda: True)
    monkeypatch.setattr(
        "runtime.engine.check_gguf_vram_budget",
        lambda *_a, **_k: calls.append("gpu"),
    )
    engine._server = None
    req = Request("r1", [1], 8, gguf=gguf)
    engine._vram_precheck_load(req)
    assert calls == ["host", "gpu"]


def test_vram_precheck_runs_when_loaded_ctx_unknown(engine, tmp_path: Path):
    gguf = engine.config.llama_model
    assert gguf is not None
    engine._server = MagicMock()
    engine._server._proc = MagicMock()
    engine._server._proc.poll.return_value = None
    engine._server.model.resolve.return_value = gguf.resolve()
    engine._loaded_vram_num_ctx = None
    req = Request("r1", [1], 8, gguf=gguf, num_ctx=2048)
    with patch("runtime.engine.gpu_vram_check_enabled", return_value=True):
        with patch("runtime.engine.check_gguf_host_budget") as host:
            with patch("runtime.engine.check_gguf_vram_budget") as gpu:
                engine._vram_precheck_load(req)
                host.assert_called_once()
                gpu.assert_called_once()


def test_vram_precheck_runs_on_larger_num_ctx(engine, tmp_path: Path):
    gguf = engine.config.llama_model
    assert gguf is not None
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


def test_vram_precheck_uses_resolved_num_ctx_from_options(engine, tmp_path: Path, monkeypatch):
    """Dequeue GPU check must not pass raw req.num_ctx when options carry num_ctx."""
    gguf = engine.config.llama_model
    assert gguf is not None
    engine._server = None
    req = Request(
        "r1",
        [1],
        8,
        gguf=gguf,
        num_ctx=None,
        vram_options={"num_ctx": 8192},
    )
    captured: dict = {}

    def _capture(*_a, **kw):
        captured["num_ctx"] = kw.get("num_ctx")

    monkeypatch.setattr("runtime.engine.gpu_vram_check_enabled", lambda: True)
    monkeypatch.setattr("runtime.engine.check_gguf_host_budget", lambda _p: None)
    monkeypatch.setattr("runtime.engine.check_gguf_vram_budget", _capture)
    engine._vram_precheck_load(req)
    assert captured.get("num_ctx") == 8192


def test_vram_precheck_runs_on_model_swap(engine, tmp_path: Path):
    a = tmp_path / "a.gguf"
    b = tmp_path / "b.gguf"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    engine._server = MagicMock()
    engine._server._proc = MagicMock()
    engine._server._proc.poll.return_value = None
    engine._server.model.resolve.return_value = a.resolve()
    req = Request("r1", [1], 8, gguf=b)
    with patch("runtime.engine.gpu_vram_check_enabled", return_value=True):
        with patch("runtime.engine.check_gguf_vram_budget") as chk:
            engine._vram_precheck_load(req)
            chk.assert_called_once()
        assert chk.call_args[0][0] == b.resolve()


def test_vram_precheck_passes_training_reserve_flag(engine):
    req = Request("r1", [1], 8, gguf=engine.config.llama_model)
    engine.coordinator.set_go_training_gpu_busy(True)
    captured: dict = {}

    def _capture(*_a, **kw):
        captured["training_reserve_active"] = kw.get("training_reserve_active")

    with patch("runtime.engine.gpu_vram_check_enabled", return_value=True):
        with patch("runtime.engine.check_gguf_host_budget", lambda _p: None):
            with patch("runtime.engine.check_gguf_vram_budget", _capture):
                engine._vram_precheck_load(req)
    assert captured.get("training_reserve_active") is True


def test_vram_check_admitting_calls_precheck(engine):
    req = Request("r1", [1], 8)
    with patch.object(engine, "_vram_precheck_load") as pre:
        with patch.object(engine.coordinator, "check_admit"):
            engine._vram_check_admitting(req)
        pre.assert_called_once()


def test_admit_one_stores_resolved_num_ctx_from_options(engine, monkeypatch):
    """Queued Request.num_ctx must match resolve_vram_num_ctx (skip/precheck parity)."""
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX", "0")
    gguf = engine.config.llama_model
    assert gguf is not None
    with patch.object(engine.loop, "tick", return_value=[MagicMock(request_id="r1")]):
        with patch.object(engine, "_check_admit_policy", return_value="normal"):
            with patch.object(engine, "_vram_precheck_enqueue"):
                with patch.object(engine.scheduler, "add_request") as add:
                    engine._admit_one(
                        "hi",
                        8,
                        gguf=gguf,
                        num_ctx=None,
                        options={"num_ctx": 8192},
                    )
                    req = add.call_args[0][0]
                    assert req.num_ctx == 8192
                    assert req.vram_options is not None
                    assert req.vram_options.get("num_ctx") == 8192


def test_enqueue_vram_precheck_before_queue(engine, tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    order: list[str] = []

    def track_host(_p):
        order.append("host")

    def track_gpu(*_a, **_k):
        order.append("gpu")

    def track_add(_req):
        order.append("queue")

    with patch("runtime.engine.check_gguf_host_budget", side_effect=track_host):
        with patch("runtime.engine.check_gguf_vram_budget", side_effect=track_gpu):
            with patch.object(engine, "_check_admit_policy", return_value="normal"):
                with patch.object(
                    engine.loop,
                    "tick",
                    return_value=[MagicMock(request_id="admitted")],
                ):
                    with patch.object(
                        engine.scheduler, "add_request", side_effect=track_add
                    ):
                        engine._admit_one("hi", 8, gguf=gguf)
    assert order == ["host", "gpu", "queue"]


def test_admit_tick_vram_budget_rejects(engine, tmp_path: Path):
    gguf = tmp_path / "big.gguf"
    gguf.write_bytes(b"x")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    try:
        with patch("runtime.engine.gpu_vram_check_enabled", return_value=True):
            with patch(
                "runtime.engine.check_gguf_vram_budget",
                side_effect=LlamaServerError("GPU memory"),
            ):
                with pytest.raises(LlamaServerError, match="GPU memory"):
                    engine._admit_one("hi", 8, gguf=gguf)
    finally:
        monkeypatch.undo()
    assert len(engine.scheduler.waiting) == 0
