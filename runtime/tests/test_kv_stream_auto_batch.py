"""Phase 15 v37 — stream auto-batch coordinator for concurrent stream generate()."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.kv.stream_auto_batch import (
    StreamAutoBatchCoordinator,
    native_stream_auto_batch_enabled,
    stream_auto_batch_eligible,
)
from runtime.scheduler.scheduler import Request


@pytest.fixture
def cfg_root(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    return RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=tmp_path,
        llama_server_bin=None,
        llama_model=gguf,
        num_blocks=64,
        block_size=16,
        device_count=1,
        llama_parallel_slots=4,
    )


def test_native_stream_auto_batch_default_off(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_KV_AUTO_BATCH_STREAM", raising=False)
    assert native_stream_auto_batch_enabled() is False


def test_native_stream_auto_batch_requires_env(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_AUTO_BATCH_STREAM", "1")
    with patch(
        "runtime.kv.native_decode_loop.native_batch_decode_available",
        return_value=False,
    ):
        assert native_stream_auto_batch_enabled() is False
    with patch(
        "runtime.kv.native_decode_loop.native_batch_decode_available",
        return_value=True,
    ):
        assert native_stream_auto_batch_enabled() is True


def test_stream_auto_batch_eligible_requires_stream_true(cfg_root, monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_AUTO_BATCH_STREAM", "1")
    eng = InferenceEngine(cfg_root)
    with patch(
        "runtime.kv.native_decode_loop.native_batch_decode_available",
        return_value=True,
    ):
        with patch.object(
            eng,
            "_resolved_llama_backend",
            return_value=__import__(
                "runtime.worker.factory", fromlist=["LlamaBackendKind"]
            ).LlamaBackendKind.INPROCESS,
        ):
            assert stream_auto_batch_eligible(eng, gguf=cfg_root.llama_model, stream=True)
            assert not stream_auto_batch_eligible(
                eng, gguf=cfg_root.llama_model, stream=False
            )


def test_stream_coordinator_batches_two_jobs(cfg_root, monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_AUTO_BATCH_STREAM", "1")
    monkeypatch.setenv("ZEROLLAMA_KV_AUTO_BATCH_MS", "50")
    eng = InferenceEngine(cfg_root)
    coord = eng._stream_auto_batch

    req_a = Request(request_id="a", prompt_tokens=[1], max_tokens=4, kv_slot=0)
    req_b = Request(request_id="b", prompt_tokens=[2], max_tokens=4, kv_slot=1)
    collected: dict[str, list[dict]] = {"a": [], "b": []}

    def _parallel(jobs):
        assert len(jobs) == 2
        for i, job in enumerate(jobs):
            yield {"request_id": job.request.request_id, "seq_idx": i, "response": "x", "stop": False}
            yield {
                "request_id": job.request.request_id,
                "seq_idx": i,
                "response": "",
                "stop": True,
            }

    eng._stream_parallel_admitted = MagicMock(side_effect=_parallel)

    def run_a():
        for chunk in coord.iter_stream(
            prompt="pa",
            request=req_a,
            n_predict=4,
            gguf=cfg_root.llama_model,
            num_ctx=4096,
            options={},
        ):
            collected["a"].append(chunk)

    def run_b():
        for chunk in coord.iter_stream(
            prompt="pb",
            request=req_b,
            n_predict=4,
            gguf=cfg_root.llama_model,
            num_ctx=4096,
            options={},
        ):
            collected["b"].append(chunk)

    t1 = threading.Thread(target=run_a)
    t2 = threading.Thread(target=run_b)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    eng._stream_parallel_admitted.assert_called_once()
    assert len(collected["a"]) == 2
    assert len(collected["b"]) == 2
    assert collected["a"][-1]["stop"] is True


def test_stream_coordinator_single_job_direct_path(cfg_root, monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_AUTO_BATCH_MS", "0")
    eng = InferenceEngine(cfg_root)
    coord = eng._stream_auto_batch
    req = Request(request_id="solo", prompt_tokens=[1], max_tokens=2, kv_slot=0)

    eng._stream_parallel_admitted = MagicMock(
        return_value=iter(
            [
                {"request_id": "solo", "seq_idx": 0, "response": "hi", "stop": False},
                {"request_id": "solo", "seq_idx": 0, "response": "", "stop": True},
            ]
        )
    )

    chunks = list(
        coord.iter_stream(
            prompt="p",
            request=req,
            n_predict=2,
            gguf=cfg_root.llama_model,
            num_ctx=2048,
            options={},
        )
    )
    assert len(chunks) == 2
    eng._stream_parallel_admitted.assert_called_once()
    assert eng._stream_parallel_admitted.call_args[0][0][0].request.request_id == "solo"


def test_health_includes_stream_auto_batch(cfg_root):
    eng = InferenceEngine(cfg_root)
    h = eng.health()
    assert "stream" in h["kv_auto_batch"]
    assert h["kv_auto_batch"]["stream"]["enabled"] is False
