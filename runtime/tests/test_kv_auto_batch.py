"""Phase 15 v32 — auto-batch coordinator for concurrent generate()."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from runtime.config import RuntimeConfig
from runtime.engine import GenerateResult, InferenceEngine
from runtime.kv.auto_batch import (
    AutoBatchCoordinator,
    auto_batch_eligible,
    native_auto_batch_enabled,
)
from runtime.scheduler.scheduler import Request, RequestState


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


def test_native_auto_batch_default_off(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_KV_AUTO_BATCH", raising=False)
    assert native_auto_batch_enabled() is False


def test_native_auto_batch_requires_env(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_AUTO_BATCH", "1")
    with patch(
        "runtime.kv.native_decode_loop.native_batch_decode_available",
        return_value=False,
    ):
        assert native_auto_batch_enabled() is False
    with patch(
        "runtime.kv.native_decode_loop.native_batch_decode_available",
        return_value=True,
    ):
        assert native_auto_batch_enabled() is True


def test_auto_batch_eligible_inprocess_multiseq(cfg_root, monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_AUTO_BATCH", "1")
    eng = InferenceEngine(cfg_root)
    with patch(
        "runtime.kv.native_decode_loop.native_batch_decode_available",
        return_value=True,
    ):
        with patch.object(
            eng, "_resolved_llama_backend", return_value=__import__(
                "runtime.worker.factory", fromlist=["LlamaBackendKind"]
            ).LlamaBackendKind.INPROCESS
        ):
            assert auto_batch_eligible(eng, gguf=cfg_root.llama_model, stream=False)
            assert not auto_batch_eligible(eng, gguf=cfg_root.llama_model, stream=True)


def test_coordinator_batches_two_jobs(cfg_root, monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_AUTO_BATCH", "1")
    monkeypatch.setenv("ZEROLLAMA_KV_AUTO_BATCH_MS", "50")
    eng = InferenceEngine(cfg_root)
    coord = eng._auto_batch

    req_a = Request(request_id="a", prompt_tokens=[1], max_tokens=4, kv_slot=0)
    req_b = Request(request_id="b", prompt_tokens=[2], max_tokens=4, kv_slot=1)
    results: list[GenerateResult] = []

    def _parallel(jobs):
        return [
            GenerateResult(content=f"out-{j.request.request_id}", request_id=j.request.request_id)
            for j in jobs
        ]

    eng._generate_parallel_admitted = MagicMock(side_effect=_parallel)
    eng._generate_one_admitted = MagicMock(
        side_effect=lambda prompt, req, **kw: GenerateResult(
            content=f"solo-{req.request_id}", request_id=req.request_id
        )
    )

    def run_a():
        results.append(
            coord.submit(
                prompt="pa",
                request=req_a,
                n_predict=4,
                gguf=cfg_root.llama_model,
                num_ctx=4096,
                options={},
            )
        )

    def run_b():
        results.append(
            coord.submit(
                prompt="pb",
                request=req_b,
                n_predict=4,
                gguf=cfg_root.llama_model,
                num_ctx=4096,
                options={},
            )
        )

    t1 = threading.Thread(target=run_a)
    t2 = threading.Thread(target=run_b)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(results) == 2
    eng._generate_parallel_admitted.assert_called_once()
    assert eng._generate_one_admitted.call_count == 0


def test_coordinator_single_job_uses_direct_path(cfg_root, monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_AUTO_BATCH_MS", "0")
    eng = InferenceEngine(cfg_root)
    coord = eng._auto_batch
    req = Request(request_id="solo", prompt_tokens=[1], max_tokens=2, kv_slot=0)

    eng._generate_one_admitted = MagicMock(
        return_value=GenerateResult(content="ok", request_id="solo")
    )
    eng._generate_parallel_admitted = MagicMock()

    out = coord.submit(
        prompt="p",
        request=req,
        n_predict=2,
        gguf=cfg_root.llama_model,
        num_ctx=2048,
        options={},
    )
    assert out.content == "ok"
    eng._generate_one_admitted.assert_called_once()
    eng._generate_parallel_admitted.assert_not_called()


def test_health_includes_kv_auto_batch(cfg_root):
    eng = InferenceEngine(cfg_root)
    h = eng.health()
    assert "kv_auto_batch" in h
    assert h["kv_auto_batch"]["non_stream"]["enabled"] is False
    assert h["kv_auto_batch"]["stream"]["enabled"] is False
