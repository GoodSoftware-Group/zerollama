from unittest.mock import patch

import pytest

from runtime.gpu.admission import AdmissionRejected
from runtime.gpu.mutex import InferenceGpuCoordinator
from runtime.gpu.priority import InferencePriority
from runtime.kv.block_pool import BlockPool
from runtime.scheduler.loop import SchedulerLoop
from runtime.scheduler.scheduler import Request, Scheduler


def test_tick_misconfig_does_not_requeue():
    pool = BlockPool(num_blocks=64, block_size=16)
    sched = Scheduler.for_pools([pool])
    coord = InferenceGpuCoordinator()
    loop = SchedulerLoop(scheduler=sched, coordinator=coord, pools=[pool])
    req = Request(
        request_id="a",
        prompt_tokens=[1],
        max_tokens=8,
    )
    sched.add_request(req)

    from runtime.gpu.admission import AdmissionMisconfigured
    from runtime.worker.llama_server import LlamaServerError

    def misconfig(_r: Request) -> None:
        raise AdmissionMisconfigured("bad env")

    with pytest.raises(LlamaServerError, match="bad env"):
        loop.tick(max_admit=1, vram_check=misconfig)
    assert len(sched.waiting) == 0


def test_cancel_waiting_removes_request():
    pool = BlockPool(num_blocks=64, block_size=16)
    sched = Scheduler.for_pools([pool])
    req = Request(request_id="x", prompt_tokens=[1], max_tokens=4)
    sched.add_request(req)
    assert sched.cancel_waiting(req)
    assert len(sched.waiting) == 0


def test_tick_vram_recheck_requeues_when_full():
    pool = BlockPool(num_blocks=64, block_size=16)
    sched = Scheduler.for_pools([pool])
    coord = InferenceGpuCoordinator()
    loop = SchedulerLoop(scheduler=sched, coordinator=coord, pools=[pool])
    req = Request(
        request_id="a",
        prompt_tokens=[1],
        max_tokens=8,
        priority=InferencePriority.NORMAL,
    )
    sched.add_request(req)

    def reject(_r: Request) -> None:
        raise AdmissionRejected("gpu full")

    admitted = loop.tick(max_admit=1, vram_check=reject)
    assert admitted == []
    assert len(sched.waiting) == 1


def test_mmap_factor_lowers_estimate(monkeypatch, tmp_path):
    from pathlib import Path

    from runtime.gpu_vram import estimate_gguf_vram_bytes

    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_EXACT", "0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_MMAP_FACTOR", "0.5")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_SCRATCH_FACTOR", "1.0")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch("runtime.host_memory.estimate_gguf_ram_bytes", return_value=1000):
        full = estimate_gguf_vram_bytes(gguf)
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_MMAP_FACTOR", "1.0")
    with patch("runtime.host_memory.estimate_gguf_ram_bytes", return_value=1000):
        assert estimate_gguf_vram_bytes(gguf) > full
