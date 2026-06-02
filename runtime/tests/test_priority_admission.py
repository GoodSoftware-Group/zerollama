import pytest

from runtime.gpu.admission import (
    AdmissionRejected,
    check_vram_admission,
    effective_min_free_for_priority,
    vram_gate_bypassed,
)
from runtime.gpu.priority import InferencePriority, parse_inference_priority
from runtime.scheduler.scheduler import Request, Scheduler


def test_parse_inference_priority():
    assert parse_inference_priority("interactive") == InferencePriority.HIGH
    assert parse_inference_priority("batch") == InferencePriority.LOW


def test_high_bypasses_vram_gate(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    check_vram_admission(1024, backlog=0, priority=InferencePriority.HIGH)


def test_low_stricter_min_free():
    assert effective_min_free_for_priority(1024**3, InferencePriority.LOW) == int(
        1.5 * 1024**3
    )


def test_low_rejected_when_borderline(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    with pytest.raises(AdmissionRejected):
        check_vram_admission(
            int(1.5 * 1024**3) - 1, backlog=0, priority=InferencePriority.LOW
        )


def test_high_priority_queue_front():
    from runtime.kv.block_pool import BlockPool

    pools = [BlockPool(num_blocks=16, block_size=16, device_id=0)]
    sched = Scheduler.for_pools(pools)
    a = Request("a", [1], 8)
    b = Request("b", [1], 8)
    sched.add_request(a, priority=InferencePriority.NORMAL)
    sched.add_request(b, priority=InferencePriority.HIGH)
    first = sched.pop_waiting()
    assert first is not None
    assert first.request_id == "b"


def test_vram_gate_bypass_high_only():
    assert vram_gate_bypassed(InferencePriority.HIGH)
    assert not vram_gate_bypassed(InferencePriority.NORMAL)
