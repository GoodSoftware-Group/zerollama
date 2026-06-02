"""Dequeue backpressure applies to batch (low) queue head only — NORMAL keeps running."""

from runtime.go_coordination import update_go_coordination
from runtime.gpu.inference_policy import inference_backpressure_blocks_low
from runtime.gpu.priority import InferencePriority
from runtime.kv.block_pool import BlockPool
from runtime.scheduler.scheduler import Request, Scheduler


def _sched() -> Scheduler:
    pools = [BlockPool(num_blocks=16, block_size=16, device_id=0)]
    return Scheduler.for_pools(pools)


def test_normal_dequeues_under_runtime_backlog():
    update_go_coordination({})
    sched = _sched()
    assert inference_backpressure_blocks_low(
        runtime_waiting=4, runtime_running=0, runtime_oldest_fifo=1
    )
    normal = Request("n", [1], 8, priority=InferencePriority.NORMAL)
    sched.add_request(normal)
    got = sched.pop_waiting_for_tick(low_backpressure=True, cross_fifo_blocked=False)
    assert got is not None
    assert got.request_id == "n"


def test_low_stalls_under_runtime_backlog():
    update_go_coordination({})
    sched = _sched()
    low = Request("l", [1], 8, priority=InferencePriority.LOW)
    sched.add_request(low)
    assert (
        sched.pop_waiting_for_tick(low_backpressure=True, cross_fifo_blocked=False)
        is None
    )


def test_high_dequeues_when_low_blocked():
    update_go_coordination({"ggml_loads_paused": True})
    sched = _sched()
    sched.add_request(Request("l", [1], 8, priority=InferencePriority.LOW))
    sched.add_request(Request("h", [1], 8, priority=InferencePriority.HIGH))
    got = sched.pop_waiting_for_tick(
        low_backpressure=True, cross_fifo_blocked=False
    )
    assert got is not None
    assert got.request_id == "h"


def test_normal_not_blocked_by_cross_fifo():
    sched = _sched()
    normal = Request("n", [1], 8, priority=InferencePriority.NORMAL)
    sched.add_request(normal)
    got = sched.pop_waiting_for_tick(
        low_backpressure=False, cross_fifo_blocked=True
    )
    assert got is not None
    assert got.request_id == "n"
