from runtime.go_coordination import (
    cross_queue_depth,
    go_fifo_oldest,
    update_go_coordination,
)
from runtime.gpu.inference_policy import cross_fifo_blocks_low
from runtime.gpu.priority import InferencePriority
from runtime.kv.block_pool import BlockPool
from runtime.scheduler.loop import SchedulerLoop
from runtime.scheduler.scheduler import Request, RequestState, Scheduler
from runtime.gpu.mutex import InferenceGpuCoordinator


def test_cross_fifo_blocks_when_ggml_is_older():
    update_go_coordination({"fifo_go_oldest_ggml": 5, "fifo_go_oldest_defer": 1})
    assert cross_fifo_blocks_low(runtime_oldest_fifo=10)
    assert not cross_fifo_blocks_low(runtime_oldest_fifo=3)


def test_cross_fifo_ignores_defer_tickets():
    """Defer-only mirror must not block runtime (inference-first; defer has other gates)."""
    update_go_coordination({"fifo_go_oldest_defer": 1, "fifo_go_oldest": 1})
    assert not cross_fifo_blocks_low(runtime_oldest_fifo=10)


def test_oldest_fifo_includes_running():
    pool = BlockPool(num_blocks=64, block_size=16)
    sched = Scheduler.for_pools([pool])
    running = Request(
        request_id="run",
        prompt_tokens=[1],
        max_tokens=4,
        fifo_seq=7,
        state=RequestState.PREFILL,
    )
    sched.running.append(running)
    assert sched.oldest_fifo_seq() == 7


def test_cross_fifo_stale_mirror_fail_open():
    update_go_coordination({})
    import runtime.go_coordination as gc

    with gc._lock:
        gc._updated_at = None
    assert not cross_fifo_blocks_low(runtime_oldest_fifo=10)


def test_scheduler_tick_waits_for_go_fifo():
    pool = BlockPool(num_blocks=64, block_size=16)
    sched = Scheduler.for_pools([pool])
    coord = InferenceGpuCoordinator()
    loop = SchedulerLoop(scheduler=sched, coordinator=coord, pools=[pool])
    update_go_coordination({"fifo_go_oldest_ggml": 1, "fifo_go_oldest": 1})
    low = Request(
        request_id="low",
        prompt_tokens=[1],
        max_tokens=4,
        priority=InferencePriority.LOW,
        fifo_seq=10,
    )
    sched.add_request(low)
    admitted = loop.tick(max_admit=1)
    assert admitted == []
    assert len(sched.waiting) == 1


def test_cross_queue_depth_includes_fifo_fields():
    update_go_coordination({"fifo_go_oldest_ggml": 7, "fifo_go_oldest_defer": 9})
    d = cross_queue_depth(runtime_waiting=1, runtime_running=0)
    assert d["fifo_go_oldest"] == 7
    assert go_fifo_oldest() == 7
