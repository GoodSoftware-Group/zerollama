from runtime.go_coordination import update_go_coordination
from runtime.gpu.inference_policy import inference_backpressure_blocks_low
from runtime.gpu.mutex import InferenceGpuCoordinator
from runtime.gpu.priority import InferencePriority
from runtime.kv.block_pool import BlockPool
from runtime.scheduler.loop import SchedulerLoop
from runtime.scheduler.scheduler import Request, Scheduler


def test_tick_admits_high_skips_low_under_backpressure():
    pool = BlockPool(num_blocks=64, block_size=16)
    sched = Scheduler.for_pools([pool])
    coord = InferenceGpuCoordinator()
    loop = SchedulerLoop(scheduler=sched, coordinator=coord, pools=[pool])
    update_go_coordination({"defer_waiting": 2, "sched_pending": 2, "sched_active": 1})
    low = Request(request_id="low", prompt_tokens=[1], max_tokens=4, priority=InferencePriority.LOW)
    high = Request(
        request_id="high", prompt_tokens=[1], max_tokens=4, priority=InferencePriority.HIGH
    )
    sched.add_request(low)
    sched.add_request(high)
    assert inference_backpressure_blocks_low(
        runtime_waiting=len(sched.waiting), runtime_running=0
    )
    admitted = loop.tick(max_admit=2)
    assert len(admitted) == 1
    assert admitted[0].priority == InferencePriority.HIGH
    assert len(sched.waiting) == 1
    assert sched.waiting[0].priority == InferencePriority.LOW
