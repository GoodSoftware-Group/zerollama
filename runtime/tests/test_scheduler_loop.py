from runtime.gpu.mutex import InferenceGpuCoordinator
from runtime.kv.block_pool import BlockPool
from runtime.scheduler.loop import SchedulerLoop
from runtime.scheduler.scheduler import Request, Scheduler


def test_tick_admits_when_running():
    pool = BlockPool(num_blocks=64, block_size=16)
    sched = Scheduler.for_pools([pool])
    coord = InferenceGpuCoordinator()
    loop = SchedulerLoop(scheduler=sched, coordinator=coord, pools=[pool])
    sched.add_request(Request(request_id="a", prompt_tokens=list(range(10)), max_tokens=32))
    admitted = loop.tick(max_admit=1)
    assert len(admitted) == 1
    assert len(sched.running) == 1


def test_tick_blocked_when_paused():
    pool = BlockPool(num_blocks=64, block_size=16)
    sched = Scheduler.for_pools([pool])
    coord = InferenceGpuCoordinator()
    coord.training_handoff()
    loop = SchedulerLoop(scheduler=sched, coordinator=coord, pools=[pool])
    sched.add_request(Request(request_id="a", prompt_tokens=[1], max_tokens=8))
    assert loop.tick() == []
