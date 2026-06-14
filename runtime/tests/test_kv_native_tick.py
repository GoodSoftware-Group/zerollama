"""Phase 15 v5: scheduler tick counter (native or python fallback)."""

from runtime.kv.native_tick import (
    record_scheduler_tick,
    reset_scheduler_tick_for_tests,
    scheduler_tick_health,
)
from runtime.scheduler.loop import SchedulerLoop
from runtime.scheduler.scheduler import Request, Scheduler
from runtime.kv.block_pool import BlockPool
from runtime.gpu.mutex import InferenceGpuCoordinator


def test_python_scheduler_tick_fallback(monkeypatch):
    reset_scheduler_tick_for_tests()
    monkeypatch.setattr(
        "runtime.kv.native_tick.native_scheduler_available", lambda: False
    )
    a = record_scheduler_tick()
    b = record_scheduler_tick()
    assert a == 1
    assert b == 2
    h = scheduler_tick_health(b)
    assert h["value"] == 2
    assert h["source"] == "python"


def test_loop_records_scheduler_tick():
    reset_scheduler_tick_for_tests()
    pool = BlockPool(num_blocks=32, block_size=16, device_id=0)
    sched = Scheduler.for_pools([pool])
    loop = SchedulerLoop(
        scheduler=sched,
        coordinator=InferenceGpuCoordinator(),
        pools=[pool],
    )
    req = Request("t", [0], 8)
    sched.add_request(req)
    admitted = loop.tick(max_admit=1)
    assert admitted
    assert loop.last_scheduler_tick is not None
    assert int(loop.last_scheduler_tick) >= 1
    # WHY cleanup: page_bind registers in native C global state; without teardown
    # subsequent tests that count active_binds see stale entries.
    try:
        from runtime.kv._kv_native import page_bind_clear

        for r in admitted:
            page_bind_clear(int(r.kv_slot or 0))
    except ImportError:
        pass
