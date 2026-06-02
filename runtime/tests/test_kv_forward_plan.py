"""Phase 15 v7: logical KV forward plan export."""

from __future__ import annotations

from runtime.gpu.mutex import InferenceGpuCoordinator
from runtime.kv.block_pool import BlockPool
from runtime.kv.forward_plan import kv_forward_plan, kv_forward_plans_for_requests
from runtime.scheduler.loop import SchedulerLoop
from runtime.scheduler.scheduler import Request, Scheduler


def test_kv_forward_plan_pages_and_reserve():
    pool = BlockPool(num_blocks=32, block_size=16, device_id=0)
    sched = Scheduler.for_pools([pool])
    loop = SchedulerLoop(
        scheduler=sched,
        coordinator=InferenceGpuCoordinator(),
        pools=[pool],
        parallel_slots=2,
        assign_llama_slots=True,
    )
    req = Request(
        request_id="fp1",
        prompt_tokens=[0] * 20,
        max_tokens=8,
        num_ctx=64,
    )
    sched.add_request(req)
    admitted = loop.tick(max_admit=1)
    assert len(admitted) == 1
    r = admitted[0]
    plan = kv_forward_plan(r, block_size=16)
    assert plan["request_id"] == "fp1"
    assert plan["kv_slot"] is not None
    assert plan["block_size"] == 16
    assert len(plan["pages"]) >= 1
    assert plan["pages"][0]["token_start"] == 0
    assert plan["pages"][0]["token_end"] == 15
    assert plan["pa_tokens_reserved"] >= plan["tokens_to_reserve"]
    loop.complete(r)


def test_kv_forward_plans_empty_when_idle():
    pool = BlockPool(num_blocks=8, block_size=16, device_id=0)
    sched = Scheduler.for_pools([pool])
    assert kv_forward_plans_for_requests(list(sched.waiting), block_size=16) == []
