"""Phase 15 v7: logical KV forward plan export."""

from __future__ import annotations

import pytest

from runtime.gpu.mutex import InferenceGpuCoordinator
from runtime.kv.backend import native_available
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
    assert "decode_prefill" in plan
    assert plan["decode_prefill"]["n_prefill_batches"] >= 1
    loop.complete(r)


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_kv_forward_plan_includes_native_page_table_after_admit():
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
        request_id="fp-native",
        prompt_tokens=[0] * 20,
        max_tokens=8,
        num_ctx=64,
    )
    sched.add_request(req)
    admitted = loop.tick(max_admit=1)
    assert len(admitted) == 1
    r = admitted[0]
    plan = kv_forward_plan(r, block_size=16)
    assert plan.get("native_page_table")
    assert plan["page_table_native_parity"] is True
    assert plan["native_page_table"] == plan["pages"]
    loop.complete(r)


def test_kv_forward_plans_empty_when_idle():
    pool = BlockPool(num_blocks=8, block_size=16, device_id=0)
    sched = Scheduler.for_pools([pool])
    assert kv_forward_plans_for_requests(list(sched.waiting), block_size=16) == []


def test_kv_continuous_batch_forward_plan_would_batch_two_running():
    from unittest.mock import patch

    from runtime.kv.forward_plan import kv_continuous_batch_forward_plan
    from runtime.scheduler.scheduler import Request, RequestState

    reqs = [
        Request(
            request_id="a",
            prompt_tokens=[0] * 10,
            max_tokens=8,
            kv_slot=0,
            state=RequestState.DECODE,
        ),
        Request(
            request_id="b",
            prompt_tokens=[0] * 12,
            max_tokens=8,
            kv_slot=1,
            state=RequestState.DECODE,
        ),
    ]
    with patch(
        "runtime.kv.native_decode_loop.native_batch_decode_available",
        return_value=True,
    ):
        plan = kv_continuous_batch_forward_plan(
            reqs,
            block_size=16,
            current_pos_by_request_id={"a": 10, "b": 12},
            parallel_slots=4,
        )
    assert plan["n_decode_candidates"] == 2
    assert plan["would_batch"] is True
    assert plan["step_plan"]["n_batch_rows"] == 2


def test_kv_continuous_batch_forward_plan_skips_prefill():
    from runtime.kv.forward_plan import kv_continuous_batch_forward_plan
    from runtime.scheduler.scheduler import Request, RequestState

    req = Request(
        request_id="a",
        prompt_tokens=[0] * 10,
        max_tokens=8,
        kv_slot=0,
        state=RequestState.PREFILL,
    )
    plan = kv_continuous_batch_forward_plan(
        [req],
        block_size=16,
        current_pos_by_request_id={"a": 5},
        parallel_slots=4,
    )
    assert plan["n_decode_candidates"] == 0
    assert plan["would_batch"] is False

