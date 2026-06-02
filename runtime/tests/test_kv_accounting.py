"""Phase 15 v1: KV scheduler accounting and num_ctx block reservation."""

from runtime.kv.accounting import kv_scheduler_snapshot
from runtime.kv.block_pool import BlockPool
from runtime.kv.slots import SlotAllocator
from runtime.scheduler.loop import SchedulerLoop
from runtime.scheduler.scheduler import Request, Scheduler
from runtime.gpu.mutex import InferenceGpuCoordinator


def test_kv_scheduler_snapshot_counts_blocks():
    pool = BlockPool(num_blocks=32, block_size=16, device_id=0)
    sched = Scheduler.for_pools([pool])
    req = Request(
        request_id="a",
        prompt_tokens=[1, 2, 3],
        max_tokens=10,
    )
    sched.add_request(req)
    req.block_table.ensure_capacity(40)
    snap = kv_scheduler_snapshot(sched, [pool], block_size=16)
    assert snap["blocks_reserved"] == 3
    assert snap["waiting"] == 1
    row = snap["requests"][0]
    assert row["blocks"] == 3
    assert row["tokens_capacity"] >= 40
    assert row["block_ids"] == req.block_table._tables[0].block_ids  # type: ignore[union-attr]


def test_slot_allocator_exhaustion():
    slots = SlotAllocator(num_slots=2)
    assert slots.acquire() == 0
    assert slots.acquire() == 1
    assert slots.acquire() is None
    slots.release(0)
    assert slots.acquire() == 0


def test_loop_reserves_num_ctx_blocks():
    block_size = 16
    num_ctx = 512
    need_blocks = (num_ctx + block_size - 1) // block_size
    pool = BlockPool(num_blocks=need_blocks + 4, block_size=block_size, device_id=0)
    sched = Scheduler.for_pools([pool])
    coord = InferenceGpuCoordinator()
    loop = SchedulerLoop(
        scheduler=sched,
        coordinator=coord,
        pools=[pool],
        parallel_slots=2,
        assign_llama_slots=True,
    )
    req = Request(
        request_id="ctx",
        prompt_tokens=[1],
        max_tokens=8,
        num_ctx=num_ctx,
    )
    sched.add_request(req)
    admitted = loop.tick(max_admit=1)
    assert len(admitted) == 1
    assert admitted[0].kv_slot == 0
    blocks = len(admitted[0].block_table._tables[0].block_ids)  # type: ignore[union-attr]
    assert blocks == need_blocks
    loop.complete(admitted[0])
    assert admitted[0].kv_slot is None
    assert loop._slots.in_use_count() == 0
