from runtime.kv.block_pool import BlockPool
from runtime.scheduler.scheduler import Request, Scheduler


def test_add_and_reserve_blocks():
    pool = BlockPool(num_blocks=64, block_size=16)
    sched = Scheduler.for_pools([pool])
    req = Request(request_id="a", prompt_tokens=[1, 2, 3], max_tokens=32)
    sched.add_request(req)
    assert len(sched.waiting) == 1
    assert req.block_table is not None
