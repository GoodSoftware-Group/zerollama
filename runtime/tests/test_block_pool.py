import pytest

from runtime.kv.block_pool import BlockPool, BlockPoolError, SequenceBlockTable


def test_allocate_and_free():
    pool = BlockPool(num_blocks=4, block_size=16)
    blocks = pool.allocate(2)
    assert len(blocks) == 2
    assert pool.num_free == 2
    pool.free(blocks)
    assert pool.num_free == 4


def test_blocks_for_tokens():
    pool = BlockPool(num_blocks=10, block_size=16)
    assert pool.blocks_for_tokens(1) == 1
    assert pool.blocks_for_tokens(16) == 1
    assert pool.blocks_for_tokens(17) == 2


def test_over_allocate_raises():
    pool = BlockPool(num_blocks=2, block_size=8)
    pool.allocate(2)
    with pytest.raises(BlockPoolError):
        pool.allocate(1)


def test_double_free_raises():
    pool = BlockPool(num_blocks=2, block_size=8)
    blocks = pool.allocate(1)
    pool.free(blocks)
    with pytest.raises(BlockPoolError):
        pool.free(blocks)


def test_sequence_block_table_grows():
    pool = BlockPool(num_blocks=8, block_size=16)
    table = SequenceBlockTable(request_id="r1", pool=pool)
    table.ensure_capacity(40)
    assert len(table.block_ids) == 3
    assert table.num_tokens_capacity >= 40
    table.release()
    assert pool.num_free == 8


def test_dual_pool_devices():
    p0 = BlockPool(num_blocks=100, block_size=16, device_id=0)
    p1 = BlockPool(num_blocks=100, block_size=16, device_id=1)
    assert p0.device_id == 0
    assert p1.device_id == 1
    p0.allocate(10)
    assert p1.num_free == 100
