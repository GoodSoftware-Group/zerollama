from runtime.kv.block_pool import BlockPool
from runtime.kv.multi_pool import MultiDeviceBlockTable


def test_multi_pool_allocates_on_both_devices():
    p0 = BlockPool(num_blocks=32, block_size=16, device_id=0)
    p1 = BlockPool(num_blocks=32, block_size=16, device_id=1)
    table = MultiDeviceBlockTable(request_id="r1", pools=[p0, p1])
    table.ensure_capacity(100)
    assert len(table._tables[0].block_ids) == len(table._tables[1].block_ids)
    assert p0.num_free == p1.num_free
    table.release()
    assert p0.num_free == 32 and p1.num_free == 32
