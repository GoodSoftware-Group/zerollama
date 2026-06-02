"""Phase 15: native block pool matches pure-Python allocator."""

from __future__ import annotations

import pytest

from runtime.kv._py_block_pool import BlockPool as PyBlockPool
from runtime.kv._py_block_pool import BlockPoolError as PyBlockPoolError
from runtime.kv._py_block_pool import SequenceBlockTable as PySequenceBlockTable

_kv_native = pytest.importorskip("runtime.kv._kv_native")
NativeBlockPool = _kv_native.BlockPool
NativeBlockPoolError = _kv_native.BlockPoolError


def _mirror_py_table(pool: NativeBlockPool) -> PySequenceBlockTable:
    py_pool = PyBlockPool(
        num_blocks=pool.num_blocks,
        block_size=pool.block_size,
        device_id=pool.device_id,
    )
    return PySequenceBlockTable(request_id="r1", pool=py_pool)


def test_native_allocate_and_free_matches_python():
    py = PyBlockPool(num_blocks=4, block_size=16)
    nat = NativeBlockPool(num_blocks=4, block_size=16)
    py_blocks = py.allocate(2)
    nat_blocks = nat.allocate(2)
    assert sorted(py_blocks) == sorted(nat_blocks)
    assert py.num_free == nat.num_free == 2
    py.free(py_blocks)
    nat.free(nat_blocks)
    assert py.num_free == nat.num_free == 4


def test_native_blocks_for_tokens():
    nat = NativeBlockPool(num_blocks=10, block_size=16)
    assert nat.blocks_for_tokens(1) == 1
    assert nat.blocks_for_tokens(16) == 1
    assert nat.blocks_for_tokens(17) == 2


def test_native_over_allocate_raises():
    nat = NativeBlockPool(num_blocks=2, block_size=8)
    nat.allocate(2)
    with pytest.raises(NativeBlockPoolError):
        nat.allocate(1)


def test_native_sequence_table_grows():
    nat = NativeBlockPool(num_blocks=8, block_size=16)
    table = PySequenceBlockTable(request_id="r1", pool=nat)  # type: ignore[arg-type]
    table.ensure_capacity(40)
    assert len(table.block_ids) == 3
    assert table.num_tokens_capacity >= 40
    table.release()
    assert nat.num_free == 8


def test_backend_factory_native(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_KV_NATIVE", "1")
    from runtime.kv.backend import (
        create_block_pool,
        kv_backend_health,
        reset_kv_backend_cache,
    )

    reset_kv_backend_cache()
    h = kv_backend_health()
    assert h["backend"] == "native"
    assert h["native_available"] is True
    pool = create_block_pool(num_blocks=4, block_size=8, device_id=0)
    assert isinstance(pool, NativeBlockPool)
