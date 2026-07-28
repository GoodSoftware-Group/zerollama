"""SWA reachable-tail store filter (vLLM #48911)."""

from __future__ import annotations

import pytest

from runtime.kv.swa_store_filter import (
    is_store_reachable_swa_chunk,
    swa_reachable_store_mask,
)


@pytest.mark.parametrize(
    (
        "absolute_chunk_index",
        "storable_chunk_count",
        "alignment_chunk_count",
        "sliding_window_chunks",
        "is_eagle_group",
        "expected",
    ),
    [
        (61, 64, 64, 2, False, False),
        (62, 64, 64, 2, False, True),
        (60, 64, 64, 2, True, False),
        (61, 64, 64, 2, True, True),
        (45, 48, 64, 2, False, False),
        (46, 48, 64, 2, False, True),
        (44, 48, 64, 2, True, False),
        (45, 48, 64, 2, True, True),
        (76, 80, 64, 3, False, False),
        (77, 80, 64, 3, False, True),
        (0, 1, None, None, False, True),
        (0, 2, 64, 2, False, True),
    ],
)
def test_is_store_reachable_swa_chunk(
    absolute_chunk_index: int,
    storable_chunk_count: int,
    alignment_chunk_count: int | None,
    sliding_window_chunks: int | None,
    is_eagle_group: bool,
    expected: bool,
):
    assert (
        is_store_reachable_swa_chunk(
            absolute_chunk_index,
            storable_chunk_count,
            alignment_chunk_count,
            sliding_window_chunks,
            is_eagle_group,
        )
        is expected
    )


def test_swa_reachable_store_mask_keeps_tail_only():
    # 8 blocks, window covers last 2 → mask stores only indices 6,7
    mask = swa_reachable_store_mask(
        8, block_size=512, sliding_window=1024, draft_extra=False
    )
    assert mask is not None
    assert mask == [False, False, False, False, False, False, True, True]


def test_swa_reachable_store_mask_dense_when_window_covers_all():
    assert (
        swa_reachable_store_mask(2, block_size=512, sliding_window=2048) is None
    )
