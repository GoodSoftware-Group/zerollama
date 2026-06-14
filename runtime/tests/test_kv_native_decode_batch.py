"""Tests for native decode batch layout (Phase 15 v8)."""

from __future__ import annotations

import pytest

from runtime.kv.backend import native_available
from runtime.kv.native_decode_batch import (
    iter_prefill_decode_chunks,
    native_decode_batch_available,
)


def test_native_decode_batch_availability_matches_ext():
    assert native_decode_batch_available() == native_available()


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_decode_batch_layout_fields():
    from runtime.kv._kv_native import decode_batch_layout

    layout = decode_batch_layout([1, 2, 3], 0, 10, 1)
    assert layout["token"] == [1, 2, 3]
    assert layout["pos"] == [10, 11, 12]
    assert layout["seq_id"] == [0, 0, 0]
    assert layout["logits"] == [0, 0, 1]


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_decode_prefill_chunks_page_aligned():
    from runtime.kv._kv_native import decode_prefill_chunks

    raw = decode_prefill_chunks(list(range(20)), 16, 0)
    assert len(raw) == 2
    assert len(raw[0][0]) == 16
    assert raw[0][1] == 0
    assert len(raw[1][0]) == 4
    assert raw[1][1] == 16


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_iter_prefill_single_chunk_short_prompt():
    chunks = iter_prefill_decode_chunks([1, 2, 3], block_size=16, pos_start=0)
    assert chunks == [([1, 2, 3], 0)]
