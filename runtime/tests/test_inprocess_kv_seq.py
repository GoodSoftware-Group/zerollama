"""Phase 15 v2: in-process multi-sequence KV helpers (no GPU)."""

import pytest

from runtime.worker.libllama_ctypes import LlamaServerError, _normalize_seq_id


def test_normalize_seq_id_defaults():
    assert _normalize_seq_id(-1, 4) == 0
    assert _normalize_seq_id(2, 4) == 2


def test_normalize_seq_id_out_of_range():
    with pytest.raises(LlamaServerError, match="out of range"):
        _normalize_seq_id(4, 4)
