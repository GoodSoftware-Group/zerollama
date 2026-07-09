"""GGUF parser resilience (truncated / corrupt files)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from runtime.gguf_estimate import (
    GGUF_MAGIC,
    _read_u64,
    gguf_file_tensor_region_bytes,
    gguf_weight_bytes,
)


def test_read_u64_short_read_raises():
    f = io.BytesIO(b"\x01\x02")
    with pytest.raises(EOFError):
        _read_u64(f)


def test_gguf_file_tensor_region_bytes_truncated(tmp_path: Path):
    path = tmp_path / "trunc.gguf"
    path.write_bytes(b"")
    assert gguf_file_tensor_region_bytes(path) is None


def test_gguf_weight_bytes_invalid_magic(tmp_path: Path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"NOPE" + b"\x00" * 12)
    assert gguf_weight_bytes(path) is None


def test_gguf_file_tensor_region_bytes_minimal_header(tmp_path: Path):
    buf = io.BytesIO()
    buf.write(GGUF_MAGIC.to_bytes(4, "little"))
    buf.write((2).to_bytes(4, "little"))  # version
    buf.write((0).to_bytes(8, "little"))  # n_tensors
    buf.write((0).to_bytes(8, "little"))  # n_kv
    path = tmp_path / "empty.gguf"
    path.write_bytes(buf.getvalue())
    assert gguf_file_tensor_region_bytes(path) is None
