import struct
from pathlib import Path

from runtime.gguf_estimate import gguf_weight_bytes


def _minimal_gguf(path: Path, payload: bytes) -> None:
    """Write a tiny valid GGUF v3 with one tensor and payload bytes."""
    alignment = 32
    name = b"test"
    kv_blob = b""
    n_kv = 0
    n_tensors = 1
    ne = len(payload) // 4  # F32 elements
    tensor_meta = (
        struct.pack("<Q", len(name))
        + name
        + struct.pack("<I", 1)  # ndim
        + struct.pack("<Q", ne)
        + struct.pack("<I", 0)  # f32 kind
        + struct.pack("<Q", 0)  # offset
    )
    header = (
        struct.pack("<I", 0x46554747)
        + struct.pack("<I", 3)
        + struct.pack("<Q", n_tensors)
        + struct.pack("<Q", n_kv)
        + kv_blob
        + tensor_meta
    )
    pad = (-len(header)) % alignment
    header += b"\0" * pad
    path.write_bytes(header + payload)


def test_gguf_weight_bytes(tmp_path: Path):
    p = tmp_path / "tiny.gguf"
    payload = b"\0" * 4096
    _minimal_gguf(p, payload)
    got = gguf_weight_bytes(p)
    assert got == len(payload)


def test_gguf_weight_bytes_non_gguf(tmp_path: Path):
    p = tmp_path / "not.gguf"
    p.write_bytes(b"not gguf")
    assert gguf_weight_bytes(p) is None
