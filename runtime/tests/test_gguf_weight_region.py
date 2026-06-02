import struct
from pathlib import Path

from runtime.gguf_estimate import gguf_file_tensor_region_bytes, gguf_weight_bytes


def _minimal_gguf(path: Path, payload: bytes, extra_pad: int = 0) -> None:
    alignment = 32
    name = b"test"
    ne = len(payload) // 4
    tensor_meta = (
        struct.pack("<Q", len(name))
        + name
        + struct.pack("<I", 1)
        + struct.pack("<Q", ne)
        + struct.pack("<I", 0)
        + struct.pack("<Q", 0)
    )
    header = (
        struct.pack("<I", 0x46554747)
        + struct.pack("<I", 3)
        + struct.pack("<Q", 1)
        + struct.pack("<Q", 0)
        + tensor_meta
    )
    pad = (-len(header)) % alignment
    path.write_bytes(header + b"\0" * pad + payload + b"\0" * extra_pad)


def test_gguf_weight_bytes_at_least_file_region(tmp_path: Path):
    p = tmp_path / "padded.gguf"
    payload = b"\0" * 4096
    _minimal_gguf(p, payload, extra_pad=128)
    region = gguf_file_tensor_region_bytes(p)
    assert region == len(payload) + 128
    assert gguf_weight_bytes(p) == region
