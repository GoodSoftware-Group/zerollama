import struct
from pathlib import Path

from runtime.gguf_estimate import (
    ggml_tensor_storage_bytes,
    gguf_tensor_infos,
)


def _tensor_meta(name: bytes, ne: int, type_id: int = 2) -> bytes:
    return (
        struct.pack("<Q", len(name))
        + name
        + struct.pack("<I", 1)
        + struct.pack("<Q", ne)
        + struct.pack("<I", type_id)
        + struct.pack("<Q", 0)
    )


def _one_tensor_gguf(path: Path, ne: int, type_id: int) -> None:
    alignment = 32
    meta = _tensor_meta(b"blk.0.weight", ne, type_id)
    header = (
        struct.pack("<I", 0x46554747)
        + struct.pack("<I", 3)
        + struct.pack("<Q", 1)
        + struct.pack("<Q", 0)
        + meta
    )
    pad = (-len(header)) % alignment
    path.write_bytes(header + b"\0" * pad + b"\0" * 64)


def test_q4_0_block_layout_bytes():
    ne = 4096
    block = ggml_tensor_storage_bytes(ne, 2)  # Q4_0
    assert block == (ne // 32) * 18
    assert block < ne * 2


def test_q4_0_tensor_infos_uses_block_layout(tmp_path: Path):
    p = tmp_path / "q4.gguf"
    ne = 1024
    _one_tensor_gguf(p, ne, type_id=2)
    infos = gguf_tensor_infos(p)
    assert infos is not None and len(infos) == 1
    assert infos[0].nbytes == (ne // 32) * 18


def test_iq4_nl_block_layout_bytes():
    ne = 4096
    assert ggml_tensor_storage_bytes(ne, 20) == (ne // 32) * 18


def test_iq2_xxs_block_layout_bytes():
    ne = 8192
    assert ggml_tensor_storage_bytes(ne, 16) == (ne // 256) * 66


def test_iq2_xs_block_layout_bytes():
    ne = 8192
    assert ggml_tensor_storage_bytes(ne, 17) == (ne // 256) * 74


def test_iq1_m_block_layout_bytes():
    ne = 512
    assert ggml_tensor_storage_bytes(ne, 29) == (ne // 256) * 56
