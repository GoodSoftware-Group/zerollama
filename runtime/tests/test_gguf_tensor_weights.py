import struct
from pathlib import Path

from runtime.gguf_estimate import estimate_gpu_weight_bytes, gguf_tensor_infos


def _tensor_meta(name: bytes, ne: int, type_id: int = 0) -> bytes:
    return (
        struct.pack("<Q", len(name))
        + name
        + struct.pack("<I", 1)
        + struct.pack("<Q", ne)
        + struct.pack("<I", type_id)
        + struct.pack("<Q", 0)
    )


def _layered_gguf(path: Path, layers: list[tuple[str, int]]) -> None:
    """GGUF v3 with named tensors (F32, ne elements each)."""
    alignment = 32
    kv_blob = b""
    meta = b"".join(
        _tensor_meta(name.encode(), ne) for name, ne in layers
    )
    header = (
        struct.pack("<I", 0x46554747)
        + struct.pack("<I", 3)
        + struct.pack("<Q", len(layers))
        + struct.pack("<Q", 0)
        + kv_blob
        + meta
    )
    pad = (-len(header)) % alignment
    payload = b"\0" * sum(ne * 4 for _, ne in layers)
    path.write_bytes(header + b"\0" * pad + payload)


def test_estimate_gpu_weight_partial_layers(tmp_path: Path):
    p = tmp_path / "m.gguf"
    _layered_gguf(
        p,
        [
            ("token_embd.weight", 100),
            ("blk.0.ffn_up.weight", 200),
            ("blk.1.ffn_up.weight", 200),
            ("output.weight", 100),
        ],
    )
    tensors = gguf_tensor_infos(p)
    assert tensors is not None and len(tensors) == 4
    full = estimate_gpu_weight_bytes(p, -1)
    half = estimate_gpu_weight_bytes(p, 1)
    none = estimate_gpu_weight_bytes(p, 0)
    assert full == 100 * 4 + 200 * 4 + 200 * 4 + 100 * 4
    # layer 0 + globals (embd, output)
    assert half == 100 * 4 + 200 * 4 + 100 * 4
    assert none == 0
