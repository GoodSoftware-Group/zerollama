from __future__ import annotations

import struct
from pathlib import Path

from runtime.gguf_estimate import (
    GGUF_MAGIC,
    GGUF_TYPE_ARRAY,
    GGUF_TYPE_UINT32,
    GgufArchHints,
    GgufTensorInfo,
    estimate_kv_cache_bytes,
    ggml_type_kv_bytes,
    gguf_arch_hints,
    gguf_head_dim,
    gguf_model_hints,
    kv_bytes_per_slot,
    kv_cache_type_summary,
    _infer_head_dims_from_tensor_infos,
)


def _write_minimal_gguf(path: Path, *, block_count: int = 40, extra_kv: list | None = None) -> None:
    kv = [
        (b"llama.block_count", GGUF_TYPE_UINT32, struct.pack("<I", block_count)),
        (b"llama.context_length", GGUF_TYPE_UINT32, struct.pack("<I", 4096)),
    ]
    if extra_kv:
        kv.extend(extra_kv)
    with path.open("wb") as f:
        f.write(struct.pack("<II", GGUF_MAGIC, 2))
        f.write(struct.pack("<QQ", 0, len(kv)))
        for key, vtype, payload in kv:
            f.write(struct.pack("<Q", len(key)))
            f.write(key)
            f.write(struct.pack("<I", vtype))
            f.write(payload)
        f.write(struct.pack("<Q", 0))


def _uint32_array_payload(values: list[int]) -> bytes:
    return (
        struct.pack("<IQ", GGUF_TYPE_UINT32, len(values))
        + b"".join(struct.pack("<I", v) for v in values)
    )


def test_gguf_model_hints_reads_block_count(tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    _write_minimal_gguf(gguf, block_count=40)
    hints = gguf_model_hints(gguf)
    assert hints.get("block_count") == 40
    assert hints.get("context_length") == 4096


def test_gguf_arch_hints_embedding_fallback_head_dim_source(tmp_path: Path):
    gguf = tmp_path / "emb.gguf"
    extra = [
        (b"llama.embedding_length", GGUF_TYPE_UINT32, struct.pack("<I", 4096)),
        (b"llama.attention.head_count", GGUF_TYPE_UINT32, struct.pack("<I", 32)),
        (b"llama.attention.head_count_kv", GGUF_TYPE_UINT32, struct.pack("<I", 8)),
    ]
    _write_minimal_gguf(gguf, block_count=4, extra_kv=extra)
    arch = gguf_arch_hints(gguf)
    assert arch.head_dim_source == "embedding_fallback"
    assert gguf_head_dim(arch) == 128


def test_gguf_model_hints_reads_attention_dims(tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    extra = [
        (b"llama.embedding_length", GGUF_TYPE_UINT32, struct.pack("<I", 4096)),
        (b"llama.attention.head_count", GGUF_TYPE_UINT32, struct.pack("<I", 32)),
        (b"llama.attention.head_count_kv", GGUF_TYPE_UINT32, struct.pack("<I", 8)),
        (b"llama.attention.key_length", GGUF_TYPE_UINT32, struct.pack("<I", 128)),
    ]
    _write_minimal_gguf(gguf, block_count=32, extra_kv=extra)
    hints = gguf_model_hints(gguf)
    assert hints["head_count"] == 32
    assert hints["head_count_kv"] == 8
    assert hints["key_length"] == 128
    assert gguf_head_dim(hints) == 128
    arch = gguf_arch_hints(gguf)
    assert arch.head_dim_source == "metadata"
    assert kv_cache_type_summary(arch)["head_dim_source"] == "metadata"


def test_estimate_kv_cache_bytes_gqa():
    arch = GgufArchHints(
        scalar={"block_count": 32, "head_count_kv": 8, "key_length": 128}
    )
    assert estimate_kv_cache_bytes(arch, 4096, elem_bytes=2) == 536_870_912


def test_estimate_kv_cache_bytes_partial_ngl():
    arch = GgufArchHints(
        scalar={"block_count": 32, "head_count_kv": 8, "key_length": 128}
    )
    full = estimate_kv_cache_bytes(arch, 4096, elem_bytes=2)
    half = estimate_kv_cache_bytes(arch, 4096, elem_bytes=2, n_gpu_layers=16)
    assert half == full // 2
    assert estimate_kv_cache_bytes(arch, 4096, n_gpu_layers=0) == 0


def test_infer_head_dims_ignores_non_attn_k_names():
    arch = GgufArchHints(
        scalar={
            "block_count": 4,
            "head_count_kv": 8,
            "embedding_length": 4096,
        }
    )
    infos = [
        GgufTensorInfo(name="blk.0.ffn_gate.weight", nbytes=1, ne=(11008, 4096)),
        GgufTensorInfo(name="blk.0.attn_k.weight", nbytes=1, ne=(1024, 4096)),
    ]
    assert _infer_head_dims_from_tensor_infos(infos, arch) == (128, 128)


def test_infer_head_dims_from_attn_tensor_shapes():
    arch = GgufArchHints(
        scalar={
            "block_count": 32,
            "head_count": 32,
            "head_count_kv": 8,
            "embedding_length": 4096,
        }
    )
    infos = [
        GgufTensorInfo(
            name="blk.0.attn_k.weight", nbytes=1, ne=(1024, 4096)
        ),
        GgufTensorInfo(
            name="blk.0.attn_v.weight", nbytes=1, ne=(1024, 4096)
        ),
    ]
    assert _infer_head_dims_from_tensor_infos(infos, arch) == (128, 128)


def test_estimate_kv_cache_bytes_derives_head_dim():
    arch = GgufArchHints(
        scalar={
            "block_count": 32,
            "head_count": 32,
            "head_count_kv": 8,
            "embedding_length": 4096,
        }
    )
    assert gguf_head_dim(arch) == 128
    assert estimate_kv_cache_bytes(arch, 4096, elem_bytes=2) == 536_870_912


def test_sliding_window_caps_kv_context():
    arch = GgufArchHints(
        scalar={
            "block_count": 32,
            "head_count_kv": 8,
            "key_length": 128,
            "sliding_window": 2048,
        }
    )
    capped = estimate_kv_cache_bytes(arch, 8192, elem_bytes=2)
    assert capped == 32 * 2048 * 8 * 128 * 4
    full = GgufArchHints(
        scalar={
            "block_count": 32,
            "head_count_kv": 8,
            "key_length": 128,
        }
    )
    assert estimate_kv_cache_bytes(full, 8192, elem_bytes=2) > capped


def test_per_layer_sliding_window_hybrid(tmp_path: Path):
    """Hybrid SWA: some layers full ctx, some capped (sum per layer, not global cap)."""
    gguf = tmp_path / "hybrid.gguf"
    extra = [
        (
            b"llama.attention.sliding_window",
            GGUF_TYPE_ARRAY,
            _uint32_array_payload([0, 2048, 0, 2048]),
        ),
        (b"llama.attention.head_count_kv", GGUF_TYPE_UINT32, struct.pack("<I", 8)),
        (b"llama.attention.key_length", GGUF_TYPE_UINT32, struct.pack("<I", 128)),
    ]
    _write_minimal_gguf(gguf, block_count=4, extra_kv=extra)
    arch = gguf_arch_hints(gguf)
    assert arch.sliding_window_per_layer == (0, 2048, 0, 2048)
    hybrid = estimate_kv_cache_bytes(arch, 8192, elem_bytes=2)
    all_swa = GgufArchHints(
        scalar={
            "block_count": 4,
            "head_count_kv": 8,
            "key_length": 128,
            "sliding_window": 2048,
        }
    )
    assert hybrid > estimate_kv_cache_bytes(all_swa, 8192, elem_bytes=2)
    full = GgufArchHints(
        scalar={"block_count": 4, "head_count_kv": 8, "key_length": 128}
    )
    assert estimate_kv_cache_bytes(full, 8192, elem_bytes=2) > hybrid


def test_ggml_type_kv_bytes_conservative():
    assert ggml_type_kv_bytes(8) == 2
    assert ggml_type_kv_bytes(999) == 2


def test_iq_kv_uses_f16_floor_over_block_layout():
    assert ggml_type_kv_bytes(16) == 2  # IQ2_XXS: max(1, F16)
    assert ggml_type_kv_bytes(34) == 2  # TQ1_0


def test_kv_block_layout_off_restores_flat_table(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_BLOCK_LAYOUT", "0")
    assert ggml_type_kv_bytes(16) == 2


def test_legacy_blck1_type_uses_flat_table_floor():
    assert ggml_type_kv_bytes(24) == 2


def test_kv_unknown_type_uses_fp16_not_one_byte_quant():
    arch = GgufArchHints(
        scalar={
            "block_count": 32,
            "head_count_kv": 8,
            "key_length": 128,
            "key_type": 2,
            "value_type": 2,
        }
    )
    assert kv_bytes_per_slot(arch) == 4


def test_per_layer_head_count_kv(tmp_path: Path):
    gguf = tmp_path / "gqa.gguf"
    extra = [
        (
            b"llama.attention.head_count_kv",
            GGUF_TYPE_ARRAY,
            _uint32_array_payload([8, 8, 4, 4]),
        ),
        (b"llama.attention.key_length", GGUF_TYPE_UINT32, struct.pack("<I", 128)),
    ]
    _write_minimal_gguf(gguf, block_count=4, extra_kv=extra)
    arch = gguf_arch_hints(gguf)
    assert arch.head_count_kv_per_layer == (8, 8, 4, 4)
    uniform = GgufArchHints(
        scalar={"block_count": 4, "head_count_kv": 8, "key_length": 128}
    )
    varied = estimate_kv_cache_bytes(arch, 4096, elem_bytes=2)
    flat = estimate_kv_cache_bytes(uniform, 4096, elem_bytes=2)
    assert varied is not None and flat is not None
    assert varied < flat


def test_per_layer_head_count_kv_partial_array(tmp_path: Path):
    """Use per-layer values when array is shorter than block_count."""
    arch = GgufArchHints(
        scalar={"block_count": 4, "head_count_kv": 8, "key_length": 128},
        head_count_kv_per_layer=(4, 4),
    )
    full = estimate_kv_cache_bytes(arch, 4096, elem_bytes=2)
    assert full is not None
    assert full < estimate_kv_cache_bytes(
        GgufArchHints(
            scalar={"block_count": 4, "head_count_kv": 8, "key_length": 128}
        ),
        4096,
        elem_bytes=2,
    )


def test_gguf_arch_hints_cached(tmp_path: Path):
    gguf = tmp_path / "c.gguf"
    _write_minimal_gguf(gguf, block_count=7)
    a = gguf_arch_hints(gguf)
    b = gguf_arch_hints(gguf)
    assert a is b
    assert a.scalar["block_count"] == 7


def test_gguf_model_hints_sliding_window_scalar(tmp_path: Path):
    gguf = tmp_path / "swa.gguf"
    extra = [(b"llama.attention.sliding_window", GGUF_TYPE_UINT32, struct.pack("<I", 4096))]
    _write_minimal_gguf(gguf, block_count=32, extra_kv=extra)
    assert gguf_model_hints(gguf).get("sliding_window") == 4096
