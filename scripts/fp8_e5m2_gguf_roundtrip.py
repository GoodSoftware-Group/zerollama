#!/usr/bin/env python3
"""CPU roundtrip smoke for GGML_TYPE_FP8_E5M2 via vendor gguf-py.

WHY: E5M2 twin of fp8_e4m3_gguf_roundtrip.py — same pack/layout, different IEEE codec.

Usage:
  PYTHONPATH=vendor/llama-cpp-86d86ed4/gguf-py python3 scripts/fp8_e5m2_gguf_roundtrip.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = ROOT / "vendor" / "llama-cpp-8f114a9b" / "gguf-py"
if GGUF_PY.is_dir():
    sys.path.insert(0, str(GGUF_PY))

from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES  # noqa: E402
from gguf.quants import FP8_E5M2, dequantize, quantize  # noqa: E402


def main() -> int:
    assert GGMLQuantizationType.FP8_E5M2 == 52
    assert GGML_QUANT_SIZES[GGMLQuantizationType.FP8_E5M2] == (32, 34)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 128), dtype=np.float32) * 0.05
    q = quantize(x, GGMLQuantizationType.FP8_E5M2)
    y = dequantize(q, GGMLQuantizationType.FP8_E5M2)
    err = float(np.max(np.abs(x - y)))
    print(f"FP8_E5M2 roundtrip max_abs_err={err:.6f} q_shape={q.shape}")
    if err > 0.02:
        print("FAIL: error too high", file=sys.stderr)
        return 1

    # Native pack: E5M2 bytes + scalar scale (ModelOpt-style)
    scale = 0.02
    qs = FP8_E5M2.fp32_to_e5m2(x / scale)
    packed = FP8_E5M2.pack_e5m2_with_scale(qs, scale)
    assert packed.shape == (8, (128 // 32) * 34)
    dq = FP8_E5M2.dequantize_blocks(packed.reshape(-1, 34)).reshape(x.shape)
    err2 = float(np.max(np.abs(x - dq)))
    print(f"FP8_E5M2 native-pack max_abs_err={err2:.6f}")
    if err2 > 0.02:
        print("FAIL: native pack error too high", file=sys.stderr)
        return 1

    # 128×128 block-scale map → per-GGML-block d (DeepSeek/FP8 HF style)
    rows, cols = 256, 256
    br, bc = 128, 128
    x3 = rng.standard_normal((rows, cols), dtype=np.float32) * 0.05
    scale_map = np.max(np.abs(x3).reshape(rows // br, br, cols // bc, bc), axis=(1, 3)) / 57344.0
    # Reconstruct block-scaled E5M2 like HF: qs = e5m2(x / scale_tile)
    tile = np.repeat(np.repeat(scale_map, br, axis=0), bc, axis=1)
    qs3 = FP8_E5M2.fp32_to_e5m2(x3 / np.maximum(tile, 1e-12))
    block_scales = FP8_E5M2.expand_block_scales_to_ggml(scale_map, (rows, cols), (br, bc))
    packed3 = FP8_E5M2.pack_e5m2_with_scale(qs3, block_scales)
    dq3 = FP8_E5M2.dequantize_blocks(packed3.reshape(-1, 34)).reshape(rows, cols)
    # Reference the F16-rounded scale actually stored in each GGML block.
    stored_scales = block_scales.astype(np.float16).astype(np.float32)
    stored_tile = np.repeat(stored_scales, 32, axis=1)
    ref3 = FP8_E5M2.e5m2_to_fp32(qs3) * stored_tile
    err3 = float(np.max(np.abs(dq3 - ref3)))
    print(f"FP8_E5M2 128x128-block-pack max_abs_err_vs_tile={err3:.6e}")
    if err3 > 1e-3:
        print("FAIL: 128x128 block pack mismatch", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
