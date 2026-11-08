#!/usr/bin/env python3
"""Microbench: native MPS SDPA vs mtlgemm sparse_attention_fwd.

Does NOT use the vendored bench's MPS→CPU→MPS SDPA bounce. That bounce is
what made fused look 5–15× faster. Our generate path uses native MPS SDPA.

Shapes match microsoft/TRELLIS.2-4B checkpoints (not guesses):
  - structure DiT (ss_flow_img_dit_*): resolution 16 → T=4096, H=12,
    head_dim=128 (model_channels 1536 / 12), attn_mode=full, N=1
  - SLat flow (slat_flow_*): H=12, head_dim=128; T is sparse occupied
    voxels — use a mid-run stand-in (2048). Head dim is the dispatch gate.

Config dtype is bfloat16; we also time fp16 (Metal tiled path). If bf16 is
unsupported on a kernel, that row records the failure and does not count as a win.

Run inside the trellis-mac venv:
  source ~/.zerollama/third_party/trellis-mac/env-mac.sh
  python scripts/mesh/bench_mps_sparse_attn.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("FLEX_GEMM_QUIET", "1")

import torch
import torch.nn.functional as F

# Checkpoint math: model_channels=1536, num_heads=12 → head_dim=128.
# Structure resolution=16 → 16**3 tokens. See TRELLIS.2 ckpt JSON under hf-cache.
STRUCTURE_T = 16**3  # 4096
STRUCTURE_H = 12
STRUCTURE_C = 1536 // 12  # 128
SLAT_T = 2048  # sparse occupied stand-in; head_dim/H match ckpt
SLAT_H = 12
SLAT_C = 1536 // 12  # 128


def _bench(label: str, fn, *, warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000.0
    print(f"  {label:48s} {ms:8.3f} ms/call")
    return ms


def _native_mps_sdpa(q, k, v, seqlens):
    """Native MPS SDPA on a single packed sequence (N=1 → no padding waste).

    Production full_attn.py pads then SDPA; with N=1 and ql==max_q the math
    is the same as this reshape (pad copy cost omitted — not the fused win).
    """
    ql = int(seqlens[0])
    q_t = q[:ql].unsqueeze(0).permute(0, 2, 1, 3)  # [1, H, T, C]
    k_t = k[:ql].unsqueeze(0).permute(0, 2, 1, 3)
    v_t = v[:ql].unsqueeze(0).permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t)
    return out.permute(0, 2, 1, 3).reshape(ql, q.shape[1], v.shape[2])


def _rand_mps(shape, dtype):
    # Some torch builds have broken MPS randn; build on CPU then transfer.
    return (torch.randn(*shape, dtype=dtype) * 0.3).to("mps")


def main() -> int:
    if not torch.backends.mps.is_available():
        print("error: MPS not available", file=sys.stderr)
        return 2

    import flex_gemm  # noqa: F401
    from flex_gemm.kernels.metal import sparse_attention_fwd

    configs = [
        # label, seqlen, H, C, dtype
        ("structure_dit_fp16_hd128_T4096", STRUCTURE_T, STRUCTURE_H, STRUCTURE_C, torch.float16),
        ("structure_dit_bf16_hd128_T4096", STRUCTURE_T, STRUCTURE_H, STRUCTURE_C, torch.bfloat16),
        ("slat_flow_fp16_hd128_T2048", SLAT_T, SLAT_H, SLAT_C, torch.float16),
        ("slat_flow_bf16_hd128_T2048", SLAT_T, SLAT_H, SLAT_C, torch.bfloat16),
    ]

    results = []
    print("=" * 80)
    print("Native MPS SDPA vs fused sparse_attention_fwd")
    print(f"  structure: T={STRUCTURE_T} H={STRUCTURE_H} C={STRUCTURE_C} (16^3, ckpt)")
    print(f"  slat:      T={SLAT_T} H={SLAT_H} C={SLAT_C} (sparse stand-in, ckpt H/C)")
    print("  N=1, no CPU-bounce SDPA baseline")
    print("=" * 80)

    for label, seqlen, H, C, dtype in configs:
        seqlens = [seqlen]
        T = seqlen
        assert C == 128 and H == 12, "bench must hit TRELLIS.2 head_dim 128 / 12 heads"
        try:
            q = _rand_mps((T, H, C), dtype)
            k = _rand_mps((T, H, C), dtype)
            v = _rand_mps((T, H, C), dtype)
            csq = torch.tensor([0, seqlen], dtype=torch.int32, device="mps")
            cskv = csq.clone()
        except Exception as e:
            print(f"\n{label}: alloc failed: {e}")
            results.append(
                {
                    "label": label,
                    "seqlen": seqlen,
                    "heads": H,
                    "head_dim": C,
                    "dtype": str(dtype).replace("torch.", ""),
                    "fused_ms": None,
                    "sdpa_ms": None,
                    "speedup_fused_over_sdpa": 0.0,
                    "parity_ok": False,
                    "parity_max_err": None,
                    "error": f"alloc: {e}",
                }
            )
            continue

        scale = 1.0 / math.sqrt(C)

        def fused():
            return sparse_attention_fwd(q, k, v, csq, cskv, seqlen, seqlen, scale)

        def sdpa():
            return _native_mps_sdpa(q, k, v, seqlens)

        print(f"\n{label}: T={T} H={H} C={C} dtype={dtype}")
        parity_ok = False
        err = None
        try:
            a = fused().detach().cpu().float()
            b = sdpa().detach().cpu().float()
            err = float((a - b).abs().max().item())
            tol = 0.35 if dtype in (torch.float16, torch.bfloat16) else 0.05
            parity_ok = err < tol
            print(f"  parity max_err={err:.4e}  {'OK' if parity_ok else 'FAIL'} (tol={tol})")
        except Exception as e:
            print(f"  parity/run failed: {e}")

        fused_ms = float("inf")
        sdpa_ms = float("inf")
        try:
            fused_ms = _bench("fused sparse_attention_fwd", fused)
        except Exception as e:
            print(f"  fused failed: {e}")
        try:
            sdpa_ms = _bench("native MPS SDPA", sdpa)
        except Exception as e:
            print(f"  sdpa failed: {e}")

        speedup = (sdpa_ms / fused_ms) if fused_ms > 0 and fused_ms < float("inf") else 0.0
        print(f"  speedup fused/SDPA: {speedup:.2f}x  ({'WIN' if speedup >= 1.15 else 'LOSE/TIE'})")
        results.append(
            {
                "label": label,
                "seqlen": seqlen,
                "heads": H,
                "head_dim": C,
                "dtype": str(dtype).replace("torch.", ""),
                "fused_ms": None if fused_ms == float("inf") else round(fused_ms, 3),
                "sdpa_ms": None if sdpa_ms == float("inf") else round(sdpa_ms, 3),
                "speedup_fused_over_sdpa": round(speedup, 3),
                "parity_ok": parity_ok,
                "parity_max_err": err,
            }
        )

    def _speedup(label: str) -> float:
        for r in results:
            if r["label"] == label:
                return float(r.get("speedup_fused_over_sdpa") or 0)
        return 0.0

    # Decision keys: fp16 structure + SLat (Metal tiled path). bf16 is informational.
    structure = _speedup("structure_dit_fp16_hd128_T4096")
    slat = _speedup("slat_flow_fp16_hd128_T2048")
    fused_wins = structure >= 1.15 and slat >= 1.15
    decision = {
        "winner": "flex_gemm_sparse_attn" if fused_wins else "sdpa",
        "reason": (
            "fused clearly faster on structure+SLat fp16 hd128"
            if fused_wins
            else "fused not clearly faster than native MPS SDPA on N=1 TRELLIS.2 shapes; keep ATTN_BACKEND=sdpa"
        ),
        "threshold_speedup": 1.15,
        "checkpoint_shapes": {
            "structure": {"T": STRUCTURE_T, "H": STRUCTURE_H, "head_dim": STRUCTURE_C, "source": "16^3, 1536/12"},
            "slat": {"T": SLAT_T, "H": SLAT_H, "head_dim": SLAT_C, "source": "sparse stand-in; 1536/12"},
        },
        "structure_fp16_speedup": structure,
        "slat_fp16_hd128_speedup": slat,
        "structure_bf16_speedup": _speedup("structure_dit_bf16_hd128_T4096"),
        "slat_bf16_speedup": _speedup("slat_flow_bf16_hd128_T2048"),
        "results": results,
    }

    out_dir = Path(
        os.environ.get(
            "TRELLIS_ROOT",
            Path.home() / ".zerollama/third_party/trellis-mac",
        )
    ) / "lab" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "attn_microbench.json"
    out_path.write_text(json.dumps(decision, indent=2) + "\n")

    print("\n" + "=" * 80)
    print(f"DECISION: {decision['winner']}")
    print(decision["reason"])
    print(f"wrote {out_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
