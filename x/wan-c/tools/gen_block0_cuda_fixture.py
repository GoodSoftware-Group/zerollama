#!/usr/bin/env python3
"""Generate block0 rematch fixture for wan-c CUDA twin.

Writes dumps/block0_cuda_fixture/:
  meta.json, x_in.f32, e0.f32, context.f32,
  py_post_sa.f32, py_post_cross.f32, py_post_ffn.f32

Usage:
  WAN_FORCE_SDPA=1 ~/.zerollama/third_party/wan/venv/bin/python \\
    tools/gen_block0_cuda_fixture.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dumps" / "block0_cuda_fixture"

# Lab geometry matching cuda_block0_real (except Tk may pad to text_len)
T, GT, GH, GW = 32, 2, 4, 4
assert GT * GH * GW == T
TK_SYNTH = 64
D, H, HD, FFN = 1536, 12, 128, 8960
SEED = 42
GEN_T = 1000.0


def main() -> int:
    ckpt = Path(
        os.environ.get(
            "WAN_CKPT",
            Path.home() / ".zerollama/third_party/wan/Wan2.1-T2V-1.3B",
        )
    )
    repo = Path(
        os.environ.get(
            "WAN_REPO",
            Path.home() / ".zerollama/third_party/wan/Wan2.1",
        )
    )
    sys.path.insert(0, str(repo))
    os.environ.setdefault("WAN_FORCE_SDPA", "1")

    import torch
    import wan.modules.attention as wan_attn  # type: ignore
    import wan.modules.model as wan_model  # type: ignore
    from wan.modules.model import WanModel, sinusoidal_embedding_1d  # type: ignore

    def _fa_f32(q, k, v, *args, **kwargs):
        kwargs["dtype"] = torch.float32
        out = wan_attn._sdpa_attention(q, k, v, *args, **kwargs)
        return out.float()

    wan_attn.flash_attention = _fa_f32
    wan_model.flash_attention = _fa_f32

    rng = np.random.default_rng(SEED)
    x_np = rng.standard_normal((T, D), dtype=np.float32) * 0.02
    # Raw T5-ish tokens before text_embedding (text_dim=4096)
    t5_np = rng.standard_normal((TK_SYNTH, 4096), dtype=np.float32) * 0.02

    device = torch.device("cpu")
    print(f"loading WanModel from {ckpt} …", flush=True)
    model = WanModel.from_pretrained(str(ckpt))
    model.eval().to(device=device, dtype=torch.float32)
    # Ensure attention stays f32 on CPU SDPA.
    for p in model.parameters():
        p.data = p.data.float()

    with torch.no_grad():
        x = torch.from_numpy(x_np).unsqueeze(0)
        te = sinusoidal_embedding_1d(
            model.freq_dim, torch.tensor([GEN_T], device=device)
        )
        e = model.time_embedding(te.float())
        e0 = model.time_projection(e).unflatten(1, (6, model.dim))  # [1,6,D]

        ctx = torch.from_numpy(t5_np)
        text_len = int(model.text_len)
        if ctx.size(0) < text_len:
            ctx = torch.cat(
                [ctx, ctx.new_zeros(text_len - ctx.size(0), ctx.size(1))], dim=0
            )
        context = model.text_embedding(ctx.unsqueeze(0))  # [1, text_len, D]
        Tk = context.size(1)

        grid_sizes = torch.tensor([[GT, GH, GW]], dtype=torch.long, device=device)
        blk = model.blocks[0]
        e_chunks = (blk.modulation.to(torch.float32) + e0).chunk(6, dim=1)

        x_sa_in = blk.norm1(x).float() * (1 + e_chunks[1]) + e_chunks[0]
        y = blk.self_attn(
            x_sa_in,
            torch.tensor([T], dtype=torch.long, device=device),
            grid_sizes,
            model.freqs,
        )
        y = y.float()
        x_sa = x + y * e_chunks[2]
        x_cx = x_sa + blk.cross_attn(blk.norm3(x_sa), context, None).float()
        y_ff = blk.ffn(blk.norm2(x_cx).float() * (1 + e_chunks[4]) + e_chunks[3])
        x_ff = x_cx + y_ff.float() * e_chunks[5]

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "T": T,
        "Tk": Tk,
        "D": D,
        "H": H,
        "HD": HD,
        "FFN": FFN,
        "grid": [GT, GH, GW],
        "gen_t": GEN_T,
        "seed": SEED,
        "ckpt": str(ckpt),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    x_np.astype(np.float32).tofile(OUT / "x_in.f32")
    e0[0].cpu().numpy().astype(np.float32).tofile(OUT / "e0.f32")  # [6,D]
    context[0].cpu().numpy().astype(np.float32).tofile(OUT / "context.f32")
    x_sa[0].cpu().numpy().astype(np.float32).tofile(OUT / "py_post_sa.f32")
    x_cx[0].cpu().numpy().astype(np.float32).tofile(OUT / "py_post_cross.f32")
    x_ff[0].cpu().numpy().astype(np.float32).tofile(OUT / "py_post_ffn.f32")
    print(f"wrote {OUT} Tk={Tk}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
