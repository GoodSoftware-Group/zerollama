#!/usr/bin/env python3
"""Compare wan-c block-0 stage dumps to Wan Python.

Expects WAN_DUMP_DIR artifacts in wan_c_dir:
  noise.f32, t5_emb.f32, meta.json, patch_tok.f32 (optional),
  b0_post_sa.f32, b0_post_cross.f32, b0_post_ffn.f32

Usage:
  WAN_FORCE_SDPA=1 ~/.zerollama/third_party/wan/venv/bin/python \\
    tools/compare_block0.py dumps/parity_c
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else float("nan")


def stats(name: str, c: np.ndarray, p: np.ndarray) -> dict:
    d = c.astype(np.float64) - p.astype(np.float64)
    out = {
        "name": name,
        "cosine": round(cos(c, p), 6),
        "mse": round(float(np.mean(d * d)), 6),
        "mae": round(float(np.mean(np.abs(d))), 6),
        "max_abs": round(float(np.max(np.abs(d))), 6),
    }
    print(
        f"{name}: cosine={out['cosine']} mse={out['mse']} "
        f"mae={out['mae']} max_abs={out['max_abs']}"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wan_c_dir", type=Path)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=None)
    args = ap.parse_args()

    meta = json.loads((args.wan_c_dir / "meta.json").read_text())
    shape = tuple(meta["latent_shape"])
    t = float(meta.get("gen_t", 1000.0))
    noise = np.fromfile(args.wan_c_dir / "noise.f32", dtype=np.float32).reshape(
        shape
    )
    t5 = np.fromfile(args.wan_c_dir / "t5_emb.f32", dtype=np.float32)
    t5_shape = tuple(meta["t5_shape"])
    t5 = t5.reshape(t5_shape)

    ckpt = args.ckpt or Path(
        os.environ.get(
            "WAN_CKPT",
            Path.home() / ".zerollama/third_party/wan/Wan2.1-T2V-1.3B",
        )
    )
    repo = args.repo or Path(
        os.environ.get(
            "WAN_REPO",
            Path.home() / ".zerollama/third_party/wan/Wan2.1",
        )
    )
    sys.path.insert(0, str(repo))
    if not os.environ.get("WAN_FORCE_SDPA"):
        os.environ["WAN_FORCE_SDPA"] = "1"

    import math
    import torch
    from wan.modules.model import WanModel, sinusoidal_embedding_1d  # type: ignore

    device = torch.device("cpu")
    model = WanModel.from_pretrained(str(ckpt))
    model.eval().to(device=device, dtype=torch.float32)

    with torch.no_grad():
        x_lat = torch.from_numpy(noise).to(device)
        pe = model.patch_embedding(x_lat.unsqueeze(0))
        grid_sizes = torch.stack(
            [torch.tensor(pe.shape[2:], dtype=torch.long, device=device)]
        )
        x = pe.flatten(2).transpose(1, 2)  # [1,T,D]
        T = x.size(1)
        seq_len = T
        # pad to seq_len (already exact for 64)
        te = sinusoidal_embedding_1d(
            model.freq_dim, torch.tensor([t], device=device)
        )
        e = model.time_embedding(te.float())
        e0 = model.time_projection(e).unflatten(1, (6, model.dim))

        ctx = torch.from_numpy(t5).to(device)
        text_len = model.text_len
        if ctx.size(0) < text_len:
            ctx = torch.cat(
                [ctx, ctx.new_zeros(text_len - ctx.size(0), ctx.size(1))], dim=0
            )
        context = model.text_embedding(ctx.unsqueeze(0))

        blk = model.blocks[0]
        # Match WanAttentionBlock.forward
        assert e0.dtype == torch.float32
        e_chunks = (blk.modulation + e0).chunk(6, dim=1)
        # self-attn
        x_sa_in = blk.norm1(x).float() * (1 + e_chunks[1]) + e_chunks[0]
        y = blk.self_attn(
            x_sa_in,
            torch.tensor([T], dtype=torch.long, device=device),
            grid_sizes,
            model.freqs,
        )
        x_sa = x + y * e_chunks[2]
        # cross
        x_cx = x_sa + blk.cross_attn(blk.norm3(x_sa), context, None)
        # ffn
        y_ff = blk.ffn(blk.norm2(x_cx).float() * (1 + e_chunks[4]) + e_chunks[3])
        x_ff = x_cx + y_ff * e_chunks[5]

    results = {}
    mapping = [
        ("b0_post_sa.f32", x_sa[0].cpu().numpy(), "post_sa"),
        ("b0_post_cross.f32", x_cx[0].cpu().numpy(), "post_cross"),
        ("b0_post_ffn.f32", x_ff[0].cpu().numpy(), "post_ffn"),
    ]
    # also patch
    pt = args.wan_c_dir / "patch_tok.f32"
    if pt.is_file():
        c_pt = np.fromfile(pt, dtype=np.float32).reshape(T, -1)
        results["patch_tok"] = stats("patch_tok", c_pt, x[0].cpu().numpy())

    for fname, py, name in mapping:
        path = args.wan_c_dir / fname
        if not path.is_file():
            print(f"{name}: missing {fname}")
            continue
        c = np.fromfile(path, dtype=np.float32).reshape(py.shape)
        results[name] = stats(name, c, py)

    (args.wan_c_dir / "block0_compare.json").write_text(
        json.dumps(results, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
