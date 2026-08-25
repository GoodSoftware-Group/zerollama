#!/usr/bin/env python3
"""Generate multi-block DiT trunk fixture (all 30 blocks) for CUDA rematch.

Reuses dumps/block0_cuda_fixture/{x_in,e0,context}.f32 if present, else regenerates
inputs. Writes py_after_blocks.f32 + py_unipc_step0.f32 + meta_multiblock.json.

UniPC step0 treats token tensors as a flat sample (composition rematch of
sched_unipc vs FlowUniPCMultistepScheduler — not a product latent path).

Usage:
  WAN_FORCE_SDPA=1 ~/.zerollama/third_party/wan/venv/bin/python \\
    tools/gen_multiblock_cuda_fixture.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "dumps" / "block0_cuda_fixture"
OUT = ROOT / "dumps" / "multiblock_cuda_fixture"

T, GT, GH, GW = 32, 2, 4, 4
D = 1536
GEN_T = 1000.0
SEED = 42
TK_SYNTH = 64


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
        return wan_attn._sdpa_attention(q, k, v, *args, **kwargs).float()

    wan_attn.flash_attention = _fa_f32
    wan_model.flash_attention = _fa_f32

    device = torch.device("cpu")
    print(f"loading WanModel from {ckpt} …", flush=True)
    model = WanModel.from_pretrained(str(ckpt))
    model.eval().to(device=device, dtype=torch.float32)
    for p in model.parameters():
        p.data = p.data.float()

    if (FIX / "x_in.f32").is_file() and (FIX / "e0.f32").is_file() and (
        FIX / "context.f32"
    ).is_file():
        x_np = np.fromfile(FIX / "x_in.f32", dtype=np.float32).reshape(T, D)
        e0_np = np.fromfile(FIX / "e0.f32", dtype=np.float32).reshape(6, D)
        ctx_np = np.fromfile(FIX / "context.f32", dtype=np.float32)
        Tk = ctx_np.size // D
        ctx_np = ctx_np.reshape(Tk, D)
        print(f"reusing block0 fixture inputs Tk={Tk}", flush=True)
    else:
        rng = np.random.default_rng(SEED)
        x_np = rng.standard_normal((T, D), dtype=np.float32) * 0.02
        t5_np = rng.standard_normal((TK_SYNTH, 4096), dtype=np.float32) * 0.02
        with torch.no_grad():
            te = sinusoidal_embedding_1d(
                model.freq_dim, torch.tensor([GEN_T], device=device)
            )
            e = model.time_embedding(te.float())
            e0 = model.time_projection(e).unflatten(1, (6, model.dim))
            e0_np = e0[0].cpu().numpy().astype(np.float32)
            ctx = torch.from_numpy(t5_np)
            text_len = int(model.text_len)
            if ctx.size(0) < text_len:
                ctx = torch.cat(
                    [ctx, ctx.new_zeros(text_len - ctx.size(0), ctx.size(1))],
                    dim=0,
                )
            context = model.text_embedding(ctx.unsqueeze(0))
            ctx_np = context[0].cpu().numpy().astype(np.float32)
            Tk = ctx_np.shape[0]

    n_blocks = len(model.blocks)
    with torch.no_grad():
        x = torch.from_numpy(x_np).unsqueeze(0)
        e0 = torch.from_numpy(e0_np).unsqueeze(0)
        context = torch.from_numpy(ctx_np).unsqueeze(0)
        grid_sizes = torch.tensor([[GT, GH, GW]], dtype=torch.long, device=device)
        seq = torch.tensor([T], dtype=torch.long, device=device)
        for i, blk in enumerate(model.blocks):
            x = blk(x, e0, seq, grid_sizes, model.freqs, context, None)
            if (i + 1) % 5 == 0 or i + 1 == n_blocks:
                print(f"  block {i + 1}/{n_blocks}", flush=True)
        out = x[0].cpu().numpy().astype(np.float32)

    # Token-space UniPC step0 vs C sched_unipc (steps=4, shift=5, order=3).
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # type: ignore

    unipc_steps = 4
    unipc_shift = 5.0
    unipc_order = 3
    sample_np = x_np.reshape(-1).astype(np.float32)
    pred_np = out.reshape(-1).astype(np.float32)
    sched = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
        solver_order=unipc_order,
        predict_x0=True,
        solver_type="bh2",
        lower_order_final=True,
        final_sigmas_type="zero",
    )
    sched.set_timesteps(unipc_steps, device=device, shift=unipc_shift)
    sample_t = torch.from_numpy(sample_np.copy()).to(device=device, dtype=torch.float32)
    pred_t = torch.from_numpy(pred_np.copy()).to(device=device, dtype=torch.float32)
    t0 = sched.timesteps[0]
    with torch.no_grad():
        prev = sched.step(pred_t, t0, sample_t, return_dict=False)[0]
    unipc_np = prev.detach().cpu().numpy().astype(np.float32).reshape(-1)

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "T": T,
        "Tk": int(Tk),
        "D": D,
        "n_blocks": n_blocks,
        "grid": [GT, GH, GW],
        "gen_t": GEN_T,
        "ckpt": str(ckpt),
        "unipc_steps": unipc_steps,
        "unipc_shift": unipc_shift,
        "unipc_order": unipc_order,
        "unipc_note": "token-space composition rematch; not product latent",
    }
    (OUT / "meta_multiblock.json").write_text(json.dumps(meta, indent=2) + "\n")
    x_np.astype(np.float32).tofile(OUT / "x_in.f32")
    e0_np.astype(np.float32).tofile(OUT / "e0.f32")
    ctx_np.astype(np.float32).tofile(OUT / "context.f32")
    out.tofile(OUT / "py_after_blocks.f32")
    unipc_np.tofile(OUT / "py_unipc_step0.f32")
    print(
        f"wrote {OUT} n_blocks={n_blocks} Tk={Tk} unipc_step0={unipc_np.shape[0]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
