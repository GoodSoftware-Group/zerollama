#!/usr/bin/env python3
"""Generate one latent UniPC denoise-step fixture for CUDA rematch.

Lab geometry: latent [16,2,8,8] → patch(1,2,2) → T=32 tokens (grid 2×4×4).
Uses FlowUniPC step0 timestep (999 for steps=4, shift=5).

Writes dumps/latent_unipc_fixture/:
  noise, x_in, e, e0, context, py_after_blocks, py_model_out, py_latent_s1, meta

Usage:
  WAN_FORCE_SDPA=1 ~/.zerollama/third_party/wan/venv/bin/python \\
    tools/gen_latent_unipc_fixture.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dumps" / "latent_unipc_fixture"

C_LAT, LT, LH, LW = 16, 2, 8, 8
PT, PH, PW = 1, 2, 2
GT, GH, GW = LT // PT, LH // PH, LW // PW
T = GT * GH * GW
assert T == 32
D = 1536
TK_SYNTH = 64
SEED = 7
UNIPC_STEPS = 4
UNIPC_SHIFT = 5.0
UNIPC_ORDER = 3


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
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # type: ignore

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

    sched = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
        solver_order=UNIPC_ORDER,
        predict_x0=True,
        solver_type="bh2",
        lower_order_final=True,
        final_sigmas_type="zero",
    )
    sched.set_timesteps(UNIPC_STEPS, device=device, shift=UNIPC_SHIFT)
    t0 = sched.timesteps[0]
    gen_t = float(t0.item())
    print(f"UniPC step0 t={gen_t} sigmas[0]={float(sched.sigmas[0])}", flush=True)

    rng = np.random.default_rng(SEED)
    noise_np = rng.standard_normal((C_LAT, LT, LH, LW), dtype=np.float32) * 0.02
    t5_np = rng.standard_normal((TK_SYNTH, 4096), dtype=np.float32) * 0.02

    with torch.no_grad():
        noise = torch.from_numpy(noise_np)
        # Patch → tokens (same as model.forward prefix)
        pe = model.patch_embedding(noise.unsqueeze(0))
        assert tuple(pe.shape[2:]) == (GT, GH, GW)
        x_tok = pe.flatten(2).transpose(1, 2)  # [1,T,D]
        x_in_np = x_tok[0].cpu().numpy().astype(np.float32)

        te = sinusoidal_embedding_1d(
            model.freq_dim, torch.tensor([gen_t], device=device)
        )
        e = model.time_embedding(te.float())
        e0 = model.time_projection(e).unflatten(1, (6, model.dim))
        e_np = e[0].cpu().numpy().astype(np.float32)
        e0_np = e0[0].cpu().numpy().astype(np.float32)

        ctx = torch.from_numpy(t5_np)
        text_len = int(model.text_len)
        if ctx.size(0) < text_len:
            ctx = torch.cat(
                [ctx, ctx.new_zeros(text_len - ctx.size(0), ctx.size(1))], dim=0
            )
        context = model.text_embedding(ctx.unsqueeze(0))
        ctx_np = context[0].cpu().numpy().astype(np.float32)
        Tk = int(ctx_np.shape[0])

        x = x_tok
        grid_sizes = torch.tensor([[GT, GH, GW]], dtype=torch.long, device=device)
        seq = torch.tensor([T], dtype=torch.long, device=device)
        for i, blk in enumerate(model.blocks):
            x = blk(x, e0, seq, grid_sizes, model.freqs, context, None)
            if (i + 1) % 5 == 0 or i + 1 == len(model.blocks):
                print(f"  block {i + 1}/{len(model.blocks)}", flush=True)
        after_blocks = x[0].cpu().numpy().astype(np.float32)

        head_tok = model.head(x, e)
        out_list = model.unpatchify(head_tok, grid_sizes)
        model_out = out_list[0].cpu().numpy().astype(np.float32)
        assert model_out.shape == (C_LAT, LT, LH, LW)

        # Cross-check full forward
        full = model(
            [noise],
            t=torch.tensor([gen_t], device=device),
            context=[torch.from_numpy(t5_np)],
            seq_len=T,
        )[0].cpu().numpy().astype(np.float32)
        cos = float(
            np.dot(model_out.ravel(), full.ravel())
            / (np.linalg.norm(model_out) * np.linalg.norm(full) + 1e-30)
        )
        print(f"staged vs full forward cosine={cos:.6f}", flush=True)

        sample_t = torch.from_numpy(noise_np.copy().reshape(-1))
        pred_t = torch.from_numpy(model_out.copy().reshape(-1))
        prev = sched.step(pred_t, t0, sample_t, return_dict=False)[0]
        latent_s1 = prev.detach().cpu().numpy().astype(np.float32).reshape(
            C_LAT, LT, LH, LW
        )

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "latent_shape": [C_LAT, LT, LH, LW],
        "patch": [PT, PH, PW],
        "grid": [GT, GH, GW],
        "T": T,
        "Tk": Tk,
        "D": D,
        "n_blocks": len(model.blocks),
        "gen_t": gen_t,
        "unipc_steps": UNIPC_STEPS,
        "unipc_shift": UNIPC_SHIFT,
        "unipc_order": UNIPC_ORDER,
        "ckpt": str(ckpt),
        "staged_vs_full_cosine": cos,
    }
    (OUT / "meta_latent_unipc.json").write_text(json.dumps(meta, indent=2) + "\n")
    noise_np.tofile(OUT / "noise.f32")
    x_in_np.tofile(OUT / "x_in.f32")
    e_np.tofile(OUT / "e.f32")
    e0_np.tofile(OUT / "e0.f32")
    ctx_np.tofile(OUT / "context.f32")
    after_blocks.tofile(OUT / "py_after_blocks.f32")
    model_out.tofile(OUT / "py_model_out.f32")
    latent_s1.tofile(OUT / "py_latent_s1.f32")
    print(f"wrote {OUT} Tk={Tk} model_out={model_out.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
