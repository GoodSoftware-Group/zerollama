#!/usr/bin/env python3
"""Compare wan-c DiT stage dumps (patch/time) to Wan Python.

Requires parity_c with noise.f32 + patch_tok.f32 + time_e*.f32 from a
WAN_DUMP_DIR run, and Wan ckpt/repo.

Usage:
  WAN_FORCE_SDPA=1 ~/.zerollama/third_party/wan/venv/bin/python \\
    tools/compare_dit_stages.py dumps/parity_c
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wan_c_dir", type=Path)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--t", type=float, default=None, help="Override timestep")
    args = ap.parse_args()

    meta = json.loads((args.wan_c_dir / "meta.json").read_text())
    shape = tuple(meta["latent_shape"])
    noise = np.fromfile(args.wan_c_dir / "noise.f32", dtype=np.float32).reshape(
        shape
    )
    t = args.t
    if t is None:
        t = float(meta.get("gen_t", 1000.0))

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
    if os.environ.get("WAN_FORCE_SDPA") is None:
        os.environ["WAN_FORCE_SDPA"] = "1"

    import torch
    from wan.modules.model import WanModel, sinusoidal_embedding_1d  # type: ignore

    device = torch.device("cpu")
    model = WanModel.from_pretrained(str(ckpt))
    model.eval().to(device=device, dtype=torch.float32)

    x = torch.from_numpy(noise).to(device)
    with torch.no_grad():
        pe = model.patch_embedding(x.unsqueeze(0))  # [1,D,F,H,W]
        grid = pe.shape[2:]
        tok = pe.flatten(2).transpose(1, 2)[0].cpu().numpy()  # [T,D]
        te = sinusoidal_embedding_1d(model.freq_dim, torch.tensor([t], device=device))
        e = model.time_embedding(te.float())[0].cpu().numpy()
        e0 = model.time_projection(torch.from_numpy(e).unsqueeze(0).to(device))
        e0 = e0.unflatten(1, (6, model.dim))[0].cpu().numpy()

    print(f"py patch_tok {tok.shape} grid={tuple(grid)} t={t}")
    out = {}
    pt = args.wan_c_dir / "patch_tok.f32"
    if pt.is_file():
        c_tok = np.fromfile(pt, dtype=np.float32).reshape(tok.shape)
        out["patch_tok"] = {
            "cosine": round(cos(c_tok, tok), 6),
            "mse": round(float(np.mean((c_tok - tok) ** 2)), 6),
            "max_abs": round(float(np.max(np.abs(c_tok - tok))), 6),
        }
        print(
            f"patch_tok: cosine={out['patch_tok']['cosine']} "
            f"mse={out['patch_tok']['mse']} max_abs={out['patch_tok']['max_abs']}"
        )
    else:
        print("patch_tok: missing C dump")

    te_p = args.wan_c_dir / "time_e.f32"
    if te_p.is_file():
        c_e = np.fromfile(te_p, dtype=np.float32)
        out["time_e"] = {
            "cosine": round(cos(c_e, e), 6),
            "mse": round(float(np.mean((c_e - e) ** 2)), 6),
            "max_abs": round(float(np.max(np.abs(c_e - e))), 6),
        }
        print(
            f"time_e: cosine={out['time_e']['cosine']} "
            f"mse={out['time_e']['mse']} max_abs={out['time_e']['max_abs']}"
        )
    e0_p = args.wan_c_dir / "time_e0.f32"
    if e0_p.is_file():
        c_e0 = np.fromfile(e0_p, dtype=np.float32).reshape(6, -1)
        out["time_e0"] = {
            "cosine": round(cos(c_e0, e0), 6),
            "mse": round(float(np.mean((c_e0 - e0) ** 2)), 6),
            "max_abs": round(float(np.max(np.abs(c_e0 - e0))), 6),
        }
        print(
            f"time_e0: cosine={out['time_e0']['cosine']} "
            f"mse={out['time_e0']['mse']} max_abs={out['time_e0']['max_abs']}"
        )

    (args.wan_c_dir / "stage_compare.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
