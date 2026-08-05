#!/usr/bin/env python3
"""Dump Wan Python T5 + noise for parity with wan-c (lab).

When WAN_REPO + ckpt exist, loads UMT5 encoder and writes:
  meta.json, t5_emb.npy [seq,4096], noise_latent.npy [16,T,H,W]

Usage:
  WAN_REPO=~/.zerollama/third_party/wan/Wan2.1 \\
  WAN_CKPT=~/.zerollama/third_party/wan/Wan2.1-T2V-1.3B \\
  python3 tools/parity_dump.py --prompt "a red apple" --out dumps/parity_py \\
    --width 64 --height 64 --frames 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def write_stub(out: Path, prompt: str, steps: int, ckpt: Path, shape: tuple) -> None:
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    c, t, h, w = shape
    meta = {
        "prompt": prompt,
        "ckpt_dir": str(ckpt),
        "steps": steps,
        "mode": "stub",
        "latent_shape": [c, t, h, w],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    rng = np.random.RandomState(0)
    np.save(out / "noise_latent.npy", rng.randn(c, t, h, w).astype(np.float32))
    np.save(out / "t5_emb_stub.npy", rng.randn(512, 4096).astype(np.float32))
    print(f"wrote stub dumps under {out}")


def try_wan_dump(
    out: Path,
    prompt: str,
    steps: int,
    ckpt: Path,
    repo: Path,
    width: int,
    height: int,
    frames: int,
    seed: int,
) -> bool:
    try:
        import numpy as np
        import torch
    except ImportError:
        return False

    sys.path.insert(0, str(repo))
    try:
        from wan.configs import WAN_CONFIGS  # type: ignore
        from wan.modules.t5 import T5EncoderModel  # type: ignore
    except Exception as exc:
        print(f"wan import failed: {exc}", file=sys.stderr)
        return False

    cfg = WAN_CONFIGS.get("t2v-1.3B") or next(iter(WAN_CONFIGS.values()))
    # Latent grid: Wan VAE stride t=4, h=w=8
    lt = (frames - 1) // 4 + 1
    lh = height // 8
    lw = width // 8
    z_channels = 16

    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    t5_path = ckpt / "models_t5_umt5-xxl-enc-bf16.pth"
    tok_path = ckpt / "google" / "umt5-xxl"
    if not t5_path.is_file():
        print(f"missing T5 weights: {t5_path}", file=sys.stderr)
        return False

    print(f"loading T5 on {device} from {t5_path} …", flush=True)
    text_encoder = T5EncoderModel(
        text_len=getattr(cfg, "text_len", 512),
        dtype=torch.float32,
        device=device,
        checkpoint_path=str(t5_path),
        tokenizer_path=str(tok_path) if tok_path.is_dir() else None,
    )
    with torch.no_grad():
        context = text_encoder([prompt], device)
    # context is list of [L, C] or tensor
    if isinstance(context, (list, tuple)):
        emb = context[0]
    else:
        emb = context[0] if context.ndim == 3 else context
    emb_np = emb.detach().float().cpu().numpy()

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    noise = torch.randn(
        z_channels, lt, lh, lw, dtype=torch.float32, generator=g
    ).numpy()

    meta = {
        "prompt": prompt,
        "ckpt_dir": str(ckpt),
        "steps": steps,
        "mode": "wan_t5_noise",
        "device": str(device),
        "t5_shape": list(emb_np.shape),
        "latent_shape": [z_channels, lt, lh, lw],
        "seed": seed,
        "width": width,
        "height": height,
        "frames": frames,
        "configs": list(WAN_CONFIGS.keys())[:8],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    np.save(out / "t5_emb.npy", emb_np.astype(np.float32))
    np.save(out / "noise_latent.npy", noise.astype(np.float32))
    print(f"wrote Wan T5+noise under {out} t5={emb_np.shape} noise={noise.shape}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ckpt-dir", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("dumps/parity_py"))
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ckpt = args.ckpt_dir or Path(
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

    shape = (16, (args.frames - 1) // 4 + 1, args.height // 8, args.width // 8)
    if repo.is_dir() and try_wan_dump(
        args.out,
        args.prompt,
        args.steps,
        ckpt,
        repo,
        args.width,
        args.height,
        args.frames,
        args.seed,
    ):
        return
    write_stub(args.out, args.prompt, args.steps, ckpt, shape)


if __name__ == "__main__":
    main()
