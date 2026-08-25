#!/usr/bin/env python3
"""Fold MiniMax Music 3 dav.pth (weight-norm) to safetensors for music-cli.

Why a separate export: C should not parse pickle. Skips if dav.pth is missing (mlx 8-bit
hear path has no dav.pth). Does not require CUDA.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path.home() / ".zerollama" / "models" / "MiniMax-Music3",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    src = args.ckpt_dir / "dav.pth"
    dst = args.out or (args.ckpt_dir / "dav.safetensors")
    if not src.is_file():
        print(f"skip: no {src}", file=sys.stderr)
        return 0
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        print("need torch + safetensors", file=sys.stderr)
        return 2
    state = torch.load(str(src), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    tensors = {}
    for key, value in state.items():
        if not key.startswith(("dec_in_proj.", "decoder.")):
            continue
        if hasattr(value, "detach"):
            tensors[key] = value.detach().contiguous().cpu()
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dst))
    print(f"wrote {dst} keys={len(tensors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
