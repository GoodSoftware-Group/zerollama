#!/usr/bin/env python3
"""Dump Wan Python pipeline intermediates for parity with wan-c.

Tries to load Wan2.1 text2video path when WAN_REPO + WAN_CKPT are set.
Always writes a structured dump dir even if Wan import fails (stub tensors).

Usage:
  WAN_CKPT=... WAN_REPO=... python3 tools/parity_dump.py --prompt "a cat" --out dumps/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def write_stub(out: Path, prompt: str, steps: int, ckpt: Path) -> None:
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "prompt": prompt,
        "ckpt_dir": str(ckpt),
        "steps": steps,
        "mode": "stub",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    # Fixed-seed noise latent for 5 frames @ 64x64 spatial latent proxy
    rng = np.random.RandomState(0)
    np.save(out / "noise_latent.npy", rng.randn(16, 2, 8, 8).astype(np.float32))
    np.save(out / "t5_emb_stub.npy", rng.randn(512, 4096).astype(np.float32))
    print(f"wrote stub dumps under {out}")


def try_wan_dump(out: Path, prompt: str, steps: int, ckpt: Path, repo: Path) -> bool:
    try:
        import numpy as np
        import torch
    except ImportError:
        return False

    sys.path.insert(0, str(repo))
    try:
        # Prefer installed wan package layout
        from wan.configs import WAN_CONFIGS  # type: ignore
    except Exception as exc:
        print(f"wan import failed: {exc}", file=sys.stderr)
        return False

    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "prompt": prompt,
        "ckpt_dir": str(ckpt),
        "steps": steps,
        "mode": "wan_partial",
        "note": "T5/DiT hooks depend on local Wan checkout; extend as needed",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # Best-effort: tokenize-free random embeddings matching UMT5 dim for harness shape.
    rng = np.random.RandomState(0)
    np.save(out / "t5_emb.npy", rng.randn(77, 4096).astype(np.float32))
    np.save(out / "noise_latent.npy", rng.randn(16, 5, 60, 104).astype(np.float32))
    print(f"wrote partial Wan-shaped dumps under {out} (configs={list(WAN_CONFIGS)[:3]}...)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ckpt-dir", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("dumps"))
    ap.add_argument("--steps", type=int, default=5)
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

    if repo.is_dir() and try_wan_dump(args.out, args.prompt, args.steps, ckpt, repo):
        return
    write_stub(args.out, args.prompt, args.steps, ckpt)


if __name__ == "__main__":
    main()
