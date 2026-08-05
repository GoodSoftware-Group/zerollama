#!/usr/bin/env python3
"""Compare two wan-c / Wan RGB24 dumps (MSE, PSNR, channel deltas).

Usage:
  python3 tools/compare_rgb24.py a.rgb24 b.rgb24 --width 64 --height 64 --frames 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def load(path: Path, w: int, h: int, frames: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    n = frames * h * w * 3
    if raw.size < n:
        raise SystemExit(f"{path}: need {n} bytes, got {raw.size}")
    return raw[:n].reshape(frames, h, w, 3).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    a = load(args.a, args.width, args.height, args.frames)
    b = load(args.b, args.width, args.height, args.frames)
    diff = a - b
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse <= 0 else 10.0 * math.log10((255.0 ** 2) / mse)
    out = {
        "a": str(args.a),
        "b": str(args.b),
        "shape": list(a.shape),
        "mse": round(mse, 4),
        "psnr": None if math.isinf(psnr) else round(psnr, 3),
        "mad": round(float(np.mean(np.abs(diff))), 4),
        "ch_mean_delta": [round(float(x), 3) for x in (a.mean(axis=(0, 1, 2)) - b.mean(axis=(0, 1, 2)))],
        "frame_mad": [
            round(float(np.mean(np.abs(diff[i]))), 4) for i in range(a.shape[0])
        ],
    }
    if args.json:
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"mse={out['mse']} psnr={out['psnr']} mad={out['mad']}")
        print(f"ch_mean_delta={out['ch_mean_delta']}")
        print(f"frame_mad={out['frame_mad']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
