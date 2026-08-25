#!/usr/bin/env python3
"""Dump MiniMax-H3 MLX-style sigma schedules for video-c rematch (no MLX/weights).

Matches minimax_h3_mlx/scheduler.py linspace+shift+dedupe (torch float32 grid).

  python3 x/video-c/tools/dump_h3_mlx_schedule.py -o x/video-c/dumps/h3_mlx_schedule.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def linspace_1_to_0(n: int) -> np.ndarray:
    start, end = 1.0, 0.0
    step = float(np.float32((end - start) / np.float32(n - 1)))
    half = n // 2
    i = np.arange(n, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    out[:half] = start + step * i[:half]
    out[half:] = end - step * (n - 1 - i[half:])
    return out.astype(np.float32)


def build_sigmas(shift: float, n: int) -> list[float]:
    base = linspace_1_to_0(n)
    shift32 = np.float32(shift)
    shifted = (shift32 * base) / (np.float32(1.0) + np.float32(shift - 1.0) * base)
    values: list[float] = []
    for v in shifted.tolist():
        if not values or v != values[-1]:
            values.append(float(v))
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    payload = {
        "note": "minimax-h3-mlx scheduler.py compatible; antirez h3_schedule_build(N) ≈ n=N+1 grid",
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "grids": {
            "n21": {
                "video": build_sigmas(12.0, 21),
                "audio": build_sigmas(3.0, 21),
            },
            "n5": {
                "video": build_sigmas(12.0, 5),
                "audio": build_sigmas(3.0, 5),
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
