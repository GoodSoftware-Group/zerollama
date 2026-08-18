#!/usr/bin/env python3
"""Summarize wan-cli raw RGB24 output for lab visual sign-off.

Usage:
  python3 tools/rgb24_stats.py out.mp4.rgb24 --width 64 --height 64 --frames 5
  python3 tools/rgb24_stats.py out.mp4.rgb24 --width 64 --height 64 --frames 5 \\
      --dump-ppm dumps/frames/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def dump_ppm(data: bytes, width: int, height: int, frames: int, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    spat = width * height * 3
    n = 0
    for f in range(frames):
        path = out_dir / f"frame_{f:03d}.ppm"
        chunk = data[f * spat : (f + 1) * spat]
        # PPM P6 binary
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        path.write_bytes(header + chunk)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rgb24", type=Path)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--dump-ppm",
        type=Path,
        default=None,
        help="Write frame_XXX.ppm under this directory (no ffmpeg needed)",
    )
    args = ap.parse_args()

    expect = args.width * args.height * args.frames * 3
    data = args.rgb24.read_bytes()
    if len(data) != expect:
        print(
            f"size mismatch: got {len(data)} want {expect}",
            file=sys.stderr,
        )
        return 1

    n = expect
    total = 0
    total_sq = 0
    hist = [0] * 256
    ch_sum = [0, 0, 0]
    for i, b in enumerate(data):
        total += b
        total_sq += b * b
        hist[b] += 1
        ch_sum[i % 3] += b

    mean = total / n
    var = total_sq / n - mean * mean
    std = var**0.5 if var > 0 else 0.0
    uniq = sum(1 for c in hist if c)
    spat = args.width * args.height * 3
    fmad = 0.0
    if args.frames > 1:
        for f in range(1, args.frames):
            a = memoryview(data)[(f - 1) * spat : f * spat]
            b = memoryview(data)[f * spat : (f + 1) * spat]
            s = 0
            for x, y in zip(a, b):
                s += abs(x - y)
            fmad += s / spat
        fmad /= args.frames - 1

    ppm_n = 0
    if args.dump_ppm is not None:
        ppm_n = dump_ppm(data, args.width, args.height, args.frames, args.dump_ppm)

    out = {
        "path": str(args.rgb24),
        "shape": [args.frames, args.height, args.width, 3],
        "mean": round(mean, 3),
        "std": round(std, 3),
        "uniq_levels": uniq,
        "ch_mean": [round(c / (n / 3), 3) for c in ch_sum],
        "frame_mad": round(fmad, 3),
        "ok_nontrivial": bool(std > 5.0 and uniq > 16 and fmad > 0.5),
        "ppm_frames": ppm_n,
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(
            f"rgb24 {args.width}x{args.height}x{args.frames} "
            f"mean={out['mean']:.1f} std={out['std']:.1f} "
            f"uniq={uniq} ch={out['ch_mean']} fmad={out['frame_mad']:.2f} "
            f"ok={out['ok_nontrivial']}"
            + (f" ppm={ppm_n}→{args.dump_ppm}" if ppm_n else "")
        )
    return 0 if out["ok_nontrivial"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
