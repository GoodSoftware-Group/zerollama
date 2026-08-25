#!/usr/bin/env python3
"""Dump MiniMax-H3 packing geometry (no MLX) for video-c rematch.

Mirrors minimax_h3_mlx/packing.py canvas/frame/latent helpers.

  python3 x/video-c/tools/dump_h3_mlx_packing.py -o x/video-c/fixtures/h3_mlx_packing.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FPS = 24
SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
CANVAS_MULTIPLE = 32
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
AUDIO_LATENTS_PER_SECOND = 40


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    ratio = aspect_width / aspect_height
    if ratio >= 1.0:
        width, height = SHORT_EDGE * ratio, float(SHORT_EDGE)
    else:
        width, height = float(SHORT_EDGE), SHORT_EDGE / ratio
    area = width * height
    if area > MAX_PIXELS:
        scale = (MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale
    m = CANVAS_MULTIPLE
    return max(m, round(height / m) * m), max(m, round(width / m) * m)


def align_num_frames(num_frames: int) -> int:
    while num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int) -> int:
    return (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    canvases = []
    for aw, ah in [(16, 9), (9, 16), (1, 1), (4, 1), (1, 4), (21, 9), (3, 2)]:
        h, w = resolve_canvas_size(aw, ah)
        canvases.append({"aw": aw, "ah": ah, "height": h, "width": w})

    frames = []
    for n in [1, 5, 6, 22, 23, 100, 360, 361]:
        aligned = align_num_frames(n)
        frames.append(
            {
                "request": n,
                "aligned": aligned,
                "video_t": video_latent_num_frames(aligned),
                "audio_t": audio_latent_num_frames(aligned),
            }
        )

    payload = {
        "note": "minimax-h3-mlx packing.py geometry; rematch vs h3_host",
        "canvases": canvases,
        "frames": frames,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
