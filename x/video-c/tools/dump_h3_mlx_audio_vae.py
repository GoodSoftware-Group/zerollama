#!/usr/bin/env python3
"""Dump MiniMax-H3 audio VAE Kaiser-sinc + hop geometry for video-c rematch.

Uses the same formula as minimax_h3_mlx.audio_vae.kaiser_sinc_filter1d
(numpy only — no MLX / weights required).

  python3 x/video-c/tools/dump_h3_mlx_audio_vae.py -o x/video-c/fixtures/h3_mlx_audio_vae.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

UPSAMPLE_RATES = (5, 5, 2, 2, 2, 2, 2)
UPSAMPLE_KERNELS = (9, 9, 4, 4, 4, 4, 4)
RESIDUAL_KERNELS = (3, 7, 11)
RESIDUAL_DILATIONS = (1, 3, 5)
SAMPLE_RATE = 32000
HOP = math.prod(UPSAMPLE_RATES)
FILTER_SIZE = 12


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> np.ndarray:
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = np.kaiser(kernel_size, beta)
    if kernel_size % 2 == 0:
        time = np.arange(-half_size, half_size, dtype=np.float64) + 0.5
    else:
        time = np.arange(kernel_size, dtype=np.float64) - half_size
    filt = 2 * cutoff * window * np.sinc(2 * cutoff * time)
    filt = filt / filt.sum()
    return filt.astype(np.float64), float(beta), float(attenuation)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    # BigVGAN Activation1d defaults: ratio 2, kernel 12
    filt, beta, attenuation = kaiser_sinc_filter1d(0.5 / 2, 0.6 / 2, FILTER_SIZE)

    pads = []
    for samples in [0, 1, 799, 800, 801, 1600, 12345]:
        pads.append(
            {
                "samples": samples,
                "padded": ((samples + HOP - 1) // HOP) * HOP,
            }
        )
    pcms = []
    for t in [1, 2, 8, 37, 107]:
        pcms.append({"latent_t": t, "pcm_samples": t * HOP})

    payload = {
        "note": "minimax-h3-mlx audio_vae Kaiser-sinc + antirez hop geometry",
        "upsample_rates": list(UPSAMPLE_RATES),
        "upsample_kernels": list(UPSAMPLE_KERNELS),
        "residual_kernels": list(RESIDUAL_KERNELS),
        "residual_dilations": list(RESIDUAL_DILATIONS),
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP,
        "filter_size": FILTER_SIZE,
        "activation_filter": {
            "ratio": 2,
            "cutoff": 0.25,
            "half_width": 0.3,
            "beta": beta,
            "attenuation": attenuation,
            "coeffs": [float(x) for x in filt],
        },
        "pad_samples": pads,
        "pcm_from_latent": pcms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
