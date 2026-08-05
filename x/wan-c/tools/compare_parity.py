#!/usr/bin/env python3
"""Compare wan-c dump (f32+meta) to Wan parity_dump npy.

Usage:
  python3 tools/compare_parity.py dumps/parity_c dumps/parity_py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_c(dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    meta = json.loads((dir / "meta.json").read_text())
    t5_shape = tuple(meta["t5_shape"])
    t5 = np.fromfile(dir / "t5_emb.f32", dtype=np.float32).reshape(t5_shape)
    noise = np.fromfile(dir / "noise.f32", dtype=np.float32)
    if "latent_shape" in meta:
        noise = noise.reshape(meta["latent_shape"])
    return t5, noise, meta


def load_py(dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    meta = json.loads((dir / "meta.json").read_text())
    t5 = np.load(dir / "t5_emb.npy")
    noise = np.load(dir / "noise_latent.npy")
    return t5, noise, meta


def stats(a: np.ndarray, b: np.ndarray, name: str) -> dict:
    if a.shape != b.shape:
        # trim to min seq for T5
        if name == "t5" and a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[1]:
            n = min(a.shape[0], b.shape[0])
            a, b = a[:n], b[:n]
        else:
            return {
                "name": name,
                "error": f"shape mismatch {a.shape} vs {b.shape}",
            }
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(np.dot(a.ravel(), b.ravel()) / denom) if denom > 0 else float("nan")
    return {
        "name": name,
        "shape": list(a.shape),
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "cosine": round(cos, 6),
        "max_abs": round(float(np.max(np.abs(diff))), 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wan_c_dir", type=Path)
    ap.add_argument("wan_py_dir", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    c_t5, c_noise, c_meta = load_c(args.wan_c_dir)
    p_t5, p_noise, p_meta = load_py(args.wan_py_dir)
    out = {
        "t5": stats(c_t5, p_t5, "t5"),
        "noise": stats(c_noise, p_noise, "noise"),
        "note": "noise RNGs differ (Box-Muller LCG vs torch); T5 should match closely",
    }
    if args.json:
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for k in ("t5", "noise"):
            s = out[k]
            if "error" in s:
                print(f"{k}: {s['error']}")
            else:
                print(
                    f"{k}: shape={s['shape']} cosine={s['cosine']} "
                    f"mse={s['mse']} mae={s['mae']} max_abs={s['max_abs']}"
                )
    # Non-zero exit if T5 cosine is poor
    t5 = out["t5"]
    if "cosine" in t5 and t5["cosine"] < 0.99:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
