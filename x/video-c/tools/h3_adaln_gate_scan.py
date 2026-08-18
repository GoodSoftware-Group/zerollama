#!/usr/bin/env python3
"""Print AdaLN gate_msa / gate_mlp RMS for video row vs t (no DiT forward)."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

PACK = Path.home() / ".zerollama/third_party/h3/dit/MiniMax-H3-FL2VA-pruned_int8_convrot.safetensors"
H = 5376
M = 3  # video, text, audio
K = 6


def load_meta(path: Path):
    raw = path.read_bytes()
    n = struct.unpack_from("<Q", raw, 0)[0]
    meta = json.loads(raw[8 : 8 + n].decode())
    return raw, 8 + n, meta


def f32(raw, base, meta, name):
    t = meta[name]
    a, b = t["data_offsets"]
    return np.frombuffer(raw, dtype=np.float32, offset=base + a, count=(b - a) // 4).reshape(t["shape"]).copy()


def lerp_table(table, t):
    g = table.shape[0]
    t = min(1.0, max(0.0, float(t)))
    x = t * (g - 1)
    i0 = int(x)
    if i0 >= g - 1:
        return table[-1]
    a = x - i0
    return (1 - a) * table[i0] + a * table[i0 + 1]


def main():
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    layers = list(range(50))
    raw, base, meta = load_meta(PACK)
    table = f32(raw, base, meta, "adaln_t_table")
    embs = {t: lerp_table(table, t) for t in ts}
    print("layer t     gate_msa_v  gate_mlp_v  gate_msa_t  gate_mlp_t")
    for L in layers:
        w = f32(raw, base, meta, f"blocks.{L}.adaln_proj.linear.weight")
        b = f32(raw, base, meta, f"blocks.{L}.adaln_proj.linear.bias")
        for t in ts:
            proj = embs[t] @ w.T + b
            chunks = proj.reshape(M, K * H).reshape(M, K, H)
            # host/Comfy: [mod, expand, H] after view(3, 6*H).chunk(6)
            g_msa = chunks[:, 2]
            g_mlp = chunks[:, 5]
            def rms(v):
                return float(np.sqrt(np.mean(v.astype(np.float64) ** 2)))
            if L in (0, 2, 24, 32, 33, 40, 49) or t == 0.0:
                print(
                    f"{L:5d} {t:.2f}  {rms(g_msa[0]):10.4g} {rms(g_mlp[0]):10.4g} "
                    f"{rms(g_msa[1]):10.4g} {rms(g_mlp[1]):10.4g}"
                )
        if L not in (0, 2, 24, 32, 33, 40, 49):
            # still print t=0 line above; skip other t
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
