#!/usr/bin/env python3
"""Rematch host H3_DUMP_L0 vs Comfy DiTBlock-0 math (CPU, no sampler).

Reads /tmp/h3_l0 from `H3_DUMP_L0=1 video-cli --dit-denoise --layers 1`.
Uses the pruned int8 ConvRot pack + comfy_kitchen eager int8_linear.
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from comfy_kitchen.backends.eager.quantization import (
    dequantize_int8_convrot_weight,
    int8_linear,
)
from comfy_kitchen.backends.eager.rope import apply_rope_split_half1

PACK = Path.home() / ".zerollama/third_party/h3/dit/MiniMax-H3-FL2VA-pruned_int8_convrot.safetensors"
DUMP = Path(os.environ.get("H3_DUMP_L0", "/tmp/h3_l0"))
H, HD, HEADS, INNER, FFN = 5376, 128, 56, 7168, 14336
GS = 256
EPS = 1e-5


def load_st(path: Path):
    raw = path.read_bytes()
    n = struct.unpack_from("<Q", raw, 0)[0]
    meta = json.loads(raw[8 : 8 + n].decode())
    base = 8 + n

    def tensor(name):
        t = meta[name]
        off0, off1 = t["data_offsets"]
        blob = raw[base + off0 : base + off1]
        dt = t["dtype"]
        shape = tuple(t["shape"])
        if dt == "F32":
            return np.frombuffer(blob, dtype=np.float32).reshape(shape).copy()
        if dt == "BF16":
            return (
                torch.frombuffer(bytearray(blob), dtype=torch.bfloat16)
                .reshape(shape)
                .float()
                .numpy()
            )
        if dt == "I8":
            return np.frombuffer(blob, dtype=np.int8).reshape(shape).copy()
        if dt == "U8":
            return blob
        raise ValueError(f"{name} {dt}")

    return tensor


def parse_meta(p: Path):
    d = {}
    for line in (p / "meta.txt").read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if "=" in parts[0]:
            for tok in parts:
                a, b = tok.split("=", 1)
                d[a] = int(b)
        elif parts[0] == "tags":
            d["tags"] = [int(x) for x in parts[1:]]
        elif parts[0] == "uniq":
            d["uniq"] = [float(x) for x in parts[1:]]
        elif parts[0] == "row_t":
            d["row_t"] = [float(x) for x in parts[1:]]
        elif parts[0] == "idx":
            d["idx"] = [int(x) for x in parts[1:]]
    return d


def rmsnorm(x, w):
    # x [S,H]
    ms = (x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True)
    return (x / np.sqrt(ms + EPS)).astype(np.float32) * w


def lerp_table(table, t):
    g, r = table.shape
    t = min(1.0, max(0.0, float(t)))
    x = t * (g - 1)
    i0 = int(x)
    if i0 >= g - 1:
        return table[-1].copy()
    a = x - i0
    return (1 - a) * table[i0] + a * table[i0 + 1]


def adaln_chunk(proj, nuniq):
    # Comfy: view(M*3, 6*H).chunk(6)
    x = proj.reshape(nuniq * 3, 6 * H)
    return [x[:, k * H : (k + 1) * H] for k in range(6)]


def sdpa(q, k, v, scale):
    # q,k,v [S, heads, hd]
    s, heads, hd = q.shape
    out = np.zeros_like(q)
    for h in range(heads):
        scores = q[:, h] @ k[:, h].T * scale
        scores = scores - scores.max(axis=-1, keepdims=True)
        w = np.exp(scores)
        w = w / w.sum(axis=-1, keepdims=True)
        out[:, h] = w @ v[:, h]
    return out


def i8_linear(x, q, scale, convrot=True):
    xt = torch.from_numpy(x.astype(np.float32))
    w = torch.from_numpy(q)
    sc = torch.from_numpy(scale.reshape(-1).astype(np.float32))
    y = int8_linear(xt, w, sc, None, torch.float32, convrot=convrot, convrot_groupsize=GS)
    return y.detach().numpy().astype(np.float32)


def f32_linear(x, q, scale):
    w = dequantize_int8_convrot_weight(
        torch.from_numpy(q), torch.from_numpy(scale.astype(np.float32)), GS
    ).numpy()
    return x @ w.T


def gemm(x, q, scale, kind):
    return i8_linear(x, q, scale) if kind == "i8" else f32_linear(x, q, scale)


def rms(a, b=None):
    if b is None:
        return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    return float(np.sqrt(np.mean((a - b).astype(np.float64) ** 2)))


def rope_freqs(pos, inv):
    per = pos[:, :, None] * inv[None, None, :]
    half = np.concatenate([per[:, 0], per[:, 1], per[:, 2]], axis=-1)
    ang = np.concatenate([half, half], axis=-1)[:, :48]
    c, s_ = np.cos(ang), np.sin(ang)
    table_r = np.stack([c, -s_, s_, c], axis=-1).reshape(pos.shape[0], 48, 2, 2)
    return torch.from_numpy(table_r.astype(np.float32)).reshape(1, pos.shape[0], 1, 48, 2, 2)


def dit_block(ten, x, idx, uniq, freqs_ck, layer, kind):
    seq = x.shape[0]
    p = f"blocks.{layer}."
    nuniq = len(uniq)
    nw1 = ten(p + "norm1.weight")
    aw = ten(p + "adaln_proj.linear.weight")
    ab = ten(p + "adaln_proj.linear.bias")
    qn = ten(p + "attn.q_norm.weight")
    kn = ten(p + "attn.k_norm.weight")
    nw2 = ten(p + "norm2.weight")
    emb = np.stack([lerp_table(ten("adaln_t_table"), t) for t in uniq], axis=0).astype(np.float32)
    proj = emb @ aw.T + ab
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_chunk(proj, nuniq)
    h = rmsnorm(x, nw1)
    for s in range(seq):
        h[s] = h[s] * (1.0 + scale_msa[idx[s]]) + shift_msa[idx[s]]
    qkv = gemm(h, ten(p + "attn.qkv_proj.weight"), ten(p + "attn.qkv_proj.weight_scale"), kind)
    q, k, v = np.split(qkv, 3, axis=-1)
    q = rmsnorm(q.reshape(-1, HD), qn).reshape(seq, HEADS, HD)
    k = rmsnorm(k.reshape(-1, HD), kn).reshape(seq, HEADS, HD)
    qt = torch.from_numpy(q.astype(np.float32)).reshape(1, seq, HEADS, HD)
    kt = torch.from_numpy(k.astype(np.float32)).reshape(1, seq, HEADS, HD)
    qr = apply_rope_split_half1(qt[..., :96], freqs_ck)
    kr = apply_rope_split_half1(kt[..., :96], freqs_ck)
    q = torch.cat([qr, qt[..., 96:]], dim=-1)[0].numpy()
    k = torch.cat([kr, kt[..., 96:]], dim=-1)[0].numpy()
    v = v.reshape(seq, HEADS, HD)
    attn = sdpa(q, k, v, 1.0 / math.sqrt(HD)).reshape(seq, INNER)
    branch = gemm(attn, ten(p + "attn.out_proj.weight"), ten(p + "attn.out_proj.weight_scale"), kind)
    x1 = x.copy()
    for s in range(seq):
        x1[s] = x[s] + gate_msa[idx[s]] * branch[s]
    h2 = rmsnorm(x1, nw2)
    for s in range(seq):
        h2[s] = h2[s] * (1.0 + scale_mlp[idx[s]]) + shift_mlp[idx[s]]
    fused = gemm(h2, ten(p + "mlp.fc1.weight"), ten(p + "mlp.fc1.weight_scale"), kind)
    gate, up = fused[:, :FFN], fused[:, FFN:]
    hid = (gate / (1.0 + np.exp(-gate))) * up
    mlp_b = gemm(hid.astype(np.float32), ten(p + "mlp.fc2.weight"), ten(p + "mlp.fc2.weight_scale"), kind)
    y = x1.copy()
    for s in range(seq):
        y[s] = x1[s] + gate_mlp[idx[s]] * mlp_b[s]
    gmsa = float(np.sqrt(np.mean(gate_msa[idx[0]].astype(np.float64) ** 2)))
    return y, gmsa


def main():
    if not (DUMP / "x.bin").is_file():
        print(f"missing {DUMP}/x.bin — run H3_DUMP_L0=1 video-cli --dit-denoise --layers 1", file=sys.stderr)
        return 2
    meta = parse_meta(DUMP)
    seq, hidden = meta["seq"], meta["H"]
    x = np.fromfile(DUMP / "x.bin", dtype=np.float32).reshape(seq, hidden)
    pos = np.fromfile(DUMP / "pos.bin", dtype=np.float32).reshape(seq, 3)
    host_h = np.fromfile(DUMP / "h.bin", dtype=np.float32).reshape(seq, hidden)
    host_qkv = np.fromfile(DUMP / "qkv.bin", dtype=np.float32).reshape(seq, 3 * INNER)
    host_y = np.fromfile(DUMP / "y.bin", dtype=np.float32).reshape(seq, hidden)
    tags, idx, uniq = meta["tags"], meta["idx"], meta["uniq"]
    nuniq = len(uniq)

    ten = load_st(PACK)
    table = ten("adaln_t_table")
    nw1 = ten("blocks.0.norm1.weight")
    aw = ten("blocks.0.adaln_proj.linear.weight")
    ab = ten("blocks.0.adaln_proj.linear.bias")
    qn = ten("blocks.0.attn.q_norm.weight")
    kn = ten("blocks.0.attn.k_norm.weight")
    inv = ten("rope.inv_freq")
    nw2 = ten("blocks.0.norm2.weight")

    h_rms = rmsnorm(x, nw1)
    emb = np.stack([lerp_table(table, t) for t in uniq], axis=0).astype(np.float32)
    proj = emb @ aw.T + ab
    chunks = adaln_chunk(proj, nuniq)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

    h = h_rms.copy()
    for s in range(seq):
        row = idx[s]
        h[s] = h[s] * (1.0 + scale_msa[row]) + shift_msa[row]

    print(f"h_mod vs host h.bin rms={rms(h, host_h):.6g} host={rms(host_h):.6g} py={rms(h):.6g}")

    qkv_q = ten("blocks.0.attn.qkv_proj.weight")
    qkv_s = ten("blocks.0.attn.qkv_proj.weight_scale")
    qkv_i8 = i8_linear(h, qkv_q, qkv_s)
    qkv_f32 = f32_linear(h, qkv_q, qkv_s)
    print(f"qkv kitchen vs host rms={rms(qkv_i8, host_qkv):.6g}  f32-dequant vs host={rms(qkv_f32, host_qkv):.6g}")
    print(f"qkv kitchen vs f32-dequant rms={rms(qkv_i8, qkv_f32):.6g}")

    qkv = qkv_f32
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(seq, HEADS, HD)
    k = k.reshape(seq, HEADS, HD)
    v = v.reshape(seq, HEADS, HD)
    q = rmsnorm(q.reshape(-1, HD), qn).reshape(seq, HEADS, HD)
    k = rmsnorm(k.reshape(-1, HD), kn).reshape(seq, HEADS, HD)

    # Comfy rope_freqs: pos[S,3] * inv[16] → cat axes → duplicate → rotation table
    per = pos[:, :, None] * inv[None, None, :]  # [S,3,16]
    half = np.concatenate([per[:, 0], per[:, 1], per[:, 2]], axis=-1)  # [S,48]
    angles = np.concatenate([half, half], axis=-1)  # [S,96]
    # apply_rope_split_half1 wants freqs [..., half, 2, 2] last dims
    ang = angles[:, :48]
    c, s_ = np.cos(ang), np.sin(ang)
    table_r = np.stack([c, -s_, s_, c], axis=-1).reshape(seq, 48, 2, 2)
    freqs_ck = torch.from_numpy(table_r.astype(np.float32)).reshape(1, seq, 1, 48, 2, 2)
    qt = torch.from_numpy(q.astype(np.float32)).reshape(1, seq, HEADS, HD)
    kt = torch.from_numpy(k.astype(np.float32)).reshape(1, seq, HEADS, HD)
    qr = apply_rope_split_half1(qt[..., :96], freqs_ck)
    kr = apply_rope_split_half1(kt[..., :96], freqs_ck)
    q = torch.cat([qr, qt[..., 96:]], dim=-1)[0].numpy()
    k = torch.cat([kr, kt[..., 96:]], dim=-1)[0].numpy()

    attn = sdpa(q, k, v, 1.0 / math.sqrt(HD)).reshape(seq, INNER)
    o_q = ten("blocks.0.attn.out_proj.weight")
    o_s = ten("blocks.0.attn.out_proj.weight_scale")
    branch = f32_linear(attn, o_q, o_s)
    x1 = x.copy()
    for s in range(seq):
        x1[s] = x[s] + gate_msa[idx[s]] * branch[s]

    h2 = rmsnorm(x1, nw2)
    for s in range(seq):
        h2[s] = h2[s] * (1.0 + scale_mlp[idx[s]]) + shift_mlp[idx[s]]
    fc1_q, fc1_s = ten("blocks.0.mlp.fc1.weight"), ten("blocks.0.mlp.fc1.weight_scale")
    fused = f32_linear(h2, fc1_q, fc1_s)
    gate, up = fused[:, :FFN], fused[:, FFN:]
    hid = (gate / (1.0 + np.exp(-gate))) * up  # silu
    fc2_q, fc2_s = ten("blocks.0.mlp.fc2.weight"), ten("blocks.0.mlp.fc2.weight_scale")
    mlp_b = f32_linear(hid.astype(np.float32), fc2_q, fc2_s)
    y = x1.copy()
    for s in range(seq):
        y[s] = x1[s] + gate_mlp[idx[s]] * mlp_b[s]

    print(f"y f32-dequant vs host y.bin rms={rms(y, host_y):.6g} host={rms(host_y):.6g} py={rms(y):.6g}")
    print(f"gate_msa row0 rms={rms(gate_msa[idx[0]]):.6g}")

    nstack = 0
    kind = "i8"
    argv = sys.argv[1:]
    if "--stack" in argv:
        nstack = int(argv[argv.index("--stack") + 1])
    if "--gemm" in argv:
        kind = argv[argv.index("--gemm") + 1]
    if nstack > 0:
        freqs_ck = rope_freqs(pos, inv)
        xs = x.copy()
        want = {0, 23, 35, 45, 46, 47, 48, 49}
        for li in range(nstack):
            xs, gmsa = dit_block(ten, xs, idx, uniq, freqs_ck, li, kind)
            if li in want or li == nstack - 1:
                print(f"stack gemm={kind} L{li} x_rms={rms(xs):.6g} gate_msa_row0={gmsa:.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
