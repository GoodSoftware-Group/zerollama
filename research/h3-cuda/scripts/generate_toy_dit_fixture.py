#!/usr/bin/env python3
"""Generate antirez-shaped toy DiT safetensors + CPU goldens for CUDA parity.

Dims match h3.c tests/test_metal.c (SEQUENCE=32, HIDDEN=256, …).
Writes misc/fixtures/h3_dit_toy_f32.safetensors under the lab tree.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

SEQUENCE = 32
HIDDEN = 256
HEADS = 4
HEAD_DIM = 32
INNER = HEADS * HEAD_DIM
FFN = 128
T_ROWS = 2
T_DIM = 32
MODALITIES = 3
MODULATION_SLOTS = 6
ROPE_HALF = 12
EPS = 1e-5


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = EPS) -> np.ndarray:
    # x [rows, width]
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * inv * w


def adaln(x, norm_w, mod, row_map, shift_slot, scale_slot):
    # mod [mod_rows, slots*width]
    n = rms_norm(x, norm_w)
    out = np.empty_like(n)
    for r in range(x.shape[0]):
        mrow = int(row_map[r])
        shift = mod[mrow, shift_slot * HIDDEN : (shift_slot + 1) * HIDDEN]
        scale = mod[mrow, scale_slot * HIDDEN : (scale_slot + 1) * HIDDEN]
        out[r] = n[r] * (1.0 + scale) + shift
    return out


def gate(residual, branch, mod, row_map, gate_slot):
    out = np.empty_like(residual)
    for r in range(residual.shape[0]):
        mrow = int(row_map[r])
        g = mod[mrow, gate_slot * HIDDEN : (gate_slot + 1) * HIDDEN]
        out[r] = residual[r] + branch[r] * g
    return out


def linear(x, w):
    # y = x @ w.T ; w [out, in]
    return x @ w.T


def silu(x):
    return x / (1.0 + np.exp(-x))


def swiglu(fused):
    # fused [rows, 2*width] = gate | up
    width = fused.shape[1] // 2
    g = fused[:, :width]
    up = fused[:, width:]
    return silu(g) * up


def qkv_rope(qkv, q_norm, k_norm, rope_cos, rope_sin):
    # qkv ungrouped [seq, 3*inner] as [Q|K|V] each [seq, heads, dim]
    seq = qkv.shape[0]
    q = np.empty((seq, HEADS, HEAD_DIM), dtype=np.float32)
    k = np.empty_like(q)
    v = np.empty_like(q)
    for r in range(seq):
        row = qkv[r]
        for h in range(HEADS):
            qb = h * HEAD_DIM
            kb = INNER + h * HEAD_DIM
            vb = 2 * INNER + h * HEAD_DIM
            qv = row[qb : qb + HEAD_DIM]
            kv = row[kb : kb + HEAD_DIM]
            vv = row[vb : vb + HEAD_DIM]
            qinv = 1.0 / np.sqrt(np.mean(qv * qv) + EPS)
            kinv = 1.0 / np.sqrt(np.mean(kv * kv) + EPS)
            qo = qv * qinv * q_norm
            ko = kv * kinv * k_norm
            for d in range(ROPE_HALF):
                c = rope_cos[r, d]
                s = rope_sin[r, d]
                q0, q1 = qo[d], qo[d + ROPE_HALF]
                k0, k1 = ko[d], ko[d + ROPE_HALF]
                qo[d] = q0 * c - q1 * s
                qo[d + ROPE_HALF] = q0 * s + q1 * c
                ko[d] = k0 * c - k1 * s
                ko[d + ROPE_HALF] = k0 * s + k1 * c
            q[r, h] = qo
            k[r, h] = ko
            v[r, h] = vv
    return q.reshape(seq, INNER), k.reshape(seq, INNER), v.reshape(seq, INNER)


def sdpa(q, k, v, scale):
    # row-major [seq, heads, dim]
    seq = q.shape[0]
    qq = q.reshape(seq, HEADS, HEAD_DIM)
    kk = k.reshape(seq, HEADS, HEAD_DIM)
    vv = v.reshape(seq, HEADS, HEAD_DIM)
    out = np.empty_like(qq)
    for h in range(HEADS):
        scores = (qq[:, h] @ kk[:, h].T) * scale  # [seq, seq]
        scores = scores - scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p = p / p.sum(axis=-1, keepdims=True)
        out[:, h] = p @ vv[:, h]
    return out.reshape(seq, INNER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "misc/fixtures/h3_dit_toy_f32.safetensors",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    def f32(*shape):
        return rng.standard_normal(shape, dtype=np.float32) * 0.05

    tensors: dict[str, np.ndarray] = {
        "x.h_in": f32(SEQUENCE, HIDDEN),
        "x.attn_in": f32(SEQUENCE, HIDDEN),
        "x.t_emb": f32(T_ROWS, T_DIM),
        "x.rope_cos": np.cos(np.linspace(0, 1, SEQUENCE * ROPE_HALF, dtype=np.float32)).reshape(
            SEQUENCE, ROPE_HALF
        ),
        "x.rope_sin": np.sin(np.linspace(0, 1, SEQUENCE * ROPE_HALF, dtype=np.float32)).reshape(
            SEQUENCE, ROPE_HALF
        ),
        "norm1.weight": np.ones(HIDDEN, dtype=np.float32),
        "norm2.weight": np.ones(HIDDEN, dtype=np.float32),
        "adaln_proj.linear.weight": f32(MODALITIES * MODULATION_SLOTS * HIDDEN, T_DIM),
        "adaln_proj.linear.bias": f32(MODALITIES * MODULATION_SLOTS * HIDDEN),
        "attn.qkv_proj.weight": f32(INNER * 3, HIDDEN),
        "attn.q_norm.weight": np.ones(HEAD_DIM, dtype=np.float32),
        "attn.k_norm.weight": np.ones(HEAD_DIM, dtype=np.float32),
        "attn.out_proj.weight": f32(HIDDEN, INNER),
        "mlp.fc1.weight": f32(FFN * 2, HIDDEN),
        "mlp.fc2.weight": f32(HIDDEN, FFN),
    }

    # 4 runs covering the sequence with modality rows 0..5
    runs = np.array(
        [
            [0, 8, 0],
            [8, 16, 1],
            [16, 24, 2],
            [24, 32, 3],
        ],
        dtype=np.int32,
    )
    tensors["x.runs"] = runs.reshape(-1)

    row_map = np.empty(SEQUENCE, dtype=np.uint32)
    for start, stop, row in runs:
        row_map[start:stop] = row

    # --- CPU reference for full block (matches test_metal full path) ---
    t_silu = silu(tensors["x.t_emb"])
    modulation = (
        linear(t_silu, tensors["adaln_proj.linear.weight"]) + tensors["adaln_proj.linear.bias"]
    )
    # modulation layout [T_ROWS, MODALITIES*SLOTS*HIDDEN] but we index by row_map
    # Expand to [T_ROWS*MODALITIES, SLOTS*HIDDEN] conceptually — linear already outputs
    # MODALITIES*SLOTS*HIDDEN per t_row → reshape to flat for adaln indexing.
    mod_flat = modulation  # [T_ROWS, MOD*SLOTS*HIDDEN]

    # For adaln, row_map indexes into T_ROWS*MODALITIES rows of [slots, hidden].
    # Metal uses modulation as contiguous [mod_rows, slots, width] with
    # mod_rows = T_ROWS * MODALITIES = 6. Our linear wrote [T_ROWS, MOD*SLOTS*HIDDEN]
    # which is the same bytes as [T_ROWS*MOD, SLOTS*HIDDEN].
    mod = mod_flat.reshape(T_ROWS * MODALITIES, MODULATION_SLOTS * HIDDEN)

    h_in = tensors["x.h_in"]
    mod_attn = adaln(h_in, tensors["norm1.weight"], mod, row_map, 0, 1)
    full_qkv = linear(mod_attn, tensors["attn.qkv_proj.weight"])
    full_q, full_k, full_v = qkv_rope(
        full_qkv,
        tensors["attn.q_norm.weight"],
        tensors["attn.k_norm.weight"],
        tensors["x.rope_cos"],
        tensors["x.rope_sin"],
    )
    scale = 1.0 / np.sqrt(float(HEAD_DIM))
    full_sdpa = sdpa(full_q, full_k, full_v, scale)
    full_attn_out = linear(full_sdpa, tensors["attn.out_proj.weight"])
    after_attn = gate(h_in, full_attn_out, mod, row_map, 2)
    mod_mlp = adaln(after_attn, tensors["norm2.weight"], mod, row_map, 3, 4)
    full_fc1 = linear(mod_mlp, tensors["mlp.fc1.weight"])
    full_swiglu = swiglu(full_fc1)
    full_mlp_out = linear(full_swiglu, tensors["mlp.fc2.weight"])
    h_out = gate(after_attn, full_mlp_out, mod, row_map, 5)

    # Plain path goldens (attn_in → mlp without AdaLN/gates) for optional checks
    plain_qkv = linear(tensors["x.attn_in"], tensors["attn.qkv_proj.weight"])
    pq, pk, pv = qkv_rope(
        plain_qkv,
        tensors["attn.q_norm.weight"],
        tensors["attn.k_norm.weight"],
        tensors["x.rope_cos"],
        tensors["x.rope_sin"],
    )
    plain_sdpa = sdpa(pq, pk, pv, scale)
    plain_attn_out = linear(plain_sdpa, tensors["attn.out_proj.weight"])
    plain_fc1 = linear(tensors["x.attn_in"], tensors["mlp.fc1.weight"])
    plain_mlp_out = linear(swiglu(plain_fc1), tensors["mlp.fc2.weight"])

    tensors["x.attn_out"] = plain_attn_out.astype(np.float32)
    tensors["x.mlp_out"] = plain_mlp_out.astype(np.float32)
    tensors["x.h_out"] = h_out.astype(np.float32)
    tensors["x.modulation"] = mod.astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Flatten 2D for safetensors consistency with Metal loader expectations
    flat = {k: (v.reshape(-1) if v.ndim > 1 and k != "x.runs" else v) for k, v in tensors.items()}
    # Keep 2D shapes in metadata via save_file — pass shaped arrays
    shaped = {k: np.asarray(v) for k, v in tensors.items()}
    save_file(shaped, str(args.output))
    size = args.output.stat().st_size
    print(f"wrote {args.output} ({size} bytes, {size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
