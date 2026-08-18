#!/usr/bin/env python3
"""Run Comfy MiniMaxH3 DiTBlocks on dumped packed x (H3_DUMP_L0)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

COMFY = Path("/Users/user1/Sites/inference/ComfyUI")
PACK = COMFY / "models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
DUMP = Path(os.environ.get("H3_DUMP_L0", "/tmp/h3_l0"))
H = 5376


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
        elif parts[0] == "idx":
            d["idx"] = [int(x) for x in parts[1:]]
    return d


def rms(a):
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))


def main() -> int:
    dtype_name = sys.argv[1] if len(sys.argv) > 1 else "fp32"
    compute = torch.bfloat16 if dtype_name in ("bf16", "bfloat16") else torch.float32
    os.chdir(COMFY)
    sys.path.insert(0, str(COMFY))
    import folder_paths  # noqa: F401
    import comfy.model_management as mm
    import comfy.sd
    from comfy.ldm.minimax.model import rope_rotation_table

    meta = parse_meta(DUMP)
    seq = meta["seq"]
    x = np.fromfile(DUMP / "x.bin", dtype=np.float32).reshape(seq, H)
    pos = np.fromfile(DUMP / "pos.bin", dtype=np.float32).reshape(seq, 3)
    host_y = np.fromfile(DUMP / "y.bin", dtype=np.float32).reshape(seq, H)
    tags = meta["tags"]
    uniq = meta["uniq"]

    device = torch.device("cpu")
    model_options = {
        "load_device": device,
        "offload_device": device,
        "dtype": compute,
    }
    print(f"loading {PACK}", flush=True)
    patcher = comfy.sd.load_diffusion_model(str(PACK), model_options=model_options)
    mm.load_models_gpu([patcher], force_full_load=True)
    dm = patcher.model.diffusion_model
    dm.eval()

    dtype = next(dm.blocks[0].parameters()).dtype
    print(f"block0 param dtype={dtype} nblocks={len(dm.blocks)}", flush=True)

    h = torch.from_numpy(x.copy()).to(device=device, dtype=compute)
    t_vals = torch.tensor(uniq, dtype=torch.float32, device=device)
    table = dm.adaln_t_table.to(device=device, dtype=torch.float32)
    pos_idx = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos_idx.floor().long().clamp(max=table.shape[0] - 2)
    t_emb = torch.lerp(table[i0], table[i0 + 1], (pos_idx - i0).unsqueeze(1))

    # contiguous runs of tags
    segs = []
    a = 0
    for i in range(1, seq + 1):
        if i == seq or tags[i] != tags[a]:
            segs.append((a, i, int(tags[a])))
            a = i

    pos_t = torch.from_numpy(pos.astype(np.float32)).to(device)
    inv = dm.rope.inv_freq.to(device=device, dtype=torch.float32)
    per = pos_t.unsqueeze(-1) * inv.view(1, 1, -1)
    t_f, h_f, w_f = per.unbind(dim=1)
    half = torch.cat((t_f, h_f, w_f), dim=-1)
    angles = torch.cat((half, half), dim=-1)
    rope_freqs = rope_rotation_table(angles, compute)

    want = {0, 23, 35, 45, 46, 47, 48, 49}
    with torch.no_grad():
        for li, block in enumerate(dm.blocks):
            h = block(h, t_emb, segs, rope_freqs)
            arr = h.detach().float().cpu().numpy()
            if li == 0:
                print(f"L0 vs host y.bin rms={rms(arr - host_y):.6g} "
                      f"host={rms(host_y):.6g} comfy={rms(arr):.6g}", flush=True)
            if li in want:
                print(f"comfy-native L{li} x_rms={rms(arr):.6g}", flush=True)
        # final video head on last token (tag 0)
        vid = [i for i, t in enumerate(tags) if t == 0]
        shift, scale = dm.final_layer.adaln_proj(t_emb)
        vrow = 0  # video modality, unique_t index 0
        va, vb = vid[0], vid[0] + 1
        hv = (dm.final_layer.norm(h[va:vb]) * (1.0 + scale[vrow]) + shift[vrow]).float()
        vout = dm.final_layer.video_out(hv).detach().cpu().numpy()
        print(f"comfy-native video_out rms={rms(vout):.6g} shape={vout.shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
