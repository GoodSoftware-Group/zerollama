#!/usr/bin/env python3
"""Dump MiniMax H3 Qwen3-VL-32B NVFP4 cond via Comfy CLIPLoader math.

Writes H3TE (magic + nt + dim + f32[nt,5120]) for video-cli --text-cond.

  ComfyUI/.venv/bin/python x/video-c/tools/h3_dump_comfy_te32.py \\
    --prompt "A red fox walking through snow" -o /tmp/h3_te32_fox.bin
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

COMFY = Path("/Users/user1/Sites/inference/ComfyUI")
TE_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DIM = 5120


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A red fox walking through snow")
    ap.add_argument("-o", "--out", default="/tmp/h3_te32_fox.bin")
    ap.add_argument("--comfy", default=str(COMFY))
    ap.add_argument("--te", default=TE_NAME)
    ap.add_argument("--device", default="cpu", choices=("cpu", "default"))
    args = ap.parse_args()

    comfy_root = Path(args.comfy).resolve()
    os.chdir(comfy_root)
    sys.path.insert(0, str(comfy_root))

    import torch
    import folder_paths
    import comfy.sd

    te_path = comfy_root / "models" / "text_encoders" / args.te
    if not te_path.is_file():
        print(f"missing TE {te_path}", file=sys.stderr)
        return 2

    model_options = {}
    if args.device == "cpu":
        model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

    print(f"loading CLIP minimax {te_path} device={args.device}", file=sys.stderr)
    clip = comfy.sd.load_clip(
        ckpt_paths=[str(te_path)],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.MINIMAX,
        model_options=model_options,
    )
    tokens = clip.tokenize(args.prompt)
    out = clip.encode_from_tokens(tokens, return_dict=True)
    cond = out["cond"]
    if hasattr(cond, "detach"):
        cond = cond.detach().float().cpu().numpy()
    if cond.ndim == 3:
        cond = cond[0]
    nt, dim = int(cond.shape[0]), int(cond.shape[1])
    if dim != DIM:
        print(f"unexpected dim {dim}", file=sys.stderr)
        return 1
    tags = out.get("minimax_token_tags")
    tag_u8 = None
    if tags is not None:
        if hasattr(tags, "detach"):
            tags = tags.detach().cpu().view(-1).to(torch.int64).numpy()
        tag_u8 = tags.astype("uint8")
        if tag_u8.shape[0] != nt:
            print(f"tag len {tag_u8.shape[0]} != nt {nt}", file=sys.stderr)
            return 1
        n0 = int((tag_u8 == 0).sum())
        n1 = int((tag_u8 == 1).sum())
        print(f"cond nt={nt} dim={dim} rms={float((cond ** 2).mean() ** 0.5):.4g} "
              f"tags video={n0} text={n1}",
              file=sys.stderr)
    else:
        print(f"cond nt={nt} dim={dim} rms={float((cond ** 2).mean() ** 0.5):.4g} tags=none",
              file=sys.stderr)
    blob = struct.pack("<4sII", b"H3TE", nt, dim) + cond.astype("float32").tobytes()
    if tag_u8 is not None:
        blob += tag_u8.tobytes()
    Path(args.out).write_bytes(blob)
    print(f"wrote {args.out} ({len(blob)} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
