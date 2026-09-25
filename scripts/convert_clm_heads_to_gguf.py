#!/usr/bin/env python3
"""Convert Contrastive-LM projection heads (.pt) to a heads-only GGUF.

One-shot tooling (not a serve dependency). Runtime inference is Go:
llm/clm_*.go over llama-server /v1/embeddings — no Torch at serve time.

  PYTHONPATH=../llama.cpp/gguf-py \\
    python3 scripts/convert_clm_heads_to_gguf.py \\
      --ckpt ~/.cache/clm/CLM_v0.1-8B.pt \\
      --outfile ~/.cache/clm/CLM_v0.1-8B.gguf

WHY custom (not convert_hf_to_gguf): checkpoint is a torch.save dict with
state_head / action_head Linear MLPs, not an HF transformer tree.
Docs: docs/clm.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError as e:
    raise SystemExit(
        "gguf package required — set PYTHONPATH to sibling llama.cpp/gguf-py "
        f"(e.g. PYTHONPATH=../llama.cpp/gguf-py). Import error: {e}"
    ) from e

try:
    import torch
except ImportError as e:
    raise SystemExit(f"torch required for one-shot convert: {e}") from e


def _f32(t) -> np.ndarray:
    return t.detach().cpu().float().contiguous().numpy()


def _add_head(writer: gguf.GGUFWriter, prefix: str, sd: dict) -> None:
    # PyTorch Linear weight is (out, in); store as-is for Go y = x @ W.T + b.
    for key, tensor in sd.items():
        name = f"clm.{prefix}.{key}"
        arr = _f32(tensor)
        writer.add_tensor(name, arr)


def convert(ckpt: Path, outfile: Path) -> None:
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = dict(ck.get("cfg") or {})
    hidden = int(ck.get("hidden_size", cfg.get("hidden_size", 4096)))
    width = int(cfg["width"])
    depth = int(cfg["depth"])
    proj = int(ck.get("projection_dim", cfg.get("projection_dim", 512)))
    activation = str(cfg.get("activation", "gelu"))
    layernorm = bool(cfg.get("layernorm", False))
    residual = bool(cfg.get("residual", False))
    logit_scale = float(torch.as_tensor(ck["logit_scale"]).float().item())

    writer = gguf.GGUFWriter(str(outfile), arch="clm")
    writer.add_name("clm-heads")
    writer.add_description("Contrastive-LM projection heads (state + action); encoder is separate")
    writer.add_uint32("clm.hidden_size", hidden)
    writer.add_uint32("clm.width", width)
    writer.add_uint32("clm.depth", depth)
    writer.add_uint32("clm.projection_dim", proj)
    writer.add_string("clm.activation", activation)
    writer.add_bool("clm.layernorm", layernorm)
    writer.add_bool("clm.residual", residual)
    # Store pre-exp logit_scale (InfoNCE); Go applies exp().clamp(max=100).
    writer.add_float32("clm.logit_scale", logit_scale)
    try:
        writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    except Exception:
        pass

    _add_head(writer, "state", ck["state_head"])
    _add_head(writer, "action", ck["action_head"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {outfile} (hidden={hidden} width={width} depth={depth} proj={proj} scale={logit_scale})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True, help="CLM_v0.1-8B.pt (or compatible)")
    ap.add_argument("--outfile", type=Path, required=True, help="output .gguf path")
    args = ap.parse_args()
    if not args.ckpt.is_file():
        raise SystemExit(f"missing checkpoint: {args.ckpt}")
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    convert(args.ckpt, args.outfile)


if __name__ == "__main__":
    main()
