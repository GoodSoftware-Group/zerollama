#!/usr/bin/env python3
"""Convert Wan 1.3B safetensors / pth shards to a minimal GGUF bundle.

Prefers torch+numpy when available; falls back to safetensors+struct writer.

Usage:
  # Full dump (huge — DiT+T5+VAE)
  python3 tools/convert_wan_to_gguf.py --ckpt-dir ~/.zerollama/.../Wan2.1-T2V-1.3B \\
      -o wan_t2v_1.3b.gguf

  # Quality-first slice: DiT blocks 0..4 + patch/text/head (~920 MiB)
  python3 tools/convert_wan_to_gguf.py --ckpt-dir ... -o wan_t2v_1.3b.gguf \\
      --only dit --max-block 4

  # DiT slice + T5 token embedding (adds ~2 GiB as f16)
  python3 tools/convert_wan_to_gguf.py --ckpt-dir ... -o wan_t2v_1.3b.gguf \\
      --only dit,t5-embed --max-block 4
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_gguf(out: Path, tensors: dict[str, tuple[list[int], int, bytes]]) -> None:
    """Write GGUF v3; tensor offsets are relative to the aligned data section.

    Matches llama.cpp / gguf-py: data_offs = data_section_start + relative_offset.
    Shapes are stored in ggml order (innermost dim first = reversed PyTorch shape).
    """
    kv = [("general.architecture", 8, "wan.t2v")]  # STRING=8
    head = bytearray()
    head += struct.pack("<I", GGUF_MAGIC)
    head += struct.pack("<I", GGUF_VERSION)
    head += struct.pack("<Q", len(tensors))
    head += struct.pack("<Q", len(kv))
    for key, typ, val in kv:
        head += pack_string(key)
        head += struct.pack("<I", typ)
        head += pack_string(val)

    align = 32
    items: list[tuple[str, list[int], int, bytes]] = [
        (n, sh, dt, raw) for n, (sh, dt, raw) in tensors.items()
    ]
    info_blobs: list[bytearray] = []
    for name, shape, dtype, _raw in items:
        info = bytearray()
        info += pack_string(name)
        info += struct.pack("<I", len(shape))
        for dim in reversed(shape):
            info += struct.pack("<Q", int(dim))
        info += struct.pack("<I", dtype)
        info += struct.pack("<Q", 0)  # placeholder relative offset
        info_blobs.append(info)

    meta_len = len(head) + sum(len(b) for b in info_blobs)
    pad = (align - (meta_len % align)) % align

    data_blob = bytearray()
    for i, (_name, _shape, _dtype, raw) in enumerate(items):
        while len(data_blob) % align:
            data_blob.append(0)
        rel_off = len(data_blob)
        info_blobs[i][-8:] = struct.pack("<Q", rel_off)
        data_blob += raw

    out.write_bytes(
        bytes(head) + b"".join(info_blobs) + (b"\x00" * pad) + bytes(data_blob)
    )
    print(f"wrote {out} ({len(tensors)} tensors, {len(data_blob)} bytes data)")


def _keep_dit(name: str, max_block: int | None) -> bool:
    """name is without dit. prefix (raw safetensors key)."""
    globals_ok = (
        name.startswith("patch_embedding.")
        or name.startswith("text_embedding.")
        or name.startswith("time_embedding.")
        or name.startswith("time_projection.")
        or name.startswith("head.")
    )
    if globals_ok:
        return True
    m = re.match(r"blocks\.(\d+)\.", name)
    if not m:
        return False
    if max_block is None:
        return True
    return int(m.group(1)) <= max_block


def _keep_t5(name: str, mode: str) -> bool:
    if mode == "t5-embed":
        return name in ("token_embedding.weight", "norm.weight")
    if mode == "t5":
        return True
    return False


def _tensor_bytes(arr, prefer_f16: bool = False) -> tuple[list[int], int, bytes]:
    import numpy as np

    a = np.asarray(arr)
    if prefer_f16 and a.dtype != np.float16:
        a = a.astype(np.float16)
    if a.dtype == np.float16:
        return list(a.shape), GGML_TYPE_F16, a.tobytes()
    a = a.astype(np.float32)
    return list(a.shape), GGML_TYPE_F32, a.tobytes()


def load_with_torch(
    ckpt_dir: Path, only: set[str], max_block: int | None
) -> dict[str, tuple[list[int], int, bytes]]:
    import torch

    tensors: dict[str, tuple[list[int], int, bytes]] = {}
    want_t5 = "t5" in only or "t5-embed" in only
    want_vae = "vae" in only
    want_dit = "dit" in only or not only

    if want_t5 or want_vae:
        for pth in sorted(ckpt_dir.glob("*.pth")):
            is_t5 = "t5" in pth.name.lower()
            is_vae = "vae" in pth.name.lower()
            if is_t5 and not want_t5:
                continue
            if is_vae and not want_vae:
                continue
            if not is_t5 and not is_vae:
                continue
            state = torch.load(pth, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                continue
            prefix = "t5" if is_t5 else "vae"
            mode = "t5-embed" if (is_t5 and "t5-embed" in only and "t5" not in only) else (
                "t5" if is_t5 else "vae"
            )
            for k, v in state.items():
                if not hasattr(v, "detach"):
                    continue
                if is_t5 and not _keep_t5(k, mode):
                    continue
                arr = v.detach().cpu()
                # Keep bf16/f16 embeds as f16 to save ~2× disk.
                prefer_f16 = is_t5 and k == "token_embedding.weight"
                if arr.dtype == torch.float16 or (
                    prefer_f16 and arr.dtype == torch.bfloat16
                ):
                    if arr.dtype == torch.bfloat16:
                        arr = arr.float().half()
                    dtype = GGML_TYPE_F16
                    raw = arr.contiguous().numpy().tobytes()
                    tensors[f"{prefix}.{k}"] = (list(arr.shape), dtype, raw)
                else:
                    arr = arr.float().contiguous()
                    tensors[f"{prefix}.{k}"] = (
                        list(arr.shape),
                        GGML_TYPE_F32,
                        arr.numpy().tobytes(),
                    )

    if want_dit:
        st = ckpt_dir / "diffusion_pytorch_model.safetensors"
        if st.is_file():
            from safetensors.torch import load_file

            state = load_file(str(st))
            for k, v in state.items():
                if not _keep_dit(k, max_block):
                    continue
                arr = v.detach().cpu()
                if arr.dtype == torch.float16:
                    tensors[f"dit.{k}"] = (
                        list(arr.shape),
                        GGML_TYPE_F16,
                        arr.contiguous().numpy().tobytes(),
                    )
                else:
                    arr = arr.float().contiguous()
                    tensors[f"dit.{k}"] = (
                        list(arr.shape),
                        GGML_TYPE_F32,
                        arr.numpy().tobytes(),
                    )
    return tensors


def load_with_safetensors(
    ckpt_dir: Path, only: set[str], max_block: int | None
) -> dict[str, tuple[list[int], int, bytes]]:
    from safetensors import safe_open

    tensors: dict[str, tuple[list[int], int, bytes]] = {}
    want_dit = "dit" in only or not only
    if not want_dit:
        return tensors
    for st in sorted(ckpt_dir.glob("*.safetensors")):
        with safe_open(st, framework="numpy") as f:
            for k in f.keys():
                if not _keep_dit(k, max_block):
                    continue
                arr = f.get_tensor(k)
                shape, dtype, raw = _tensor_bytes(arr)
                tensors[f"dit.{k}"] = (shape, dtype, raw)
    return tensors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, default=Path("wan_t2v_1.3b.gguf"))
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma list: dit, t5, t5-embed, vae (empty = all via torch)",
    )
    ap.add_argument(
        "--max-block",
        type=int,
        default=None,
        help="Include dit.blocks.0 .. dit.blocks.MAX (inclusive). Default: all",
    )
    args = ap.parse_args()
    if not args.ckpt_dir.is_dir():
        print(f"missing ckpt dir: {args.ckpt_dir}", file=sys.stderr)
        raise SystemExit(1)

    only = {p.strip() for p in args.only.split(",") if p.strip()}
    meta = args.ckpt_dir / "config.json"
    if meta.is_file():
        print(json.loads(meta.read_text()).get("model_type", "wan"), file=sys.stderr)
    if only:
        print(f"filter only={sorted(only)} max_block={args.max_block}", file=sys.stderr)

    tensors: dict[str, tuple[list[int], int, bytes]] = {}
    try:
        tensors = load_with_torch(args.ckpt_dir, only, args.max_block)
    except Exception as exc:
        print(f"torch path failed ({exc}); trying safetensors", file=sys.stderr)
        tensors = load_with_safetensors(args.ckpt_dir, only, args.max_block)

    if not tensors:
        print("no tensors found", file=sys.stderr)
        raise SystemExit(1)

    write_gguf(args.out, tensors)


if __name__ == "__main__":
    main()
