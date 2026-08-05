#!/usr/bin/env python3
"""Build a tiny JSON index for T5 embed tensors inside a torch .pth zip.

Does not copy weights — only records absolute zip_offset / nbytes / dtype / shape
so wan-c can mmap the .pth as-is.

  python3 tools/build_t5_embed_index.py \\
    ~/.zerollama/third_party/wan/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth \\
    -o indices/t5_embed_index.json
"""
from __future__ import annotations

import argparse
import json
import struct
import zipfile
from pathlib import Path

import torch


def zip_entry_data_abs(zpath: Path, member: str) -> tuple[int, int]:
    z = zipfile.ZipFile(zpath)
    info = z.getinfo(member)
    with open(zpath, "rb") as f:
        f.seek(info.header_offset + 26)
        nlen, elen = struct.unpack("<HH", f.read(4))
        data_start = info.header_offset + 30 + nlen + elen
        if info.compress_type != 0:
            raise SystemExit(f"compressed member not supported: {member}")
        return data_start, info.file_size


def tensor_raw_prefix(t: torch.Tensor, n: int = 64) -> bytes:
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes()[:n]
    if t.dtype == torch.float16:
        return t.numpy().tobytes()[:n]
    if t.dtype == torch.float32:
        return t.numpy().tobytes()[:n]
    return t.view(torch.uint8).numpy().tobytes()[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pth", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument(
        "--keys",
        default="token_embedding.weight,norm.weight",
        help="Comma-separated state_dict keys (no t5. prefix)",
    )
    args = ap.parse_args()
    want = {k.strip() for k in args.keys.split(",") if k.strip()}
    state = torch.load(args.pth, map_location="cpu", weights_only=True, mmap=True)
    z = zipfile.ZipFile(args.pth)
    out: dict = {}
    for k, t in state.items():
        if k not in want or not torch.is_tensor(t):
            continue
        nbytes = t.numel() * t.element_size()
        raw = tensor_raw_prefix(t)
        matched = None
        cands = [
            i
            for i in z.infolist()
            if "/data/" in i.filename
            and i.file_size == nbytes
            and i.compress_type == 0
        ]
        for info in cands:
            abs_off, sz = zip_entry_data_abs(args.pth, info.filename)
            with open(args.pth, "rb") as f:
                f.seek(abs_off)
                head = f.read(min(64, nbytes))
            if head[:16] == raw[:16]:
                matched = (info.filename, abs_off, sz)
                break
        if not matched and len(cands) == 1:
            abs_off, sz = zip_entry_data_abs(args.pth, cands[0].filename)
            matched = (cands[0].filename, abs_off, sz)
        if not matched:
            raise SystemExit(f"failed to locate storage for {k}")
        out[f"t5.{k}"] = {
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "zip_offset": matched[1],
            "nbytes": matched[2],
            "member": matched[0],
        }
        print("OK", f"t5.{k}", out[f"t5.{k}"]["dtype"], out[f"t5.{k}"]["shape"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
