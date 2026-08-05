#!/usr/bin/env python3
"""Inspect (or optionally rewrite) Ollama gpt-oss GGUF tensor type ids.

IMPORTANT: Do **not** remap type 4 → 39 for blobs loaded via ollama-engine.
Ollama's published gpt-oss:20b uses tensor type **4** to mean MXFP4 in its own
type table (`fs/ggml/type.go`). Upstream llama.cpp uses GGML_TYPE_MXFP4 = 39;
the on-disk bytes for Ollama type 4 are **not** interchangeable with type 39.

zerollama routes gptoss/gpt-oss to ollama-engine (see llm/server.go), which
understands type 4. Remapping to 39 produces garbage logits.

Use --rewrite-to-39 only for experiments with a llama-server build that also
has fused-MoE support (not the default path).
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

GGML_TYPE_MXFP4 = 39
OLLAMA_LEGACY_MXFP4 = 4


def scan(path: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"not GGUF: {path}")
        struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def read_str() -> None:
            (n,) = struct.unpack("<Q", f.read(8))
            f.read(n)

        def skip_val(t: int) -> None:
            if t == 8:
                read_str()
                return
            if t == 9:
                (at,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                if at == 8:
                    for _ in range(n):
                        read_str()
                else:
                    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                    f.read(sizes[at] * n)
                return
            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            f.read(sizes[t])

        for _ in range(n_kv):
            read_str()
            (t,) = struct.unpack("<I", f.read(4))
            skip_val(t)
        for _ in range(n_tensors):
            read_str()
            (ndims,) = struct.unpack("<I", f.read(4))
            f.read(8 * ndims)
            (tt,) = struct.unpack("<I", f.read(4))
            f.read(8)
            counts[tt] = counts.get(tt, 0) + 1
    return counts


def rewrite(path: Path, src: int, dst: int) -> int:
    n = 0
    with path.open("r+b") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"not GGUF: {path}")
        struct.unpack("<I", f.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def read_str() -> None:
            (nlen,) = struct.unpack("<Q", f.read(8))
            f.read(nlen)

        def skip_val(t: int) -> None:
            if t == 8:
                read_str()
                return
            if t == 9:
                (at,) = struct.unpack("<I", f.read(4))
                (nlen,) = struct.unpack("<Q", f.read(8))
                if at == 8:
                    for _ in range(nlen):
                        read_str()
                else:
                    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                    f.read(sizes[at] * nlen)
                return
            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            f.read(sizes[t])

        for _ in range(n_kv):
            read_str()
            (t,) = struct.unpack("<I", f.read(4))
            skip_val(t)
        for _ in range(n_tensors):
            read_str()
            (ndims,) = struct.unpack("<I", f.read(4))
            f.read(8 * ndims)
            pos = f.tell()
            (tt,) = struct.unpack("<I", f.read(4))
            f.read(8)
            if tt == src:
                cur = f.tell()
                f.seek(pos)
                f.write(struct.pack("<I", dst))
                f.seek(cur)
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", type=Path, nargs="+")
    ap.add_argument(
        "--rewrite-to-39",
        action="store_true",
        help="DANGEROUS for ollama-engine: rewrite type 4 → 39",
    )
    ap.add_argument(
        "--rewrite-to-4",
        action="store_true",
        help="Rewrite type 39 → 4 (restore Ollama-engine-native encoding)",
    )
    args = ap.parse_args()
    if args.rewrite_to_39 and args.rewrite_to_4:
        raise SystemExit("pick at most one rewrite flag")
    for p in args.gguf:
        before = scan(p)
        print(f"{p}: types={before}")
        if args.rewrite_to_39:
            n = rewrite(p, OLLAMA_LEGACY_MXFP4, GGML_TYPE_MXFP4)
            print(f"  remapped {n} tensors 4→39; now={scan(p)}")
            print("  WARN: ollama-engine expects type 4 for Ollama gpt-oss blobs", file=sys.stderr)
        elif args.rewrite_to_4:
            n = rewrite(p, GGML_TYPE_MXFP4, OLLAMA_LEGACY_MXFP4)
            print(f"  remapped {n} tensors 39→4; now={scan(p)}")


if __name__ == "__main__":
    main()
