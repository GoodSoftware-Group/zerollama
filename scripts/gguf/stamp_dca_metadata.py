#!/usr/bin/env python3
"""Write a Dual Chunk Attention sidecar JSON next to a GGUF from HF config.json.

Prefer stamping via convert_hf_to_gguf.py (writes GGUF KV directly). This helper
emits a sidecar for operators who already have a GGUF and need the dca.* values
recorded for re-stamp / debugging.

Native path: patched llama-server (llama/patches 0095–0097) reads
{arch}.attention.dca.* and runs DualChunk RoPE + 3× FA. SGLang is the logit
oracle only (docs/dca-dual-chunk-attention.md, scripts/dca_oracle_logits.py).

Usage:
  python3 scripts/gguf/stamp_dca_metadata.py \\
    --hf /path/to/Qwen2.5-14B-Instruct-1M \\
    --gguf /path/to/model.gguf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf", required=True, type=Path, help="HF model dir with config.json")
    ap.add_argument("--gguf", required=True, type=Path, help="GGUF path (sidecar written beside it)")
    ap.add_argument("--arch", default="qwen2", help="GGUF architecture prefix (default qwen2)")
    args = ap.parse_args()

    cfg_path = args.hf / "config.json"
    if not cfg_path.is_file():
        print(f"missing {cfg_path}", file=sys.stderr)
        return 1
    cfg = json.loads(cfg_path.read_text())
    dca = cfg.get("dual_chunk_attention_config")
    if not isinstance(dca, dict) or not dca:
        print("no dual_chunk_attention_config in HF config — nothing to stamp", file=sys.stderr)
        return 2
    if not args.gguf.is_file():
        print(f"missing {args.gguf}", file=sys.stderr)
        return 1

    sidecar = Path(str(args.gguf) + ".dca.json")
    payload = {
        "arch": args.arch,
        "keys": {
            f"{args.arch}.attention.dca.chunk_size": dca.get("chunk_size"),
            f"{args.arch}.attention.dca.local_size": dca.get("local_size"),
            f"{args.arch}.attention.dca.original_context_length": dca.get(
                "original_max_position_embeddings"
            ),
        },
        "hf": str(args.hf.resolve()),
        "gguf": str(args.gguf.resolve()),
        "note": "Sidecar only — re-convert or use gguf-py add_dca_* to embed KV in GGUF",
    }
    sidecar.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
