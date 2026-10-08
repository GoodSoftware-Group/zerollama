#!/usr/bin/env python3
"""Convert Cloudflare Clef with its trained joint head, using pinned llama.cpp.

Produces an Ollama-compatible GGUF: backbone tensors under the Qwen3.5 arch plus
`clef.*` float32 joint-head tensors and `{arch}.decision.*` metadata that
`llama/clef/clef.cpp` loads at score time.

Usage:
  python3 llama/clef/convert.py --llama-cpp vendor/llama-cpp-b11351 \
    /path/to/Cloudflare/clef-flash --outtype q8_0 --outfile clef-flash.gguf

For the vision projector, add --mmproj --outtype f16 and use a separate outfile.
All remaining arguments are passed to convert_hf_to_gguf.py.

Note: ggml-org/Clef-Flash-GGUF uses a native `clef` arch + `decision.*` tensors
and is not interchangeable with this convert path without a head loader update.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Convert Cloudflare Clef HF → GGUF with joint head (Ollama wire)",
    add_help=True,
)
parser.add_argument(
    "--llama-cpp",
    type=Path,
    required=True,
    help="Pinned llama.cpp tree containing convert_hf_to_gguf.py + conversion/",
)
parser.add_argument(
    "--check-only",
    action="store_true",
    help="Validate model dir + import convert stack; do not write a GGUF",
)
args, remaining = parser.parse_known_args()

llama_cpp = args.llama_cpp.resolve()
if not (llama_cpp / "convert_hf_to_gguf.py").is_file():
    sys.exit(f"error: missing convert_hf_to_gguf.py under {llama_cpp}")
if not (llama_cpp / "conversion" / "qwen.py").is_file():
    sys.exit(f"error: missing conversion/qwen.py under {llama_cpp}")

sys.path[:0] = [str(llama_cpp), str(llama_cpp / "gguf-py")]

# Locate HF model dir (first non-flag positional among remaining).
model_dir: Path | None = None
for i, tok in enumerate(remaining):
    if tok.startswith("-"):
        # skip flag and its value when not --flag=value / boolean-ish
        if "=" not in tok and i + 1 < len(remaining) and not remaining[i + 1].startswith("-"):
            # common convert flags that take a value
            if tok in {
                "--outfile",
                "--outtype",
                "--vocab-type",
                "--concurrency",
                "--remote-hf-model-id",
                "--remote-hf-token",
                "--remote-hf-cache-dir",
            }:
                continue
        continue
    model_dir = Path(tok)
    break

if model_dir is None:
    sys.exit("error: MODEL directory required (Cloudflare/clef-flash checkout)")
if not model_dir.is_dir():
    sys.exit(f"error: model dir not found: {model_dir}")
for req in ("joint_head.safetensors", "joint_head_config.json", "config.json"):
    if not (model_dir / req).is_file():
        sys.exit(f"error: missing {req} in {model_dir}")

cfg = json.loads((model_dir / "joint_head_config.json").read_text())
for key in ("hidden_size", "width", "heads", "routing_layers", "layers"):
    if key not in cfg:
        sys.exit(f"error: joint_head_config.json missing {key}")

try:
    import numpy as np
    from safetensors import safe_open
except ImportError as e:
    sys.exit(f"error: need numpy + safetensors ({e})")

from conversion.qwen import Qwen3_5TextModel
import convert_hf_to_gguf

original_tensors = Qwen3_5TextModel.prepare_tensors
original_metadata = Qwen3_5TextModel.set_gguf_parameters


def prepare_tensors(self):
    original_tensors(self)
    with safe_open(self.dir_model / "joint_head.safetensors", framework="pt", device="cpu") as head:
        for name in head.keys():
            # Keep the small head in float32, including its scalar gates.
            data = head.get_tensor(name).float().numpy()
            self.gguf_writer.add_tensor("clef." + name, np.atleast_1d(data))


def set_gguf_parameters(self):
    original_metadata(self)
    config = json.loads((self.dir_model / "joint_head_config.json").read_text())
    self.gguf_writer.add_string(f"{self.gguf_writer.arch}.decision.type", "clef")
    for name, value in config.items():
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.decision.{name}", int(value))
    # Per-token hidden states for score_fields (LAST pooling collapses the head).
    self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.pooling_type", 0)


Qwen3_5TextModel.prepare_tensors = prepare_tensors
Qwen3_5TextModel.set_gguf_parameters = set_gguf_parameters

if args.check_only:
    print(
        "OK: convert stack imports; model dir valid "
        f"({model_dir}, head hidden={cfg['hidden_size']} width={cfg['width']})"
    )
    raise SystemExit(0)

sys.argv = [sys.argv[0], "--no-mtp", *remaining]
convert_hf_to_gguf.main()
