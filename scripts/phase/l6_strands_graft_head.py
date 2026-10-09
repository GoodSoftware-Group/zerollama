#!/usr/bin/env python3
"""Graft Strands pointer-head tensors + decision meta onto a Qwen3.5 GGUF.

Product path: fabricant451 strands Q8 backbone (decision.type=strands, no head)
+ StrandsAgents head.safetensors + hobson_config.json.

Synth path: --synth grafts a random head for tip wire smoke.

Usage:
  python3 scripts/phase/l6_strands_graft_head.py \\
    -i /root/models/strands-hobson-v19-gguf/strands-q8_0.gguf \\
    --head /root/models/strands-hobson-v19/head.safetensors \\
    --config /root/models/strands-hobson-v19/hobson_config.json \\
    -o /root/models/strands-hobson-v19-gguf/strands-ollama-q8_0.gguf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO / "vendor/llama-cpp-b11351/gguf-py"),
    str(REPO / "llama/llama.cpp/gguf-py"),
]

from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType  # noqa: E402


def _contents(reader: GGUFReader, key: str):
    field = reader.get_field(key)
    if field is None:
        raise KeyError(key)
    return field.contents()


def _copy_kv(reader: GGUFReader, writer: GGUFWriter) -> None:
    skip_keys = {
        "general.architecture",
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
    }
    for key, field in reader.fields.items():
        if key in skip_keys or key.startswith("GGUF.") or ".decision." in key:
            continue
        if key.endswith(".pooling_type"):
            continue
        # Published packs set embedding_length_out=512 for an unused cls head;
        # pointer head needs full backbone width (same as embedding_length).
        if key.endswith(".embedding_length_out"):
            continue
        try:
            contents = field.contents()
            if field.types[0] == GGUFValueType.ARRAY:
                subtype = field.types[1]
                if subtype == GGUFValueType.STRING:
                    writer.add_array(key, [str(x) for x in contents])
                elif subtype in (GGUFValueType.INT32, GGUFValueType.UINT32, GGUFValueType.INT64):
                    writer.add_array(key, [int(x) for x in contents])
                elif subtype == GGUFValueType.FLOAT32:
                    writer.add_array(key, [float(x) for x in contents])
                elif subtype == GGUFValueType.BOOL:
                    writer.add_array(key, [bool(x) for x in contents])
                else:
                    print(f"  skip array kv {key} subtype={subtype}")
                continue
            vtype = field.types[0]
            if vtype == GGUFValueType.STRING:
                writer.add_string(key, str(contents))
            elif vtype == GGUFValueType.UINT32:
                writer.add_uint32(key, int(contents))
            elif vtype == GGUFValueType.INT32:
                writer.add_int32(key, int(contents))
            elif vtype == GGUFValueType.UINT64:
                writer.add_uint64(key, int(contents))
            elif vtype == GGUFValueType.INT64:
                writer.add_int64(key, int(contents))
            elif vtype == GGUFValueType.FLOAT32:
                writer.add_float32(key, float(contents))
            elif vtype == GGUFValueType.FLOAT64:
                writer.add_float64(key, float(contents))
            elif vtype == GGUFValueType.BOOL:
                writer.add_bool(key, bool(contents))
            else:
                print(f"  skip kv {key} type={vtype}")
        except Exception as e:
            print(f"  skip kv {key}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--head", type=Path, help="head.safetensors (norm/q/k FP32)")
    ap.add_argument("--config", type=Path, help="hobson_config.json / strands_decider_config.json")
    ap.add_argument("--synth", action="store_true", help="random head for wire smoke")
    ap.add_argument("--pointer-dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.synth and (args.head is None or args.config is None):
        sys.exit("error: --head and --config required unless --synth")

    reader = GGUFReader(str(args.input))
    arch = str(_contents(reader, "general.architecture"))
    hidden = int(_contents(reader, f"{arch}.embedding_length"))

    cfg: dict = {
        "pointer_dim": args.pointer_dim,
        "temperature": 1.0,
        "temperature_by_kind": {},
        "ordinal_smoothing": 0.0,
        "head_type": "pointer",
    }
    if args.config:
        cfg.update(json.loads(args.config.read_text()))
    pointer_dim = int(cfg.get("pointer_dim") or args.pointer_dim)
    if cfg.get("head_type") not in (None, "pointer"):
        sys.exit(f"error: unsupported head_type {cfg.get('head_type')}")

    weights: dict[str, np.ndarray] = {}
    if args.synth:
        rng = np.random.default_rng(args.seed)
        weights["strands.norm.weight"] = np.ones((hidden,), dtype=np.float32)
        weights["strands.norm.bias"] = np.zeros((hidden,), dtype=np.float32)
        weights["strands.q.weight"] = rng.normal(0, 0.02, size=(pointer_dim, hidden)).astype(np.float32)
        weights["strands.q.bias"] = np.zeros((pointer_dim,), dtype=np.float32)
        weights["strands.k.weight"] = rng.normal(0, 0.02, size=(pointer_dim, hidden)).astype(np.float32)
        weights["strands.k.bias"] = np.zeros((pointer_dim,), dtype=np.float32)
    else:
        from safetensors import safe_open

        with safe_open(args.head, framework="pt") as f:
            for name in (
                "norm.weight",
                "norm.bias",
                "q.weight",
                "q.bias",
                "k.weight",
                "k.bias",
            ):
                t = f.get_tensor(name).float().numpy()
                weights["strands." + name] = np.ascontiguousarray(t, dtype=np.float32)
        if weights["strands.norm.weight"].shape != (hidden,):
            sys.exit(
                f"error: head hidden {weights['strands.norm.weight'].shape} != GGUF embedding {hidden}"
            )
        if weights["strands.q.weight"].shape != (pointer_dim, hidden):
            sys.exit(
                f"error: q.weight shape {weights['strands.q.weight'].shape} != ({pointer_dim}, {hidden})"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(args.output), arch=arch)
    _copy_kv(reader, writer)

    prefix = f"{arch}.decision"
    writer.add_string(f"{prefix}.type", "strands")
    writer.add_uint32(f"{prefix}.pointer_dim", pointer_dim)
    writer.add_float32(f"{prefix}.temperature", float(cfg.get("temperature") or 1.0))
    tbk = cfg.get("temperature_by_kind") or {}
    for kind in ("noul", "choice", "score"):
        if kind in tbk:
            writer.add_float32(f"{prefix}.temperature.{kind}", float(tbk[kind]))
    if "ordinal_smoothing" in cfg:
        writer.add_float32(f"{prefix}.ordinal_smoothing", float(cfg["ordinal_smoothing"]))
    writer.add_uint32(f"{arch}.pooling_type", 0)
    writer.add_uint32(f"{arch}.embedding_length_out", hidden)

    for tensor in reader.tensors:
        # Drop unused bert-style cls.* (not in tip qwen35 weight map) and any
        # prior strands/bare head tensors before re-adding the pointer head.
        if (
            tensor.name.startswith("strands.")
            or tensor.name.startswith("cls.")
            or tensor.name
            in {
                "norm.weight",
                "norm.bias",
                "q.weight",
                "q.bias",
                "k.weight",
                "k.bias",
            }
        ):
            continue
        writer.add_tensor(
            tensor.name,
            np.asarray(tensor.data),
            raw_dtype=GGMLQuantizationType(tensor.tensor_type),
        )
    for name, data in weights.items():
        writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    check = GGUFReader(str(args.output))
    dtype = str(_contents(check, f"{arch}.decision.type"))
    n = sum(1 for t in check.tensors if t.name.startswith("strands."))
    assert dtype == "strands" and n == 6, (dtype, n)
    print(f"OK: {arch}.decision.type=strands strands_tensors={n} pointer_dim={pointer_dim} -> {args.output}")


if __name__ == "__main__":
    main()
