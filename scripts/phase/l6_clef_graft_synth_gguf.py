#!/usr/bin/env python3
"""Graft a synthetic Clef joint head onto a small GGUF for tip wire smoke.

WHY: Cloudflare Clef is 27B / Flash ~9B; bartowski GGUFs omit the joint head.
This builds a lab-only model that exercises tip llama-server score_fields +
clef_head without downloading production weights. Logits are not meaningful.

Usage:
  python3 scripts/phase/l6_clef_graft_synth_gguf.py \\
    -i /root/models/tiny-agent/Tiny-Agent-a-0.5B.F16.gguf \\
    -o /tmp/clef-synth-tiny.gguf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO / "vendor/llama-cpp-b11351/gguf-py"),
    str(REPO / "llama/llama.cpp/gguf-py"),
]

from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType  # noqa: E402


def _field_contents(reader: GGUFReader, key: str):
    field = reader.get_field(key)
    if field is None:
        raise KeyError(key)
    return field.contents()


def _arch(reader: GGUFReader) -> str:
    return str(_field_contents(reader, "general.architecture"))


def _hidden(reader: GGUFReader, arch: str) -> int:
    for key in (
        f"{arch}.embedding_length",
        f"{arch}.hidden_size",
        "general.embedding_length",
    ):
        field = reader.get_field(key)
        if field is not None:
            return int(field.contents())
    raise SystemExit(f"cannot find embedding length for arch={arch}")


def _tensor_data(reader: GGUFReader, name: str) -> np.ndarray:
    for t in reader.tensors:
        if t.name == name:
            return np.asarray(t.data)
    raise KeyError(name)


def _add_norm(weights: dict[str, np.ndarray], key: str, dim: int, rng: np.random.Generator):
    weights[f"{key}.weight"] = rng.normal(0, 0.02, size=(dim,)).astype(np.float32)
    weights[f"{key}.bias"] = np.zeros((dim,), dtype=np.float32)


def _add_linear(
    weights: dict[str, np.ndarray],
    key: str,
    out_f: int,
    in_f: int,
    rng: np.random.Generator,
    bias: bool = False,
):
    weights[f"{key}.weight"] = rng.normal(0, 0.02, size=(out_f, in_f)).astype(np.float32)
    if bias:
        weights[f"{key}.bias"] = np.zeros((out_f,), dtype=np.float32)


def _add_attention(weights: dict[str, np.ndarray], key: str, width: int, rng: np.random.Generator):
    weights[f"{key}.in_proj_weight"] = rng.normal(0, 0.02, size=(3 * width, width)).astype(np.float32)
    weights[f"{key}.in_proj_bias"] = np.zeros((3 * width,), dtype=np.float32)
    _add_linear(weights, f"{key}.out_proj", width, width, rng, bias=True)


def synth_clef_weights(
    hidden: int,
    width: int,
    heads: int,
    layers: int,
    routing_layers: int,
    feedforward: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if width % heads != 0:
        raise SystemExit("width must be divisible by heads")
    rng = np.random.default_rng(seed)
    w: dict[str, np.ndarray] = {}
    _add_norm(w, "hidden_norm", hidden, rng)
    _add_linear(w, "memory_projection", width, hidden, rng)
    _add_linear(w, "option_context_projection", width, hidden, rng)
    _add_linear(w, "option_lexical_projection", width, hidden, rng)
    _add_linear(w, "option_question_projection", width, hidden, rng)
    _add_linear(w, "question_projection", width, hidden, rng)
    _add_linear(w, "global_projection", width, hidden, rng)
    _add_norm(w, "option_summary_norm", width, rng)
    _add_norm(w, "field_norm", width, rng)
    _add_norm(w, "option_norm", width, rng)
    w["type_embedding.weight"] = rng.normal(0, 0.02, size=(3, width)).astype(np.float32)
    w["prior_logit_scale"] = np.array([0.0], dtype=np.float32)
    w["joint_logit_scale"] = np.array([0.0], dtype=np.float32)
    w["residual_gate"] = np.array([0.0], dtype=np.float32)
    _add_linear(w, "residual_scorer.0", feedforward, 4 * width, rng, bias=True)
    _add_linear(w, "residual_scorer.3", 1, feedforward, rng, bias=True)

    for i in range(routing_layers):
        key = f"evidence_layers.{i}"
        _add_norm(w, f"{key}.query_norm", width, rng)
        _add_norm(w, f"{key}.memory_norm", width, rng)
        _add_attention(w, f"{key}.attention", width, rng)
        _add_norm(w, f"{key}.feedforward_norm", width, rng)
        _add_linear(w, f"{key}.feedforward.0", feedforward, width, rng, bias=True)
        _add_linear(w, f"{key}.feedforward.3", width, feedforward, rng, bias=True)

    for i in range(layers):
        key = f"layers.{i}"
        _add_norm(w, f"{key}.norm1", width, rng)
        _add_norm(w, f"{key}.norm2", width, rng)
        _add_norm(w, f"{key}.norm3", width, rng)
        _add_attention(w, f"{key}.self_attn", width, rng)
        _add_attention(w, f"{key}.multihead_attn", width, rng)
        _add_linear(w, f"{key}.linear1", feedforward, width, rng, bias=True)
        _add_linear(w, f"{key}.linear2", width, feedforward, rng, bias=True)
    return w


def graft(src: Path, dst: Path, width: int, heads: int, layers: int, routing: int, ff: int, seed: int) -> None:
    reader = GGUFReader(str(src))
    arch = _arch(reader)
    hidden = _hidden(reader, arch)
    print(f"source arch={arch} hidden={hidden} tensors={len(reader.tensors)}")

    writer = GGUFWriter(str(dst), arch=arch)

    # Copy scalar/string/array metadata. Skip writer-owned / duplicate keys.
    skip_keys = {
        "general.architecture",
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
    }
    for key, field in reader.fields.items():
        if key in skip_keys or key.startswith("GGUF.") or ".decision." in key:
            continue
        try:
            contents = field.contents()
            if field.types[0] == GGUFValueType.ARRAY:
                subtype = field.types[1]
                if subtype == GGUFValueType.STRING:
                    writer.add_array(key, [str(x) for x in contents])
                elif subtype == GGUFValueType.INT32:
                    writer.add_array(key, [int(x) for x in contents])
                elif subtype == GGUFValueType.UINT32:
                    writer.add_array(key, [int(x) for x in contents])
                elif subtype == GGUFValueType.FLOAT32:
                    writer.add_array(key, [float(x) for x in contents])
                elif subtype == GGUFValueType.INT64:
                    writer.add_array(key, [int(x) for x in contents])
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

    prefix = f"{arch}.decision"
    writer.add_string(f"{prefix}.type", "clef")
    for name, value in (
        ("hidden_size", hidden),
        ("width", width),
        ("heads", heads),
        ("layers", layers),
        ("routing_layers", routing),
        ("feedforward", ff),
    ):
        writer.add_uint32(f"{prefix}.{name}", int(value))
    # NONE (0) — marks GGUF as embedding-capable so Go launch passes --embedding
    # for score_fields without requiring a separate Modelfile flag.
    writer.add_uint32(f"{arch}.pooling_type", 0)

    has_output = any(t.name == "output.weight" for t in reader.tensors)
    for t in reader.tensors:
        writer.add_tensor(t.name, np.asarray(t.data), raw_dtype=GGMLQuantizationType(t.tensor_type))

    if not has_output:
        for t in reader.tensors:
            if t.name == "token_embd.weight":
                print(f"  synthesizing output.weight from token_embd.weight dtype={t.tensor_type}")
                writer.add_tensor(
                    "output.weight",
                    np.asarray(t.data),
                    raw_dtype=GGMLQuantizationType(t.tensor_type),
                )
                break
        else:
            raise SystemExit("no token_embd.weight to synthesize output.weight")

    clef = synth_clef_weights(hidden, width, heads, layers, routing, ff, seed)
    for name, arr in sorted(clef.items()):
        # gguf stores 2d as (ne0=cols, ne1=rows) via shape; 1d ok
        writer.add_tensor(f"clef.{name}", arr.astype(np.float32, copy=False))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"OK wrote {dst} (+{len(clef)} clef tensors, width={width})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--routing-layers", type=int, default=1)
    ap.add_argument("--feedforward", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    graft(
        args.input,
        args.output,
        width=args.width,
        heads=args.heads,
        layers=args.layers,
        routing=args.routing_layers,
        ff=args.feedforward,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
