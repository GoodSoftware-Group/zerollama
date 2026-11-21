#!/usr/bin/env python3
"""Regenerate checkpoint/ + fixture.gpatch with DAWE-compatible patch_id hashing."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

FORMAT = "geodesic-patch"
VERSION = 1
HASH_CHUNK = 16 << 20
CONFIG_FILES = ("config.json", "generation_config.json")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
)

HERE = Path(__file__).resolve().parent
ROOT = HERE / "checkpoint"


def f32s(vals: list[float]) -> bytes:
    return b"".join(struct.pack("<f", float(v)) for v in vals)


def write_checkpoint() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "config.json").write_text(
        '{"architectures":["TinyForCausalLM"],"model_type":"tiny"}'
    )
    (ROOT / "tokenizer_config.json").write_text('{"model_max_length":128}')
    embed = f32s([i + 0.5 for i in range(8)])
    other = f32s([10 + i for i in range(4)])
    header = {
        "model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [4, 2],
            "data_offsets": [0, len(embed)],
        },
        "other.weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [len(embed), len(embed) + len(other)],
        },
    }
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((-len(hj)) % 8)
    blob = struct.pack("<Q", len(hj)) + hj + embed + other
    (ROOT / "model.safetensors").write_bytes(blob)


def sha256_range(path: Path, start: int, end: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        f.seek(start)
        left = end - start
        while left:
            b = f.read(min(HASH_CHUNK, left))
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def main() -> None:
    write_checkpoint()
    st = ROOT / "model.safetensors"
    data = st.read_bytes()
    n = struct.unpack_from("<Q", data, 0)[0]
    header = json.loads(data[8 : 8 + n])
    data_start = 8 + n
    tensors = {}
    where = {}
    for name, d in header.items():
        dtype, shape = d["dtype"], tuple(d["shape"])
        a, b = d["data_offsets"]
        tensors[name] = (dtype, shape, a, b)
        where[name] = (st, data_start + a, data_start + b)

    names = sorted(tensors)
    layout = hashlib.sha256(
        "\n".join(f"{n}\t{tensors[n][0]}\t{','.join(map(str, tensors[n][1]))}" for n in names).encode()
    ).hexdigest()
    hashes = {n: sha256_range(*where[n]) for n in names}
    fp = hashlib.sha256(
        "\n".join(
            f"{n}\t{tensors[n][0]}\t{','.join(map(str, tensors[n][1]))}\t{hashes[n]}" for n in names
        ).encode()
    ).hexdigest()
    config = {f: sha256_file(ROOT / f) for f in CONFIG_FILES if (ROOT / f).is_file()}
    tokenizer = {f: sha256_file(ROOT / f) for f in TOKENIZER_FILES if (ROOT / f).is_file()}

    name = "model.embed_tokens.weight"
    dtype, shape, _, _ = tensors[name]
    itemsize = 4
    span = where[name]
    payload = data[span[1] : span[2]]
    indices = [2, 3]
    old = b"".join(payload[i * itemsize : (i + 1) * itemsize] for i in indices)
    new = f32s([100.0, 101.0])
    result = bytearray(payload)
    for pos, i in enumerate(indices):
        result[i * itemsize : (i + 1) * itemsize] = new[pos * itemsize : (pos + 1) * itemsize]
    result_sha = hashlib.sha256(result).hexdigest()
    result_hashes = dict(hashes)
    result_hashes[name] = result_sha
    result_fp = hashlib.sha256(
        "\n".join(
            f"{n}\t{tensors[n][0]}\t{','.join(map(str, tensors[n][1]))}\t{result_hashes[n]}"
            for n in names
        ).encode()
    ).hexdigest()

    def pack_idx(idxs: list[int]) -> bytes:
        return struct.pack(f"<{len(idxs)}I", *idxs)

    def payload_digest(idxs: list[int], old_b: bytes, new_b: bytes) -> str:
        h = hashlib.sha256()
        h.update(pack_idx(idxs))
        h.update(old_b)
        h.update(new_b)
        return h.hexdigest()

    manifest = {
        "name": "fixture-embed-row",
        "description": "checked-in weightedit fixture",
        "format": FORMAT,
        "version": VERSION,
        "source": {
            "fingerprint": fp,
            "layout_sha256": layout,
            "n_tensors": len(names),
            "config": config,
            "tokenizer": tokenizer,
        },
        "result": {"fingerprint": result_fp},
        "tensors": [
            {
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "numel": shape[0] * shape[1],
                "changed": len(indices),
                "source_sha256": hashes[name],
                "result_sha256": result_sha,
            }
        ],
        "changed_weights": len(indices),
        "payload_sha256": payload_digest(indices, old, new),
    }
    man_for_id = dict(manifest)
    manifest["patch_id"] = hashlib.sha256(_canonical(man_for_id)).hexdigest()

    tensors_out = {
        f"index/{name}": ("U32", [len(indices)], pack_idx(indices)),
        f"old/{name}": (dtype, [len(indices)], old),
        f"new/{name}": (dtype, [len(indices)], new),
    }
    header_out: dict = {
        "__metadata__": {
            "geodesic_patch": json.dumps(manifest, sort_keys=True),
            "format": f"{FORMAT}/{VERSION}",
        }
    }
    offset = 0
    order = list(tensors_out.keys())
    for n in order:
        dtype_i, shape_i, raw_i = tensors_out[n]
        header_out[n] = {
            "dtype": dtype_i,
            "shape": shape_i,
            "data_offsets": [offset, offset + len(raw_i)],
        }
        offset += len(raw_i)
    blob = json.dumps(header_out, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    out = HERE / "fixture.gpatch"
    with out.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for n in order:
            f.write(tensors_out[n][2])
    print(f"wrote {out} patch_id={manifest['patch_id'][:16]}… size={out.stat().st_size}")


if __name__ == "__main__":
    main()
