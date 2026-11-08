#!/usr/bin/env python3
"""Inspect a GLB for room-PBR fields, or run a self-test with no args.

Usage:
  python3 scripts/mesh/test_glb_inspect.py                 # synthetic self-test
  python3 scripts/mesh/test_glb_inspect.py path/to/file.glb
  python3 scripts/mesh/test_glb_inspect.py --require-pbr path.glb
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import tempfile
from pathlib import Path


def inspect_glb(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 12 or data[0:4] != b"glTF":
        return {"error": "not a GLB", "path": str(path)}
    _magic, version, length = struct.unpack_from("<III", data, 0)
    chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != 0x4E4F534A:
        return {"error": "missing JSON chunk", "version": version, "length": length, "path": str(path)}
    js = json.loads(data[20 : 20 + chunk_len].decode("utf-8"))
    mats = js.get("materials") or []
    has_base = False
    has_mr = False
    alpha_modes = []
    mime_types = []
    for m in mats:
        pbr = m.get("pbrMetallicRoughness") or {}
        if "baseColorTexture" in pbr:
            has_base = True
        if "metallicRoughnessTexture" in pbr:
            has_mr = True
        alpha_modes.append(m.get("alphaMode", "OPAQUE"))
    for im in js.get("images") or []:
        mime_types.append(im.get("mimeType") or im.get("uri", "")[:32])
    accessors = js.get("accessors") or []
    triangles = 0
    for mesh in js.get("meshes") or []:
        for prim in mesh.get("primitives") or []:
            idx = prim.get("indices")
            if idx is not None and idx < len(accessors):
                triangles += int(accessors[idx].get("count", 0)) // 3
    return {
        "path": str(path),
        "materials": len(mats),
        "baseColorTexture": has_base,
        "metallicRoughnessTexture": has_mr,
        "alphaModes": alpha_modes,
        "meshes": len(js.get("meshes") or []),
        "triangles": triangles,
        "imageMimeTypes": mime_types,
        "bytes": length,
    }


def _write_minimal_glb(path: Path, *, with_pbr: bool, alpha_mode: str = "BLEND") -> None:
    """Tiny valid GLB with optional PBR texture refs (no image bytes needed for JSON inspect)."""
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max": [1, 1, 0], "min": [0, 0, 0]},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
        ],
        "buffers": [{"byteLength": 42}],
        "materials": [],
    }
    if with_pbr:
        gltf["materials"] = [
            {
                "alphaMode": alpha_mode,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1},
                },
            }
        ]
        gltf["textures"] = [{"source": 0}, {"source": 1}]
        gltf["images"] = [{"mimeType": "image/png", "uri": "data:,"}, {"mimeType": "image/png", "uri": "data:,"}]
        gltf["meshes"][0]["primitives"][0]["material"] = 0

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4:
        json_bytes += b" "
    bin_blob = b"\x00" * 42
    while len(bin_blob) % 4:
        bin_blob += b"\x00"
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
    out = bytearray()
    out += struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(json_bytes), 0x4E4F534A)
    out += json_bytes
    out += struct.pack("<II", len(bin_blob), 0x004E4942)
    out += bin_blob
    path.write_bytes(out)


def _self_test() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        good = td / "good.glb"
        bad = td / "bad.glb"
        _write_minimal_glb(good, with_pbr=True, alpha_mode="MASK")
        _write_minimal_glb(bad, with_pbr=False)
        g = inspect_glb(good)
        b = inspect_glb(bad)
        assert g["baseColorTexture"] and g["metallicRoughnessTexture"], g
        assert g["alphaModes"] == ["MASK"], g
        assert not b["baseColorTexture"] and not b["metallicRoughnessTexture"], b
        print("test_glb_inspect: PASS", g)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb", nargs="*", help="GLB path(s) to inspect")
    ap.add_argument("--require-pbr", action="store_true", help="Exit 1 if baseColor or metallicRoughness missing")
    ap.add_argument(
        "--require-png",
        action="store_true",
        help="Exit 1 unless every image mimeType is image/png (and at least one image exists)",
    )
    ap.add_argument("--self-test", action="store_true", help="Run synthetic unit test (default with no paths)")
    args = ap.parse_args()

    if args.self_test or not args.glb:
        return _self_test()

    rc = 0
    for raw in args.glb:
        path = Path(raw).expanduser()
        if not path.is_file():
            print(f"error: not a file: {path}", file=sys.stderr)
            rc = 2
            continue
        report = inspect_glb(path)
        print(json.dumps(report, indent=2))
        if args.require_pbr and (
            not report.get("baseColorTexture") or not report.get("metallicRoughnessTexture")
        ):
            print(f"error: missing PBR textures: {path}", file=sys.stderr)
            rc = 1
        if args.require_png:
            mimes = report.get("imageMimeTypes") or []
            if not mimes or any(m != "image/png" for m in mimes):
                print(f"error: expected PNG image mime types, got {mimes}: {path}", file=sys.stderr)
                rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
