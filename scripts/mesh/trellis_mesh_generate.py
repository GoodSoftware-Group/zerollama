#!/usr/bin/env python3
"""TRELLIS.2 image → PBR GLB (lab wrapper for astra / one 4090).

Generate, delete the pipeline, then bake. Room defaults: ~100k faces, texture
1024, PNG. Sets alphaMode from the alpha channel (stock to_glb leaves OPAQUE).

Lab-only: uses stock o_voxel.postprocess.to_glb (nvdiffrast). Do not put this
venv on the zerollama serve LD_LIBRARY_PATH.

Examples:
  source ~/.zerollama/third_party/trellis/env.sh
  python scripts/mesh/trellis_mesh_generate.py --image still.png --out trellis2.glb
  python scripts/mesh/trellis_mesh_generate.py --image still.png --resolution 1024 \\
      --decimation 100000 --texture-size 1024 --note-out note.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _progress(msg: str) -> None:
    print(f"PROGRESS: {msg}", flush=True)


def _ensure_import_path() -> Path:
    root = Path(os.environ.get("TRELLIS_ROOT", Path.home() / ".zerollama/third_party/trellis"))
    src = root / "TRELLIS.2"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def _install_dinov3_mirror() -> str:
    """Route gated facebook/dinov3 → public mirror (or local path via DINOV3_MODEL).

    Official facebook/dinov3-vitl16-pretrain-lvd1689m is gated. Lab default is the
    camenduru mirror with the same weights; DINOv3 license still applies.

    Also patches extract_features for transformers≥5 where encoder layers live at
    ``model.model.layer`` instead of ``model.layer``.
    """
    mirror = os.environ.get("DINOV3_MODEL", "camenduru/dinov3-vitl16-pretrain-lvd1689m")
    import torch
    import torch.nn.functional as F
    import trellis2.modules.image_feature_extractor as ife

    orig = ife.DinoV3FeatureExtractor.__init__

    def _init(self, model_name: str, image_size=512):  # type: ignore[no-untyped-def]
        if "dinov3" in model_name.lower() and not Path(model_name).expanduser().exists():
            model_name = mirror
        return orig(self, model_name, image_size=image_size)

    def _extract_features(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore[no-untyped-def]
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)
        layers = getattr(self.model, "layer", None)
        if layers is None:
            layers = self.model.model.layer
        for layer_module in layers:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )
        return F.layer_norm(hidden_states, hidden_states.shape[-1:])

    ife.DinoV3FeatureExtractor.__init__ = _init  # type: ignore[method-assign]
    ife.DinoV3FeatureExtractor.extract_features = _extract_features  # type: ignore[method-assign]
    return mirror


def _install_rembg_stub_or_mirror(*, stub: bool) -> str:
    """Avoid gated briaai/RMBG-2.0 when loading the pipeline.

    Pipeline.json pins RMBG-2.0. That is a different model from
    ZhengPeng7/BiRefNet (not a weight mirror). Lab stills are usually already
    rembg'd RGBA, so we stub rembg when ``stub`` is true and never download
    either network. With ``--preprocess-image``, remap to ``RMBG_MODEL``
    (default BiRefNet) — quality differs from stock RMBG-2.0.
    """
    from trellis2.pipelines.rembg.BiRefNet import BiRefNet

    if stub:

        def _init_stub(self, model_name: str = "stub"):  # type: ignore[no-untyped-def]
            self.model = None
            self.transform_image = None

        def _call_stub(self, image):  # type: ignore[no-untyped-def]
            return image

        def _noop(self, *a, **k):  # type: ignore[no-untyped-def]
            return self

        BiRefNet.__init__ = _init_stub  # type: ignore[method-assign]
        BiRefNet.__call__ = _call_stub  # type: ignore[method-assign]
        BiRefNet.to = _noop  # type: ignore[method-assign]
        BiRefNet.cuda = _noop  # type: ignore[method-assign]
        BiRefNet.cpu = _noop  # type: ignore[method-assign]
        return "stub(skip; still has alpha)"

    mirror = os.environ.get("RMBG_MODEL", "ZhengPeng7/BiRefNet")
    orig = BiRefNet.__init__

    def _init(self, model_name: str = "ZhengPeng7/BiRefNet"):  # type: ignore[no-untyped-def]
        name = model_name or mirror
        if "briaai" in name.lower() or name.upper().startswith("RMBG"):
            name = mirror
        return orig(self, model_name=name)

    BiRefNet.__init__ = _init  # type: ignore[method-assign]
    return mirror


def _image_has_useful_alpha(image) -> bool:
    """True when the still already carries a non-trivial alpha cutout."""
    import numpy as np

    if getattr(image, "mode", "") != "RGBA":
        return False
    alpha = np.asarray(image.split()[-1], dtype="float32") / 255.0
    return float(alpha.min()) < 0.99 and float(alpha.mean()) < 0.999


def _alpha_mode_from_material(mesh) -> str:
    """Return MASK/BLEND/OPAQUE from baked alpha texture statistics."""
    try:
        mat = mesh.visual.material
    except Exception:
        return "OPAQUE"
    tex = getattr(mat, "baseColorTexture", None)
    if tex is None:
        return "OPAQUE"
    import numpy as np

    arr = np.asarray(tex)
    if arr.ndim != 3 or arr.shape[2] < 4:
        return "OPAQUE"
    alpha = arr[:, :, 3].astype("float32") / 255.0
    amin = float(alpha.min())
    amean = float(alpha.mean())
    if amin >= 0.99:
        return "OPAQUE"
    # Mostly binary cutouts → MASK; soft edges → BLEND.
    if amean > 0.85 and amin < 0.1:
        return "MASK"
    return "BLEND"


def _apply_room_material(mesh, alpha_mode: str) -> None:
    try:
        mat = mesh.visual.material
    except Exception:
        return
    if hasattr(mat, "alphaMode"):
        mat.alphaMode = alpha_mode


def _inspect_glb(path: Path) -> dict:
    """Lightweight glTF report without requiring a GPU."""
    import struct

    data = path.read_bytes()
    if len(data) < 12 or data[0:4] != b"glTF":
        return {"error": "not a GLB"}
    # GLB: 12-byte header, then JSON chunk
    _magic, version, length = struct.unpack_from("<III", data, 0)
    chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != 0x4E4F534A:  # JSON
        return {"error": "missing JSON chunk", "version": version, "length": length}
    js = json.loads(data[20 : 20 + chunk_len].decode("utf-8"))
    mats = js.get("materials") or []
    has_base = False
    has_mr = False
    alpha_modes = []
    for m in mats:
        pbr = m.get("pbrMetallicRoughness") or {}
        if "baseColorTexture" in pbr:
            has_base = True
        if "metallicRoughnessTexture" in pbr:
            has_mr = True
        alpha_modes.append(m.get("alphaMode", "OPAQUE"))
    return {
        "materials": len(mats),
        "baseColorTexture": has_base,
        "metallicRoughnessTexture": has_mr,
        "alphaModes": alpha_modes,
        "meshes": len(js.get("meshes") or []),
        "bytes": length,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="Object PNG (prefer alpha / rembg'd)")
    ap.add_argument("--out", default="trellis2.glb", help="Output GLB path")
    ap.add_argument("--resolution", type=int, default=1024, choices=(512, 1024, 1536))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--decimation", type=int, default=100_000, help="Target face count (~100k room default)")
    ap.add_argument("--texture-size", type=int, default=1024)
    ap.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    ap.add_argument(
        "--low-vram",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pipeline.low_vram (default on for resolution>=1024 on <=24GB cards)",
    )
    ap.add_argument("--no-remesh", action="store_true")
    ap.add_argument("--preprocess-image", action="store_true", help="Run pipeline rembg if image has no alpha")
    ap.add_argument("--note-out", default="", help="Write JSON note (seed, VRAM, textures)")
    ap.add_argument("--skip-bake", action="store_true", help="Generate only (debug)")
    args = ap.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.is_file():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 2

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Sparse attention needs xformers or flash_attn (sdpa is dense-only).
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", os.environ.get("ATTN_BACKEND", "xformers"))

    src = _ensure_import_path()
    _progress(f"trellis2_src={src}")

    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        print("error: CUDA required", file=sys.stderr)
        return 2

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
    _progress(f"gpu={gpu_name} vram_mib={gpu_mem} visible={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

    # Import trellis2 first so the DINOv3 mirror patch can bind the class.
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    image = Image.open(image_path)
    has_alpha = _image_has_useful_alpha(image)
    # Only run pipeline rembg when asked AND the still lacks alpha.
    do_preprocess = bool(args.preprocess_image) and not has_alpha
    dinov3 = _install_dinov3_mirror()
    rembg = _install_rembg_stub_or_mirror(stub=not do_preprocess)
    _progress(f"dinov3_model={dinov3} rembg={rembg} has_alpha={has_alpha} preprocess={do_preprocess}")

    pipeline_type = {
        512: "512",
        1024: "1024_cascade",
        1536: "1536_cascade",
    }[args.resolution]

    low_vram = args.low_vram
    if low_vram is None:
        # 1024 cascade + decode peaks past 24 GB without CPU offload.
        low_vram = args.resolution >= 1024 and gpu_mem <= 24576

    t0 = time.time()
    _progress(f"load_model={args.model} low_vram={low_vram}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.low_vram = bool(low_vram)
    pipeline.cuda()

    _progress(f"run resolution={args.resolution} pipeline_type={pipeline_type} seed={args.seed}")
    outputs = pipeline.run(
        image,
        seed=args.seed,
        preprocess_image=do_preprocess,
        pipeline_type=pipeline_type,
    )
    mesh = outputs[0]
    # Soft pre-bake simplify only. The hard ~100k room cap is bake
    # ``decimation_target`` below — this call just shrinks huge cascades first.
    try:
        soft_cap = max(args.decimation * 3, 1_000_000)
        mesh.simplify(soft_cap)
        _progress(f"pre_bake_simplify_cap={soft_cap}")
    except Exception as e:
        _progress(f"simplify_warn={e}")

    peak_gen = torch.cuda.max_memory_allocated() // (1024**2)
    _progress(f"generate_done peak_alloc_mib={peak_gen}")

    # Free weights before bake (bake is a separate VRAM peak).
    del pipeline
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _progress("pipeline_deleted")

    note = {
        "model": args.model,
        "image": str(image_path),
        "out": str(out_path),
        "seed": args.seed,
        "resolution": args.resolution,
        "pipeline_type": pipeline_type,
        "decimation_target": args.decimation,
        "texture_size": args.texture_size,
        "low_vram": bool(low_vram),
        "dinov3_model": dinov3,
        "rembg": rembg,
        "preprocess_image": do_preprocess,
        "gpu": gpu_name,
        "gpu_vram_mib": gpu_mem,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "peak_generate_alloc_mib": peak_gen,
        "extractor": "nvdiffrast",  # stock lab bake; service replacement is later
        "license_note": "nvdiffrast is NVIDIA Source Code License (1-Way Commercial); lab evaluation only",
    }

    if args.skip_bake:
        note["elapsed_sec"] = round(time.time() - t0, 2)
        if args.note_out:
            Path(args.note_out).write_text(json.dumps(note, indent=2) + "\n")
        _progress("skip_bake")
        return 0

    _progress(f"bake decimation={args.decimation} texture={args.texture_size}")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation,
        texture_size=args.texture_size,
        remesh=not args.no_remesh,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    alpha_mode = _alpha_mode_from_material(glb)
    _apply_room_material(glb, alpha_mode)
    # PNG textures (not webp) for broad glTF loaders / model-viewer.
    glb.export(str(out_path), extension_webp=False)
    # Peak after reset_peak_memory_stats → bake-only high-water, not process lifetime.
    peak_bake = torch.cuda.max_memory_allocated() // (1024**2)
    note["alphaMode"] = alpha_mode
    note["peak_bake_alloc_mib"] = peak_bake
    note["elapsed_sec"] = round(time.time() - t0, 2)
    note["glb_inspect"] = _inspect_glb(out_path)
    _progress(f"wrote {out_path} alphaMode={alpha_mode} inspect={note['glb_inspect']}")

    note_path = Path(args.note_out).expanduser() if args.note_out else out_path.with_suffix(".json")
    note_path.write_text(json.dumps(note, indent=2) + "\n")

    inspect = note["glb_inspect"]
    if not inspect.get("baseColorTexture") or not inspect.get("metallicRoughnessTexture"):
        print(
            f"error: GLB missing PBR textures: {inspect}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
