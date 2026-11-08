#!/usr/bin/env python3
"""TRELLIS text XL → color GLB (+ optional gaussian ply). Lab compare half.

Uses stock postprocessing_utils.to_glb (nvdiffrast). Output is base-color only
(roughnessFactor=1, no metal map). Unload the pipeline before bake.

Requires the separate text venv from:
  ./scripts/mesh/install_trellis_astra.sh --with-text
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
    src = Path(os.environ.get("TRELLIS_TEXT_SRC", root / "TRELLIS"))
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="trellis-text.glb")
    ap.add_argument("--ply-out", default="", help="Optional gaussian .ply path")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--texture-size", type=int, default=1024)
    ap.add_argument("--simplify", type=float, default=0.95)
    ap.add_argument("--model", default="microsoft/TRELLIS-text-xlarge")
    args = ap.parse_args()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "flash-attn")

    src = _ensure_import_path()
    _progress(f"trellis_text_src={src}")

    import torch
    from trellis.pipelines import TrellisTextTo3DPipeline
    from trellis.utils import postprocessing_utils

    if not torch.cuda.is_available():
        print("error: CUDA required", file=sys.stderr)
        return 2

    gpu_name = torch.cuda.get_device_name(0)
    _progress(f"gpu={gpu_name} seed={args.seed}")

    t0 = time.time()
    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    _progress("run_text")
    outputs = pipeline.run(args.prompt, seed=args.seed)

    gs = outputs["gaussian"][0]
    mesh = outputs["mesh"][0]
    peak_gen = torch.cuda.max_memory_allocated() // (1024**2)

    if args.ply_out:
        ply_path = Path(args.ply_out).expanduser().resolve()
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        gs.save_ply(str(ply_path))
        _progress(f"wrote_ply={ply_path}")

    del pipeline
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _progress("pipeline_deleted")

    _progress(f"bake texture={args.texture_size}")
    glb = postprocessing_utils.to_glb(
        gs,
        mesh,
        simplify=args.simplify,
        texture_size=args.texture_size,
        verbose=True,
    )
    glb.export(str(out_path))
    # Peak after reset → bake-only high-water, not process lifetime.
    peak_bake = torch.cuda.max_memory_allocated() // (1024**2)

    note = {
        "model": args.model,
        "prompt": args.prompt,
        "out": str(out_path),
        "seed": args.seed,
        "texture_size": args.texture_size,
        "simplify": args.simplify,
        "gpu": gpu_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "peak_generate_alloc_mib": peak_gen,
        "peak_bake_alloc_mib": peak_bake,
        "elapsed_sec": round(time.time() - t0, 2),
        "pbr": False,
        "extractor": "nvdiffrast",
        "license_note": "stock color bake uses nvdiffrast (non-commercial); ply is the MIT gaussian artifact",
    }
    out_path.with_suffix(".json").write_text(json.dumps(note, indent=2) + "\n")
    _progress(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
