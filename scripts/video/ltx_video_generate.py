#!/usr/bin/env python3
"""LTX / LTXV text-to-video wrapper for zerollama run_script jobs (Wan2GP backend).

Contract (mirrors wan_video_generate.py):
  - Env from server/video_generate.go (LTX_* / WAN2GP_* / VIDEO_*).
  - Output only at LTX_OUTPUT_PATH / VIDEO_OUTPUT_PATH (no latest-mp4 fallback).
  - PROGRESS: lines + TRAINING_COMPLETE for the training worker.
  - LTX_DRY_RUN=1 validates settings/weights and exits 0 without allocating the DiT.
  - LTX-2.5: model_type ltx2_25_22B_distilled; optional image_start/image_end (keyframes).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def progress(pct: float, msg: str) -> None:
    print(f"PROGRESS:{pct:.1f}:{msg}", flush=True)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def truthy(name: str) -> bool:
    return env(name).lower() in ("1", "true", "yes", "on")


def require(name: str) -> str:
    v = env(name)
    if not v:
        raise SystemExit(f"missing required env {name}")
    return v


def is_ltx2(model_type: str) -> bool:
    t = (model_type or "").lower()
    return t.startswith("ltx2") or "ltx2_" in t or "ltx-2" in t


def is_ltxv_2b(model_type: str) -> bool:
    """LTXV 2B distilled only — not LTX-2.x."""
    t = (model_type or "").lower()
    if is_ltx2(t):
        return False
    return "2b" in t or t == "ltxv_2b_distilled"


def weight_names(model_type: str) -> list[str]:
    if is_ltx2(model_type):
        return [
            "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors",
            "ltx-2.5-22b_video_vae_bf16.safetensors",
            "ltx-2.5-22b_audio_vae_bf16.safetensors",
            "ltx-2.5-22b_vocoder_bf16.safetensors",
            "ltx-2.5-22b_text_embedding_projection_bf16.safetensors",
            "ltx-2.5-22b_video_embeddings_connector_int8_convrot.safetensors",
            "ltx-2.5-22b_audio_embeddings_connector_int8_convrot.safetensors",
            "gemma4-12b-ltx-v1/gemma4-12b-ltx-v1_int8_convrot.safetensors",
            "gemma4-12b-ltx-v1/tokenizer.json",
        ]
    shared = [
        "ltxv_0.9.7_VAE.safetensors",
        "ltxv_scheduler.json",
        "T5_xxl_1.1/T5_xxl_1.1_enc_quanto_bf16_int8.safetensors",
    ]
    if is_ltxv_2b(model_type):
        return [
            "ltxv-2b-0.9.8-distilled-fp8.safetensors",
            "ltxv_0.9.8_spatial_upscaler.safetensors",
            *shared,
        ]
    return [
        "ltxv_0.9.8_13B_distilled_quanto_bf16_int8.safetensors",
        "ltxv_0.9.7_spatial_upscaler.safetensors",
        *shared,
    ]


def check_weights(ckpt: Path, model_type: str = "") -> list[str]:
    missing = [n for n in weight_names(model_type) if not (ckpt / n).is_file()]
    # Accept bf16 DiT as alternate to int8 for LTX-2.5.
    if is_ltx2(model_type):
        dit = "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
        if dit in missing and (ckpt / "ltx-2.5-22b-distilled_diffusion_model_bf16.safetensors").is_file():
            missing = [n for n in missing if n != dit]
        if dit in missing and (ckpt / "ltx-2.5-22b-distilled_diffusion_model_nvfp4.safetensors").is_file():
            missing = [n for n in missing if n != dit]
    if is_ltxv_2b(model_type) and "ltxv_0.9.8_spatial_upscaler.safetensors" in missing:
        if (ckpt / "ltxv-spatial-upscaler-0.9.8.safetensors").is_file():
            missing = [n for n in missing if n != "ltxv_0.9.8_spatial_upscaler.safetensors"]
    return missing


def stills_in_dir(dir_path: str) -> list[str]:
    if not dir_path:
        return []
    d = Path(dir_path)
    if not d.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    out = sorted(
        str(p) for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    return out


def build_settings() -> dict:
    model_type = env("LTX_MODEL_TYPE", "ltxv_distilled")
    prompt = require("LTX_PROMPT") if env("LTX_PROMPT") else require("WAN_PROMPT")
    default_size = "1280x704" if is_ltx2(model_type) else "768x512"
    default_frames = "97" if is_ltx2(model_type) else "17"
    default_steps = "8" if is_ltx2(model_type) else "6"
    size = env("LTX_SIZE") or env("VIDEO_SIZE") or env("WAN_SIZE") or default_size
    frames = int(env("LTX_FRAMES") or env("VIDEO_FRAMES") or env("WAN_FRAMES") or default_frames)
    steps = int(env("LTX_STEPS") or env("WAN_STEPS") or default_steps)
    seed_s = env("LTX_SEED") or env("VIDEO_SEED") or env("WAN_SEED")
    settings: dict = {
        "model_type": model_type,
        "prompt": prompt,
        "resolution": size.replace("*", "x"),
        "video_length": frames,
        "num_inference_steps": steps,
        "force_fps": int(env("LTX_FPS", "24" if is_ltx2(model_type) else "30") or ("24" if is_ltx2(model_type) else "30")),
    }
    if seed_s:
        settings["seed"] = int(seed_s)

    # Control: start/end stills (LTX-2) — from env paths or keyframe staging dir.
    image_start = env("LTX_IMAGE_START") or env("LTX_IMAGE") or env("VIDEO_IMAGE")
    image_end = env("LTX_IMAGE_END")
    keyframe_dir = env("VIDEO_KEYFRAME_DIR")
    if keyframe_dir and not image_start:
        stills = stills_in_dir(keyframe_dir)
        if len(stills) >= 1:
            image_start = stills[0]
        if len(stills) >= 2 and not image_end:
            image_end = stills[-1]
    if image_start:
        settings["image_start"] = image_start
    if image_end:
        settings["image_end"] = image_end
    if env("LTX_VIDEO_GUIDE"):
        settings["video_guide"] = env("LTX_VIDEO_GUIDE")
    if env("LTX_IMAGE_REFS"):
        # Comma-separated paths for reference images.
        refs = [p.strip() for p in env("LTX_IMAGE_REFS").split(",") if p.strip()]
        if refs:
            settings["image_refs"] = refs
    return settings


def defaults_path(repo: Path, model_type: str) -> Path:
    mt = str(model_type)
    if is_ltx2(mt):
        # defaults/<model_type>.json (e.g. ltx2_25_22B_distilled.json)
        cand = repo / "defaults" / f"{mt}.json"
        if cand.is_file():
            return cand
        return repo / "defaults" / "ltx2_25_22B_distilled.json"
    if is_ltxv_2b(mt):
        defaults = repo / "finetunes" / "ltxv_2b_distilled.json"
        if defaults.is_file():
            return defaults
        return repo / "defaults" / "ltxv_2b_distilled.json"
    return repo / "defaults" / "ltxv_distilled.json"


def dry_run(repo: Path, ckpt: Path, settings: dict) -> int:
    progress(5.0, "dry-run: checking weights")
    model_type = str(settings.get("model_type") or env("LTX_MODEL_TYPE", "ltxv_distilled"))
    missing = check_weights(ckpt, model_type)
    if missing:
        eprint("missing LTX weights:")
        for m in missing:
            eprint(f"  {ckpt / m}")
        if is_ltx2(model_type):
            eprint("reinstall: ./scripts/video/install_ltx2_wan2gp.sh --weights-only")
        else:
            eprint("reinstall: ./scripts/video/install_ltx_wan2gp.sh --weights-only (or --2b-only)")
        return 1
    defaults = defaults_path(repo, model_type)
    if not defaults.is_file():
        eprint(f"missing Wan2GP model def at {defaults}")
        return 1
    progress(40.0, "dry-run: settings ok")
    out = {
        "ok": True,
        "dry_run": True,
        "repo": str(repo),
        "ckpt": str(ckpt),
        "settings": settings,
        "defaults": str(defaults),
    }
    print(json.dumps(out, indent=2), flush=True)
    progress(100.0, "dry-run complete")
    print("TRAINING_COMPLETE", flush=True)
    return 0


def run_generate(repo: Path, ckpt: Path, settings: dict, output: Path) -> int:
    progress(5.0, "importing Wan2GP API")
    sys.path.insert(0, str(repo))
    # Shared freest-GPU picker lives next to this wrapper.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cuda_device import apply_cuda_device  # noqa: WPS433

    link = repo / "ckpts"
    if not link.exists():
        try:
            link.symlink_to(ckpt)
        except OSError:
            eprint(f"warning: could not link {link} -> {ckpt}")

    # Best-fit (+ migrate irodori only under solvable contention). See cuda_device.py.
    apply_cuda_device(override_env="LTX_CUDA_DEVICE", label="LTX")
    from shared.api import init  # type: ignore

    profile = env("LTX_MMGP_PROFILE") or env("WAN2GP_PROFILE") or "5"
    attention = env("LTX_ATTENTION") or "sdpa"
    progress(12.0, f"init Wan2GP profile={profile} attention={attention}")
    session = init(
        root=repo,
        cli_args=["--attention", attention, "--profile", profile],
    )
    progress(20.0, f"submit {settings.get('model_type', 'ltx')} task")
    job = session.submit_task(settings)
    last = 20.0
    # mmgp load phases report non-monotonic sub-step %; pin floors so clients do not see thrash.
    load_floors = {
        "loading_model": 27.0,
        "encoding_text": 40.0,
        "loading": 27.0,
    }
    for event in job.events.iter(timeout=0.5):
        if event.kind == "progress":
            p = event.data
            phase = str(getattr(p, "phase", None) or "diffusing")
            if phase in load_floors:
                floor = load_floors[phase]
                pct = floor if last < floor else min(last + 0.05, 55.0)
            else:
                frac = 0.0
                try:
                    if getattr(p, "total_steps", 0):
                        frac = float(p.current_step) / float(p.total_steps)
                    elif getattr(p, "progress", None) is not None:
                        frac = float(p.progress) / 100.0
                except Exception:
                    frac = 0.0
                pct = 20.0 + max(0.0, min(1.0, frac)) * 70.0
            if pct > last:
                progress(pct, phase)
                last = pct
        elif event.kind == "stream":
            line = event.data
            text = getattr(line, "text", "") or ""
            if text:
                # mmgp re-hooks the same modules many times while thrashing — keep journal readable.
                if "Hooked to model" in text or "Async loading plan" in text:
                    continue
                if text.count("Loading Model") and "ltx-2.5" in text:
                    # one-line breadcrumb instead of a reload flood
                    if last < 30.0:
                        eprint(text.rstrip())
                    continue
                eprint(text.rstrip())
    result = job.result()
    if not result.success:
        for err in result.errors or []:
            eprint(getattr(err, "message", str(err)))
        return 1

    files = list(result.generated_files or [])
    if not files:
        eprint("Wan2GP returned success but no generated_files")
        return 1
    src = Path(files[0])
    if not src.is_file():
        eprint(f"generated file missing: {src}")
        return 1
    progress(94.0, "copying artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, output)
    if not output.is_file() or output.stat().st_size < 1:
        eprint(f"failed to write {output}")
        return 1
    progress(100.0, "done")
    print("TRAINING_COMPLETE", flush=True)
    print(f"artifact={output}", flush=True)
    return 0


def main() -> int:
    repo = Path(require("WAN2GP_REPO")).expanduser().resolve()
    ckpt = Path(env("WAN2GP_CKPT_DIR") or (repo / "ckpts")).expanduser().resolve()
    output = Path(
        env("LTX_OUTPUT_PATH") or env("VIDEO_OUTPUT_PATH") or env("WAN_OUTPUT_PATH")
    ).expanduser()
    if not str(output):
        raise SystemExit("LTX_OUTPUT_PATH / VIDEO_OUTPUT_PATH required")

    job_id = env("TRAINING_JOB_ID") or env("JOB_ID")
    if "{job_id}" in str(output) and job_id:
        output = Path(str(output).replace("{job_id}", job_id))

    settings = build_settings()
    progress(1.0, "ltx wrapper start")

    if truthy("LTX_DRY_RUN") or "--dry-run" in sys.argv:
        return dry_run(repo, ckpt, settings)

    model_type = settings.get("model_type") or env("LTX_MODEL_TYPE")
    missing = check_weights(ckpt, model_type)
    if missing:
        if is_ltx2(str(model_type)):
            eprint("missing LTX-2.5 weights — run ./scripts/video/install_ltx2_wan2gp.sh --weights-only")
        else:
            eprint("missing LTXV weights — run ./scripts/video/install_ltx_wan2gp.sh --weights-only (or --2b-only)")
        for m in missing:
            eprint(f"  {ckpt / m}")
        return 1

    return run_generate(repo, ckpt, settings, output)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        eprint("interrupted")
        raise SystemExit(130)
