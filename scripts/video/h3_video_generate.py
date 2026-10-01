#!/usr/bin/env python3
"""MiniMax-H3 T2VA wrapper for zerollama run_script jobs (Wan2GP CUDA backend).

Contract (mirrors ltx_video_generate.py):
  - Env from server/video_generate.go (H3_* / WAN2GP_* / VIDEO_*).
  - Output only at H3_OUTPUT_PATH / VIDEO_OUTPUT_PATH.
  - PROGRESS: lines + TRAINING_COMPLETE for the training worker.
  - H3_DRY_RUN=1 validates settings/weights without allocating the DiT.
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


def is_pruned(model_type: str) -> bool:
    t = (model_type or "").lower()
    return "pruned" in t


def dit_name(model_type: str) -> str:
    if is_pruned(model_type):
        return "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors"
    return "MiniMax-H3-FL2VA_int8_convrot.safetensors"


def weight_names(model_type: str) -> list[str]:
    return [
        dit_name(model_type),
        "minimax_h3_video_vae_fp8mix.safetensors",
        "MiniMax-H3-audio_vae_fp32.safetensors",
    ]


def te_present(ckpt: Path) -> bool:
    te = ckpt / "Qwen3-VL-32B-Instruct"
    cands = [
        te / "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf",
        te / "qwen3vl-32B-MiniMax-H3-Q2_K.gguf",
        te / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        te / "Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors",
    ]
    return any(p.is_file() for p in cands)


def check_weights(ckpt: Path, model_type: str = "") -> list[str]:
    missing = [n for n in weight_names(model_type) if not (ckpt / n).is_file()]
    if not te_present(ckpt):
        missing.append("Qwen3-VL-32B-Instruct/<Q4|Q2|NVFP4 TE>")
    return missing


def build_settings() -> dict:
    model_type = env("H3_MODEL_TYPE", "minimax_h3_fl2va")
    prompt = require("H3_PROMPT") if env("H3_PROMPT") else require("WAN_PROMPT")
    size = env("H3_SIZE") or env("VIDEO_SIZE") or env("WAN_SIZE") or "480x832"
    frames = int(env("H3_FRAMES") or env("VIDEO_FRAMES") or env("WAN_FRAMES") or "17")
    steps = int(env("H3_STEPS") or env("WAN_STEPS") or "8")
    seed_s = env("H3_SEED") or env("VIDEO_SEED") or env("WAN_SEED")
    settings: dict = {
        "model_type": model_type,
        "prompt": prompt,
        "resolution": size.replace("*", "x"),
        "video_length": frames,
        "num_inference_steps": steps,
        "force_fps": int(env("H3_FPS", "24") or "24"),
    }
    # Prefer fp8mix VAE when present (install script); Wan2GP may still use int8_convrot default.
    if env("H3_VIDEO_VAE"):
        settings["video_vae_file"] = env("H3_VIDEO_VAE")
    elif env("H3_VIDEO_VAE_MODE"):
        settings["video_vae_mode"] = env("H3_VIDEO_VAE_MODE")  # fp8mix | int8_convrot | bf16
    else:
        settings["video_vae_mode"] = "fp8mix"
    if env("H3_TEXT_ENCODER"):
        settings["text_encoder_variant"] = env("H3_TEXT_ENCODER")
    if seed_s:
        settings["seed"] = int(seed_s)
    return settings


def dry_run(repo: Path, ckpt: Path, settings: dict) -> int:
    progress(5.0, "dry-run: checking weights")
    missing = check_weights(ckpt, str(settings.get("model_type") or ""))
    if missing:
        eprint("missing MiniMax-H3 weights:")
        for m in missing:
            eprint(f"  {ckpt / m}")
        eprint("reinstall: ./scripts/video/install_h3_wan2gp.sh --weights-only")
        return 1
    model_type = str(settings.get("model_type") or "minimax_h3_fl2va")
    defaults = repo / "defaults" / f"{model_type}.json"
    if not defaults.is_file():
        # Wan2GP may register via models/minimax_h3 without defaults JSON.
        handler = repo / "models" / "minimax_h3"
        if not handler.is_dir():
            eprint(f"missing Wan2GP MiniMax-H3 handler at {handler}")
            return 1
        defaults = handler
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
    link = repo / "ckpts"
    if not link.exists():
        try:
            link.symlink_to(ckpt)
        except OSError:
            eprint(f"warning: could not link {link} -> {ckpt}")

    # Prefer GPU 0 unless operator set CUDA_VISIBLE_DEVICES.
    if not env("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = env("H3_CUDA_DEVICE", "0") or "0"

    from shared.api import init  # type: ignore

    profile = env("H3_MMGP_PROFILE") or env("WAN2GP_PROFILE") or "5"
    attention = env("H3_ATTENTION") or "sdpa"
    progress(12.0, f"init Wan2GP profile={profile} attention={attention}")
    session = init(
        root=repo,
        cli_args=["--attention", attention, "--profile", profile],
    )
    progress(20.0, f"submit {settings.get('model_type', 'minimax_h3')} task")
    job = session.submit_task(settings)
    last = 20.0
    for event in job.events.iter(timeout=0.5):
        if event.kind == "progress":
            p = event.data
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
                phase = getattr(p, "phase", None) or "diffusing"
                progress(pct, str(phase))
                last = pct
        elif event.kind == "stream":
            line = event.data
            text = getattr(line, "text", "") or ""
            if text:
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
        env("H3_OUTPUT_PATH") or env("VIDEO_OUTPUT_PATH") or env("WAN_OUTPUT_PATH")
    ).expanduser()
    if not str(output):
        raise SystemExit("H3_OUTPUT_PATH / VIDEO_OUTPUT_PATH required")

    job_id = env("TRAINING_JOB_ID") or env("JOB_ID")
    if "{job_id}" in str(output) and job_id:
        output = Path(str(output).replace("{job_id}", job_id))

    settings = build_settings()
    progress(1.0, "h3 wan2gp wrapper start")

    if truthy("H3_DRY_RUN") or "--dry-run" in sys.argv:
        return dry_run(repo, ckpt, settings)

    missing = check_weights(ckpt, settings.get("model_type") or env("H3_MODEL_TYPE"))
    if missing:
        eprint("missing MiniMax-H3 weights — run ./scripts/video/install_h3_wan2gp.sh --weights-only")
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
