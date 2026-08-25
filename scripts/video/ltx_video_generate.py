#!/usr/bin/env python3
"""LTXV text-to-video wrapper for zerollama run_script jobs (Wan2GP backend).

Contract (mirrors wan_video_generate.py):
  - Env from server/video_generate.go (LTX_* / WAN2GP_* / VIDEO_*).
  - Output only at LTX_OUTPUT_PATH / VIDEO_OUTPUT_PATH (no latest-mp4 fallback).
  - PROGRESS: lines + TRAINING_COMPLETE for the training worker.
  - LTX_DRY_RUN=1 validates settings/weights and exits 0 without allocating the DiT.
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


def check_weights(ckpt: Path) -> list[str]:
    need = [
        "ltxv_0.9.8_13B_distilled_quanto_bf16_int8.safetensors",
        "ltxv_0.9.7_VAE.safetensors",
        "ltxv_0.9.7_spatial_upscaler.safetensors",
        "ltxv_scheduler.json",
        "T5_xxl_1.1/T5_xxl_1.1_enc_quanto_bf16_int8.safetensors",
    ]
    missing = [n for n in need if not (ckpt / n).is_file()]
    return missing


def build_settings() -> dict:
    model_type = env("LTX_MODEL_TYPE", "ltxv_distilled")
    prompt = require("LTX_PROMPT") if env("LTX_PROMPT") else require("WAN_PROMPT")
    size = env("LTX_SIZE") or env("VIDEO_SIZE") or env("WAN_SIZE") or "768x512"
    frames = int(env("LTX_FRAMES") or env("VIDEO_FRAMES") or env("WAN_FRAMES") or "17")
    steps = int(env("LTX_STEPS") or env("WAN_STEPS") or "6")
    seed_s = env("LTX_SEED") or env("VIDEO_SEED") or env("WAN_SEED")
    settings: dict = {
        "model_type": model_type,
        "prompt": prompt,
        "resolution": size.replace("*", "x"),
        "video_length": frames,
        "num_inference_steps": steps,
        "force_fps": int(env("LTX_FPS", "30") or "30"),
    }
    if seed_s:
        settings["seed"] = int(seed_s)
    return settings


def dry_run(repo: Path, ckpt: Path, settings: dict) -> int:
    progress(5.0, "dry-run: checking weights")
    missing = check_weights(ckpt)
    if missing:
        eprint("missing LTXV weights:")
        for m in missing:
            eprint(f"  {ckpt / m}")
        eprint("reinstall: ./scripts/video/install_ltx_wan2gp.sh --weights-only")
        return 1
    defaults = repo / "defaults" / "ltxv_distilled.json"
    if not defaults.is_file():
        eprint(f"missing Wan2GP defaults at {defaults}")
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
    # Ensure ckpts visible from Wan2GP root.
    link = repo / "ckpts"
    if not link.exists():
        try:
            link.symlink_to(ckpt)
        except OSError:
            eprint(f"warning: could not link {link} -> {ckpt}")

    from shared.api import init  # type: ignore

    profile = env("LTX_MMGP_PROFILE") or env("WAN2GP_PROFILE") or "5"
    attention = env("LTX_ATTENTION") or "sdpa"
    progress(12.0, f"init Wan2GP profile={profile} attention={attention}")
    session = init(
        root=repo,
        cli_args=["--attention", attention, "--profile", profile],
    )
    progress(20.0, "submit ltxv_distilled task")
    job = session.submit_task(settings)
    last = 20.0
    for event in job.events.iter(timeout=0.5):
        if event.kind == "progress":
            p = event.data
            # Map diffusion into 20–90%.
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
        env("LTX_OUTPUT_PATH") or env("VIDEO_OUTPUT_PATH") or env("WAN_OUTPUT_PATH")
    ).expanduser()
    if not str(output):
        raise SystemExit("LTX_OUTPUT_PATH / VIDEO_OUTPUT_PATH required")

    # Expand {job_id} if training worker left the token (Go may pre-expand).
    job_id = env("TRAINING_JOB_ID") or env("JOB_ID")
    if "{job_id}" in str(output) and job_id:
        output = Path(str(output).replace("{job_id}", job_id))

    settings = build_settings()
    progress(1.0, "ltx wrapper start")

    if truthy("LTX_DRY_RUN") or "--dry-run" in sys.argv:
        return dry_run(repo, ckpt, settings)

    missing = check_weights(ckpt)
    if missing:
        eprint("missing LTXV weights — run ./scripts/video/install_ltx_wan2gp.sh --weights-only")
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
