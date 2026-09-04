#!/usr/bin/env python3
"""run_script entry for ltx-mlx (Apple Silicon LTX-Video 0.9.8 2B).

Why this wrapper: /v1/videos jobs are training run_script; embed CPython has no mlx.
Why not Wan2GP: that is CUDA/PyTorch. Fast Mac path is community ltx-mlx (T5 + 2B DiT).

Env: LTX_MLX_MODEL_DIR, LTX_PROMPT, LTX_OUTPUT_PATH, LTX_SIZE, LTX_FRAMES, LTX_STEPS,
LTX_SEED, LTX_MODEL (2b|13b), LTX_IMAGE / VIDEO_IMAGE (I2V first frame).
"""
from __future__ import annotations

import os
import shutil
import subprocess
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


def parse_size(size: str) -> tuple[int, int]:
    raw = (size or "").strip().lower().replace("×", "x")
    if "*" in raw:
        w_s, h_s = raw.split("*", 1)
    elif "x" in raw:
        w_s, h_s = raw.split("x", 1)
    else:
        return 768, 480
    try:
        w, h = int(w_s.strip()), int(h_s.strip())
    except ValueError:
        return 768, 480
    if w < 32 or h < 32:
        return 768, 480
    return w, h


def check_model_dir(model_dir: Path, dit: str = "2b") -> list[str]:
    missing: list[str] = []
    dit = (dit or "2b").strip().lower()
    if dit == "13b":
        p13 = model_dir / "LTX 13B" / "ltxv-13b-0.9.8-distilled.safetensors"
        p13_flat = model_dir / "ltxv-13b-0.9.8-distilled.safetensors"
        if not p13.is_file() and not p13_flat.is_file() and not list(
            model_dir.glob("**/ltxv-13b-0.9.8-distilled*.safetensors")
        ):
            missing.append("LTX 13B/ltxv-13b-0.9.8-distilled.safetensors")
    elif not list(model_dir.glob("ltxv-2b-0.9.8-distilled*.safetensors")) and not list(
        model_dir.glob("*.safetensors")
    ):
        missing.append("ltxv-2b-0.9.8-distilled.safetensors")
    if not (model_dir / "tokenizer" / "spiece.model").is_file():
        missing.append("tokenizer/spiece.model")
    if not (model_dir / "text_encoder" / "config.json").is_file():
        missing.append("text_encoder/config.json")
    return missing


def build_cmd(environ: dict[str, str]) -> list[str] | None:
    model_dir = environ.get("LTX_MLX_MODEL_DIR") or ""
    prompt = environ.get("LTX_PROMPT") or environ.get("WAN_PROMPT") or ""
    out = environ.get("LTX_OUTPUT_PATH") or environ.get("VIDEO_OUTPUT_PATH") or ""
    if not model_dir or not prompt or not out:
        return None
    w, h = parse_size(environ.get("LTX_SIZE") or environ.get("VIDEO_SIZE") or "768x480")
    frames = environ.get("LTX_FRAMES") or environ.get("VIDEO_FRAMES") or "17"
    steps = environ.get("LTX_STEPS") or "4"
    seed = environ.get("LTX_SEED") or environ.get("VIDEO_SEED") or "42"
    python = environ.get("LTX_MLX_PYTHON") or sys.executable
    dit = (environ.get("LTX_MODEL") or "2b").strip().lower()
    if dit not in ("2b", "13b"):
        dit = "2b"
    cmd = [
        python,
        "-m",
        "ltx_mlx.cli",
        "--prompt",
        prompt,
        "--model-dir",
        model_dir,
        "--model",
        dit,
        "--width",
        str(w),
        "--height",
        str(h),
        "--frames",
        str(frames),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--output",
        out,
    ]
    image = environ.get("LTX_IMAGE") or environ.get("VIDEO_IMAGE") or ""
    if image:
        cmd.extend(["--image", image])
    return cmd


def main() -> int:
    model_dir = Path(env("LTX_MLX_MODEL_DIR") or "models/LTX").expanduser()
    prompt = env("LTX_PROMPT") or env("WAN_PROMPT")
    out = Path(env("LTX_OUTPUT_PATH") or env("VIDEO_OUTPUT_PATH") or "")
    if not prompt or not str(out):
        eprint("LTX_PROMPT and LTX_OUTPUT_PATH required")
        return 1
    job_id = env("TRAINING_JOB_ID") or env("JOB_ID")
    if "{job_id}" in str(out) and job_id:
        out = Path(str(out).replace("{job_id}", job_id))
        os.environ["LTX_OUTPUT_PATH"] = str(out)
        os.environ["VIDEO_OUTPUT_PATH"] = str(out)

    missing = check_model_dir(model_dir, env("LTX_MODEL") or "2b")
    if missing:
        eprint("missing ltx-mlx weights:")
        for m in missing:
            eprint(f"  {model_dir / m}")
        eprint("install: ./scripts/video/install_ltx_mlx.sh")
        return 1

    if truthy("LTX_DRY_RUN") or "--dry-run" in sys.argv:
        progress(40.0, "dry-run: weights ok")
        print({"ok": True, "dry_run": True, "model_dir": str(model_dir)}, flush=True)
        progress(100.0, "dry-run complete")
        print("TRAINING_COMPLETE", flush=True)
        return 0

    cmd = build_cmd({**os.environ, "LTX_OUTPUT_PATH": str(out)})
    if not cmd:
        eprint("failed to build ltx-mlx command")
        return 1
    if shutil.which(cmd[0]) is None and not Path(cmd[0]).is_file():
        eprint(f"python missing: {cmd[0]}")
        return 1

    progress(5.0, "starting ltx-mlx")
    print(" ".join(cmd), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
        if not out.is_file() or out.stat().st_size < 1:
            eprint(f"ltx-mlx produced no file at {out}")
            return 1
        progress(100.0, "done")
        print("TRAINING_COMPLETE", flush=True)
        print(f"artifact={out}", flush=True)
        return 0
    finally:
        if truthy("VIDEO_CLEANUP_KEYFRAME_DIR"):
            kf = env("VIDEO_KEYFRAME_DIR")
            if kf:
                shutil.rmtree(kf, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
