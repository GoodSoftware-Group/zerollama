#!/usr/bin/env python3
"""MiniMax Music 3 song wrapper for zerollama run_script jobs (Wan2GP CUDA).

The Mac path stays scripts/audio/music3_mlx_generate.py. That script re-execs
this file when MUSIC3_MLX_MODEL is a Wan2GP checkpoint directory.

Contract:
  - Env from server/music_generate.go (MUSIC3_*).
  - Output only at MUSIC3_OUTPUT_PATH (.wav).
  - PROGRESS: lines + TRAINING_COMPLETE for the training worker.
  - MUSIC3_DRY_RUN=1 checks weights and does not allocate the model.
"""
from __future__ import annotations

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


TRANSFORMERS = (
    "MiniMax_Music3_transformer_int8_convrot.safetensors",
    "MiniMax_Music3_transformer_bf16.safetensors",
)
TEXT_ENCODERS = (
    "MiniMaxMusic3-Qwen3/MiniMaxMusic3-Qwen3_int8_convrot.safetensors",
    "MiniMaxMusic3-Qwen3/MiniMaxMusic3-Qwen3_bf16.safetensors",
)
SUPPORT = (
    "MiniMax-Music3/vocoder_fp32.safetensors",
    "MiniMax-Music3/condition_encoder_fp32.safetensors",
    "MiniMax-Music3/rvq_depth_decoder_int8_convrot.safetensors",
    "MiniMaxMusic3-Qwen3/tokenizer.json",
)


def is_ckpt(path: Path) -> bool:
    return any((path / name).is_file() for name in TRANSFORMERS)


def resolve_layout() -> tuple[Path, Path]:
    ckpt_s = env("WAN2GP_CKPT_DIR") or env("MUSIC3_MLX_MODEL")
    if not ckpt_s:
        raise SystemExit("missing WAN2GP_CKPT_DIR or MUSIC3_MLX_MODEL")
    ckpt = Path(ckpt_s).expanduser().resolve()
    if not is_ckpt(ckpt):
        raise SystemExit(f"not a MiniMax Music 3 Wan2GP pack: {ckpt}")
    repo_s = env("WAN2GP_REPO")
    repo = Path(repo_s).expanduser().resolve() if repo_s else (ckpt.parent / "repo")
    if not (repo / "wgp.py").is_file() and not (repo / "models" / "TTS" / "minimax_music3").is_dir():
        raise SystemExit(f"Wan2GP repo missing at {repo}")
    return repo, ckpt


def missing_weights(ckpt: Path) -> list[str]:
    missing: list[str] = []
    if not any((ckpt / name).is_file() for name in TRANSFORMERS):
        missing.append(TRANSFORMERS[0])
    if not any((ckpt / name).is_file() for name in TEXT_ENCODERS):
        missing.append(TEXT_ENCODERS[0])
    for name in SUPPORT:
        if not (ckpt / name).is_file():
            missing.append(name)
    return missing


def build_settings() -> dict:
    lyrics = require("MUSIC3_LYRICS")
    caption = env("MUSIC3_CAPTION") or "Warm acoustic pop, 96 BPM, intimate vocal"
    duration = float(env("MUSIC3_DURATION") or "10")
    steps = int(env("MUSIC3_STEPS") or "30")
    if duration <= 0:
        raise SystemExit("MUSIC3_DURATION must be positive")
    if steps <= 0:
        raise SystemExit("MUSIC3_STEPS must be positive")
    settings: dict = {
        "model_type": env("MUSIC3_MODEL_TYPE") or "minimax_music3",
        "prompt": lyrics,
        "alt_prompt": caption,
        "duration_seconds": duration,
        "num_inference_steps": steps,
        "guidance_scale": float(env("MUSIC3_GUIDANCE") or "1.7"),
        "audio_prompt_type": "",
        "prompt_enhancer": "",
    }
    seed_s = env("MUSIC3_SEED")
    if seed_s:
        settings["seed"] = int(seed_s)
    return settings


def expand_output() -> Path:
    output = Path(require("MUSIC3_OUTPUT_PATH")).expanduser()
    job_id = env("TRAINING_JOB_ID") or env("JOB_ID")
    if "{job_id}" in str(output):
        if not job_id:
            raise SystemExit("MUSIC3_OUTPUT_PATH contains {job_id} but TRAINING_JOB_ID is empty")
        output = Path(str(output).replace("{job_id}", job_id))
    return output


def dry_run(repo: Path, ckpt: Path, settings: dict) -> int:
    progress(5.0, "dry-run: checking weights")
    missing = missing_weights(ckpt)
    if missing:
        eprint("missing MiniMax Music 3 weights:")
        for name in missing:
            eprint(f"  {ckpt / name}")
        return 1
    progress(40.0, "dry-run: settings ok")
    print(
        f"ok repo={repo} ckpt={ckpt} model={settings['model_type']} "
        f"duration={settings['duration_seconds']} steps={settings['num_inference_steps']}",
        flush=True,
    )
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

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "video"))
    from cuda_device import apply_cuda_device  # noqa: WPS433

    need = int(env("MUSIC3_VRAM_NEED_MIB") or "20000")
    apply_cuda_device(override_env="MUSIC3_CUDA_DEVICE", label="Music3", need_mib=need)

    from shared.api import init  # type: ignore

    # Profile 1 keeps the int8 pack on a 24GB card. Profiles 2/4/5 set
    # tiny mmgp budgets (the handler offloads the 8B) and each second of
    # audio takes many minutes.
    profile = env("MUSIC3_MMGP_PROFILE") or env("WAN2GP_PROFILE") or "1"
    attention = env("MUSIC3_ATTENTION") or "sdpa"
    work = Path(env("MUSIC3_WORK_DIR") or "/tmp/zerollama-music3").expanduser()
    work.mkdir(parents=True, exist_ok=True)
    progress(12.0, f"init Wan2GP profile={profile} attention={attention}")
    session = init(
        root=repo,
        output_dir=work,
        cli_args=["--attention", attention, "--profile", profile],
    )
    progress(20.0, "submit minimax_music3 task")
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
                phase = getattr(p, "phase", None) or "generating"
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

    files = [Path(p) for p in (result.generated_files or []) if Path(p).is_file()]
    if not files:
        eprint("Wan2GP returned success but no generated_files")
        return 1
    src = files[0]
    progress(94.0, "copying artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".wav":
        shutil.copy2(src, output)
    else:
        import subprocess

        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), str(output)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            eprint(proc.stderr[-2000:])
            eprint(f"ffmpeg failed to convert {src} -> {output}")
            return 1
    if not output.is_file() or output.stat().st_size < 44:
        eprint(f"failed to write {output}")
        return 1
    progress(100.0, "done")
    print("TRAINING_COMPLETE", flush=True)
    print(f"artifact={output}", flush=True)
    return 0


def main() -> int:
    repo, ckpt = resolve_layout()
    output = expand_output()
    settings = build_settings()
    progress(1.0, "music3 wan2gp wrapper start")
    if truthy("MUSIC3_DRY_RUN") or "--dry-run" in sys.argv:
        return dry_run(repo, ckpt, settings)
    missing = missing_weights(ckpt)
    if missing:
        eprint("missing MiniMax Music 3 weights:")
        for name in missing:
            eprint(f"  {ckpt / name}")
        return 1
    return run_generate(repo, ckpt, settings, output)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        eprint("interrupted")
        raise SystemExit(130)
