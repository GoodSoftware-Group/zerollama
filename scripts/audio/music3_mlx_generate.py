#!/usr/bin/env python3
"""Lab MiniMax Music 3 generate via mlx-audio (no ComfyUI).

Why mlx-audio: native MLX on Apple Silicon; Omni acoustic needs CUDA; Comfy is GPL.
Why a wrapper: training run_script + MUSIC3_* env; expand {job_id} if the worker
left the token (Go does not know the uuid at submit).

Pin (until PyPI ships Music 3):
  uv pip install --python .venv-music \
    "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git@784b29e2691a93ca7483147d86f61859dfaa6296"

Does not bind 11434/8081. Default clip is 10 seconds.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MLX_AUDIO_PIN = "784b29e2691a93ca7483147d86f61859dfaa6296"
DEFAULT_MODEL = "mlx-community/MiniMax-Music3-8bit"
DEFAULT_LYRICS = (
    "[Verse]\n"
    "Morning light filtering through the pine\n"
    "[Chorus]\n"
    "Softly the world begins to breathe"
)
DEFAULT_CAPTION = "Warm acoustic pop, 96 BPM, intimate female vocal"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("MUSIC3_MLX_MODEL", DEFAULT_MODEL))
    p.add_argument("--caption", default=os.environ.get("MUSIC3_CAPTION", DEFAULT_CAPTION))
    lyrics = p.add_mutually_exclusive_group()
    lyrics.add_argument("--lyrics", default=None)
    lyrics.add_argument("--lyrics-file", type=Path, default=None)
    p.add_argument(
        "--duration",
        type=float,
        default=float(os.environ.get("MUSIC3_DURATION", "10")),
        help="Cap in seconds (25 frames/s)",
    )
    p.add_argument("--steps", type=int, default=int(os.environ.get("MUSIC3_STEPS", "30")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("MUSIC3_SEED", "7")))
    p.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("MUSIC3_OUTPUT_PATH", "/tmp/music3_10s.wav")),
    )
    p.add_argument("--quiet", action="store_true")
    return p


def _wan2gp_ckpt(model: str) -> Path | None:
    path = Path(model).expanduser()
    for name in (
        "MiniMax_Music3_transformer_int8_convrot.safetensors",
        "MiniMax_Music3_transformer_bf16.safetensors",
    ):
        if (path / name).is_file():
            return path
    return None


def _dispatch_wan2gp(ckpt: Path) -> int:
    """Re-exec the CUDA pack with the Wan2GP venv. MLX stays the Mac path."""
    root = ckpt.parent
    venv = Path(os.environ.get("WAN2GP_VENV") or (root / "venv"))
    py = venv / "bin" / "python3"
    script = Path(__file__).resolve().parent / "music3_wan2gp_generate.py"
    if not py.is_file():
        print(f"Wan2GP python missing: {py}", file=sys.stderr)
        return 2
    if not script.is_file():
        print(f"Wan2GP music wrapper missing: {script}", file=sys.stderr)
        return 2
    os.environ["WAN2GP_CKPT_DIR"] = str(ckpt)
    os.environ.setdefault("WAN2GP_REPO", str(root / "repo"))
    os.environ.setdefault("WAN2GP_VENV", str(venv))
    if Path(sys.executable).resolve() != py.resolve():
        os.execv(str(py), [str(py), str(script)])
    import runpy

    runpy.run_path(str(script), run_name="__main__")
    return 0


def main() -> int:
    args = _parser().parse_args()
    ckpt = _wan2gp_ckpt(args.model)
    if ckpt is not None or os.environ.get("MUSIC3_BACKEND", "").lower() in ("wan2gp", "cuda"):
        if ckpt is None:
            raw = os.environ.get("WAN2GP_CKPT_DIR") or args.model
            ckpt = Path(raw).expanduser()
        return _dispatch_wan2gp(ckpt)
    lyrics = args.lyrics
    if args.lyrics_file is not None:
        lyrics = args.lyrics_file.read_text(encoding="utf-8")
    if not lyrics:
        lyrics = os.environ.get("MUSIC3_LYRICS") or DEFAULT_LYRICS
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")

    out = Path(args.out)
    out_s = str(out)
    if "{job_id}" in out_s:
        jid = os.environ.get("TRAINING_JOB_ID") or os.environ.get("JOB_ID") or ""
        if jid:
            out = Path(out_s.replace("{job_id}", jid))
    out.parent.mkdir(parents=True, exist_ok=True)
    args.out = out

    try:
        from mlx_audio.music.generate import generate_music
    except ImportError:
        print(
            "mlx-audio is not installed. Pin:\n"
            f'  uv pip install --python .venv-music "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git@{MLX_AUDIO_PIN}"',
            file=sys.stderr,
        )
        return 2

    dest = generate_music(
        caption=args.caption,
        lyrics=lyrics,
        model=args.model,
        duration=args.duration,
        steps=args.steps,
        seed=args.seed,
        output_path=args.out,
        verbose=not args.quiet,
    )
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
