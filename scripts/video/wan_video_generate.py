#!/usr/bin/env python3
"""Wan text-to-video wrapper for zerollama run_script jobs.

Why a wrapper instead of calling generate.py from Go:
  - Upstream CLI flags differ by Wan version (e.g. Wan2.2 --convert_model_dtype).
  - We need PROGRESS: lines and TRAINING_COMPLETE for the existing training worker protocol.
  - The Wan venv must run generate.py; the embedded training interpreter does not ship PyTorch/Wan.

Contract:
  - Env from server/video_generate.go (WAN_*); output at WAN_OUTPUT_PATH (--save_file).
  - No "find latest mp4" fallback—serving a stale file from an old run would be a security/UX bug.
  - Inner wait bounded by WAN_SUBPROCESS_TIMEOUT (aligned with job timeout).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def progress(pct: float, msg: str) -> None:
    print(f"PROGRESS:{pct:.1f}:{msg}", flush=True)


# Upstream log lines → progress before tqdm diffusion (5–30%).
_LOAD_LOG_PROGRESS: list[tuple[re.Pattern[str], float, str]] = [
    (re.compile(r"Creating WanT2V pipeline", re.I), 8, "creating pipeline"),
    (re.compile(r"loading .*t5", re.I), 12, "loading T5 encoder"),
    (re.compile(r"loading .*VAE", re.I), 18, "loading VAE"),
    (re.compile(r"Creating WanModel", re.I), 22, "loading diffusion model"),
    (re.compile(r"Generating video", re.I), 26, "starting diffusion"),
    (re.compile(r"released T5 encoder", re.I), 28, "prompt encoded"),
    (re.compile(r"VAE decode on CPU", re.I), 91, "decoding video"),
    (re.compile(r"Saving generated video", re.I), 94, "writing mp4"),
]


def progress_from_log_line(line: str, last_pct: float) -> float | None:
    for pattern, pct, msg in _LOAD_LOG_PROGRESS:
        if pattern.search(line):
            if pct > last_pct:
                progress(pct, msg)
                return pct
            return None
    return None


def map_diffusion_progress(tqdm_pct: float) -> float:
    """Map Wan 0–100% diffusion bar into wrapper band 30–90%."""
    return 30.0 + max(0.0, min(100.0, tqdm_pct)) * 0.60


def wan_size(size: str) -> str:
    """Wan CLI uses WxH with asterisk, e.g. 832*480."""
    return size.replace("x", "*").replace("X", "*")


def profile_task(profile: str) -> str:
    p = profile.strip().lower()
    if p in ("wan2.1-t2v-1.3b", "wan2.1-t2v", "t2v-1.3b"):
        return "t2v-1.3B"
    if p in ("wan2.2-ti2v-5b", "wan2.2-ti2v", "ti2v-5b"):
        return "ti2v-5B"
    raise ValueError(f"unsupported WAN_PROFILE: {profile}")


def wan_python() -> str:
    """Return the Python interpreter to use for generate.py.

    Priority (highest first):
    1. WAN_PYTHON env var — operator-specified interpreter or venv path
    2. WAN_VENV env var — directory of a venv; uses <venv>/bin/python3
    3. <WAN_REPO>/../venv/bin/python3 — convention from install_wan_video.sh
    4. sys.executable — fallback (likely wrong if running from embedded Python)
    """
    if p := os.environ.get("WAN_PYTHON", "").strip():
        return p
    if venv := os.environ.get("WAN_VENV", "").strip():
        return str(Path(venv) / "bin" / "python3")
    wan_repo = os.environ.get("WAN_REPO", "").strip()
    if wan_repo:
        default_venv = Path(wan_repo).parent / "venv" / "bin" / "python3"
        if default_venv.is_file():
            return str(default_venv)
    return sys.executable


def wan_subprocess_env() -> dict[str, str]:
    """Env for generate.py: Wan venv torch only, not embed training PYTHONPATH."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    if venv := os.environ.get("WAN_VENV", "").strip():
        env["VIRTUAL_ENV"] = str(Path(venv).expanduser())
    repo = os.environ.get("WAN_REPO", "").strip()
    if repo:
        env["PYTHONPATH"] = str(Path(repo).expanduser())
    # SM120 flash_attn builds often SIGABRT; SDPA is stable (see scripts/video/patch_wan_attention_sdpa.py).
    if env.get("WAN_FORCE_SDPA", "").lower() not in ("1", "true", "yes"):
        if os.environ.get("ZEROLLAMA_WAN_FORCE_SDPA", "1").lower() not in ("0", "false", "no"):
            env["WAN_FORCE_SDPA"] = "1"
    # SM120 cuDNN conv workaround: see scripts/video/wan_torch_compat.py (probe at install).
    if v := os.environ.get("WAN_DISABLE_CUDNN", "").strip():
        env["WAN_DISABLE_CUDNN"] = v
    elif os.environ.get("ZEROLLAMA_WAN_DISABLE_CUDNN", "").strip():
        env["WAN_DISABLE_CUDNN"] = os.environ["ZEROLLAMA_WAN_DISABLE_CUDNN"].strip()
    if os.environ.get("ZEROLLAMA_WAN_VAE_CPU", "1").lower() in ("0", "false", "no"):
        if v := os.environ.get("WAN_VAE_CPU", "").strip():
            env["WAN_VAE_CPU"] = v
        else:
            env["WAN_VAE_CPU"] = "0"
    else:
        # Default on: GPU VAE needs ~15G contiguous VRAM on 16g cards (overrides stale job env).
        env["WAN_VAE_CPU"] = "1"
    if v := os.environ.get("WAN_UNLOAD_T5", "").strip():
        env["WAN_UNLOAD_T5"] = v
    elif os.environ.get("ZEROLLAMA_WAN_UNLOAD_T5", "1").lower() not in ("0", "false", "no"):
        env.setdefault("WAN_UNLOAD_T5", "1")
    return env


def prepare_wan_subprocess_env(python_bin: str) -> dict[str, str]:
    """Build env for wan_generate_entry; sanitize LD so torch uses bundled cuDNN."""
    env = wan_subprocess_env()
    from wan_torch_compat import sanitize_ld_library_path_for_pytorch

    return sanitize_ld_library_path_for_pytorch(env, python=python_bin)


def list_keyframe_images(keyframe_dir: Path) -> list[Path]:
    if not keyframe_dir.is_dir():
        return []
    files = [
        p
        for p in sorted(keyframe_dir.iterdir())
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    ]
    return files


def build_generate_cmd(
    *,
    python_bin: str,
    entry_py: Path,
    task: str,
    size: str,
    ckpt_path: Path,
    prompt: str,
    frames: str,
    steps: str,
    out: Path,
    seed: str,
    offload: bool,
    t5_cpu: bool,
    convert_dtype: bool,
    profile: str,
    image: Path | None,
) -> list[str]:
    cmd = [
        python_bin,
        str(entry_py),
        "--task",
        task,
        "--size",
        wan_size(size),
        "--ckpt_dir",
        str(ckpt_path),
        "--prompt",
        prompt,
        "--frame_num",
        str(frames),
        "--sample_steps",
        str(steps),
        "--save_file",
        str(out),
    ]
    if seed:
        cmd.extend(["--base_seed", seed])
    cmd.extend(["--offload_model", "True" if offload else "False"])
    if t5_cpu:
        cmd.append("--t5_cpu")
    if convert_dtype and profile.strip().lower().startswith("wan2.2"):
        cmd.append("--convert_model_dtype")
    if image is not None:
        cmd.extend(["--image", str(image)])
    return cmd


def run_generate(
    cmd: list[str],
    *,
    repo_path: Path,
    sub_env: dict[str, str],
    progress_lo: float,
    progress_hi: float,
) -> int:
    """Run one generate.py invocation; map tqdm into [progress_lo, progress_hi]."""
    eprint("running: " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_path),
        env=sub_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    progress_re = re.compile(r"^\s*(\d+)%\s*\|", re.I)
    progress_line_re = re.compile(r"^PROGRESS:(\d+(?:\.\d+)?):?(.*)$")
    span = max(0.1, progress_hi - progress_lo)
    last_pct = progress_lo

    def remap(wrapper_pct: float) -> float:
        # Map legacy 5–100 wrapper band into this segment's band.
        frac = max(0.0, min(1.0, (wrapper_pct - 5.0) / 95.0))
        return progress_lo + frac * span

    for line in proc.stdout:
        line = line.rstrip()
        pm = progress_line_re.match(line)
        if pm:
            pct = remap(float(pm.group(1)))
            msg = (pm.group(2) or "").strip() or "generating"
            if pct > last_pct:
                last_pct = pct
                progress(pct, msg)
            continue
        if line:
            print(line, flush=True)
        if bumped := progress_from_log_line(line, last_pct):
            # progress_from_log_line already emitted absolute pct; remap roughly
            last_pct = max(last_pct, remap(bumped))
            continue
        m = progress_re.match(line)
        if m:
            tqdm_pct = float(m.group(1))
            pct = progress_lo + (tqdm_pct / 100.0) * span * 0.9
            if pct > last_pct:
                last_pct = pct
                progress(pct, "generating")

    timeout_raw = os.environ.get("WAN_SUBPROCESS_TIMEOUT", "").strip()
    wait_timeout: float | None = None
    if timeout_raw:
        try:
            wait_timeout = max(1.0, float(timeout_raw))
        except ValueError:
            pass

    try:
        code = proc.wait(timeout=wait_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        eprint("generate.py timed out")
        return 1
    return code


def parse_wxh(size: str) -> tuple[int, int]:
    s = size.lower().replace("*", "x")
    parts = s.split("x")
    if len(parts) != 2:
        return 832, 480
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return 832, 480


def ffmpeg_concat(segment_paths: list[Path], out: Path) -> int:
    """Concatenate segment MP4s; try stream copy then re-encode fallback.

    WHY -c copy first: cheap when Wan segments share codec/timebase.
    WHY libx264 fallback: per-segment Wan outputs often disagree on SPS/PPS;
    failing concat after expensive generation is worse than a re-encode.
    """
    list_file = out.parent / f"{out.stem}_concat.txt"
    lines = []
    for p in segment_paths:
        esc = str(p.resolve()).replace("'", "'\\''")
        lines.append(f"file '{esc}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run_concat(extra: list[str]) -> subprocess.CompletedProcess[str]:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            *extra,
            str(out),
        ]
        eprint("running: " + " ".join(cmd))
        return subprocess.run(cmd, capture_output=True, text=True)

    proc = run_concat(["-c", "copy"])
    if proc.returncode != 0:
        eprint(proc.stdout or "")
        eprint(proc.stderr or "")
        eprint("ffmpeg stream-copy concat failed; retrying with re-encode")
        proc = run_concat(
            [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-preset",
                "veryfast",
                "-an",
            ]
        )
        if proc.returncode != 0:
            eprint(proc.stdout or "")
            eprint(proc.stderr or "")
            eprint(f"ffmpeg concat failed with code {proc.returncode}")
            return proc.returncode
    try:
        list_file.unlink(missing_ok=True)
    except OSError:
        pass
    return 0


def ffmpeg_still_clip(image: Path, out: Path, size: str, duration_sec: float = 0.5) -> int:
    """Encode a short still clip from an image so the timeline can end on the final keyframe."""
    w, h = parse_wxh(size)
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image),
        "-t",
        str(duration_sec),
        "-vf",
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "16",
        "-an",
        str(out),
    ]
    eprint("running: " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        eprint(proc.stdout or "")
        eprint(proc.stderr or "")
        eprint(f"ffmpeg still clip failed with code {proc.returncode}")
    return proc.returncode


def cleanup_keyframe_dir() -> None:
    # WHY opt-in VIDEO_CLEANUP_KEYFRAME_DIR: Go sets this when it staged the dir;
    # refuse to rmtree paths outside .../keyframes/ so a mis-set env cannot wipe arbitrary trees.
    if os.environ.get("VIDEO_CLEANUP_KEYFRAME_DIR", "").lower() not in ("1", "true", "yes"):
        return
    raw = os.environ.get("WAN_KEYFRAME_DIR") or os.environ.get("VIDEO_KEYFRAME_DIR") or ""
    if not raw:
        return
    path = Path(raw).expanduser()
    # Only remove dirs we staged under .../generated/keyframes/
    if "keyframes" not in path.parts:
        eprint(f"refusing to cleanup keyframe dir outside keyframes/: {path}")
        return
    try:
        import shutil

        shutil.rmtree(path, ignore_errors=True)
        eprint(f"cleaned keyframe staging dir {path}")
    except OSError as err:
        eprint(f"keyframe cleanup failed: {err}")


def main() -> int:
    try:
        return _main_impl()
    finally:
        cleanup_keyframe_dir()


def _main_impl() -> int:
    profile = os.environ.get("WAN_PROFILE", "")
    repo = os.environ.get("WAN_REPO", "")
    ckpt = os.environ.get("WAN_CKPT_DIR", "")
    prompt = os.environ.get("WAN_PROMPT", "")
    size = os.environ.get("WAN_SIZE") or os.environ.get("VIDEO_SIZE") or "832x480"
    frames = os.environ.get("WAN_FRAMES") or os.environ.get("VIDEO_FRAMES") or "49"
    steps = os.environ.get("WAN_STEPS", "25")
    out_path = os.environ.get("WAN_OUTPUT_PATH") or os.environ.get("VIDEO_OUTPUT_PATH") or ""
    offload = os.environ.get("WAN_OFFLOAD_MODEL", "false").lower() in ("1", "true", "yes")
    t5_cpu = os.environ.get("WAN_T5_CPU", "false").lower() in ("1", "true", "yes")
    seed = os.environ.get("WAN_SEED") or os.environ.get("VIDEO_SEED") or ""
    image_env = os.environ.get("WAN_IMAGE") or os.environ.get("VIDEO_IMAGE") or ""
    keyframe_dir_env = (
        os.environ.get("WAN_KEYFRAME_DIR") or os.environ.get("VIDEO_KEYFRAME_DIR") or ""
    )
    python_bin = wan_python()

    if not repo or not ckpt or not prompt or not out_path:
        eprint("WAN_REPO, WAN_CKPT_DIR, WAN_PROMPT, and WAN_OUTPUT_PATH are required")
        return 1

    repo_path = Path(repo).expanduser().resolve()
    ckpt_path = Path(ckpt).expanduser().resolve()
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    generate_py = repo_path / "generate.py"
    entry_py = Path(__file__).resolve().parent / "wan_generate_entry.py"
    if not generate_py.is_file():
        eprint(f"upstream generate.py not found under {repo_path}")
        return 1
    if not entry_py.is_file():
        eprint(f"entry script not found: {entry_py}")
        return 1
    if not ckpt_path.is_dir():
        eprint(f"checkpoint dir not found: {ckpt_path}")
        return 1

    try:
        task = profile_task(profile)
    except ValueError as err:
        eprint(str(err))
        return 1

    keyframes = list_keyframe_images(Path(keyframe_dir_env).expanduser()) if keyframe_dir_env else []
    single_image: Path | None = None
    if image_env:
        single_image = Path(image_env).expanduser()
        if not single_image.is_file():
            eprint(f"WAN_IMAGE not found: {single_image}")
            return 1

    # Multi-keyframe: N images → N−1 start-conditioned segments, then append final still
    # so the timeline ends on the last keyframe (true FLF end-conditioning is a later runner).
    #
    # WHY not skip the final label: earlier drafts only conditioned on starts 0..N-2, leaving
    # the agent's last keyframe unused. The still is visual end-frame enforcement until FLF.
    if len(keyframes) >= 2:
        if "ti2v" not in task:
            eprint("multi-keyframe generation requires a TI2V task profile")
            return 1
        progress(0, f"starting Wan multi-keyframe ({len(keyframes)} images)")
        eprint(f"using python: {python_bin}")
        sub_env = prepare_wan_subprocess_env(python_bin)
        convert_dtype = os.environ.get("WAN_CONVERT_MODEL_DTYPE", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        segments: list[Path] = []
        n_seg = len(keyframes) - 1
        for i in range(n_seg):
            seg_out = out.parent / f"{out.stem}_seg{i:03d}.mp4"
            lo = 5.0 + (85.0 * i / n_seg)
            hi = 5.0 + (85.0 * (i + 1) / n_seg)
            progress(lo, f"segment {i + 1}/{n_seg} (start={keyframes[i].name})")
            cmd = build_generate_cmd(
                python_bin=python_bin,
                entry_py=entry_py,
                task=task,
                size=size,
                ckpt_path=ckpt_path,
                prompt=prompt,
                frames=frames,
                steps=steps,
                out=seg_out,
                seed=seed,
                offload=offload,
                t5_cpu=t5_cpu,
                convert_dtype=convert_dtype,
                profile=profile,
                image=keyframes[i],
            )
            code = run_generate(
                cmd, repo_path=repo_path, sub_env=sub_env, progress_lo=lo, progress_hi=hi
            )
            if code != 0:
                eprint(f"generate.py exited with code {code} on segment {i}")
                return code
            if not seg_out.is_file():
                eprint(f"expected segment output not found at {seg_out}")
                return 1
            segments.append(seg_out)

        progress(90, "encoding final keyframe still")
        still_out = out.parent / f"{out.stem}_final_still.mp4"
        code = ffmpeg_still_clip(keyframes[-1], still_out, size)
        if code != 0:
            return code
        segments.append(still_out)

        progress(94, "concatenating segments")
        code = ffmpeg_concat(segments, out)
        if code != 0:
            return code
        for seg in segments:
            try:
                seg.unlink(missing_ok=True)
            except OSError:
                pass
        progress(100, "done")
        print("TRAINING_COMPLETE", flush=True)
        return 0

    progress(0, "starting Wan generation")
    eprint(f"using python: {python_bin}")
    sub_env = prepare_wan_subprocess_env(python_bin)
    convert_dtype = os.environ.get("WAN_CONVERT_MODEL_DTYPE", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    image = single_image
    if image is None and len(keyframes) == 1:
        image = keyframes[0]
    if image is not None and "ti2v" not in task and "i2v" not in task:
        eprint("WAN_IMAGE / keyframes require a TI2V (or I2V) task profile")
        return 1

    progress(5, "launching generate.py")
    cmd = build_generate_cmd(
        python_bin=python_bin,
        entry_py=entry_py,
        task=task,
        size=size,
        ckpt_path=ckpt_path,
        prompt=prompt,
        frames=frames,
        steps=steps,
        out=out,
        seed=seed,
        offload=offload,
        t5_cpu=t5_cpu,
        convert_dtype=convert_dtype,
        profile=profile,
        image=image,
    )
    code = run_generate(cmd, repo_path=repo_path, sub_env=sub_env, progress_lo=5.0, progress_hi=96.0)
    if code != 0:
        eprint(f"generate.py exited with code {code}")
        return code

    progress(96, "collecting output")
    if not out.is_file():
        eprint(f"expected output not found at {out} (--save_file was passed to generate.py)")
        return 1

    progress(100, "done")
    print("TRAINING_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
