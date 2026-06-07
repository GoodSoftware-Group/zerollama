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
    # SM120 flash_attn builds often SIGABRT; SDPA is stable (see scripts/patch_wan_attention_sdpa.py).
    if env.get("WAN_FORCE_SDPA", "").lower() not in ("1", "true", "yes"):
        if os.environ.get("ZEROLLAMA_WAN_FORCE_SDPA", "1").lower() not in ("0", "false", "no"):
            env["WAN_FORCE_SDPA"] = "1"
    # SM120 cuDNN conv workaround: see scripts/wan_torch_compat.py (probe at install).
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


def main() -> int:
    profile = os.environ.get("WAN_PROFILE", "")
    repo = os.environ.get("WAN_REPO", "")
    ckpt = os.environ.get("WAN_CKPT_DIR", "")
    prompt = os.environ.get("WAN_PROMPT", "")
    size = os.environ.get("WAN_SIZE", "832x480")
    frames = os.environ.get("WAN_FRAMES", "49")
    steps = os.environ.get("WAN_STEPS", "25")
    out_path = os.environ.get("WAN_OUTPUT_PATH", "")
    offload = os.environ.get("WAN_OFFLOAD_MODEL", "false").lower() in ("1", "true", "yes")
    t5_cpu = os.environ.get("WAN_T5_CPU", "false").lower() in ("1", "true", "yes")
    seed = os.environ.get("WAN_SEED", "")
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

    progress(0, "starting Wan generation")

    eprint(f"using python: {python_bin}")
    sub_env = prepare_wan_subprocess_env(python_bin)
    eprint(f"WAN_FORCE_SDPA={sub_env.get('WAN_FORCE_SDPA', '')}")
    eprint(f"WAN_VAE_CPU={sub_env.get('WAN_VAE_CPU', '')}")
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
        # Tell Wan exactly where to write the output so we don't need to scrape
        "--save_file",
        str(out),
    ]
    if seed:
        cmd.extend(["--base_seed", seed])
    # Always pass offload explicitly; Wan defaults offload=True when omitted, which is
    # confusing when the manifest says false. On 16g, manifest/Go force both flags on.
    cmd.extend(["--offload_model", "True" if offload else "False"])
    if t5_cpu:
        cmd.append("--t5_cpu")

    # Wan2.2 TI2V README recommends --convert_model_dtype for VRAM on consumer GPUs.
    convert_dtype = os.environ.get("WAN_CONVERT_MODEL_DTYPE", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    if convert_dtype and profile.strip().lower().startswith("wan2.2"):
        cmd.append("--convert_model_dtype")

    eprint("running: " + " ".join(cmd))
    progress(5, "launching generate.py")

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
    last_pct = 5.0
    for line in proc.stdout:
        line = line.rstrip()
        pm = progress_line_re.match(line)
        if pm:
            pct = float(pm.group(1))
            msg = (pm.group(2) or "").strip() or "generating"
            if pct > last_pct:
                last_pct = pct
                progress(pct, msg)
            continue
        if line:
            print(line, flush=True)
        if bumped := progress_from_log_line(line, last_pct):
            last_pct = bumped
            continue
        m = progress_re.match(line)
        if m:
            tqdm_pct = float(m.group(1))
            pct = map_diffusion_progress(tqdm_pct)
            if pct > last_pct:
                last_pct = pct
                progress(pct, "generating")
            if tqdm_pct >= 100 and last_pct < 91:
                last_pct = 91.0
                progress(91, "decoding video")

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
