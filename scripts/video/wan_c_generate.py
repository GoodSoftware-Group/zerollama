#!/usr/bin/env python3
"""run_script entry for Pure-C video-cli (ZEROLLAMA_VIDEO_CLI / ZEROLLAMA_WAN_CLI).

Mirrors wan_video_generate.py: read WAN_* env, emit PROGRESS, exec video-cli.
Clients never set the runner — operator env / manifest backend_paths only.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def parse_size(size: str) -> tuple[str, str]:
    raw = (size or "").strip().lower().replace("×", "x")
    if "*" in raw:
        w_s, h_s = raw.split("*", 1)
    elif "x" in raw:
        w_s, h_s = raw.split("x", 1)
    else:
        return "480", "832"
    w_s, h_s = w_s.strip(), h_s.strip()
    if not w_s or not h_s:
        return "480", "832"
    return w_s, h_s


def build_cmd(env: dict[str, str]) -> list[str] | None:
    cli = (
        env.get("VIDEO_CLI")
        or env.get("ZEROLLAMA_VIDEO_CLI")
        or env.get("WAN_CLI")
        or env.get("ZEROLLAMA_WAN_CLI")
        or ""
    )
    if not cli:
        return None
    out = env.get("WAN_OUTPUT_PATH") or env.get("VIDEO_OUTPUT_PATH")
    ckpt = env.get("WAN_CKPT_DIR")
    prompt = env.get("WAN_PROMPT")
    if not out or not ckpt or not prompt:
        return None
    w_s, h_s = parse_size(env.get("WAN_SIZE") or env.get("VIDEO_SIZE") or "")
    frames = env.get("WAN_FRAMES") or env.get("VIDEO_FRAMES") or "81"
    steps = env.get("WAN_STEPS", "50")
    cfg = env.get("WAN_CFG", "5.0")
    shift = env.get("WAN_SHIFT", "5.0")
    seed = env.get("WAN_SEED") or env.get("VIDEO_SEED")
    vocab = env.get("WAN_C_VOCAB")
    sock = env.get("UMA_SOCK", "/tmp/uma_daemon.sock")
    neg = env.get("WAN_NEG_PROMPT")
    family = env.get("VIDEO_FAMILY", "wan")
    cmd = [
        cli,
        "--family",
        family,
        "--ckpt-dir",
        ckpt,
        "--prompt",
        prompt,
        "--width",
        w_s,
        "--height",
        h_s,
        "--frames",
        frames,
        "--steps",
        steps,
        "--cfg",
        cfg,
        "--shift",
        shift,
        "--uma-sock",
        sock,
        "--out",
        out,
    ]
    if seed:
        cmd.extend(["--seed", seed])
    if vocab and Path(vocab).is_file():
        cmd.extend(["--vocab", vocab])
    if neg:
        cmd.extend(["--negative-prompt", neg])
    if family == "h3":
        cmd.append("--generate")
        layers = env.get("VIDEO_H3_LAYERS") or env.get("H3_DIT_LAYERS") or "50"
        cmd.extend(["--layers", layers])
        reuse = env.get("VIDEO_H3_REUSE") or env.get("H3_REUSE")
        if reuse:
            cmd.extend(["--reuse", reuse])
    return cmd


def main() -> int:
    cli = (
        os.environ.get("VIDEO_CLI")
        or os.environ.get("ZEROLLAMA_VIDEO_CLI")
        or os.environ.get("WAN_CLI")
        or os.environ.get("ZEROLLAMA_WAN_CLI")
    )
    if not cli or not Path(cli).is_file():
        print(f"ERROR: VIDEO_CLI/WAN_CLI missing or not a file: {cli!r}", file=sys.stderr)
        return 1
    cmd = build_cmd(dict(os.environ))
    if not cmd:
        print("ERROR: WAN_OUTPUT_PATH, WAN_CKPT_DIR, WAN_PROMPT required", file=sys.stderr)
        return 1
    print("PROGRESS: 5 starting video-cli", flush=True)
    print(" ".join(cmd), flush=True)
    run_env = os.environ.copy()
    if run_env.get("VIDEO_FAMILY", "wan") == "h3":
        try:
            nsteps = int(run_env.get("WAN_STEPS") or run_env.get("VIDEO_STEPS") or "0")
        except ValueError:
            nsteps = 0
        if nsteps >= 8:
            run_env.setdefault("H3_SAMPLER", "res_multistep")
    rc = subprocess.call(cmd, env=run_env)
    if rc == 0:
        print("PROGRESS: 100 done", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
