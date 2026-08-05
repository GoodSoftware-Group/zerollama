#!/usr/bin/env python3
"""run_script entry for Pure-C wan-cli (ZEROLLAMA_WAN_CLI / WAN_CLI).

Mirrors wan_video_generate.py: read WAN_* env, emit PROGRESS, exec wan-cli.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    cli = os.environ.get("WAN_CLI") or os.environ.get("ZEROLLAMA_WAN_CLI")
    if not cli or not Path(cli).is_file():
        print(f"ERROR: WAN_CLI missing or not a file: {cli!r}", file=sys.stderr)
        return 1

    out = os.environ.get("WAN_OUTPUT_PATH")
    ckpt = os.environ.get("WAN_CKPT_DIR")
    prompt = os.environ.get("WAN_PROMPT")
    if not out or not ckpt or not prompt:
        print("ERROR: WAN_OUTPUT_PATH, WAN_CKPT_DIR, WAN_PROMPT required", file=sys.stderr)
        return 1

    size = os.environ.get("WAN_SIZE", "480*832")
    if "*" in size:
        w_s, h_s = size.split("*", 1)
    else:
        w_s, h_s = "480", "832"
    frames = os.environ.get("WAN_FRAMES", "81")
    steps = os.environ.get("WAN_STEPS", "50")
    cfg = os.environ.get("WAN_CFG", "5.0")
    shift = os.environ.get("WAN_SHIFT", "5.0")
    seed = os.environ.get("WAN_SEED")
    vocab = os.environ.get("WAN_C_VOCAB")
    sock = os.environ.get("UMA_SOCK", "/tmp/uma_daemon.sock")
    neg = os.environ.get("WAN_NEG_PROMPT")

    cmd = [
        cli,
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

    print("PROGRESS: 5 starting wan-cli", flush=True)
    print(" ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc == 0:
        print("PROGRESS: 100 done", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
