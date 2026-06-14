"""Phase 15 v25 — native ext build / auto-link policy tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent.parent


def test_setup_forced_link_fails_without_libllama():
    """ZEROLLAMA_KV_DECODE_LOOP=1 must exit non-zero when libllama is absent."""
    env = os.environ.copy()
    env["ZEROLLAMA_KV_DECODE_LOOP"] = "1"
    env["LLAMA_CPP_ROOT"] = "/nonexistent/llama.cpp"
    env.pop("LLAMA_CPP_LIB", None)
    proc = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=RUNTIME_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined
    assert "libllama not found" in combined


def test_setup_unlinked_build_succeeds_without_libllama():
    """ZEROLLAMA_KV_DECODE_LOOP=0 builds the base ext on machines without libllama."""
    env = os.environ.copy()
    env["ZEROLLAMA_KV_DECODE_LOOP"] = "0"
    env["LLAMA_CPP_ROOT"] = "/nonexistent/llama.cpp"
    env.pop("LLAMA_CPP_LIB", None)
    proc = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=RUNTIME_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
