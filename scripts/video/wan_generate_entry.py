#!/usr/bin/env python3
"""Run upstream Wan generate.py with SM120-safe PyTorch settings.

Probes cuDNN conv on Blackwell (5080 class) and disables cuDNN only when the
installed torch+cuDNN build fails (``free(): invalid pointer``). Native CUDA
conv still uses the GPU. Override with WAN_DISABLE_CUDNN=1|0.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from wan_memory_hooks import apply_memory_hooks  # noqa: E402
from wan_mps_compat import apply_before_wan_import  # noqa: E402
from wan_torch_compat import apply_torch_workarounds, sanitize_ld_library_path_for_pytorch  # noqa: E402


def _normalize_vae_cpu_env() -> None:
    """Default CPU VAE on unless explicitly disabled (16g cards OOM on GPU decode)."""
    if os.environ.get("WAN_VAE_CPU", "1").lower() not in ("0", "false", "no"):
        os.environ["WAN_VAE_CPU"] = "1"


def main() -> int:
    sanitize_ld_library_path_for_pytorch()
    _normalize_vae_cpu_env()
    print("PROGRESS:6.0:checking GPU compatibility", flush=True)
    status = apply_torch_workarounds()
    if status.get("cudnn_disabled"):
        print(
            f"WAN: cuDNN disabled for SM120 ({status.get('torch')}); using native CUDA conv",
            file=sys.stderr,
            flush=True,
        )

    repo = os.environ.get("WAN_REPO", "").strip()
    if not repo:
        print("WAN_REPO is required", file=sys.stderr)
        return 1
    repo_path = Path(repo).expanduser().resolve()
    generate_py = repo_path / "generate.py"
    if not generate_py.is_file():
        print(f"generate.py not found under {repo_path}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(repo_path))
    print("PROGRESS:7.0:configuring Wan runtime", flush=True)
    apply_before_wan_import(repo_path)
    apply_memory_hooks()
    sys.argv = [str(generate_py), *sys.argv[1:]]
    runpy.run_path(str(generate_py), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
