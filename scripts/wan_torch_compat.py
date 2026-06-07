#!/usr/bin/env python3
"""SM120 (RTX 50xx) PyTorch/cuDNN compatibility for Wan video generation.

On torch 2.11+cu128 with RTX 5080 (compute 12.0), cuDNN-backed convolutions can
SIGABRT with ``free(): invalid pointer``. PyTorch's native CUDA conv path (cuDNN
disabled) is correct but slower. We probe once per torch/cuDNN build and cache
the result so runtime and install can apply the right workaround automatically.

Set WAN_DISABLE_CUDNN=1|0 to override probe/cache.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROBE_VERSION = 1
DEFAULT_PROBE_REL = ".zerollama/third_party/wan/.wan_torch_probe.json"


def torch_bundled_lib_dir(*, python: str | None = None) -> Path | None:
    """Locate site-packages/torch/lib without importing torch."""
    py = Path(python or sys.executable).resolve()
    venv_root = py.parent.parent if py.name.startswith("python") else py.parent
    for candidate in sorted(venv_root.glob("lib/python*/site-packages/torch/lib")):
        if candidate.is_dir():
            return candidate
    return None


def _path_may_shadow_cudnn(entry: str) -> bool:
    p = entry.strip()
    if not p:
        return False
    if "hostlibs" in p:
        return True
    if "cudnn" in Path(p).name.lower():
        return True
    try:
        d = Path(p)
        if d.is_dir() and any(d.glob("libcudnn.so*")):
            return True
    except OSError:
        pass
    return False


def sanitize_ld_library_path_for_pytorch(
    env: dict[str, str] | None = None,
    *,
    python: str | None = None,
) -> dict[str, str]:
    """Prepend torch/lib and drop LD entries that shadow bundled cuDNN.

    Why: zerollama serve sets LD_LIBRARY_PATH for ggml (often /usr/hostlibs with
    older libcudnn). PyTorch wheels ship a matching cuDNN; shadowing raises
    RuntimeError before the SM120 conv probe can run.
    """
    target = env if env is not None else os.environ
    torch_lib = torch_bundled_lib_dir(python=python)
    if torch_lib is None:
        return target
    prefix = str(torch_lib)
    raw = target.get("LD_LIBRARY_PATH", "")
    parts = [p for p in raw.split(":") if p and p != prefix]
    filtered = [p for p in parts if not _path_may_shadow_cudnn(p)]
    target["LD_LIBRARY_PATH"] = ":".join([prefix, *filtered]) if filtered else prefix
    return target


def safe_cudnn_version() -> int | None:
    import torch

    if not torch.cuda.is_available():
        return None
    try:
        return torch.backends.cudnn.version()
    except RuntimeError:
        return None


def probe_cache_path() -> Path:
    env = os.environ.get("WAN_TORCH_PROBE_CACHE", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / DEFAULT_PROBE_REL


def _probe_script() -> str:
    return """
import torch
import torch.nn as nn
if not torch.cuda.is_available():
    raise SystemExit(2)
cap = torch.cuda.get_device_capability(0)
if cap[0] < 12:
    raise SystemExit(3)
torch.backends.cudnn.enabled = True
x = torch.randn(1, 3, 16, 16, device="cuda")
nn.Conv2d(3, 8, 3).cuda()(x)
print("ok")
"""


def run_cudnn_conv_probe(python: str | None = None) -> dict[str, Any]:
    """Run isolated subprocess conv test; return probe record."""
    py = python or sys.executable
    sanitize_ld_library_path_for_pytorch(python=py)
    import torch

    record: dict[str, Any] = {
        "probe_version": PROBE_VERSION,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": safe_cudnn_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None,
        "cudnn_conv_ok": None,
        "disable_cudnn_recommended": False,
    }
    if not torch.cuda.is_available():
        record["cudnn_conv_ok"] = None
        record["note"] = "no cuda"
        return record
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 12:
        record["cudnn_conv_ok"] = True
        record["note"] = "pre-sm120"
        return record

    try:
        probe_env = os.environ.copy()
        sanitize_ld_library_path_for_pytorch(probe_env, python=py)
        proc = subprocess.run(
            [py, "-c", _probe_script()],
            capture_output=True,
            text=True,
            timeout=60,
            env=probe_env,
        )
    except subprocess.TimeoutExpired:
        record["cudnn_conv_ok"] = False
        record["disable_cudnn_recommended"] = True
        record["note"] = "probe timeout"
        return record

    if proc.returncode == 0 and "ok" in proc.stdout:
        record["cudnn_conv_ok"] = True
        record["disable_cudnn_recommended"] = False
        return record

    record["cudnn_conv_ok"] = False
    record["disable_cudnn_recommended"] = True
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    record["probe_error"] = err[-1] if err else f"exit {proc.returncode}"
    return record


def load_probe_cache(path: Path | None = None) -> dict[str, Any] | None:
    p = path or probe_cache_path()
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("probe_version") != PROBE_VERSION:
        return None
    import torch

    if data.get("torch") != torch.__version__:
        return None
    cudnn = safe_cudnn_version()
    if cudnn is None or data.get("cudnn") != cudnn:
        return None
    return data


def save_probe_cache(record: dict[str, Any], path: Path | None = None) -> Path:
    p = path or probe_cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2) + "\n")
    return p


def should_disable_cudnn(
    *,
    python: str | None = None,
    cache_path: Path | None = None,
) -> bool:
    """Return True when cuDNN conv must be disabled for this torch/GPU combo."""
    override = os.environ.get("WAN_DISABLE_CUDNN", "").strip().lower()
    if override in ("1", "true", "yes"):
        return True
    if override in ("0", "false", "no"):
        return False

    cached = load_probe_cache(cache_path)
    if cached is not None and cached.get("disable_cudnn_recommended") is not None:
        return bool(cached["disable_cudnn_recommended"])

    record = run_cudnn_conv_probe(python)
    try:
        save_probe_cache(record, cache_path)
    except OSError:
        pass
    return bool(record.get("disable_cudnn_recommended"))


def apply_torch_workarounds(*, python: str | None = None) -> dict[str, Any]:
    """Apply SM120 workarounds in the current process. Returns status dict."""
    sanitize_ld_library_path_for_pytorch(python=python)
    import torch

    status: dict[str, Any] = {
        "torch": torch.__version__,
        "cudnn_disabled": False,
        "vae_cpu_patched": False,
    }
    if should_disable_cudnn(python=python):
        torch.backends.cudnn.enabled = False
        status["cudnn_disabled"] = True
    return status


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    python = sys.executable
    sanitize_ld_library_path_for_pytorch(python=python)
    cache = probe_cache_path()
    if args and args[0] == "--print-cache":
        data = load_probe_cache(cache)
        print(json.dumps(data or {}, indent=2))
        return 0
    record = run_cudnn_conv_probe(python)
    path = save_probe_cache(record, cache)
    print(json.dumps(record, indent=2))
    print(f"cache: {path}", file=sys.stderr)
    if record.get("disable_cudnn_recommended"):
        print(
            "SM120 cuDNN conv probe FAILED — Wan will disable cuDNN at runtime "
            "(native CUDA conv). Track upstream: torch+cuDNN on compute 12.0.",
            file=sys.stderr,
        )
        return 1
    print("SM120 cuDNN conv probe OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
