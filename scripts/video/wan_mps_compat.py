"""Apple Silicon (MPS) shims for upstream Wan2.1 (CUDA-only defaults).

Why this file exists:
  Wan's `t5.py` evaluates `device=torch.cuda.current_device()` at *class body*
  import time — MPS torch has no CUDA, so import AssertionError's before generate
  runs. `text2video.py` hardcodes `cuda:{id}` for the DiT/VAE device.

Applied from `wan_generate_entry.py` before `apply_memory_hooks()` / generate.py.
No-op when CUDA is available.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def mps_available() -> bool:
    try:
        import torch

        return bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def needs_mps_shim() -> bool:
    try:
        import torch

        if torch.cuda.is_available():
            return False
    except Exception:
        return False
    return mps_available() or sys.platform == "darwin"


def _replace_once(path: Path, old: str, new: str, label: str) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if label in text:
        return False
    if old not in text:
        return False
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"WAN MPS: patched {path.name} ({label})", file=sys.stderr, flush=True)
    return True


def patch_wan_sources(repo: Path) -> None:
    """Idempotent source patches so Wan imports and runs on MPS."""
    # 1) Class-default must not call cuda.current_device() at import time.
    _replace_once(
        repo / "wan" / "modules" / "t5.py",
        "device=torch.cuda.current_device(),",
        "device=None,  # zerollama-mps:t5-default",
        "zerollama-mps:t5-default",
    )

    # 2) Device picker — keep in sync if an older helper is already present.
    t2v = repo / "wan" / "text2video.py"
    if t2v.is_file():
        text = t2v.read_text(encoding="utf-8")
        new_helper = '''def _zerollama_wan_device(device_id=0):
    import os
    # Prefer CUDA; Darwin defaults to MPS + float32 DiT. WAN_FORCE_CPU=1 for CPU-only.
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    if os.environ.get("WAN_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return torch.device("cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
'''
        if "zerollama-mps:device-helper" in text:
            import re

            text2, n = re.subn(
                r"def _zerollama_wan_device\(device_id=0\):.*?return torch\.device\(\"cpu\"\)\n",
                new_helper,
                text,
                count=1,
                flags=re.S,
            )
            if n:
                t2v.write_text(text2, encoding="utf-8")
                print("WAN MPS: refreshed text2video.py device helper", file=sys.stderr, flush=True)
            text = text2 if n else text
        else:
            helper = "\n# zerollama-mps:device-helper\n" + new_helper + "\n"
            needle = "class WanT2V:"
            if needle in text:
                text = text.replace(needle, helper + needle, 1)
                text = text.replace(
                    'self.device = torch.device(f"cuda:{device_id}")',
                    "self.device = _zerollama_wan_device(device_id)  # zerollama-mps:device",
                    1,
                )
                t2v.write_text(text, encoding="utf-8")
                print("WAN MPS: patched text2video.py device", file=sys.stderr, flush=True)

    # 3) VAE default device — CUDA or CPU (MPS DiT is opt-in via WAN_ALLOW_MPS).
    _replace_once(
        repo / "wan" / "modules" / "vae.py",
        'device="cuda"):',
        'device=("cuda" if torch.cuda.is_available() else "cpu"):  # zerollama-mps:vae-device',
        "zerollama-mps:vae-device",
    )

    # 4) RoPE freqs / sinusoidal embeddings use float64 — MPS rejects f64.
    model_py = repo / "wan" / "modules" / "model.py"
    if model_py.is_file():
        text = model_py.read_text(encoding="utf-8")
        if "zerollama-mps:freqs" not in text:
            old = "self.freqs = self.freqs.to(device)"
            new = (
                "self.freqs = (self.freqs.float().to(device) if getattr(device, 'type', '') == 'mps' "
                "else self.freqs.to(device))  # zerollama-mps:freqs"
            )
            if old in text:
                text = text.replace(old, new, 1)
        if "zerollama-mps:sinusoid" not in text:
            old_fn = '''def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x'''
            new_fn = '''def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    # zerollama-mps:sinusoid — f64 on CPU then cast back (MPS has no float64)
    dev = position.device
    position = position.detach().float().cpu().double()
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).double().div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).float()
    return x.to(dev)'''
            if old_fn in text:
                text = text.replace(old_fn, new_fn, 1)
                print("WAN MPS: patched model.py sinusoidal_embedding_1d", file=sys.stderr, flush=True)
        if "zerollama-mps:rope-apply" not in text:
            old_rope = '''        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))'''
            new_rope = '''        # zerollama-mps:rope-apply — MPS has no float64/complex128
        _rope_dtype = torch.float32 if getattr(x.device, "type", "") == "mps" else torch.float64
        x_i = torch.view_as_complex(x[i, :seq_len].to(_rope_dtype).reshape(
            seq_len, n, -1, 2))'''
            if old_rope in text:
                text = text.replace(old_rope, new_rope, 1)
                # also cast freqs slices to complex64 on mps inside the loop — full replace done in-tree
                print("WAN MPS: patched model.py rope_apply float64", file=sys.stderr, flush=True)
        model_py.write_text(text, encoding="utf-8")
        if "zerollama-mps:freqs" in text or "zerollama-mps:rope-apply" in text:
            print("WAN MPS: model.py freqs/sinusoid/rope patches present", file=sys.stderr, flush=True)


def apply_runtime_shims() -> None:
    """Make cuda.empty_cache/synchronize safe no-ops / MPS equivalents + dtype fix."""
    if not needs_mps_shim():
        return

    import torch

    if getattr(torch.cuda, "_zerollama_mps_shim", False):
        return

    _empty = torch.cuda.empty_cache
    _sync = torch.cuda.synchronize

    def empty_cache(*_a, **_k):
        if torch.cuda.is_available():
            return _empty()
        if mps_available() and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def synchronize(*_a, **_k):
        if torch.cuda.is_available():
            return _sync()
        if mps_available() and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    torch.cuda.empty_cache = empty_cache  # type: ignore[method-assign]
    torch.cuda.synchronize = synchronize  # type: ignore[method-assign]
    torch.cuda._zerollama_mps_shim = True  # type: ignore[attr-defined]

    # bf16×fp32 matmul fails on CPU/MPS — cast DiT to fp32 after T5 unload (see memory hooks).
    # Context dtype cast is also applied there so unload + encode wrappers compose.
    try:
        from wan import text2video as t2v

        if not getattr(t2v.WanT2V.__init__, "_zerollama_mps_dtype", False):
            _orig = t2v.WanT2V.__init__

            def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                _orig(self, *args, **kwargs)
                if torch.cuda.is_available():
                    return
                # Mark only — heavy float32 materialize happens after T5 release.
                self._zerollama_need_fp32 = True  # type: ignore[attr-defined]

            __init__._zerollama_mps_dtype = True  # type: ignore[attr-defined]
            t2v.WanT2V.__init__ = __init__  # type: ignore[method-assign]
    except Exception as e:
        print(f"WAN MPS: dtype shim skipped: {e}", file=sys.stderr, flush=True)

    print(f"WAN MPS: runtime shims; device will be {_resolve_device()}", file=sys.stderr, flush=True)


def _resolve_device(device_id: int = 0):
    import torch

    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    if os.environ.get("WAN_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        return torch.device("cpu")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_before_wan_import(repo: str | Path) -> None:
    """Call before importing wan.* — patches source + env."""
    if not needs_mps_shim():
        return
    repo_path = Path(repo).expanduser().resolve()
    patch_wan_sources(repo_path)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    apply_runtime_shims()
    print("WAN MPS: Apple Silicon path enabled", file=sys.stderr, flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Apply Wan Darwin/MPS source patches")
    ap.add_argument(
        "--repo",
        default=str(Path.home() / ".zerollama/third_party/wan/Wan2.1"),
        help="Path to Wan2.1 checkout",
    )
    ns = ap.parse_args()
    patch_wan_sources(Path(ns.repo))
    print("done", ns.repo)
