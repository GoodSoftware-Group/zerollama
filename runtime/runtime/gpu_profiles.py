"""Per-GPU llama-server autotune (ROADMAP borrowings L1).

WHY this module exists
----------------------
Phase 13 estimates VRAM fit and suggests ``num_ctx``. Operators still need
hardware-appropriate *throughput* flags (batch, parallel slots, flash-attn,
MTP draft bounds). Eliza-v3 ships JSON per GPU class; we port that data without
requiring the elizaOS/llama.cpp fork (L2).

WHAT it does
------------
Loads JSON under ``runtime/configs/gpu/``, selects a profile by:

- **NVIDIA** — ``nvidia-smi`` name or VRAM bucket (``single_gpu.yaml`` /
  ``dual_4090.yaml`` only).
- **Intel Arc (Vulkan)** — ``vulkaninfo`` device name or ``ZEROLLAMA_GPU_PROFILE_ID``.
- **Apple Silicon** — ``hw.memsize`` tier (``apple_silicon.yaml`` only).

Merges sanitized flags into ``RuntimeConfig`` and ``llama_server_args()``.

STOCK LLAMA.CPP SAFETY (default)
--------------------------------
When fork is off (``runtime.llama_fork``), fork KV types → ``q8_0`` and
``ctx_checkpoints`` are stripped. When fork is on, ``_eliza_fork_*`` JSON
sections merge and emit fork argv. See ``docs/gpu-profiles-l1.md`` / ``l2.md``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.nvidia_probe import (
    detect_gpu_total_vram_gb,
    detect_nvidia_gpu_name,
)
from runtime.llama_fork import (
    ELIZA_FORK_CACHE_TYPES,
    llama_fork_enabled,
    normalize_cache_type,
)

# elizaOS/llama.cpp kernel cache types — not in stock ggml yet (ROADMAP L2).
# WHY rewrite at load: stock llama-server rejects unknown --cache-type-* values.
_FORK_ONLY_CACHE_TYPES = ELIZA_FORK_CACHE_TYPES
# eliza fork llama-server flags — kept in JSON _fork_only_* for L2, never argv.
# WHY: --ctx-checkpoints crashes stock llama-server with "unknown argument".
_FORK_ONLY_FLAG_KEYS = frozenset(
    {
        "ctx_checkpoints",
        "ctx_checkpoint_interval",
    }
)
_STOCK_CACHE_FALLBACK = "q8_0"


def _resolve_llama_server_bin(config: object) -> Path | None:
    raw = getattr(config, "llama_server_bin", None)
    if raw is None:
        return None
    p = Path(str(raw))
    return p if p.is_file() else None


def flags_from_gpu_config(
    cfg: dict[str, Any],
    *,
    fork_enabled: bool,
) -> tuple[dict[str, Any], bool]:
    """Build active llama_server flag dict from profile JSON.

    Returns ``(flags, cache_types_fallback)``. ``cache_types_fallback`` is True
    when stock sanitize rewrote fork KV types to ``q8_0``; always False on fork path.
    """
    base = dict(cfg.get("llama_server_flags") or {})
    if not fork_enabled:
        return sanitize_llama_flags(base)

    merged = dict(base)
    merged.update(cfg.get("_eliza_fork_llama_server_flags") or {})
    # Legacy key from early L1 port (checkpoints only).
    merged.update(cfg.get("_fork_only_llama_server_flags") or {})
    for field in ("cache_type_k", "cache_type_v"):
        raw = merged.get(field)
        if raw:
            merged[field] = normalize_cache_type(str(raw))
    return merged, False


@dataclass(frozen=True)
class SelectedGpuProfile:
    id: str
    name: str
    source: str  # match | bucket | platform
    flags: dict[str, Any]
    cache_types_fallback: bool
    bucket_label: str | None = None
    runtime_kv: dict[str, Any] | None = None


def _runtime_kv_from_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    raw = cfg.get("runtime_kv")
    if not isinstance(raw, dict) or not raw:
        return None
    return dict(raw)


def gpu_profiles_enabled(yaml_data: dict[str, Any] | None = None) -> bool:
    env = os.environ.get("ZEROLLAMA_GPU_PROFILE", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    if yaml_data:
        lp = yaml_data.get("llama_profile")
        if lp is False or lp == "off":
            return False
        if isinstance(lp, dict) and lp.get("enabled") is False:
            return False
    return True


def profile_emit_options(yaml_data: dict[str, Any] | None = None) -> dict[str, bool]:
    """Which profile fields become llama-server argv (operator overrides)."""
    opts = {"ctx_size": True, "mlock": True}
    env_ctx = os.environ.get("ZEROLLAMA_GPU_PROFILE_CTX", "").strip().lower()
    if env_ctx in ("0", "false", "no", "off"):
        opts["ctx_size"] = False
    env_mlock = os.environ.get("ZEROLLAMA_GPU_PROFILE_MLOCK", "").strip().lower()
    if env_mlock in ("0", "false", "no", "off"):
        opts["mlock"] = False
    elif env_mlock in ("1", "true", "yes", "on"):
        opts["mlock"] = True
    if yaml_data:
        lp = yaml_data.get("llama_profile")
        if isinstance(lp, dict):
            if lp.get("apply_ctx_size") is False:
                opts["ctx_size"] = False
            if lp.get("apply_mlock") is False:
                opts["mlock"] = False
            elif lp.get("apply_mlock") is True:
                opts["mlock"] = True
    return opts


def configs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "gpu"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"gpu profile must be object: {path}")
    return raw


def load_index() -> dict[str, Any]:
    path = configs_dir() / "index.json"
    return _load_json(path)


def load_gpu_config(config_id: str) -> dict[str, Any]:
    index = load_index()
    for entry in index.get("configs") or []:
        if entry.get("id") == config_id:
            return _load_json(configs_dir() / str(entry["file"]))
    raise KeyError(f"unknown gpu profile id: {config_id}")


def _match_name(name: str, match_names: list[str]) -> bool:
    upper = name.upper()
    return any(m.upper() in upper for m in match_names if m)


def _sanitize_cache_type(value: str) -> tuple[str, bool]:
    key = normalize_cache_type(value.strip())
    if key in _FORK_ONLY_CACHE_TYPES:
        return _STOCK_CACHE_FALLBACK, True
    return key, False


def sanitize_llama_flags(flags: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return flags copy with stock-safe cache types and no fork-only argv keys."""
    out = dict(flags)
    fallback = False
    for field in ("cache_type_k", "cache_type_v"):
        raw = out.get(field)
        if not raw:
            continue
        safe, fb = _sanitize_cache_type(str(raw))
        out[field] = safe
        fallback = fallback or fb
    for key in _FORK_ONLY_FLAG_KEYS:
        out.pop(key, None)
    return out, fallback


def _normalize_n_gpu_layers(raw: Any) -> int:
    """Map eliza-style 999 (all layers) to llama.cpp -1."""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return -1
    if n >= 999:
        return -1
    return n


def match_gpu_config_by_name(name: str) -> dict[str, Any] | None:
    index = load_index()
    for entry in index.get("configs") or []:
        cfg = _load_json(configs_dir() / str(entry["file"]))
        names = cfg.get("match_names") or []
        if names and _match_name(name, names):
            return cfg
    return None


def select_by_vram_bucket(vram_gb: float) -> tuple[dict[str, Any] | None, str | None, float]:
    index = load_index()
    buckets = index.get("fallback_buckets") or []
    for bucket in buckets:
        cap = float(bucket.get("max_vram_gb", 0))
        if vram_gb <= cap:
            cid = bucket.get("config_id")
            scale = float(bucket.get("parallel_scale", 1.0))
            label = str(bucket.get("label") or "")
            if cid is None:
                return None, label, scale
            return load_gpu_config(str(cid)), label, scale
    return None, None, 1.0


def darwin_unified_memory_gb() -> float | None:
    from runtime.host_memory import darwin_total_memory_bytes

    total = darwin_total_memory_bytes()
    if total is None:
        return None
    return total / (1024**3)


def select_by_apple_memory_bucket(
    unified_gb: float,
) -> tuple[dict[str, Any] | None, str | None, float]:
    index = load_index()
    buckets = index.get("apple_unified_memory_buckets") or []
    for bucket in buckets:
        cap = float(bucket.get("max_unified_gb", 0))
        if unified_gb <= cap:
            cid = bucket.get("config_id")
            scale = float(bucket.get("parallel_scale", 1.0))
            label = str(bucket.get("label") or "")
            if cid is None:
                return None, label, scale
            return load_gpu_config(str(cid)), label, scale
    return None, None, 1.0


def select_apple_silicon_profile(*, fork_enabled: bool = False) -> SelectedGpuProfile | None:
    unified_gb = darwin_unified_memory_gb()
    bucket_cfg: dict[str, Any] | None = None
    label: str | None = None
    scale = 1.0
    source = "platform"

    if unified_gb is not None:
        bucket_cfg, label, scale = select_by_apple_memory_bucket(unified_gb)
        if bucket_cfg is not None:
            source = "apple_memory"

    if bucket_cfg is None:
        try:
            bucket_cfg = load_gpu_config("apple-silicon-48g")
            label = "48g-fallback"
        except KeyError:
            return None

    flags, cache_fb = flags_from_gpu_config(bucket_cfg, fork_enabled=fork_enabled)
    n_par = flags.get("n_parallel")
    if n_par is not None and scale != 1.0:
        flags = dict(flags)
        flags["n_parallel"] = max(1, int(int(n_par) * scale))

    return SelectedGpuProfile(
        id=str(bucket_cfg.get("id", "apple-silicon")),
        name=str(bucket_cfg.get("name", "Apple Silicon unified memory")),
        source=source,
        flags=flags,
        cache_types_fallback=cache_fb,
        bucket_label=label,
        runtime_kv=_runtime_kv_from_config(bucket_cfg),
    )


def detect_vulkan_gpu_name(main_gpu: int = 0) -> str | None:
    """Best-effort Vulkan device name (Intel Arc, etc.) via vulkaninfo."""
    if not shutil.which("vulkaninfo"):
        return None
    try:
        out = subprocess.check_output(
            ["vulkaninfo", "--summary"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    names: list[str] = []
    for line in out.splitlines():
        m = re.search(r"deviceName\s*=\s*(.+)", line)
        if not m:
            continue
        name = m.group(1).strip()
        if "llvmpipe" in name.lower() or "lavapipe" in name.lower():
            continue
        names.append(name)
    if not names:
        return None
    idx = max(0, min(main_gpu, len(names) - 1))
    return names[idx]


def select_vulkan_gpu_profile(*, main_gpu: int = 0, fork_enabled: bool = False) -> SelectedGpuProfile | None:
    name = detect_vulkan_gpu_name(main_gpu)
    if not name:
        return None
    matched = match_gpu_config_by_name(name)
    if matched is None:
        return None
    flags, cache_fb = flags_from_gpu_config(matched, fork_enabled=fork_enabled)
    return SelectedGpuProfile(
        id=str(matched.get("id", "vulkan")),
        name=str(matched.get("name", name)),
        source="match",
        flags=flags,
        cache_types_fallback=cache_fb,
        runtime_kv=_runtime_kv_from_config(matched),
    )


def select_nvidia_gpu_profile(*, main_gpu: int = 0, fork_enabled: bool = False) -> SelectedGpuProfile | None:
    """Match NVIDIA GPU by name or VRAM bucket (no Apple Silicon fallback)."""
    name = detect_nvidia_gpu_name(main_gpu)
    vram_gb = detect_gpu_total_vram_gb(main_gpu)

    if name:
        matched = match_gpu_config_by_name(name)
        if matched is not None:
            flags, cache_fb = flags_from_gpu_config(matched, fork_enabled=fork_enabled)
            return SelectedGpuProfile(
                id=str(matched.get("id", "unknown")),
                name=str(matched.get("name", name)),
                source="match",
                flags=flags,
                cache_types_fallback=cache_fb,
                runtime_kv=_runtime_kv_from_config(matched),
            )

    if vram_gb is not None:
        bucket_cfg, label, scale = select_by_vram_bucket(vram_gb)
        if bucket_cfg is not None:
            flags, cache_fb = flags_from_gpu_config(bucket_cfg, fork_enabled=fork_enabled)
            n_par = flags.get("n_parallel")
            if n_par is not None and scale != 1.0:
                flags = dict(flags)
                flags["n_parallel"] = max(1, int(int(n_par) * scale))
            return SelectedGpuProfile(
                id=str(bucket_cfg.get("id", "bucket")),
                name=str(bucket_cfg.get("name", label or "VRAM bucket")),
                source="bucket",
                flags=flags,
                cache_types_fallback=cache_fb,
                bucket_label=label,
                runtime_kv=_runtime_kv_from_config(bucket_cfg),
            )
    return None


def select_gpu_profile(*, main_gpu: int = 0, fork_enabled: bool = False) -> SelectedGpuProfile | None:
    if sys.platform == "darwin":
        if detect_nvidia_gpu_name(main_gpu) is None:
            return select_apple_silicon_profile(fork_enabled=fork_enabled)

    profile = select_nvidia_gpu_profile(main_gpu=main_gpu, fork_enabled=fork_enabled)
    if profile is not None:
        return profile

    profile = select_vulkan_gpu_profile(main_gpu=main_gpu, fork_enabled=fork_enabled)
    if profile is not None:
        return profile

    if sys.platform == "darwin":
        return select_apple_silicon_profile(fork_enabled=fork_enabled)
    return None


def _select_profile_for_config(
    config_path: Path | None, *, main_gpu: int = 0, fork_enabled: bool = False
) -> SelectedGpuProfile | None:
    """Pick profile family from YAML path.

    WHY gate on config filename: darwin autoconfig loads apple_silicon.yaml,
    but tests and explicit ZEROLLAMA_RUNTIME_CONFIG=single_gpu.yaml must not
    apply Metal RAM tiers (and vice versa on Linux CI).
    """
    name = (config_path.name if config_path else "").lower()
    if name == "apple_silicon.yaml":
        return select_apple_silicon_profile(fork_enabled=fork_enabled)
    if name in ("single_gpu.yaml", "dual_4090.yaml"):
        return select_nvidia_gpu_profile(main_gpu=main_gpu, fork_enabled=fork_enabled)
    if name == "arc_a380.yaml":
        forced = os.environ.get("ZEROLLAMA_GPU_PROFILE_ID", "").strip()
        if forced:
            try:
                cfg = load_gpu_config(forced)
            except KeyError:
                cfg = None
            if cfg is not None:
                flags, cache_fb = flags_from_gpu_config(cfg, fork_enabled=fork_enabled)
                return SelectedGpuProfile(
                    id=str(cfg.get("id", forced)),
                    name=str(cfg.get("name", forced)),
                    source="env",
                    flags=flags,
                    cache_types_fallback=cache_fb,
                    runtime_kv=_runtime_kv_from_config(cfg),
                )
        profile = select_vulkan_gpu_profile(main_gpu=main_gpu, fork_enabled=fork_enabled)
        if profile is not None:
            return profile
        try:
            cfg = load_gpu_config("arc-a380")
            flags, cache_fb = flags_from_gpu_config(cfg, fork_enabled=fork_enabled)
            return SelectedGpuProfile(
                id=str(cfg.get("id", "arc-a380")),
                name=str(cfg.get("name", "Intel Arc A380")),
                source="platform",
                flags=flags,
                cache_types_fallback=cache_fb,
                runtime_kv=_runtime_kv_from_config(cfg),
            )
        except KeyError:
            return None
    return select_gpu_profile(main_gpu=main_gpu, fork_enabled=fork_enabled)


def llama_argv_from_profile_flags(
    flags: dict[str, Any],
    *,
    emit: dict[str, bool] | None = None,
) -> list[str]:
    """Turn profile flag dict into llama-server argv (excluding -np/-sm/-mg/-ts)."""
    emit_opts = emit or {"ctx_size": True, "mlock": True}
    args: list[str] = []
    if emit_opts.get("ctx_size", True):
        ctx = flags.get("ctx_size")
        if ctx is not None:
            args.extend(["-c", str(int(ctx))])
    batch = flags.get("batch_size")
    if batch is not None:
        args.extend(["-b", str(int(batch))])
    ubatch = flags.get("ubatch_size")
    if ubatch is not None:
        args.extend(["-ub", str(int(ubatch))])
    if flags.get("flash_attn"):
        args.extend(["-fa", "on"])
    ck = flags.get("cache_type_k")
    if ck:
        args.extend(["--cache-type-k", str(ck)])
    cv = flags.get("cache_type_v")
    if cv:
        args.extend(["--cache-type-v", str(cv)])
    if emit_opts.get("mlock", True) and flags.get("mlock"):
        args.append("--mlock")
    if flags.get("no_mmap"):
        args.append("--no-mmap")
    if flags.get("no_kv_offload"):
        args.append("--no-kv-offload")
    ckpt = flags.get("ctx_checkpoints")
    if ckpt is not None:
        args.extend(["--ctx-checkpoints", str(int(ckpt))])
    ckpt_iv = flags.get("ctx_checkpoint_interval")
    if ckpt_iv is not None:
        # WHY -cpent: elizaOS/llama.cpp fork @ 96dd1a8 renamed interval flag; old
        # --ctx-checkpoint-interval crashes llama-server at startup on CUDA sign-off.
        args.extend(["--checkpoint-every-n-tokens", str(int(ckpt_iv))])
    draft_p = flags.get("draft_p_min")
    if draft_p is not None:
        args.extend(["--spec-draft-p-min", str(float(draft_p))])
    return args


def profile_n_gpu_layers(flags: dict[str, Any]) -> int:
    return _normalize_n_gpu_layers(flags.get("n_gpu_layers", -1))


def apply_profile_to_config(
    config: object,
    profile: SelectedGpuProfile,
    *,
    emit: dict[str, bool],
    fork_enabled: bool = False,
) -> None:
    """Mutate RuntimeConfig fields from a selected profile."""
    flags = profile.flags
    n_par = flags.get("n_parallel")
    if n_par is not None:
        config.llama_parallel_slots = max(1, int(n_par))  # type: ignore[attr-defined]

    sm = flags.get("split_mode")
    if sm and config.tensor_parallel <= 1:  # type: ignore[attr-defined]
        config.split_mode = str(sm)  # type: ignore[attr-defined]

    mg = flags.get("main_gpu")
    if mg is not None:
        config.main_gpu = int(mg)  # type: ignore[attr-defined]

    config.n_gpu_layers_default = profile_n_gpu_layers(flags)  # type: ignore[attr-defined]

    spec = config.speculative  # type: ignore[attr-defined]
    dmax = flags.get("draft_max")
    dmin = flags.get("draft_min")
    if dmax is not None:
        spec.draft_n_max = int(dmax)
    if dmin is not None:
        spec.draft_n_min = int(dmin)

    if profile.runtime_kv and not os.environ.get("ZEROLLAMA_KV_NUM_BLOCKS", "").strip():
        nb = profile.runtime_kv.get("num_blocks_per_device")
        if nb is None:
            nb = profile.runtime_kv.get("num_blocks")
        if nb is not None:
            config.num_blocks = max(1, int(nb))  # type: ignore[attr-defined]
        if not os.environ.get("ZEROLLAMA_KV_BLOCK_SIZE", "").strip():
            bs = profile.runtime_kv.get("block_size")
            if bs is not None:
                config.block_size = max(1, int(bs))  # type: ignore[attr-defined]

    config.gpu_profile = {  # type: ignore[attr-defined]
        "id": profile.id,
        "name": profile.name,
        "source": profile.source,
        "bucket_label": profile.bucket_label,
        "cache_types_fallback": profile.cache_types_fallback,
        "n_parallel": config.llama_parallel_slots,  # type: ignore[attr-defined]
        "ctx_size_default": flags.get("ctx_size"),
        "emit_ctx_size": emit.get("ctx_size", True),
        "emit_mlock": emit.get("mlock", True),
        "llama_fork": fork_enabled,
        "kv_num_blocks": getattr(config, "num_blocks", None),
        "kv_block_size": getattr(config, "block_size", None),
    }
    if sys.platform == "darwin":
        unified_gb = darwin_unified_memory_gb()
        if unified_gb is not None:
            config.gpu_profile["unified_memory_gb"] = round(unified_gb, 1)  # type: ignore[index]
    config._gpu_profile_flags = dict(flags)  # type: ignore[attr-defined]
    config._gpu_profile_emit = dict(emit)  # type: ignore[attr-defined]


def maybe_apply_gpu_profile(
    config: object, yaml_data: dict[str, Any], *, config_path: Path | None = None
) -> None:
    if not gpu_profiles_enabled(yaml_data):
        config.gpu_profile = None  # type: ignore[attr-defined]
        config._gpu_profile_flags = {}  # type: ignore[attr-defined]
        config._gpu_profile_emit = {}  # type: ignore[attr-defined]
        return
    emit = profile_emit_options(yaml_data)
    main_gpu = int(getattr(config, "main_gpu", 0) or 0)
    bin_path = _resolve_llama_server_bin(config)
    fork_on = llama_fork_enabled(llama_server_bin=bin_path)
    forced_id = os.environ.get("ZEROLLAMA_GPU_PROFILE_ID", "").strip()
    if forced_id:
        try:
            cfg = load_gpu_config(forced_id)
            flags, cache_fb = flags_from_gpu_config(cfg, fork_enabled=fork_on)
            selected = SelectedGpuProfile(
                id=str(cfg.get("id", forced_id)),
                name=str(cfg.get("name", forced_id)),
                source="env",
                flags=flags,
                cache_types_fallback=cache_fb,
                runtime_kv=_runtime_kv_from_config(cfg),
            )
            apply_profile_to_config(config, selected, emit=emit, fork_enabled=fork_on)
            return
        except KeyError:
            pass
    selected = _select_profile_for_config(
        config_path, main_gpu=main_gpu, fork_enabled=fork_on
    )
    if selected is None:
        config.gpu_profile = None  # type: ignore[attr-defined]
        config._gpu_profile_flags = {}  # type: ignore[attr-defined]
        config._gpu_profile_emit = {}  # type: ignore[attr-defined]
        return
    apply_profile_to_config(config, selected, emit=emit, fork_enabled=fork_on)
