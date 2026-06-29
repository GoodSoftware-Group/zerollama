"""Centralized runtime env parsing with platform/backend smart defaults.

WHY: L3 and cache modules each called ``os.environ.get`` with different default
semantics — operator shells (e.g. ``ZEROLLAMA_LLAMA_CACHE_DISK=0`` from Metal)
leaked into unrelated CUDA smokes. One module owns tri-state bools, hints from
``RuntimeConfig``, and test reset.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

TriState = Optional[bool]  # None = env unset → use smart default


@dataclass(frozen=True)
class L3Settings:
    """YAML ``l3:`` block; env overrides individual fields."""

    radix_share: bool = False
    block_size: int = 512
    trace: bool = False
    trace_dir: str | None = None
    lmcache_uri: str | None = None
    retention_interval: int | None = None


@dataclass(frozen=True)
class RuntimeEnvHints:
    """Process hints set once from ``RuntimeConfig`` (engine init)."""

    llama_backend: str = "subprocess"
    n_parallel: int = 1


_hints: RuntimeEnvHints | None = None
_l3: L3Settings | None = None


def configure_l3_settings(raw: dict[str, object] | None) -> None:
    """Load ``l3:`` from runtime YAML (called from ``RuntimeConfig.from_file``)."""
    global _l3
    if not raw or not isinstance(raw, dict):
        _l3 = L3Settings()
        return
    block_size = 512
    if (bs := raw.get("block_size")) is not None:
        try:
            block_size = max(1, int(bs))
        except (TypeError, ValueError):
            pass
    retention = raw.get("retention_interval")
    ri: int | None = None
    if retention is not None:
        try:
            ri = int(retention)
        except (TypeError, ValueError):
            ri = None
    trace_dir = raw.get("trace_dir")
    lmcache_uri = raw.get("lmcache_uri")
    _l3 = L3Settings(
        radix_share=bool(raw.get("radix_share", False)),
        block_size=block_size,
        trace=bool(raw.get("trace", False)),
        trace_dir=str(trace_dir).strip() if trace_dir else None,
        lmcache_uri=str(lmcache_uri).strip() if lmcache_uri else None,
        retention_interval=ri,
    )


def l3_settings() -> L3Settings:
    return _l3 or L3Settings()


def configure_runtime_env(
    *,
    llama_backend: str | None = None,
    n_parallel: int | None = None,
) -> None:
    """Wire YAML/GPU profile context for smart env defaults."""
    global _hints
    backend = (llama_backend or os.environ.get("ZEROLLAMA_RUNTIME_LLAMA_BACKEND") or "subprocess").strip().lower()
    parallel = n_parallel
    if parallel is None:
        raw = os.environ.get("ZEROLLAMA_LLAMA_PARALLEL_SLOTS", "").strip()
        parallel = int(raw) if raw.isdigit() else 1
    _hints = RuntimeEnvHints(llama_backend=backend, n_parallel=max(1, int(parallel)))


def reset_runtime_env_for_tests() -> None:
    """Test helper: drop engine hints."""
    global _hints, _l3
    _hints = None
    _l3 = None


def runtime_hints() -> RuntimeEnvHints:
    if _hints is not None:
        return _hints
    return RuntimeEnvHints(
        llama_backend=(os.environ.get("ZEROLLAMA_RUNTIME_LLAMA_BACKEND") or "subprocess").strip().lower(),
        n_parallel=_int_env("ZEROLLAMA_LLAMA_PARALLEL_SLOTS", default=1, minimum=1),
    )


def _int_env(name: str, *, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def env_tri_state(name: str) -> TriState:
    """Parse ``1/on/yes`` → True, ``0/off/no`` → False, unset → None."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    v = raw.strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return None


def env_bool(name: str, *, default: bool) -> bool:
    t = env_tri_state(name)
    return default if t is None else t


def llama_cache_enabled() -> bool:
    return env_bool("ZEROLLAMA_LLAMA_CACHE", default=True)


def llama_cache_disk_default(*, backend: str | None = None) -> bool:
    """Smart default when ``ZEROLLAMA_LLAMA_CACHE_DISK`` is unset.

    WHY: Metal/in-process paths are latency-sensitive (disk off); Linux CUDA
    subprocess uses ``--slot-save-path`` (disk on). Explicit env always wins.
    """
    plat = sys.platform
    bk = (backend or runtime_hints().llama_backend).strip().lower()
    if plat == "darwin":
        return False
    if bk == "inprocess":
        return plat.startswith("linux")
    return True


def llama_cache_disk_enabled(*, backend: str | None = None) -> bool:
    if not llama_cache_enabled():
        return False
    explicit = env_tri_state("ZEROLLAMA_LLAMA_CACHE_DISK")
    if explicit is not None:
        return explicit
    return llama_cache_disk_default(backend=backend)


def radix_prefix_share_enabled() -> bool:
    """Cross-slot Radix seed (v1). Default off — WHY: opt-in cross-slot KV copy.

    Auto-enables prefix block pool when on. Prefer ``ZEROLLAMA_L3_PROFILE=agent``
    over raw env; explicit ``ZEROLLAMA_RADIX_PREFIX_SHARE=0/1`` wins over YAML.
    """
    explicit = env_tri_state("ZEROLLAMA_RADIX_PREFIX_SHARE")
    if explicit is not None:
        return explicit
    return l3_settings().radix_share


def radix_hybrid_seq_copy_enabled() -> bool:
    """Allow Radix ``seq_cp`` on Gemma-style hybrid when prefix fits SWA window (L3-R5).

    WHY default on: ``kind=hybrid`` in GGUF metadata usually means full+SWA layers
    (Gemma), not attn+recurrent — v1's blanket skip blocked those models.
    WHY env exists: LFM2 / true recurrent memory may still abort ``seq_cp``; set
    ``ZEROLLAMA_RADIX_HYBRID_SEQ_COPY=0`` until a model-specific live gate passes.
    """
    return env_bool("ZEROLLAMA_RADIX_HYBRID_SEQ_COPY", default=True)


def lmcache_tier_enabled() -> bool:
    """LMCache metadata tier: on when ``ZEROLLAMA_LMCACHE_URI`` or YAML ``l3.lmcache_uri`` set."""
    if os.environ.get("ZEROLLAMA_LMCACHE_URI", "").strip():
        return True
    if l3_settings().lmcache_uri:
        return True
    return env_bool("ZEROLLAMA_LMCACHE_TIER", default=False)


def lmcache_uri() -> str:
    raw = os.environ.get("ZEROLLAMA_LMCACHE_URI", "").strip()
    if raw:
        return raw
    yaml_uri = l3_settings().lmcache_uri
    if yaml_uri:
        return yaml_uri
    return "file://~/.cache/zerollama/lmcache"


def lmcache_ttl_sec() -> int | None:
    raw = os.environ.get("ZEROLLAMA_LMCACHE_TTL_SEC", "").strip()
    if not raw:
        return None
    try:
        sec = int(raw)
        return sec if sec > 0 else None
    except ValueError:
        return None


def prefix_cache_block_size() -> int:
    raw = os.environ.get("ZEROLLAMA_PREFIX_CACHE_BLOCK_SIZE", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, l3_settings().block_size)


def prefix_cache_retention_interval() -> int | None:
    raw = os.environ.get("ZEROLLAMA_PREFIX_CACHE_RETENTION_INTERVAL", "").strip()
    if raw:
        if raw.lower() in ("0", "false", "no", "off"):
            return 0
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return l3_settings().retention_interval


def prefix_cache_trace_enabled() -> bool:
    explicit = env_tri_state("ZEROLLAMA_PREFIX_CACHE_TRACE")
    if explicit is not None:
        return explicit
    if "l3" in debug_tags():
        return True
    return l3_settings().trace


def prefix_cache_trace_dir() -> Path:
    raw = os.environ.get("ZEROLLAMA_PREFIX_CACHE_TRACE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    yaml_dir = l3_settings().trace_dir
    if yaml_dir:
        return Path(yaml_dir).expanduser()
    return Path.home() / ".cache" / "zerollama" / "prefix-cache-traces"


def decode_graph_invalidate_enabled() -> bool:
    return env_bool("ZEROLLAMA_DECODE_GRAPH_INVALIDATE", default=True)


def decode_graph_trace_enabled() -> bool:
    if env_bool("ZEROLLAMA_DECODE_GRAPH_TRACE", default=False):
        return True
    return "l3" in debug_tags()


def prefix_block_pool_enabled(*, n_parallel: int | None = None) -> bool:
    """Hash-chained prefix blocks for L3 verification / Radix donor lookup.

    Auto-on when: Radix share, LMCache URI, explicit ``PREFIX_BLOCK_POOL=1``,
    or L3 cache + ``n_parallel > 1`` (multi-slot agent workloads).
    """
    explicit = env_tri_state("ZEROLLAMA_PREFIX_BLOCK_POOL")
    if explicit is False:
        return False
    if radix_prefix_share_enabled():
        return True
    if lmcache_tier_enabled():
        return True
    if explicit is True:
        return True
    if not llama_cache_enabled():
        return False
    slots = n_parallel if n_parallel is not None else runtime_hints().n_parallel
    return slots > 1


def prefix_block_pool_max_entries() -> int:
    return _int_env("ZEROLLAMA_PREFIX_BLOCK_POOL_MAX", default=8192, minimum=64)


# --- L3 profile presets (YAML bundles; env overrides fields) ---

_L3_PROFILE_CONFIGS: dict[str, str] = {
    "agent": "l3_agent_subprocess.yaml",
}


def l3_profile_name() -> str | None:
    raw = os.environ.get("ZEROLLAMA_L3_PROFILE", "").strip().lower()
    return raw or None


def resolve_l3_profile_config_path() -> Path | None:
    """``ZEROLLAMA_L3_PROFILE=agent`` → ``runtime/configs/l3_agent_subprocess.yaml``."""
    name = l3_profile_name()
    if not name:
        return None
    fname = _L3_PROFILE_CONFIGS.get(name)
    if not fname:
        return None
    path = Path(__file__).resolve().parents[1] / "configs" / fname
    return path if path.is_file() else None


def debug_tags() -> frozenset[str]:
    """``ZEROLLAMA_DEBUG=l3,infer`` — comma/space separated debug tiers."""
    raw = os.environ.get("ZEROLLAMA_DEBUG", "").strip().lower()
    if not raw:
        return frozenset()
    return frozenset(part for part in raw.replace(",", " ").split() if part)


def infer_trace_enabled() -> bool:
    if env_bool("ZEROLLAMA_INFER_TRACE", default=False):
        return True
    tags = debug_tags()
    return "infer" in tags or "trace" in tags


def llama_cache_root() -> Path:
    override = os.environ.get("ZEROLLAMA_LLAMA_CACHE_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        return Path(xdg) / "zerollama" / "llama-cache"
    return Path.home() / ".cache" / "zerollama" / "llama-cache"


def default_slot_ttl_ms(*, default_ms: int = 3_600_000) -> int:
    raw = os.environ.get("ZEROLLAMA_LLAMA_CACHE_TTL_MS", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return default_ms


def default_cache_salt() -> str | None:
    raw = os.environ.get("ZEROLLAMA_CACHE_SALT", "").strip()
    return raw or None


def runtime_env_health() -> dict[str, object]:
    """Operator snapshot for ``/health`` — effective L3/env without shell archaeology."""
    hints = runtime_hints()
    l3 = l3_settings()
    from runtime.vram_yaml_defaults import apply_status

    return {
        "l3_profile": l3_profile_name(),
        "l3_profile_config": str(resolve_l3_profile_config_path() or ""),
        "debug_tags": sorted(debug_tags()),
        "llama_cache_disk": llama_cache_disk_enabled(backend=hints.llama_backend),
        "llama_cache_disk_explicit": env_tri_state("ZEROLLAMA_LLAMA_CACHE_DISK"),
        "prefix_block_pool": prefix_block_pool_enabled(),
        "radix_share": radix_prefix_share_enabled(),
        "lmcache_tier": lmcache_tier_enabled(),
        "n_parallel_hint": hints.n_parallel,
        "llama_backend_hint": hints.llama_backend,
        "l3_yaml": {
            "radix_share": l3.radix_share,
            "block_size": l3.block_size,
            "trace": l3.trace,
        },
        "kv": kv_env_health(),
        "vram": vram_env_health(),
        "vram_yaml_apply": apply_status(),
    }


# --- VRAM / admission (YAML may pre-fill via vram_yaml_defaults) ---


def vram_probe_mode_raw() -> str:
    return os.environ.get("ZEROLLAMA_RUNTIME_VRAM_PROBE", "auto").strip().lower()


def vram_nvml_unified_fallback_enabled() -> bool:
    explicit = env_tri_state("ZEROLLAMA_RUNTIME_VRAM_UNIFIED_FALLBACK")
    return True if explicit is None else explicit


def vram_check_gpu_explicit() -> TriState:
    return env_tri_state("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM")


def vram_margin() -> float:
    raw = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_MARGIN", "").strip()
    if not raw:
        return 1.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.0


def vram_inference_policy() -> str:
    return os.environ.get("ZEROLLAMA_RUNTIME_INFERENCE_POLICY", "inference-first").strip().lower()


def inference_first_policy_enabled() -> bool:
    return vram_inference_policy() not in ("off", "0", "false", "no", "disabled")


def runtime_shared_python_embedded() -> bool:
    """True when training and runtime share one in-process CPython."""
    v = os.environ.get("ZEROLLAMA_RUNTIME_SHARED_PYTHON", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    import sys

    return "ollama_training_native" in sys.modules


def vram_ram_overhead() -> float:
    return _float_env("ZEROLLAMA_RUNTIME_RAM_OVERHEAD", default=1.12, minimum=1.0)


def vram_ram_margin() -> float:
    return _float_env("ZEROLLAMA_RUNTIME_RAM_MARGIN", default=1.0, minimum=0.0)


def vram_weight_tensor_mode() -> str:
    raw = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_WEIGHT_TENSOR", "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return "off"
    if raw in ("1", "true", "yes", "on"):
        return "on"
    return "auto"


def vram_weight_tensor_use_per_tensor(n_gpu_layers: int | None) -> bool:
    mode = vram_weight_tensor_mode()
    if mode == "off":
        return False
    if mode == "on":
        return True
    return n_gpu_layers is not None and n_gpu_layers >= 0


def vram_suggest_ctx_max_cap() -> int:
    raw = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_SUGGEST_CTX_MAX", "131072").strip()
    try:
        return max(512, int(raw))
    except ValueError:
        return 131072


def vram_clamp_num_ctx_raw() -> str:
    return os.environ.get("ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX", "0").strip()


def vram_num_ctx_clamp_enabled() -> bool:
    v = vram_clamp_num_ctx_raw().lower()
    if v in ("0", "false", "no", "off", ""):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    from runtime.gpu_vram import gpu_vram_check_enabled

    return gpu_vram_check_enabled()


def vram_probe_calibrate_raw() -> str:
    return os.environ.get("ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE", "auto").strip().lower()


def vram_probe_calibrate_enabled() -> bool:
    v = vram_probe_calibrate_raw()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    from runtime.gpu_vram import gpu_vram_check_enabled

    return gpu_vram_check_enabled()


def vram_estimate_autotune_enabled() -> bool:
    explicit = vram_estimate_autotune_explicit()
    if explicit is False:
        return False
    if explicit is True:
        return True
    return vram_probe_calibrate_enabled()


def vram_apply_exported_env_enabled() -> bool:
    v = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_APPLY_EXPORTED_ENV", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def vram_apply_exported_env_path() -> str:
    return os.environ.get("ZEROLLAMA_RUNTIME_VRAM_APPLY_EXPORTED_ENV_PATH", "").strip()


def vram_autotune_persist_explicit() -> TriState:
    return env_tri_state("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST")


def vram_autotune_persist_enabled() -> bool:
    explicit = vram_autotune_persist_explicit()
    if explicit is False:
        return False
    return vram_estimate_autotune_enabled()


def vram_runtime_state_dir() -> str:
    return os.environ.get("ZEROLLAMA_RUNTIME_STATE_DIR", "").strip()


def vram_autotune_state_path_override() -> str:
    return os.environ.get("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_STATE", "").strip()


def vram_kv_block_layout_enabled() -> bool:
    return env_bool("ZEROLLAMA_RUNTIME_VRAM_KV_BLOCK_LAYOUT", default=True)


def vram_weight_block_layout_enabled() -> bool:
    return env_bool("ZEROLLAMA_RUNTIME_VRAM_WEIGHT_BLOCK_LAYOUT", default=True)


def vram_kv_elem_bytes() -> int:
    return _int_env("ZEROLLAMA_RUNTIME_VRAM_KV_ELEM_BYTES", default=2, minimum=1)


def _float_env(
    name: str,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def vram_num_ctx_override() -> int | None:
    raw = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_NUM_CTX", "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except ValueError:
        return None


def vram_mmap_factor() -> float:
    return _float_env("ZEROLLAMA_RUNTIME_VRAM_MMAP_FACTOR", default=1.0, minimum=0.0, maximum=1.0)


def vram_layer_scale_enabled() -> bool:
    return env_bool("ZEROLLAMA_RUNTIME_VRAM_LAYER_SCALE", default=True)


def vram_layer_base() -> int:
    return _int_env("ZEROLLAMA_RUNTIME_VRAM_LAYER_BASE", default=32, minimum=1)


def vram_kv_exact_enabled() -> bool:
    return env_bool("ZEROLLAMA_RUNTIME_VRAM_KV_EXACT", default=True)


def vram_ngram_scratch_bytes_default() -> int:
    raw = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_NGRAM_SCRATCH_BYTES", "").strip()
    if not raw:
        return 128 * 1024 * 1024
    try:
        from runtime.gpu.admission import parse_size_bytes

        parsed = parse_size_bytes(raw)
        return parsed if parsed is not None and parsed > 0 else 128 * 1024 * 1024
    except Exception:
        return 128 * 1024 * 1024


def vram_scratch_factor() -> float:
    return _float_env("ZEROLLAMA_RUNTIME_VRAM_SCRATCH_FACTOR", default=1.05, minimum=1.0)


def vram_estimate_factor() -> float:
    return _float_env("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR", default=1.0, minimum=0.1)


def vram_estimate_autotune_explicit() -> TriState:
    return env_tri_state("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE")


def vram_kv_factor(*, kv_exact: bool) -> float:
    default = 1.10 if kv_exact else 1.15
    return _float_env("ZEROLLAMA_RUNTIME_VRAM_KV_FACTOR", default=default, minimum=0.1)


def vram_env_health() -> dict[str, object]:
    """Effective VRAM env for ``/health`` (YAML-applied values visible after engine init)."""
    return {
        "probe": vram_probe_mode_raw(),
        "check_gpu_vram_explicit": vram_check_gpu_explicit(),
        "margin": vram_margin(),
        "inference_policy": vram_inference_policy(),
        "inference_first": inference_first_policy_enabled(),
        "nvml_unified_fallback": vram_nvml_unified_fallback_enabled(),
        "num_ctx_override": vram_num_ctx_override(),
        "mmap_factor": vram_mmap_factor(),
        "kv_exact": vram_kv_exact_enabled(),
        "kv_factor": vram_kv_factor(kv_exact=True),
        "estimate_factor": vram_estimate_factor(),
        "estimate_autotune_explicit": vram_estimate_autotune_explicit(),
        "estimate_autotune": vram_estimate_autotune_enabled(),
        "scratch_factor": vram_scratch_factor(),
        "layer_scale": vram_layer_scale_enabled(),
        "layer_base": vram_layer_base(),
        "weight_tensor_mode": vram_weight_tensor_mode(),
        "kv_block_layout": vram_kv_block_layout_enabled(),
        "weight_block_layout": vram_weight_block_layout_enabled(),
        "kv_elem_bytes": vram_kv_elem_bytes(),
        "ngram_scratch_bytes": vram_ngram_scratch_bytes_default(),
        "ram_overhead": vram_ram_overhead(),
        "ram_margin": vram_ram_margin(),
        "suggest_ctx_max": vram_suggest_ctx_max_cap(),
        "clamp_num_ctx": vram_clamp_num_ctx_raw(),
        "clamp_num_ctx_enabled": vram_num_ctx_clamp_enabled(),
        "probe_calibrate": vram_probe_calibrate_raw(),
        "probe_calibrate_enabled": vram_probe_calibrate_enabled(),
        "apply_exported_env": vram_apply_exported_env_enabled(),
        "autotune_persist": vram_autotune_persist_enabled(),
        "shared_python_embedded": runtime_shared_python_embedded(),
    }


# --- Phase 15 KV native decode / auto-batch (default-on except auto-batch) ---


def kv_native_decode_enabled() -> bool:
    return env_bool("ZEROLLAMA_KV_NATIVE_DECODE", default=True)


def kv_native_batch_enabled() -> bool:
    return env_bool("ZEROLLAMA_KV_NATIVE_BATCH", default=True)


def kv_native_sample_enabled() -> bool:
    return env_bool("ZEROLLAMA_KV_NATIVE_SAMPLE", default=True)


def kv_auto_batch_enabled() -> bool:
    """Opt-in: coalesce concurrent in-process ``generate()`` within a short window."""
    return env_bool("ZEROLLAMA_KV_AUTO_BATCH", default=False)


def kv_auto_batch_window_ms() -> int:
    return _int_env("ZEROLLAMA_KV_AUTO_BATCH_MS", default=5, minimum=0)


def kv_env_health() -> dict[str, object]:
    return {
        "native_decode": kv_native_decode_enabled(),
        "native_batch": kv_native_batch_enabled(),
        "native_sample": kv_native_sample_enabled(),
        "auto_batch": kv_auto_batch_enabled(),
        "auto_batch_ms": kv_auto_batch_window_ms(),
    }
