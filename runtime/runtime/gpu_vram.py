"""NVIDIA GPU free VRAM probe and heuristic headroom checks before llama-server start.

Why this module exists: CUDA OOM after subprocess start is slow and opaque; we fail
early with actionable errors using free-memory probes plus coarse KV/layer scaling.

Why heuristics (not exact llama.cpp math): tensor layouts and mmap behavior vary;
margins and factors are operator-tunable via ZEROLLAMA_RUNTIME_VRAM_* env vars.

Why NVML before nvidia-smi: fewer subprocess spawns; optional nvidia-ml-py via
pip install -e 'runtime/.[gpu]'. Unified-memory fallback matches ggml on iGPU hosts
where NVML reports NOT_SUPPORTED.
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runtime.gpu.priority import InferencePriority

_SMI_CACHE_TTL_S = 0.25
_smi_cache: dict[int, tuple[float, int | None]] = {}
_smi_lock = threading.Lock()
_active_vram_probe: str | None = None
_session_autotune_factor: float | None = None
_session_autotune_model: str | None = None
_autotune_lock = threading.Lock()
_shared_no_smi_warned = False
_shared_no_smi_warn_lock = threading.Lock()


def shared_interpreter_embedded() -> bool:
    """True when training and runtime share one in-process CPython (see docs/bugs/)."""
    v = os.environ.get("ZEROLLAMA_RUNTIME_SHARED_PYTHON", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return "ollama_training_native" in sys.modules


def _vram_probe_env_raw() -> str:
    return os.environ.get("ZEROLLAMA_RUNTIME_VRAM_PROBE", "auto").strip().lower()


def _shared_auto_without_smi() -> bool:
    """Shared CPython + probe auto + no nvidia-smi: avoid pynvml (can stall the GIL)."""
    if _vram_probe_env_raw() != "auto":
        return False
    if not shared_interpreter_embedded():
        return False
    return not nvidia_smi_available()


def warn_shared_interpreter_no_smi_once() -> None:
    global _shared_no_smi_warned
    if not _shared_auto_without_smi():
        return
    with _shared_no_smi_warn_lock:
        if _shared_no_smi_warned:
            return
        _shared_no_smi_warned = True
    from runtime.logutil import get_logger

    get_logger("gpu_vram").warning(
        "embedded training+runtime share one Python interpreter but nvidia-smi is "
        "unavailable; skipping NVML VRAM probes (install nvidia-smi or set "
        "ZEROLLAMA_RUNTIME_VRAM_PROBE=smi). GPU admission checks may be fail-open."
    )


def vram_probe_mode() -> str:
    """Configured probe mode: auto | nvml | smi (shared embed may remap auto → smi)."""
    v = _vram_probe_env_raw()
    if v in ("nvml", "pynvml"):
        return "nvml"
    if v in ("smi", "nvidia-smi", "nvidia_smi"):
        return "smi"
    # NVML/pynvml can hold the GIL during driver calls; subprocess smi releases it while waiting.
    if v == "auto" and shared_interpreter_embedded():
        if nvidia_smi_available():
            return "smi"
        warn_shared_interpreter_no_smi_once()
    return "auto"


def vram_probe_effective() -> str:
    """Operational probe for /health: last backend used, configured mode, or skipped."""
    if _shared_auto_without_smi():
        return "skipped"
    raw = _vram_probe_env_raw()
    if raw in ("nvml", "pynvml"):
        return "nvml"
    if raw in ("smi", "nvidia-smi", "nvidia_smi"):
        return "smi"
    last = active_vram_probe()
    if last:
        return last
    return vram_probe_mode()


def active_vram_probe() -> str | None:
    """Last successful probe backend used (nvml or nvidia-smi)."""
    return _active_vram_probe


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


_pynvml_mod = None
_nvml_init_failed_at = 0.0
_nvml_lock = threading.Lock()
_NVML_INIT_RETRY_COOLDOWN_S = 30.0
_CTX_BASELINE = 4096


def _nvml_unified_fallback_enabled() -> bool:
    v = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_UNIFIED_FALLBACK", "").strip().lower()
    if v in ("0", "false", "no"):
        return False
    if v in ("1", "true", "yes"):
        return True
    return True


def _is_nvml_not_supported(exc: BaseException) -> bool:
    try:
        import pynvml  # type: ignore[import-untyped]

        if isinstance(exc, pynvml.NVMLError):
            return int(getattr(exc, "value", -1)) == int(
                pynvml.NVML_ERROR_NOT_SUPPORTED
            )
    except Exception:
        pass
    return "not supported" in str(exc).lower()


def _host_unified_free_vram_bytes() -> int | None:
    """Linux fallback when NVML reports no dedicated framebuffer (unified memory GPUs)."""
    from runtime.host_memory import read_linux_host_memory

    mem = read_linux_host_memory()
    if mem is None or mem.available_bytes <= 0:
        return None
    return mem.available_bytes


def _reset_nvml() -> None:
    global _pynvml_mod
    with _nvml_lock:
        if _pynvml_mod is not None:
            try:
                _pynvml_mod.nvmlShutdown()
            except Exception:
                pass
        _pynvml_mod = None


def _pynvml():
    global _pynvml_mod, _nvml_init_failed_at
    with _nvml_lock:
        if _pynvml_mod is not None:
            return _pynvml_mod
        if _nvml_init_failed_at > 0:
            if time.monotonic() - _nvml_init_failed_at < _NVML_INIT_RETRY_COOLDOWN_S:
                return None
        try:
            import pynvml  # type: ignore[import-untyped]  # optional: nvidia-ml-py

            pynvml.nvmlInit()
            _pynvml_mod = pynvml
            _nvml_init_failed_at = 0.0
            return _pynvml_mod
        except Exception:
            _nvml_init_failed_at = time.monotonic()
            return None


def nvml_available() -> bool:
    return _pynvml() is not None


def llama_vram_device_indices(
    main_gpu: int,
    tensor_parallel: int,
    device_count: int,
) -> list[int]:
    """Device indices that participate in llama-server VRAM budgeting."""
    tp = max(1, tensor_parallel)
    if tp == 1:
        return [main_gpu]
    n = min(tp, max(device_count, tp))
    from_main = [main_gpu + i for i in range(n)]
    upper = max(device_count, main_gpu + 1)
    if all(0 <= i < upper for i in from_main):
        return from_main
    return list(range(n))


def _query_nvidia_smi_free_vram_bytes(device_index: int) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        line = out.strip().splitlines()[0].strip()
        mib = int(line)
        return mib * 1024 * 1024
    except (subprocess.SubprocessError, ValueError, IndexError, OSError):
        return None


def _query_nvml_free_vram_bytes(device_index: int) -> int | None:
    global _active_vram_probe
    nvml = _pynvml()
    if nvml is None:
        return None
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.free)
    except Exception as exc:
        if _nvml_unified_fallback_enabled() and _is_nvml_not_supported(exc):
            val = _host_unified_free_vram_bytes()
            if val is not None:
                _active_vram_probe = "host-unified"
            return val
        return None


def _query_free_vram_bytes(device_index: int) -> tuple[int | None, str | None]:
    """Return (free bytes, probe name) for device_index."""
    global _active_vram_probe
    if _shared_auto_without_smi():
        warn_shared_interpreter_no_smi_once()
        return None, None
    mode = vram_probe_mode()
    if mode == "nvml":
        val = _query_nvml_free_vram_bytes(device_index)
        if val is not None:
            _active_vram_probe = "nvml"
        return val, "nvml" if val is not None else None
    if mode == "smi":
        val = _query_nvidia_smi_free_vram_bytes(device_index)
        if val is not None:
            _active_vram_probe = "nvidia-smi"
        return val, "nvidia-smi" if val is not None else None
    val = _query_nvml_free_vram_bytes(device_index)
    if val is not None:
        _active_vram_probe = "nvml"
        return val, "nvml"
    val = _query_nvidia_smi_free_vram_bytes(device_index)
    if val is not None:
        _active_vram_probe = "nvidia-smi"
        return val, "nvidia-smi"
    return None, None


def nvidia_free_vram_bytes(device_index: int = 0, *, fresh: bool = False) -> int | None:
    """Return free VRAM in bytes for device_index, or None if unavailable.

    When ``fresh`` is True, bypass the short TTL cache (used for load calibration).
    """
    if _shared_auto_without_smi():
        warn_shared_interpreter_no_smi_once()
        return None
    mode = vram_probe_mode()
    if mode == "smi" and not nvidia_smi_available():
        return None
    if mode == "nvml" and not nvml_available():
        return None
    if not fresh:
        now = time.monotonic()
        with _smi_lock:
            if device_index in _smi_cache:
                at, val = _smi_cache[device_index]
                if now - at < _SMI_CACHE_TTL_S:
                    return val
    val, _ = _query_free_vram_bytes(device_index)
    with _smi_lock:
        _smi_cache[device_index] = (time.monotonic(), val)
    return val


def resolve_num_ctx(options: dict | None) -> int | None:
    """Read num_ctx from Ollama-shaped request options."""
    if not options:
        return None
    raw = options.get("num_ctx")
    if raw is None:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def resolve_vram_num_ctx(
    options: dict | None = None,
    gguf: Path | None = None,
    *,
    hints: dict[str, int] | None = None,
    explicit: int | None = None,
    llama_args: list[str] | None = None,
) -> int | None:
    """Context for KV scaling: explicit/request > env > llama -c > GGUF context_length."""
    if explicit is not None and explicit > 0:
        return explicit
    n = resolve_num_ctx(options)
    if n is not None:
        return n
    raw = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_NUM_CTX", "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    from runtime.llama_args import parse_llama_server_args

    cli_ctx = parse_llama_server_args(llama_args).num_ctx
    if cli_ctx is not None and cli_ctx > 0:
        return cli_ctx
    h = hints
    if h is None and gguf is not None:
        from runtime.gguf_estimate import gguf_model_hints

        h = gguf_model_hints(gguf)
    if h:
        ctx = h.get("context_length") or 0
        if ctx > 0:
            return ctx
    return None


def _vram_mmap_weight_scale() -> float:
    """Scale weight bytes when GGUF is mmap'd (llama.cpp may not resident all tensors on GPU)."""
    raw = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_MMAP_FACTOR", "").strip()
    if not raw:
        return 1.0
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 1.0


def _gpu_weight_scale(block_count: int | None, n_gpu_layers: int | None) -> float:
    """Fraction of weight tensors expected on GPU (-ngl / layer count)."""
    if n_gpu_layers is None or n_gpu_layers < 0:
        return 1.0
    if n_gpu_layers == 0:
        return 0.0
    layers = block_count if block_count and block_count > 0 else 32
    return min(1.0, n_gpu_layers / layers)


def _vram_weight_block_layout_enabled() -> bool:
    from runtime.gguf_estimate import _vram_block_layout_enabled

    return _vram_block_layout_enabled()


def _vram_weight_tensor_enabled(n_gpu_layers: int | None) -> bool:
    """Use per-tensor GGUF sums when partial offload is configured."""
    v = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_WEIGHT_TENSOR", "1").strip().lower()
    if v in ("0", "false", "no"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    # auto: tensor path only when -ngl is explicit (0 or partial), not full -1 default.
    return n_gpu_layers is not None and n_gpu_layers >= 0


def _vram_kv_gpu_fraction(
    n_gpu_layers: int | None, block_count: int | None
) -> float:
    """Fraction of KV on GPU: per-layer exact path; linear fallback for heuristics."""
    if n_gpu_layers is None or n_gpu_layers < 0:
        return 1.0
    if n_gpu_layers == 0:
        return 0.0
    layers = block_count if block_count and block_count > 0 else 32
    return min(1.0, n_gpu_layers / layers)


def _estimate_gpu_weight_bytes(
    gguf: Path,
    n_gpu_layers: int | None,
) -> int | None:
    from runtime.gguf_estimate import estimate_gpu_weight_bytes

    if n_gpu_layers is None:
        return None
    return estimate_gpu_weight_bytes(gguf, n_gpu_layers)


def _weight_bytes_for_vram(
    gguf: Path,
    *,
    n_gpu_layers: int | None,
    block_count: int | None,
) -> tuple[int, str]:
    """Return (weight bytes, estimate path: tensor | linear)."""
    from runtime.host_memory import estimate_gguf_ram_bytes

    overhead = float(os.environ.get("ZEROLLAMA_RUNTIME_RAM_OVERHEAD", "1.12"))
    mmap = _vram_mmap_weight_scale()
    if _vram_weight_tensor_enabled(n_gpu_layers) and n_gpu_layers is not None:
        raw = _estimate_gpu_weight_bytes(gguf, n_gpu_layers)
        if raw is not None:
            return int(raw * overhead * mmap), "tensor"
    scale = _gpu_weight_scale(block_count, n_gpu_layers)
    return int(estimate_gguf_ram_bytes(gguf) * scale * mmap), "linear"


def _resolve_parallel_slots(
    llama_args: list[str] | None,
    *,
    default: int = 1,
    llama_backend: str | None = None,
) -> int:
    from runtime.kv.live_physical import effective_parallel_slots
    from runtime.worker.factory import resolve_llama_backend

    backend = llama_backend
    if backend is None:
        backend = resolve_llama_backend().value
    return effective_parallel_slots(
        llama_args,
        default=default,
        backend=backend,
    )


def vram_ctx_scale(
    num_ctx: int | None = None,
    *,
    gguf: Path | None = None,
    hints: dict[str, int] | None = None,
    options: dict | None = None,
    llama_args: list[str] | None = None,
) -> float:
    """Scale KV budget vs a 4096-token baseline."""
    ctx = resolve_vram_num_ctx(
        options, gguf, hints=hints, explicit=num_ctx, llama_args=llama_args
    )
    if ctx is None or ctx <= 0:
        ctx = _CTX_BASELINE
    return max(1.0, ctx / _CTX_BASELINE)


def nvidia_free_vram_by_device(device_indices: list[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for idx in device_indices:
        free = nvidia_free_vram_bytes(idx)
        if free is not None:
            out[idx] = free
    return out


def gguf_layer_kv_scale(
    gguf: Path,
    hints: dict[str, int] | None = None,
) -> float:
    """Scale KV headroom from GGUF layer count vs a 32-layer baseline."""
    if os.environ.get("ZEROLLAMA_RUNTIME_VRAM_LAYER_SCALE", "1").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return 1.0
    if hints is None:
        from runtime.gguf_estimate import gguf_model_hints

        hints = gguf_model_hints(gguf)
    blocks = hints.get("block_count")
    if not blocks or blocks <= 0:
        return 1.0
    baseline = int(os.environ.get("ZEROLLAMA_RUNTIME_VRAM_LAYER_BASE", "32"))
    if baseline <= 0:
        baseline = 32
    return max(1.0, blocks / baseline)


def _vram_kv_exact_enabled() -> bool:
    v = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_KV_EXACT", "1").strip().lower()
    return v not in ("0", "false", "no")


def _ngram_scratch_bytes(llama_args: list[str] | None) -> int:
    """Fixed headroom when ngram speculative decode is enabled (no draft GGUF)."""
    from runtime.llama_args import parse_llama_server_args

    spec = parse_llama_server_args(llama_args).spec_type
    if not spec or not spec.startswith("ngram"):
        return 0
    raw = os.environ.get("ZEROLLAMA_RUNTIME_VRAM_NGRAM_SCRATCH_BYTES", "").strip()
    if not raw:
        return 128 * 1024 * 1024
    try:
        from runtime.gpu.admission import parse_size_bytes

        parsed = parse_size_bytes(raw)
        return parsed if parsed is not None and parsed > 0 else 128 * 1024 * 1024
    except Exception:
        return 128 * 1024 * 1024


def _vram_scratch_factor() -> float:
    """Headroom for activations / CUDA context beyond weights+KV (exact path)."""
    try:
        return max(1.0, float(os.environ.get("ZEROLLAMA_RUNTIME_VRAM_SCRATCH_FACTOR", "1.05")))
    except ValueError:
        return 1.05


def _vram_estimate_factor() -> float:
    """Operator calibration multiplier on the final VRAM estimate (default 1.0)."""
    try:
        return max(0.1, float(os.environ.get("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR", "1.0")))
    except ValueError:
        return 1.0


def vram_estimate_autotune_enabled() -> bool:
    """On when GPU VRAM checks run; per-model calibration persists automatically."""
    v = os.environ.get(
        "ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE", ""
    ).strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    from runtime.vram_calibration import vram_probe_calibrate_enabled

    return vram_probe_calibrate_enabled()


def _model_autotune_key(model: str | Path) -> str:
    from runtime.vram_autotune_persist import model_autotune_key

    return model_autotune_key(model)


def _try_model_autotune_key(model: str | Path) -> str | None:
    from runtime.vram_autotune_persist import try_model_autotune_key

    return try_model_autotune_key(model)


def session_vram_estimate_factor(*, model: str | Path | None = None) -> float | None:
    """In-process factor for model (or last calibrated model in session)."""
    with _autotune_lock:
        if model is not None:
            key = _try_model_autotune_key(model)
            if (
                key is not None
                and _session_autotune_model == key
                and _session_autotune_factor is not None
            ):
                return _session_autotune_factor
        elif _session_autotune_factor is not None:
            return _session_autotune_factor
    if not vram_estimate_autotune_enabled():
        return None
    from runtime.vram_autotune_persist import load_persisted_autotune

    return load_persisted_autotune(model)


def set_session_vram_estimate_factor(
    factor: float | None,
    *,
    model: str | Path | None = None,
) -> None:
    """Set or clear session autotune (does not modify process environment)."""
    global _session_autotune_factor, _session_autotune_model
    with _autotune_lock:
        if factor is None:
            _session_autotune_factor = None
            _session_autotune_model = None
            if vram_estimate_autotune_enabled():
                from runtime.vram_autotune_persist import clear_persisted_autotune

                clear_persisted_autotune(model)
        else:
            if not vram_estimate_autotune_enabled():
                return
            _session_autotune_factor = max(0.1, min(3.0, float(factor)))
            _session_autotune_model = (
                _try_model_autotune_key(model) if model is not None else None
            )
            from runtime.vram_autotune_persist import save_persisted_autotune

            save_persisted_autotune(_session_autotune_factor, model=model)


def effective_vram_estimate_factor(*, gguf: Path | str | None = None) -> float:
    """Env factor unless autotune has a per-model or last-calibrated sample."""
    if not vram_estimate_autotune_enabled():
        return _vram_estimate_factor()
    loaded = session_vram_estimate_factor(model=gguf)
    if loaded is not None:
        return loaded
    return _vram_estimate_factor()


def vram_estimate_factor_source(*, gguf: Path | str | None = None) -> str:
    """Where ``estimate_factor_effective`` came from: env, session, or catalog."""
    if not vram_estimate_autotune_enabled():
        return "env"
    if gguf is not None:
        key = _try_model_autotune_key(gguf)
        if key is not None:
            with _autotune_lock:
                if (
                    _session_autotune_factor is not None
                    and _session_autotune_model == key
                ):
                    return "session"
            from runtime.vram_autotune_persist import load_persisted_autotune

            if load_persisted_autotune(gguf) is not None:
                return "catalog"
        return "env"
    with _autotune_lock:
        if _session_autotune_factor is not None:
            return "session"
    from runtime.vram_autotune_persist import load_persisted_autotune

    if load_persisted_autotune(None) is not None:
        return "catalog"
    return "env"


def vram_estimate_autotune_status() -> dict[str, Any]:
    """/health summary: autotune needs PROBE_CALIBRATE + at least one load."""
    from runtime.vram_autotune_persist import persist_status
    from runtime.vram_calibration import vram_probe_calibrate_enabled
    from runtime.vram_factor_export import export_status
    from runtime.vram_env_apply import apply_status

    enabled = vram_estimate_autotune_enabled()
    with _autotune_lock:
        session = _session_autotune_factor
        session_model = _session_autotune_model
    calibrate = vram_probe_calibrate_enabled()
    persist = persist_status(
        session_factor=session,
        session_model=session_model,
    )
    has_sample = session is not None or (
        persist.get("model_count", 0) > 0 if enabled else False
    )
    return {
        "enabled": enabled,
        "env_factor": _vram_estimate_factor(),
        "effective_factor": effective_vram_estimate_factor(),
        "session_factor": session,
        "session_model": session_model,
        "pending_first_calibration": enabled and not has_sample,
        "probe_calibrate_enabled": calibrate,
        "probe_calibrate_required": enabled and not calibrate,
        "persist": persist,
        "export": export_status(),
        "apply_exported_env": apply_status(),
        "note": (
            "Product defaults: autotune + persist + .env export when GPU VRAM checks "
            "and probe calibration run. Autotune overrides VRAM_ESTIMATE_FACTOR env."
            if enabled
            else None
        ),
    }


def _resolve_draft_gguf(
    llama_args: list[str] | None,
    draft_gguf: Path | None,
) -> Path | None:
    if draft_gguf is not None and draft_gguf.is_file():
        return draft_gguf
    from runtime.llama_args import parse_llama_server_args

    raw = parse_llama_server_args(llama_args).draft_model
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_file() else None


def estimate_gguf_vram_bytes(
    gguf: Path,
    *,
    tensor_parallel: int = 1,
    num_ctx: int | None = None,
    options: dict | None = None,
    llama_args: list[str] | None = None,
    parallel_slots_default: int = 1,
    llama_backend: str | None = None,
    n_gpu_layers_default: int = -1,
    draft_gguf: Path | None = None,
    draft_n_gpu_layers: int = -1,
    include_speculative_draft: bool = True,
    _apply_estimate_factor: bool = True,
) -> int:
    """Best-effort VRAM for weights on GPU (tensor payload + KV headroom)."""
    from runtime.gguf_estimate import estimate_kv_cache_bytes, gguf_arch_hints
    from runtime.host_memory import estimate_gguf_ram_bytes
    from runtime.llama_args import parse_llama_server_args

    arch = gguf_arch_hints(gguf)
    hints = arch.scalar
    cli = parse_llama_server_args(llama_args)
    n_gpu_layers = (
        cli.n_gpu_layers
        if cli.n_gpu_layers is not None
        else n_gpu_layers_default
    )
    parallel_slots = _resolve_parallel_slots(
        llama_args,
        default=parallel_slots_default,
        llama_backend=llama_backend,
    )
    base, _weight_path = _weight_bytes_for_vram(
        gguf,
        n_gpu_layers=n_gpu_layers,
        block_count=hints.get("block_count"),
    )
    blocks = hints.get("block_count")
    kv_layer_frac = _vram_kv_gpu_fraction(n_gpu_layers, blocks)
    ctx = resolve_vram_num_ctx(
        options, gguf, hints=hints, explicit=num_ctx, llama_args=llama_args
    )
    if ctx is None or ctx <= 0:
        ctx = _CTX_BASELINE

    kv_exact: int | None = None
    if _vram_kv_exact_enabled():
        kv_exact = estimate_kv_cache_bytes(arch, ctx, n_gpu_layers=n_gpu_layers)
        if kv_exact is not None and parallel_slots > 1:
            kv_exact *= parallel_slots

    if kv_exact is not None:
        # Weights + explicit K/V; avoid multiplying ctx/layer heuristics again.
        kv_factor = float(os.environ.get("ZEROLLAMA_RUNTIME_VRAM_KV_FACTOR", "1.10"))
        kv_term = int(kv_exact * kv_factor)
        total = int((base + kv_term) * _vram_scratch_factor())
    else:
        kv_factor = float(os.environ.get("ZEROLLAMA_RUNTIME_VRAM_KV_FACTOR", "1.15"))
        ctx_layer = (
            vram_ctx_scale(
                ctx,
                gguf=gguf,
                hints=hints,
                options=options,
                llama_args=llama_args,
            )
            * gguf_layer_kv_scale(gguf, hints=hints)
        )
        weight_part = base * kv_factor
        # KV grows with ctx/layer; parallel_slots scales KV only (matches exact path).
        kv_extra = weight_part * max(0.0, ctx_layer - 1.0) * kv_layer_frac
        total = int(
            (weight_part + kv_extra * parallel_slots) * _vram_scratch_factor()
        )
    tp = max(1, tensor_parallel)
    if tp > 1:
        total = (total + tp - 1) // tp

    if include_speculative_draft:
        draft_path = _resolve_draft_gguf(llama_args, draft_gguf)
        if draft_path is not None:
            d_ngl = draft_n_gpu_layers
            if cli.draft_n_gpu_layers is not None:
                d_ngl = cli.draft_n_gpu_layers
            # Draft shares llama-server slots; do not multiply KV by -np again.
            total += estimate_gguf_vram_bytes(
                draft_path,
                tensor_parallel=tensor_parallel,
                num_ctx=num_ctx,
                options=options,
                llama_args=None,
                parallel_slots_default=1,
                n_gpu_layers_default=d_ngl,
                include_speculative_draft=False,
                _apply_estimate_factor=False,
            )
    total += _ngram_scratch_bytes(llama_args)
    if _apply_estimate_factor:
        factor = effective_vram_estimate_factor(gguf=gguf)
        if factor != 1.0:
            total = int(total * factor)
    return total


def gpu_vram_check_enabled() -> bool:
    v = os.environ.get("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "").strip().lower()
    if v in ("0", "false", "no"):
        return False
    if v in ("1", "true", "yes"):
        return True
    mode = vram_probe_mode()
    if mode == "nvml":
        return nvml_available()
    if mode == "smi":
        return nvidia_smi_available()
    return nvml_available() or nvidia_smi_available()


def describe_vram_estimate(
    gguf: Path,
    *,
    num_ctx: int | None = None,
    options: dict | None = None,
    tensor_parallel: int = 1,
    llama_args: list[str] | None = None,
    parallel_slots_default: int = 1,
    llama_backend: str | None = None,
    n_gpu_layers_default: int = -1,
    draft_gguf: Path | None = None,
    draft_n_gpu_layers: int = -1,
) -> dict[str, int | str | None]:
    """Summary for /health and debugging (no GPU probe)."""
    from runtime.gguf_estimate import (
        estimate_kv_cache_bytes,
        gguf_arch_hints,
        kv_cache_type_summary,
    )
    from runtime.llama_args import parse_llama_server_args

    arch = gguf_arch_hints(gguf)
    hints = arch.scalar
    cli = parse_llama_server_args(llama_args)
    ctx = resolve_vram_num_ctx(
        options, gguf, hints=hints, explicit=num_ctx, llama_args=llama_args
    )
    n_gl = cli.n_gpu_layers if cli.n_gpu_layers is not None else n_gpu_layers_default
    kv_b = (
        estimate_kv_cache_bytes(arch, ctx, n_gpu_layers=n_gl)
        if ctx and ctx > 0
        else None
    )
    slots = _resolve_parallel_slots(
        llama_args,
        default=parallel_slots_default,
        llama_backend=llama_backend,
    )
    if kv_b is not None and slots > 1:
        kv_b *= slots
    exact = _vram_kv_exact_enabled() and kv_b is not None
    gpu_w, weight_path = _weight_bytes_for_vram(
        gguf,
        n_gpu_layers=n_gl,
        block_count=hints.get("block_count"),
    )
    kv_layer_frac = _vram_kv_gpu_fraction(n_gl, hints.get("block_count"))
    draft_path = _resolve_draft_gguf(llama_args, draft_gguf)
    draft_required: int | None = None
    if draft_path is not None:
        d_ngl = (
            cli.draft_n_gpu_layers
            if cli.draft_n_gpu_layers is not None
            else draft_n_gpu_layers
        )
        try:
            draft_required = estimate_gguf_vram_bytes(
                draft_path,
                tensor_parallel=tensor_parallel,
                num_ctx=num_ctx,
                options=options,
                parallel_slots_default=1,
                n_gpu_layers_default=d_ngl,
                include_speculative_draft=False,
                _apply_estimate_factor=False,
            )
        except OSError:
            draft_required = None
    try:
        required = estimate_gguf_vram_bytes(
            gguf,
            tensor_parallel=tensor_parallel,
            num_ctx=num_ctx,
            options=options,
            llama_args=llama_args,
            parallel_slots_default=parallel_slots_default,
            llama_backend=llama_backend,
            n_gpu_layers_default=n_gpu_layers_default,
            draft_gguf=draft_gguf,
            draft_n_gpu_layers=draft_n_gpu_layers,
        )
    except OSError:
        required = None
    return {
        "gguf": str(gguf.resolve()) if gguf.is_file() else str(gguf),
        "path": "exact_kv" if exact else "heuristic",
        "num_ctx": ctx,
        "parallel_slots": slots,
        "n_gpu_layers": n_gl,
        "weight_estimate": weight_path,
        "gpu_weight_bytes": gpu_w,
        "kv_gpu_fraction": kv_layer_frac,
        "n_gpu_layers_kv": n_gl,
        "draft_model": str(draft_path) if draft_path else None,
        "draft_required_per_gpu_bytes": draft_required,
        "kv_cache_bytes": kv_b,
        "required_per_gpu_bytes": required,
        "ngram_scratch_bytes": _ngram_scratch_bytes(llama_args),
        "mmap_weight_scale": _vram_mmap_weight_scale(),
        "weight_block_layout": _vram_weight_block_layout_enabled(),
        "estimate_factor_env": _vram_estimate_factor(),
        "estimate_factor_effective": effective_vram_estimate_factor(gguf=gguf),
        "estimate_factor_source": vram_estimate_factor_source(gguf=gguf),
        "estimate_factor_autotune": session_vram_estimate_factor(model=gguf),
        "sliding_window": hints.get("sliding_window"),
        "sliding_window_per_layer": (
            list(arch.sliding_window_per_layer)
            if arch.sliding_window_per_layer
            else None
        ),
        "head_count_kv_per_layer": (
            list(arch.head_count_kv_per_layer)
            if arch.head_count_kv_per_layer
            else None
        ),
        **kv_cache_type_summary(arch),
    }


def format_vram_reject_kv_hint(desc: dict[str, Any]) -> str:
    """KV slice for pre-load reject messages (from ``describe_vram_estimate``)."""
    from runtime.host_memory import format_bytes

    kv_b = desc.get("kv_cache_bytes")
    if not isinstance(kv_b, int) or kv_b <= 0:
        return ""
    per_slot = desc.get("kv_bytes_per_slot")
    slot_note = (
        f", {per_slot} B/slot"
        if isinstance(per_slot, int) and per_slot > 0
        else ""
    )
    return f"; kv_cache≈{format_bytes(kv_b)}{slot_note}"


def vram_budget_health(
    vram_estimate: dict[str, Any] | None,
    *,
    gpu_free_bottleneck: int | None,
    inference_paused_for_reserve: bool = False,
    suggest_profile: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Compare model estimate to bottleneck free VRAM (for /health dashboards)."""
    from runtime.host_memory import format_bytes

    if not vram_estimate or gpu_free_bottleneck is None:
        return None
    req = vram_estimate.get("required_per_gpu_bytes")
    if not isinstance(req, int) or req <= 0:
        return None
    margin = float(os.environ.get("ZEROLLAMA_RUNTIME_VRAM_MARGIN", "1.0"))
    req_margin = int(req * margin)
    headroom = gpu_free_bottleneck - req
    headroom_margin = gpu_free_bottleneck - req_margin
    out: dict[str, Any] = {
        "model_gguf": vram_estimate.get("gguf"),
        "required_per_gpu_bytes": req,
        "vram_margin": margin,
        "required_with_margin_bytes": req_margin,
        "free_bottleneck_bytes": gpu_free_bottleneck,
        "headroom_bytes": headroom,
        "headroom_with_margin_bytes": headroom_margin,
        "fits": headroom >= 0,
        "fits_with_margin": headroom_margin >= 0,
        "required_per_gpu": format_bytes(req),
        "required_with_margin": format_bytes(req_margin),
        "free_bottleneck": format_bytes(gpu_free_bottleneck),
        "headroom": format_bytes(abs(headroom)),
        "estimate_path": vram_estimate.get("path"),
        "estimate_factor_effective": vram_estimate.get("estimate_factor_effective"),
    }
    kv_b = vram_estimate.get("kv_cache_bytes")
    if isinstance(kv_b, int) and kv_b > 0:
        out["kv_cache_bytes"] = kv_b
        out["kv_cache"] = format_bytes(kv_b)
    hds = vram_estimate.get("head_dim_source")
    if isinstance(hds, str):
        out["head_dim_source"] = hds
    from runtime.gpu.admission import (
        admission_vram_gate_enabled,
        min_free_vram_for_admission,
        training_vram_reserve_bytes,
    )

    if admission_vram_gate_enabled():
        min_free = min_free_vram_for_admission()
        if min_free is not None:
            try:
                reserve = training_vram_reserve_bytes(
                    inference_paused=inference_paused_for_reserve
                )
            except Exception:
                reserve = 0
            effective = max(0, gpu_free_bottleneck - reserve)
            out["admission_min_free_bytes"] = min_free
            out["admission_training_reserve_bytes"] = reserve
            out["admission_effective_free_bytes"] = effective
            out["admission_fits"] = effective >= min_free
            out["admission_min_free"] = format_bytes(min_free)
            out["admission_effective_free"] = format_bytes(effective)
    st = vram_estimate_autotune_status()
    if st.get("pending_first_calibration"):
        out["autotune_pending"] = True

    eff = out.get("admission_effective_free_bytes")
    if not isinstance(eff, int) or eff <= 0:
        fb = out.get("free_bottleneck_bytes")
        if isinstance(fb, int) and fb > 0:
            eff = fb
    if isinstance(eff, int) and eff > 0:
        gguf_raw = vram_estimate.get("gguf")
        if isinstance(gguf_raw, str) and gguf_raw:
            from runtime.vram_suggest import build_suggest_profile, suggest_max_num_ctx

            profile = build_suggest_profile(
                vram_estimate,
                **(suggest_profile or {}),
            )
            min_free_for_suggest = out.get("admission_min_free_bytes")
            if not isinstance(min_free_for_suggest, int):
                min_free_for_suggest = None
            try:
                suggested = suggest_max_num_ctx(
                    Path(gguf_raw),
                    eff,
                    margin=margin,
                    min_free_bytes=min_free_for_suggest,
                    **profile,
                )
            except OSError:
                suggested = None
            if suggested is not None:
                out["suggested_max_num_ctx"] = suggested
                cur = vram_estimate.get("num_ctx")
                if isinstance(cur, int) and cur > suggested:
                    out["num_ctx_over_budget"] = True
    return out


def effective_vram_free_after_reserve(
    *,
    main_gpu: int = 0,
    tensor_parallel: int = 1,
    device_count: int = 1,
    inference_paused_for_reserve: bool = False,
) -> int | None:
    """Bottleneck free VRAM minus training reserve (for suggest / clamp)."""
    indices = llama_vram_device_indices(main_gpu, tensor_parallel, device_count)
    free_by_dev = nvidia_free_vram_by_device(indices)
    if not free_by_dev:
        return None
    free = min(free_by_dev.values())
    reserve = 0
    if inference_paused_for_reserve:
        from runtime.gpu.admission import training_vram_reserve_bytes

        try:
            reserve = training_vram_reserve_bytes(inference_paused=True)
        except Exception:
            reserve = 0
    return max(0, free - reserve)


def training_reserve_bytes_for_load(*, active: bool) -> int:
    """Bytes to subtract from free VRAM before load pre-check (parity with admission)."""
    if not active:
        return 0
    from runtime.gpu.admission import AdmissionMisconfigured, training_vram_reserve_bytes
    from runtime.worker.llama_server import LlamaServerError

    try:
        return training_vram_reserve_bytes(inference_paused=True)
    except AdmissionMisconfigured as e:
        raise LlamaServerError(str(e)) from e


def check_gguf_vram_budget(
    gguf: Path,
    *,
    main_gpu: int = 0,
    tensor_parallel: int = 1,
    device_count: int = 1,
    margin: float | None = None,
    num_ctx: int | None = None,
    options: dict | None = None,
    llama_args: list[str] | None = None,
    parallel_slots_default: int = 1,
    llama_backend: str | None = None,
    n_gpu_layers_default: int = -1,
    draft_gguf: Path | None = None,
    draft_n_gpu_layers: int = -1,
    training_reserve_active: bool = False,
    priority: InferencePriority | None = None,
) -> None:
    """Raise LlamaServerError if GGUF likely exceeds free GPU memory."""
    from runtime.gpu.admission import (
        admission_vram_gate_enabled,
        effective_min_free_for_priority,
        min_free_vram_for_admission,
        vram_gate_bypassed,
    )
    from runtime.gpu.priority import priority_from_options
    from runtime.host_memory import format_bytes
    from runtime.worker.llama_server import LlamaServerError

    if not gpu_vram_check_enabled():
        return
    if priority is None:
        priority = priority_from_options(options)
    indices = llama_vram_device_indices(main_gpu, tensor_parallel, device_count)
    free_by_dev = nvidia_free_vram_by_device(indices)
    # Skip when any GPU is unreadable (avoid treating a failed query as 0 bytes free).
    if not free_by_dev or len(free_by_dev) < len(indices):
        from runtime.gpu.admission import admission_vram_gate_enabled

        if admission_vram_gate_enabled():
            raise LlamaServerError(
                "GPU free VRAM unavailable while runtime VRAM checks are enabled "
                "(install nvidia-smi / nvidia-ml-py or set ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM=0)"
            )
        return
    if margin is None:
        margin = float(os.environ.get("ZEROLLAMA_RUNTIME_VRAM_MARGIN", "1.0"))
    required_per_gpu = int(
        estimate_gguf_vram_bytes(
            gguf,
            tensor_parallel=tensor_parallel,
            num_ctx=num_ctx,
            options=options,
            llama_args=llama_args,
            parallel_slots_default=parallel_slots_default,
            llama_backend=llama_backend,
            n_gpu_layers_default=n_gpu_layers_default,
            draft_gguf=draft_gguf,
            draft_n_gpu_layers=draft_n_gpu_layers,
        )
        * margin
    )
    # Fold 1 GiB (and 1.5x for low) into model budget so enqueue does not double-check.
    if admission_vram_gate_enabled() and not vram_gate_bypassed(priority):
        min_free = min_free_vram_for_admission()
        if min_free is not None:
            required_per_gpu = max(
                required_per_gpu,
                effective_min_free_for_priority(min_free, priority),
            )
    raw_free_by_dev = dict(free_by_dev)
    reserve = training_reserve_bytes_for_load(active=training_reserve_active)
    if reserve > 0:
        for idx in indices:
            free_by_dev[idx] = max(0, raw_free_by_dev.get(idx, 0) - reserve)
    bottleneck = min(indices, key=lambda i: free_by_dev.get(i, 0))
    effective_free = free_by_dev.get(bottleneck, 0)
    free_raw = raw_free_by_dev.get(bottleneck, 0)
    if required_per_gpu > effective_free:
        kv_hint = ""
        try:
            desc = describe_vram_estimate(
                gguf,
                num_ctx=num_ctx,
                options=options,
                tensor_parallel=tensor_parallel,
                llama_args=llama_args,
                parallel_slots_default=parallel_slots_default,
                llama_backend=llama_backend,
                n_gpu_layers_default=n_gpu_layers_default,
                draft_gguf=draft_gguf,
                draft_n_gpu_layers=draft_n_gpu_layers,
            )
            kv_hint = format_vram_reject_kv_hint(desc)
            factor = desc.get("estimate_factor_effective")
            if isinstance(factor, (int, float)) and factor != 1.0:
                kv_hint += f"; estimate_factor={factor}"
            hds = desc.get("head_dim_source")
            if isinstance(hds, str):
                kv_hint += f"; head_dim_source={hds}"
            if vram_estimate_autotune_status().get("pending_first_calibration"):
                kv_hint += "; autotune pending (load once with probe calibrate)"
            from runtime.gpu.admission import min_free_vram_for_admission
            from runtime.vram_suggest import format_suggest_num_ctx_hint

            ctx_hint = format_suggest_num_ctx_hint(
                gguf,
                effective_free,
                margin=margin,
                min_free_bytes=min_free_vram_for_admission(),
                priority=priority,
                num_ctx=num_ctx,
                options=options,
                tensor_parallel=tensor_parallel,
                llama_args=llama_args,
                parallel_slots_default=parallel_slots_default,
                llama_backend=llama_backend,
                n_gpu_layers_default=n_gpu_layers_default,
                draft_gguf=draft_gguf,
                draft_n_gpu_layers=draft_n_gpu_layers,
            )
            if ctx_hint:
                kv_hint += ctx_hint
        except OSError:
            pass
        reserve_note = (
            f" (effective free {format_bytes(effective_free)} after "
            f"{format_bytes(reserve)} training reserve)"
            if reserve
            else ""
        )
        if len(indices) == 1:
            raise LlamaServerError(
                f"model requires about {format_bytes(required_per_gpu)} GPU memory "
                f"but only {format_bytes(free_raw)} is free on GPU {bottleneck}"
                f"{reserve_note}{kv_hint}"
            )
        raise LlamaServerError(
            f"model requires about {format_bytes(required_per_gpu)} per GPU "
            f"(tensor_parallel={tensor_parallel}) "
            f"but GPU {bottleneck} only has {format_bytes(free_raw)} free"
            f"{reserve_note}{kv_hint}"
        )
