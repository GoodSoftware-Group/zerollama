"""Inference admission policy (Phase 11) — opinionated defaults, minimal env surface.

Operator knobs (policy):
  - ZEROLLAMA_RUNTIME_INFERENCE_POLICY=off  — disable inference-first backpressure (defer/ggml/batch gates)
  - ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM=0      — disable GPU/host VRAM pre-check (load + enqueue + 1 GiB floor)

VRAM headroom (when checks on; size strings like 1GiB, 512MiB):
  - ZEROLLAMA_RUNTIME_VRAM_MIN_FREE          — admission floor (default 1 GiB)
  - ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE  — held while training busy (default 2 GiB)

Optional safety cap: ZEROLLAMA_RUNTIME_MAX_QUEUE (default 512, code constant).

Backpressure thresholds: defaults in inference_policy.py; optional ZEROLLAMA_RUNTIME_* overrides (see phase11-runtime-admission.md).
"""

from __future__ import annotations

import logging
import re
import threading

from runtime.gpu.inference_policy import (
    CROSS_QUEUE_PRESSURE_ON,
    DEFER_WAITING_MIN,
    GGML_SCHED_BACKLOG_MIN,
    LOW_PRIORITY_VRAM_FACTOR,
    RUNTIME_BACKLOG_BATCH_MIN,
    VRAM_MIN_FREE_DEFAULT_BYTES,
    backpressure_snapshot,
    batch_priority_subject_to_backpressure,
    inference_backpressure_blocks_low,
    inference_first_enabled,
)
from runtime.gpu.priority import InferencePriority
from runtime.host_memory import format_bytes

_log = logging.getLogger(__name__)

# Headroom reserved while training holds the GPU (handoff, pause, or Go training-gpu-busy).
TRAINING_VRAM_RESERVE_BYTES = 2 * 1024**3

# Waiting-queue cap (override via ZEROLLAMA_RUNTIME_MAX_QUEUE only if needed).
DEFAULT_MAX_RUNTIME_QUEUE = 512

_SIZE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(b|kib|mib|gib|kb|mb|gb|ki|mi|gi)?\s*$",
    re.IGNORECASE,
)

_admission_warn_once = threading.Lock()
_admission_warned: set[str] = set()


class AdmissionRejected(Exception):
    """Request cannot enter the runtime scheduler queue."""


class AdmissionMisconfigured(Exception):
    """VRAM admission cannot run (e.g. GPU probe unavailable while checks are on)."""


def max_runtime_queue() -> int:
    import os

    raw = os.environ.get("ZEROLLAMA_RUNTIME_MAX_QUEUE", "").strip()
    if not raw:
        return DEFAULT_MAX_RUNTIME_QUEUE
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_RUNTIME_QUEUE


def parse_size_bytes(raw: str) -> int | None:
    """Parse a size string for operator env (``1GiB``, ``512MiB``; ``GB``/``MB`` use 1000-based)."""
    m = _SIZE_RE.match(raw.strip())
    if not m:
        return None
    num = float(m.group(1))
    unit = (m.group(2) or "b").lower()
    mult = {
        "b": 1,
        "kib": 1024,
        "ki": 1024,
        "kb": 1000,
        "mib": 1024**2,
        "mi": 1024**2,
        "mb": 1000**2,
        "gib": 1024**3,
        "gi": 1024**3,
        "gb": 1000**3,
    }.get(unit)
    if mult is None:
        return None
    return int(num * mult)


def _warn_admission_once(key: str, msg: str, *args: object) -> None:
    with _admission_warn_once:
        if key in _admission_warned:
            return
        _admission_warned.add(key)
    _log.warning(msg, *args)


def admission_vram_gate_enabled() -> bool:
    """1 GiB headroom gate tracks load pre-check (CHECK_GPU_VRAM), not inference-first.

    Why separate from INFERENCE_POLICY: operators may disable defer/ggml throttling
    without turning off VRAM safety rails on a full GPU.
    """
    from runtime.gpu_vram import gpu_vram_check_enabled

    return gpu_vram_check_enabled()


def admission_config_error() -> str | None:
    return None


def _env_size_bytes(name: str, default: int) -> int:
    import os

    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    parsed = parse_size_bytes(raw)
    if parsed is None:
        _warn_admission_once(
            name,
            "invalid %s=%r; using default %s",
            name,
            raw,
            format_bytes(default),
        )
        return default
    return max(0, parsed)


def configured_min_free_vram_bytes() -> int:
    """Minimum free VRAM for admission (default 1 GiB; override via VRAM_MIN_FREE).

    Why env, not only code constants: 5080 operators measure under real chat+training
    load; rebuilding Python to change 1 GiB was the wrong feedback loop.
    """
    return _env_size_bytes(
        "ZEROLLAMA_RUNTIME_VRAM_MIN_FREE", VRAM_MIN_FREE_DEFAULT_BYTES
    )


def configured_training_vram_reserve_bytes() -> int:
    """Training headroom while GPU is busy (default 2 GiB; override via TRAINING_VRAM_RESERVE).

    Why subtract at load/suggest: training, handoff, and Go ``training-gpu-busy`` can
    hold the card while inference requests still arrive on :8081 — reserve avoids
    admitting work that cannot fit beside a training job.
    """
    return _env_size_bytes(
        "ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE", TRAINING_VRAM_RESERVE_BYTES
    )


def min_free_vram_for_admission() -> int | None:
    if not admission_vram_gate_enabled():
        return None
    return configured_min_free_vram_bytes()


def training_vram_reserve_bytes(*, inference_paused: bool = False) -> int:
    """Bytes held back for training while inference is paused or Go reports GPU busy."""
    if not inference_paused:
        return 0
    return configured_training_vram_reserve_bytes()


def vram_gate_bypassed(priority: InferencePriority) -> bool:
    return priority == InferencePriority.HIGH


def effective_min_free_for_priority(
    min_free: int, priority: InferencePriority
) -> int:
    if priority == InferencePriority.LOW:
        return int(min_free * LOW_PRIORITY_VRAM_FACTOR)
    return min_free


def defer_backlog_admission_mode() -> str:
    return "on" if inference_first_enabled() else "off"


def defer_backlog_policy_active() -> bool:
    return inference_first_enabled()


def defer_backlog_blocks_admission_now() -> bool:
    if not inference_first_enabled():
        return False
    from runtime.go_coordination import go_coordination_is_fresh, go_defer_waiting

    if not go_coordination_is_fresh():
        return False
    return go_defer_waiting() >= DEFER_WAITING_MIN


def ggml_backlog_admission_mode() -> str:
    return "on" if inference_first_enabled() else "off"


def ggml_backlog_policy_active() -> bool:
    return inference_first_enabled()


def ggml_backlog_blocks_admission_now() -> bool:
    if not inference_first_enabled():
        return False
    from runtime.go_coordination import go_coordination_is_fresh, go_ggml_sched_backlog

    if not go_coordination_is_fresh():
        return False
    return go_ggml_sched_backlog(include_loaded=False) >= GGML_SCHED_BACKLOG_MIN


def ggml_paused_admission_mode() -> str:
    return "on" if inference_first_enabled() else "off"


def ggml_paused_policy_active() -> bool:
    return inference_first_enabled()


def ggml_paused_blocks_admission_now() -> bool:
    if not inference_first_enabled():
        return False
    from runtime.go_coordination import go_coordination_is_fresh, go_ggml_loads_paused

    if not go_coordination_is_fresh():
        return False
    return go_ggml_loads_paused()


def runtime_backlog_admission_mode() -> str:
    return "on" if inference_first_enabled() else "off"


def runtime_backlog_policy_active() -> bool:
    return inference_first_enabled()


def runtime_backlog_blocks_admission_now(*, waiting: int, running: int) -> bool:
    if not inference_first_enabled():
        return False
    return (max(0, waiting) + max(0, running)) >= RUNTIME_BACKLOG_BATCH_MIN


def admission_vram_gate_mode() -> str:
    return "on" if admission_vram_gate_enabled() else "off"


def check_inference_first_admission(
    *,
    waiting: int,
    running: int,
    gpu_free_bytes: int | None = None,
    inference_paused: bool = False,
    priority: InferencePriority = InferencePriority.NORMAL,
    runtime_oldest_fifo: int = 0,
    skip_generic_vram_gate: bool = False,
) -> None:
    """Metrics-driven inference-first: throttle batch (low) work under pressure.

    Why skip_generic_vram_gate: when a GGUF path is known, engine._vram_precheck_enqueue
    runs check_gguf_vram_budget (model + min floor). Avoid a second bottleneck-only check.
    """
    metrics = backpressure_snapshot(
        runtime_waiting=waiting, runtime_running=running
    )
    backlog = int(metrics["runtime_backlog"])

    if not vram_gate_bypassed(priority) and batch_priority_subject_to_backpressure(
        priority
    ):
        if inference_backpressure_blocks_low(
            runtime_waiting=waiting,
            runtime_running=running,
            runtime_oldest_fifo=runtime_oldest_fifo,
        ):
            reason = "inference backpressure"
            from runtime.gpu.inference_policy import cross_fifo_blocks_low

            if cross_fifo_blocks_low(runtime_oldest_fifo=runtime_oldest_fifo):
                from runtime.go_coordination import go_fifo_oldest_ggml

                reason = (
                    f"cross-queue FIFO (ggml ticket {go_fifo_oldest_ggml()} "
                    f"< runtime {runtime_oldest_fifo or 'head'})"
                )
            elif backlog >= RUNTIME_BACKLOG_BATCH_MIN:
                reason = (
                    f"runtime queue backlog ({backlog} >= {RUNTIME_BACKLOG_BATCH_MIN})"
                )
            elif metrics.get("go_mirror_fresh"):
                if metrics.get("ggml_loads_paused"):
                    reason = "go ggml loads paused"
                elif int(metrics.get("go_defer_waiting", 0)) >= DEFER_WAITING_MIN:
                    reason = (
                        f"go training defer backlog ({metrics['go_defer_waiting']} waiting)"
                    )
                elif int(metrics.get("go_ggml_backlog", 0)) >= GGML_SCHED_BACKLOG_MIN:
                    reason = f"go ggml scheduler backlog ({metrics['go_ggml_backlog']})"
                elif metrics.get("cross_queue_pressure_latched"):
                    reason = (
                        f"cross-queue pressure {metrics['cross_queue_pressure']} "
                        f"(>= {CROSS_QUEUE_PRESSURE_ON})"
                    )
            raise AdmissionRejected(
                f"{reason}; inference-first (priority={priority.value})"
            )

    if gpu_free_bytes is not None and not skip_generic_vram_gate:
        check_vram_admission(
            gpu_free_bytes,
            backlog=backlog,
            inference_paused=inference_paused,
            priority=priority,
        )


def check_ggml_paused_admission(
    priority: InferencePriority = InferencePriority.NORMAL,
) -> None:
    check_inference_first_admission(waiting=0, running=0, priority=priority)


def check_runtime_backlog_admission(
    *,
    waiting: int,
    running: int,
    priority: InferencePriority = InferencePriority.NORMAL,
) -> None:
    check_inference_first_admission(
        waiting=waiting, running=running, priority=priority
    )


def check_ggml_backlog_admission(
    priority: InferencePriority = InferencePriority.NORMAL,
) -> None:
    check_inference_first_admission(waiting=0, running=0, priority=priority)


def check_training_defer_admission(
    priority: InferencePriority = InferencePriority.NORMAL,
) -> None:
    check_inference_first_admission(waiting=0, running=0, priority=priority)


def check_vram_admission(
    gpu_free_bytes: int | None,
    *,
    backlog: int,
    inference_paused: bool = False,
    priority: InferencePriority = InferencePriority.NORMAL,
) -> None:
    if not admission_vram_gate_enabled():
        return

    min_free = configured_min_free_vram_bytes()

    if gpu_free_bytes is None:
        raise AdmissionMisconfigured(
            "GPU free VRAM unavailable while ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM is enabled"
        )

    if vram_gate_bypassed(priority):
        return

    reserve = training_vram_reserve_bytes(inference_paused=inference_paused)
    min_free = effective_min_free_for_priority(min_free, priority)
    effective_free = max(0, gpu_free_bytes - reserve)
    if effective_free < min_free:
        raise AdmissionRejected(
            f"gpu free {format_bytes(effective_free)} below admission minimum "
            f"{format_bytes(min_free)}"
            + (f" (reserve {format_bytes(reserve)} for training)" if reserve else "")
            + f"; backlog={backlog}; priority={priority.value}"
        )
