"""GPU coordination between inference runtime and embedded training (training.py)."""

from __future__ import annotations

import threading
from enum import Enum
from typing import Any, Callable

from runtime.gpu.admission import (
    AdmissionMisconfigured,
    AdmissionRejected,
    check_inference_first_admission,
    configured_min_free_vram_bytes,
    configured_training_vram_reserve_bytes,
    max_runtime_queue,
    min_free_vram_for_admission,
    training_vram_reserve_bytes,
    vram_gate_bypassed,
)
from runtime.gpu.inference_policy import backpressure_snapshot, inference_first_enabled
from runtime.gpu.priority import InferencePriority


class InferenceState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    UNLOADED = "unloaded"


UnloadHook = Callable[[], None]


class InferenceGpuCoordinator:
    """Mirrors Go scheduler: PauseNewLoads → UnloadAllRunners → ResumeLoads.

    Phase 4 wires this to Go via CGO or IPC; Phase 0–3 uses it inside Python only.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = InferenceState.RUNNING
        self._unload_hook: UnloadHook | None = None
        self._max_waiting = max_runtime_queue()
        # Set during training-handoff / pause; drives TRAINING_VRAM_RESERVE_ONLY_WHEN_PAUSED.
        self._training_handoff_active = False
        # Set by Go training GPU policy via POST /internal/training-gpu-busy.
        self._go_training_gpu_busy = False

    @property
    def state(self) -> InferenceState:
        with self._lock:
            return self._state

    def set_unload_hook(self, hook: UnloadHook | None) -> None:
        with self._lock:
            self._unload_hook = hook

    def pause_inference(self) -> None:
        with self._lock:
            if self._state == InferenceState.UNLOADED:
                return
            self._state = InferenceState.PAUSED
            self._training_handoff_active = True

    def unload_all(self) -> None:
        with self._lock:
            self._state = InferenceState.UNLOADED
            hook = self._unload_hook
        if hook is not None:
            hook()

    def resume_inference(self) -> None:
        """Allow new runtime loads after handoff without restarting zerollama."""
        with self._lock:
            if self._state == InferenceState.UNLOADED:
                self._state = InferenceState.RUNNING
            elif self._state == InferenceState.PAUSED:
                self._state = InferenceState.RUNNING
            self._training_handoff_active = False

    def accepts_new_loads(self) -> bool:
        with self._lock:
            return self._state == InferenceState.RUNNING

    def inference_paused_for_vram_reserve(self) -> bool:
        """Training handoff or non-running state (legacy name for reserve helpers)."""
        with self._lock:
            return (
                self._training_handoff_active
                or self._state != InferenceState.RUNNING
            )

    def training_reserve_active(self) -> bool:
        """True when admission/load should subtract TRAINING_VRAM_RESERVE (handoff, pause, or Go busy)."""
        return self._training_reserve_active()

    def set_go_training_gpu_busy(self, busy: bool) -> None:
        """Mirror Go trainingOccupiesGPU for VRAM reserve while runtime may still be RUNNING."""
        with self._lock:
            self._go_training_gpu_busy = busy

    @property
    def go_training_gpu_busy(self) -> bool:
        with self._lock:
            return self._go_training_gpu_busy

    def _training_reserve_active(self) -> bool:
        """True when training handoff, non-RUNNING, or Go reports training on GPU."""
        with self._lock:
            return (
                self._training_handoff_active
                or self._state != InferenceState.RUNNING
                or self._go_training_gpu_busy
            )

    def check_admit(
        self,
        *,
        waiting: int,
        running: int,
        gpu_free_bytes: int | None = None,
        priority: InferencePriority = InferencePriority.NORMAL,
        runtime_oldest_fifo: int = 0,
        skip_generic_vram_gate: bool = False,
    ) -> None:
        """Raise AdmissionRejected when inference should not accept more work."""
        with self._lock:
            if self._state != InferenceState.RUNNING:
                raise AdmissionRejected("inference paused for training")
            backlog = waiting + running
            if backlog >= self._max_waiting:
                raise AdmissionRejected(
                    f"runtime queue full ({backlog} >= {self._max_waiting})"
                )
            reserve_active = self._training_reserve_active()
        check_inference_first_admission(
            waiting=waiting,
            running=running,
            gpu_free_bytes=gpu_free_bytes,
            inference_paused=reserve_active,
            priority=priority,
            runtime_oldest_fifo=runtime_oldest_fifo,
            skip_generic_vram_gate=skip_generic_vram_gate,
        )

    def policy_snapshot(
        self,
        *,
        waiting: int,
        running: int,
        gpu_free_bytes: int | None = None,
        priority: InferencePriority = InferencePriority.NORMAL,
    ) -> dict[str, Any]:
        from runtime.gpu.admission import (
            admission_config_error,
            admission_vram_gate_enabled,
            defer_backlog_blocks_admission_now,
            ggml_backlog_blocks_admission_now,
            ggml_paused_blocks_admission_now,
            runtime_backlog_blocks_admission_now,
        )

        with self._lock:
            snap: dict[str, Any] = {
                "state": self._state.value,
                "accepts_new_loads": self._state == InferenceState.RUNNING,
                "max_queue": self._max_waiting,
                "waiting": waiting,
                "running": running,
                "backlog": waiting + running,
            }
            reserve_active = self._training_reserve_active()
            snap["training_handoff_active"] = self._training_handoff_active
            snap["go_training_gpu_busy"] = self._go_training_gpu_busy
        snap["inference_policy"] = inference_first_enabled()
        snap["backpressure"] = backpressure_snapshot(
            runtime_waiting=waiting, runtime_running=running
        )
        from runtime.gpu.inference_policy import inference_backpressure_blocks_low

        low_pressure = inference_backpressure_blocks_low(
            runtime_waiting=waiting, runtime_running=running
        )
        backlog_pressure = runtime_backlog_blocks_admission_now(
            waiting=waiting, running=running
        )
        defer_low = defer_backlog_blocks_admission_now()
        ggml_sched_low = ggml_backlog_blocks_admission_now()
        ggml_paused_low = ggml_paused_blocks_admission_now()
        # /health: true = signal on, NOT "all traffic blocked".
        # Only priority=low is rejected at enqueue or stalled when queue head is low.
        # runtime_backlog_pressure does not block normal chat.
        snap["gates_active"] = {
            "runtime_backlog_pressure": backlog_pressure,
            "defer_would_block_low": defer_low,
            "ggml_sched_would_block_low": ggml_sched_low,
            "ggml_paused_would_block_low": ggml_paused_low,
            "low_would_wait": low_pressure,
        }
        snap["gates_active_compat"] = {
            "runtime_backlog": backlog_pressure,
            "defer": defer_low,
            "ggml_backlog": ggml_sched_low,
            "ggml_paused": ggml_paused_low,
            "low_backpressure": low_pressure,
            "batch_backpressure": low_pressure,
        }
        snap["vram_gate"] = admission_vram_gate_enabled()
        snap["vram_gate_bypassed"] = vram_gate_bypassed(priority)
        snap["priority"] = priority.value
        snap["vram_min_free"] = min_free_vram_for_admission()
        snap["vram_min_free_configured"] = configured_min_free_vram_bytes()
        snap["vram_training_reserve_configured"] = (
            configured_training_vram_reserve_bytes()
        )
        snap["vram_config_error"] = admission_config_error()
        try:
            snap["vram_training_reserve"] = training_vram_reserve_bytes(
                inference_paused=reserve_active
            )
        except AdmissionMisconfigured as e:
            snap["vram_training_reserve"] = 0
            snap["vram_config_error"] = snap.get("vram_config_error") or str(e)
        if gpu_free_bytes is not None:
            snap["gpu_free_bytes"] = gpu_free_bytes
        return snap

    def training_handoff(self) -> None:
        """Unload inference GPU (training OOM bridge + operator “free VRAM”).

        Name is from training’s perspective, not “training beats inference” in product
        policy—inference-first means default UX favors chat/generate; this only runs when
        training or an operator must evict the runtime llama-server subprocess.
        """
        self.pause_inference()
        self.unload_all()
