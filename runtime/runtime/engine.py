"""Inference engine: PA scheduler + llama-server worker (Phase 2)."""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from runtime.config import RuntimeConfig
from runtime.logutil import get_logger
from runtime.gpu.admission import AdmissionMisconfigured, AdmissionRejected
from runtime.gpu.priority import InferencePriority, priority_from_options
from runtime.gpu.model_swap import ModelSwapGate
from runtime.gpu.mutex import InferenceGpuCoordinator, InferenceState
from runtime.gpu_vram import (
    active_vram_probe,
    check_gguf_vram_budget,
    describe_vram_estimate,
    gpu_vram_check_enabled,
    llama_vram_device_indices,
    nvidia_free_vram_by_device,
    vram_budget_health,
)
from runtime.host_memory import check_gguf_host_budget, format_bytes, read_host_memory
from runtime.kv.accounting import kv_scheduler_snapshot
from runtime.kv.block_pool import create_block_pool, kv_backend_health
from runtime.scheduler.loop import SchedulerLoop
from runtime.scheduler.scheduler import Request, RequestState, Scheduler
from runtime.worker.factory import (
    LlamaBackendKind,
    LlamaForwardWorker,
    create_llama_worker,
    llama_backend_source,
    resolve_llama_backend,
)
from runtime.worker.llama_server import LlamaServerError
from runtime.worker.sampler_options import sampler_options_from_dict


@dataclass
class GenerateResult:
    content: str
    request_id: str
    llama: dict[str, Any] = field(default_factory=dict)
    vram_num_ctx: dict[str, Any] | None = None
    kv_decode_steps: int | None = None


def _vram_calibration_for_health() -> dict[str, Any] | None:
    from runtime.vram_calibration import vram_calibration_health

    return vram_calibration_health()


def _vram_autotune_for_health() -> dict[str, Any]:
    from runtime.gpu_vram import vram_estimate_autotune_status

    return vram_estimate_autotune_status()


def _vram_num_ctx_policy_for_health() -> dict[str, Any]:
    from runtime.vram_suggest import vram_num_ctx_policy_health

    return vram_num_ctx_policy_health()


class InferenceEngine:
    """Owns block pools, scheduler, and optional llama-server subprocess."""

    _log = get_logger("engine")
    _HEALTH_CACHE_TTL_S = 0.4

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig.from_env()
        self._llama_backend_override: LlamaBackendKind | None = None
        if self.config.llama_model and self.config.llama_model.is_file():
            InferenceEngine._log.info(
                "runtime configured for model %s (gguf=%s)",
                self.config.llama_model.name,
                self.config.llama_model.resolve(),
            )
        self.coordinator = InferenceGpuCoordinator()
        self._model_swap = ModelSwapGate()
        self.pools = [
            create_block_pool(
                num_blocks=self.config.num_blocks,
                block_size=self.config.block_size,
                device_id=i,
            )
            for i in range(self.config.device_count)
        ]
        kv_pools = self.pools[: self.config.active_kv_pools()]
        self.scheduler = Scheduler.for_pools(kv_pools)
        llama_slots = self._effective_llama_parallel_slots()
        self.loop = SchedulerLoop(
            scheduler=self.scheduler,
            coordinator=self.coordinator,
            pools=self.pools,
            parallel_slots=llama_slots,
            assign_llama_slots=self._resolved_llama_backend()
            in (LlamaBackendKind.SUBPROCESS, LlamaBackendKind.INPROCESS),
        )
        self._loop_vram_check = self._vram_check_admitting
        self._server: LlamaForwardWorker | None = None
        self._loaded_vram_num_ctx: int | None = None
        self._health_cache: dict[str, Any] | None = None
        self._health_cache_at: float = 0.0
        self._health_cache_lock = threading.Lock()
        self._health_build_lock = threading.Lock()
        self._vocab_sessions: dict[str, Any] = {}
        self._vocab_lock = threading.RLock()
        if self.config.llama_model and self.config.llama_model.is_file():
            if self._llama_backend_enabled():
                self._server = self._create_llama_worker(self.config.llama_model)
                self.coordinator.set_unload_hook(self._stop_server)

    def _clear_vocab_sessions(self) -> None:
        with self._vocab_lock:
            for session in self._vocab_sessions.values():
                session.close()
            self._vocab_sessions.clear()

    def _stop_server(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None
        self._clear_vocab_sessions()
        self._loaded_vram_num_ctx = None
        self._model_swap.reset()

    _VOCAB_CACHE_MAX = 4

    def _new_vocab_session(self, resolved: Path) -> Any:
        """Vocab-only session for render tokenize (backend-aware)."""
        from runtime.worker.factory import LlamaBackendKind

        backend = self._resolved_llama_backend()
        if backend == LlamaBackendKind.LLAMA_CPP_PYTHON:
            from runtime.worker.llama_cpp_python import LlamaCppVocabSession

            return LlamaCppVocabSession(resolved, main_gpu=self.config.main_gpu)
        from runtime.worker.libllama_ctypes import LlamaVocabSession, resolve_libllama_path

        try:
            resolve_libllama_path(self.config.llama_cpp_lib, self.config.llama_cpp_root)
        except LlamaServerError as e:
            if backend == LlamaBackendKind.SUBPROCESS:
                try:
                    from runtime.worker.llama_cpp_python import LlamaCppVocabSession

                    return LlamaCppVocabSession(
                        resolved, main_gpu=self.config.main_gpu
                    )
                except LlamaServerError:
                    pass
            raise e
        return LlamaVocabSession(
            resolved,
            lib_path=self.config.llama_cpp_lib,
            cpp_root=self.config.llama_cpp_root,
            default_main_gpu=self.config.main_gpu,
        )

    def tokenize_gguf_text(
        self,
        gguf: Path,
        text: str,
        *,
        add_special: bool = True,
    ) -> list[int]:
        """Tokenize for Go /internal/render-chat (loaded worker or vocab-only cache).

        Why reuse loaded worker when paths match: avoids a second full/vocab load while
        the same GGUF is already in VRAM for inference. Why vocab cache otherwise: render
        may tokenize a different manifest path than LLAMA_MODEL without loading weights.
        """
        resolved = gguf.resolve()
        if not resolved.is_file():
            raise LlamaServerError(f"model not found: {resolved}")

        with self._model_swap.hold(resolved):
            srv = self._server
            if srv is not None and srv.is_running():
                if getattr(srv, "model", None) is not None:
                    try:
                        if srv.model.resolve() == resolved:
                            tokenize = getattr(srv, "tokenize_text", None)
                            if callable(tokenize):
                                return tokenize(text, add_special=add_special)
                    except OSError:
                        pass
                session = getattr(srv, "_session", None)
                if session is not None and session.model_path == resolved:
                    return session.tokenize_text(text, add_special=add_special)

        key = str(resolved)
        with self._vocab_lock:
            session = self._vocab_sessions.get(key)
            if session is None:
                if len(self._vocab_sessions) >= self._VOCAB_CACHE_MAX:
                    old_key = next(iter(self._vocab_sessions))
                    self._vocab_sessions.pop(old_key).close()
                session = self._new_vocab_session(resolved)
                self._vocab_sessions[key] = session
            return session.tokenize_text(text, add_special=add_special)

    def _resolved_llama_backend(self) -> LlamaBackendKind:
        if self._llama_backend_override is not None:
            return self._llama_backend_override
        return resolve_llama_backend(self.config)

    def _requested_llama_backend(self) -> LlamaBackendKind:
        return resolve_llama_backend(self.config)

    def _inprocess_fallback_enabled(self) -> bool:
        raw = os.environ.get("ZEROLLAMA_RUNTIME_INPROCESS_FALLBACK", "").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return False
        if raw in ("1", "true", "yes", "on", "subprocess"):
            return True
        if sys.platform != "darwin":
            return False
        bin_p = self.config.llama_server_bin
        return bin_p is not None and bin_p.is_file()

    def _start_server_with_fallback(self, extra_args: list[str]) -> None:
        if self._server is None:
            raise LlamaServerError("llama forward not configured")
        try:
            self._server.start(extra_args=extra_args)
        except LlamaServerError as exc:
            if (
                self._llama_backend_override is None
                and self._requested_llama_backend() == LlamaBackendKind.INPROCESS
                and self._inprocess_fallback_enabled()
            ):
                bin_p = self.config.llama_server_bin
                model = getattr(self._server, "model", None)
                if (
                    bin_p is not None
                    and bin_p.is_file()
                    and model is not None
                ):
                    self._log.warning(
                        "inprocess load failed; falling back to subprocess llama-server: %s",
                        exc,
                    )
                    self._server.stop()
                    self._llama_backend_override = LlamaBackendKind.SUBPROCESS
                    self._server = create_llama_worker(
                        kind=LlamaBackendKind.SUBPROCESS,
                        binary=bin_p,
                        model=model,
                        host=self.config.host,
                        port=self.config.port + 1,
                        lib_path=self.config.llama_cpp_lib,
                        cpp_root=self.config.llama_cpp_root,
                        main_gpu=self.config.main_gpu,
                        config=self.config,
                    )
                    self._server.start(extra_args=extra_args)
                    self.invalidate_health_cache()
                    return
            raise

    def _llama_backend_enabled(self) -> bool:
        try:
            backend = self._resolved_llama_backend()
        except LlamaServerError:
            return False
        if backend in (
            LlamaBackendKind.INPROCESS,
            LlamaBackendKind.LLAMA_CPP_PYTHON,
        ):
            return True
        return self.config.llama_server_bin is not None

    def _health_llama_backend(self) -> str:
        try:
            return self._resolved_llama_backend().value
        except LlamaServerError:
            raw = str(self.config.llama_backend or "subprocess").strip().lower()
            return raw or "subprocess"

    def _health_llama_cpp(self) -> dict[str, Any] | None:
        if self._resolved_llama_backend() != LlamaBackendKind.LLAMA_CPP_PYTHON:
            return None
        from runtime.worker.llama_cpp_python import llama_cpp_wheel_health

        worker = (
            self._server
            if self._server is not None and self._server.is_running()
            else None
        )
        return llama_cpp_wheel_health(worker)

    def _effective_llama_parallel_slots(self) -> int:
        """Match ``SlotAllocator`` / in-process ``n_seq_max`` to llama ``-np`` (argv over YAML)."""
        from runtime.kv.live_physical import effective_parallel_slots

        return effective_parallel_slots(
            self.config.llama_server_args(),
            default=self.config.llama_parallel_slots,
            backend=self._health_llama_backend(),
        )

    def _kv_live_physical_health(self) -> dict[str, Any]:
        from runtime.kv.live_physical import kv_live_physical_health

        return kv_live_physical_health(
            self.config.llama_server_args(),
            default=self.config.llama_parallel_slots,
            backend=self._health_llama_backend(),
        )

    def _health_inprocess_n_seq_max(self) -> int | None:
        """Shared-context sequence count when in-process multi-seq KV is active (>1)."""
        if self._resolved_llama_backend() != LlamaBackendKind.INPROCESS:
            return None
        slots = self._effective_llama_parallel_slots()
        if slots <= 1:
            return None
        srv = self._server
        if srv is None:
            return slots
        session = getattr(srv, "_session", None)
        if session is None:
            return slots
        n = getattr(session, "n_seq_max", None)
        return int(n) if n is not None else slots

    def _create_llama_worker(self, model: Path) -> LlamaForwardWorker:
        return create_llama_worker(
            kind=self._resolved_llama_backend(),
            binary=self.config.llama_server_bin,
            model=model,
            host=self.config.host,
            port=self.config.port + 1,
            lib_path=self.config.llama_cpp_lib,
            cpp_root=self.config.llama_cpp_root,
            main_gpu=self.config.main_gpu,
            config=self.config,
        )

    def _ensure_server(self) -> LlamaForwardWorker:
        if self._server is None:
            raise LlamaServerError(
                "llama forward not configured (LLAMA_MODEL + LLAMA_SERVER_BIN or "
                "ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess|llama-cpp-python)"
            )
        if not self._server.is_running():
            self._start_server_with_fallback(extra_args=self.config.llama_server_args())
        return self._server

    @staticmethod
    def _vram_options(
        num_ctx: int | None, options: dict | None
    ) -> dict | None:
        merged = dict(options) if options else {}
        merged.pop("priority", None)
        if num_ctx is not None:
            merged["num_ctx"] = num_ctx
        return merged or None

    def _llama_server_start_args(
        self,
        gguf: Path,
        *,
        num_ctx: int | None = None,
        options: dict | None = None,
    ) -> list[str]:
        """Config llama-server argv with request ``num_ctx`` when it differs from defaults."""
        from runtime.llama_args import with_llama_num_ctx

        base = self.config.llama_server_args()
        ctx = self._vram_num_ctx_for_load(gguf, num_ctx=num_ctx, options=options)
        if ctx is not None and ctx > 0:
            return with_llama_num_ctx(base, ctx)
        return base

    def _vram_llama_kwargs(self) -> dict[str, Any]:
        """Flags passed to llama-server (for VRAM estimate parity with subprocess)."""
        from runtime.speculative import resolve_method

        kw: dict[str, Any] = {
            "llama_args": self.config.llama_server_args(),
            "parallel_slots_default": self.config.llama_parallel_slots,
            "llama_backend": self._health_llama_backend(),
            "n_gpu_layers_default": -1,
        }
        spec = self.config.speculative
        if (
            spec.draft_model is not None
            and spec.draft_model.is_file()
            and resolve_method(spec.method).startswith("draft")
        ):
            kw["draft_gguf"] = spec.draft_model
            kw["draft_n_gpu_layers"] = spec.draft_n_gpu_layers
        return kw

    def _vram_num_ctx_for_load(
        self,
        gguf: Path,
        *,
        num_ctx: int | None = None,
        options: dict | None = None,
    ) -> int | None:
        from runtime.gpu_vram import resolve_vram_num_ctx

        return resolve_vram_num_ctx(
            options,
            gguf,
            explicit=num_ctx,
            llama_args=self.config.llama_server_args(),
        )

    def _vram_num_ctx_for_request(self, req: Request, gguf: Path) -> int | None:
        return self._vram_num_ctx_for_load(
            gguf, num_ctx=req.num_ctx, options=req.vram_options
        )

    def _vram_precheck_skippable_for(
        self,
        resolved: Path,
        *,
        num_ctx: int | None,
        vram_options: dict | None,
    ) -> bool:
        """True when loaded weights match and context does not require a larger KV."""
        if self._server is None or not self._server.is_running():
            return False
        try:
            if self._server.model.resolve() != resolved:
                return False
        except OSError:
            return False
        needed = self._vram_num_ctx_for_load(
            resolved, num_ctx=num_ctx, options=vram_options
        )
        loaded = self._loaded_vram_num_ctx
        if needed is None or loaded is None:
            # Do not skip when ctx is unknown — precheck must still run host/GPU budget checks.
            return False
        return needed <= loaded

    def _vram_precheck_skippable(self, req: Request, resolved: Path) -> bool:
        return self._vram_precheck_skippable_for(
            resolved, num_ctx=req.num_ctx, vram_options=req.vram_options
        )

    def _vram_precheck_gguf(
        self,
        resolved: Path,
        *,
        num_ctx: int | None,
        vram_options: dict | None,
        skippable: bool,
        priority: InferencePriority,
    ) -> None:
        """Host + GPU budget check (enqueue and dequeue)."""
        if not resolved.is_file():
            return
        if skippable:
            return
        check_gguf_host_budget(resolved)
        if not gpu_vram_check_enabled():
            return
        resolved_ctx = self._vram_num_ctx_for_load(
            resolved, num_ctx=num_ctx, options=vram_options
        )
        check_gguf_vram_budget(
            resolved,
            main_gpu=self.config.main_gpu,
            tensor_parallel=self.config.tensor_parallel,
            device_count=self.config.device_count,
            num_ctx=resolved_ctx,
            options=vram_options,
            training_reserve_active=self.coordinator.training_reserve_active(),
            priority=priority,
            **self._vram_llama_kwargs(),
        )

    def _vram_precheck_enqueue(
        self,
        gguf: Path | None,
        *,
        num_ctx: int | None,
        vram_options: dict | None,
        priority: InferencePriority,
    ) -> None:
        """Fail before queueing when the GGUF cannot fit host/GPU budget.

        Why before add_request: a full queue of KV reservations is expensive to unwind;
        operators get a clear 503 instead of stuck work at dequeue.
        """
        if gguf is None:
            return
        try:
            resolved = gguf.resolve()
        except OSError:
            return
        skippable = self._vram_precheck_skippable_for(
            resolved, num_ctx=num_ctx, vram_options=vram_options
        )
        self._vram_precheck_gguf(
            resolved,
            num_ctx=num_ctx,
            vram_options=vram_options,
            skippable=skippable,
            priority=priority,
        )

    def ensure_gguf_loaded(
        self,
        gguf: Path | None,
        *,
        num_ctx: int | None = None,
        options: dict | None = None,
    ) -> LlamaForwardWorker:
        """Load or swap GGUF weights when a manifest path is provided."""
        with self._model_swap.hold(gguf):
            return self._ensure_gguf_loaded_unlocked(
                gguf, num_ctx=num_ctx, options=self._vram_options(num_ctx, options)
            )

    def _ensure_gguf_loaded_unlocked(
        self,
        gguf: Path | None,
        *,
        num_ctx: int | None = None,
        options: dict | None = None,
    ) -> LlamaForwardWorker:
        if gguf is None:
            return self._ensure_server()
        resolved = gguf.resolve()
        if not resolved.is_file():
            raise LlamaServerError(f"model not found: {resolved}")
        check_gguf_host_budget(resolved)
        backend = self._resolved_llama_backend()
        bin_path = self.config.llama_server_bin
        if backend == LlamaBackendKind.SUBPROCESS and (
            bin_path is None or not bin_path.is_file()
        ):
            raise LlamaServerError(
                "llama-server not configured (set LLAMA_SERVER_BIN or "
                "ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess|llama-cpp-python)"
            )
        current = (
            self._server.model.resolve()
            if self._server is not None
            else None
        )
        if self._server is None or current != resolved:
            if self._server is not None:
                self._stop_server()
            self.config.llama_model = resolved
            self._server = self._create_llama_worker(resolved)
            self.coordinator.set_unload_hook(self._stop_server)
            current = resolved
        proc_alive = self._server.is_running()
        needed_ctx = self._vram_num_ctx_for_load(
            resolved, num_ctx=num_ctx, options=options
        )
        loaded_ctx = self._loaded_vram_num_ctx
        ctx_grew = (
            proc_alive
            and current == resolved
            and needed_ctx is not None
            and loaded_ctx is not None
            and needed_ctx > loaded_ctx
        )
        if ctx_grew:
            vram_kw = self._vram_llama_kwargs()
            check_gguf_vram_budget(
                resolved,
                main_gpu=self.config.main_gpu,
                tensor_parallel=self.config.tensor_parallel,
                device_count=self.config.device_count,
                num_ctx=needed_ctx,
                options=options,
                training_reserve_active=self.coordinator.training_reserve_active(),
                priority=priority_from_options(options),
                **vram_kw,
            )
            self._stop_server()
            self.config.llama_model = resolved
            self._server = self._create_llama_worker(resolved)
            self.coordinator.set_unload_hook(self._stop_server)
            proc_alive = False
        elif proc_alive and current == resolved:
            return self._server
        if not self._server.is_running():
            from runtime.gpu_vram import (
                active_vram_probe,
                estimate_gguf_vram_bytes,
                nvidia_free_vram_bytes,
            )
            from runtime.vram_calibration import maybe_record_vram_after_load

            vram_kw = self._vram_llama_kwargs()
            est_kw: dict[str, Any] = {
                "tensor_parallel": self.config.tensor_parallel,
                "num_ctx": needed_ctx,
                "options": options,
                **vram_kw,
            }
            free_before = nvidia_free_vram_bytes(self.config.main_gpu, fresh=True)
            estimated_raw = estimate_gguf_vram_bytes(
                resolved, _apply_estimate_factor=False, **est_kw
            )
            estimated_effective = estimate_gguf_vram_bytes(resolved, **est_kw)
            probe_at_load = active_vram_probe()
            # After any unload/swap so free VRAM reflects the target load.
            check_gguf_vram_budget(
                resolved,
                main_gpu=self.config.main_gpu,
                tensor_parallel=self.config.tensor_parallel,
                device_count=self.config.device_count,
                num_ctx=needed_ctx,
                options=options,
                training_reserve_active=self.coordinator.training_reserve_active(),
                priority=priority_from_options(options),
                **vram_kw,
            )
            self._start_server_with_fallback(
                extra_args=self._llama_server_start_args(
                    resolved, num_ctx=needed_ctx, options=options
                )
            )
            self._loaded_vram_num_ctx = needed_ctx
            maybe_record_vram_after_load(
                model_path=resolved,
                device_index=self.config.main_gpu,
                free_before=free_before,
                estimated_raw_bytes=estimated_raw,
                estimated_effective_bytes=estimated_effective,
                tensor_parallel=self.config.tensor_parallel,
                probe=probe_at_load,
            )
        return self._server

    def vram_estimate_and_budget(
        self,
        gguf: Path,
        *,
        num_ctx: int | None = None,
        options: dict | None = None,
        prefer_loaded_num_ctx: bool = False,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """VRAM estimate + budget for a GGUF (shared by /health and /internal/vram-estimate)."""
        from runtime.gpu_vram import resolve_vram_num_ctx

        vram_kw = self._vram_llama_kwargs()
        if prefer_loaded_num_ctx and self._loaded_vram_num_ctx is not None:
            ctx = self._loaded_vram_num_ctx
        else:
            ctx = resolve_vram_num_ctx(
                options,
                gguf,
                explicit=num_ctx,
                llama_args=vram_kw.get("llama_args"),
            )
        try:
            resolved = gguf.resolve()
        except OSError:
            resolved = gguf
        try:
            est = describe_vram_estimate(
                resolved,
                num_ctx=ctx,
                options=options,
                tensor_parallel=self.config.tensor_parallel,
                **vram_kw,
            )
        except OSError:
            return None, None
        from runtime.vram_suggest import build_suggest_profile

        budget = vram_budget_health(
            est,
            gpu_free_bottleneck=self._gpu_free_for_admission(),
            inference_paused_for_reserve=self.coordinator.training_reserve_active(),
            suggest_profile=build_suggest_profile(
                est,
                tensor_parallel=self.config.tensor_parallel,
                options=options,
                **vram_kw,
            ),
        )
        from runtime.host_memory import host_ram_budget_snapshot

        host_ram = host_ram_budget_snapshot(resolved)
        if host_ram is not None:
            if budget is None:
                budget = {}
            budget["host_ram"] = host_ram
        return est, budget

    def _health_gguf_path(self) -> Path | None:
        """Loaded model path, else config / model-swap gate (pre-load estimate)."""
        if self._server is not None:
            try:
                resolved = self._server.model.resolve()
                if resolved.is_file():
                    return resolved
            except OSError:
                pass
        if self.config.llama_model:
            p = Path(self.config.llama_model)
            if p.is_file():
                return p
        loaded = self._model_swap.stats().get("loaded_gguf")
        if loaded and loaded != "__default__":
            p = Path(loaded)
            if p.is_file():
                return p
        return None

    def invalidate_health_cache(self) -> None:
        """Drop cached /health after inference or training GPU state changes."""
        with self._health_cache_lock:
            self._health_cache = None
            self._health_cache_at = 0.0

    def health(self) -> dict[str, Any]:
        """Runtime /health snapshot (cached briefly; single-flight rebuild under load)."""
        now = time.monotonic()
        with self._health_cache_lock:
            cached = self._health_cache
            if cached is not None and (now - self._health_cache_at) < self._HEALTH_CACHE_TTL_S:
                return dict(cached)

        with self._health_build_lock:
            now = time.monotonic()
            with self._health_cache_lock:
                cached = self._health_cache
                if cached is not None and (now - self._health_cache_at) < self._HEALTH_CACHE_TTL_S:
                    return dict(cached)
            body = self._health_body()
            with self._health_cache_lock:
                self._health_cache = body
                self._health_cache_at = time.monotonic()
            return dict(body)

    def _health_body(self) -> dict[str, Any]:
        waiting = len(self.scheduler.waiting)
        running = len(self.scheduler.running)
        mem = read_host_memory()
        host_mem: dict[str, Any] | None = None
        if mem is not None:
            host_mem = {
                "available": format_bytes(mem.available_bytes),
                "swap_free": format_bytes(mem.swap_free_bytes),
                "load_budget": format_bytes(mem.load_budget_bytes),
            }
        dev_indices = llama_vram_device_indices(
            self.config.main_gpu,
            self.config.tensor_parallel,
            self.config.device_count,
        )
        free_by_dev = nvidia_free_vram_by_device(dev_indices)
        gpu_free_bottleneck = (
            min(free_by_dev.values()) if free_by_dev else None
        )
        gpu_mem: list[dict[str, Any]] | None = None
        if free_by_dev:
            probe = active_vram_probe()
            gpu_mem = [
                {
                    "device": idx,
                    "free": format_bytes(free_by_dev[idx]),
                    **({"probe": probe} if probe and idx == dev_indices[0] else {}),
                }
                for idx in dev_indices
                if idx in free_by_dev
            ]
        vram_est: dict[str, Any] | None = None
        vram_budget: dict[str, Any] | None = None
        model_path = self._health_gguf_path()
        if model_path is not None:
            vram_est, vram_budget = self.vram_estimate_and_budget(
                model_path,
                prefer_loaded_num_ctx=True,
            )
        from runtime.go_coordination import go_coordination_health

        go_coord = go_coordination_health()
        admission = self.coordinator.policy_snapshot(
            waiting=waiting,
            running=running,
            gpu_free_bytes=gpu_free_bottleneck,
        )
        if vram_budget:
            admission["vram_load_fits"] = vram_budget.get("fits")
            admission["vram_load_fits_margin"] = vram_budget.get("fits_with_margin")
            if "admission_fits" in vram_budget:
                admission["vram_admission_fits"] = vram_budget["admission_fits"]
        from runtime.gpu_vram import (
            shared_interpreter_embedded,
            vram_probe_effective,
            vram_probe_mode,
        )

        from runtime.autoconfig import autoconfig_health

        llama_cpp_health = self._health_llama_cpp()
        embed_boot = os.environ.get("ZEROLLAMA_RUNTIME_EMBED_BOOT", "").strip()
        body: dict[str, Any] = {
            "status": "ok",
            "autoconfig": autoconfig_health(main_gpu=self.config.main_gpu),
            "inference_state": self.coordinator.state.value,
            "accepts_new_loads": self.coordinator.accepts_new_loads(),
            "admission": admission,
            "go_coordination": go_coord if go_coord else None,
            "model_swap": self._model_swap.stats(),
            "shared_interpreter": shared_interpreter_embedded(),
            "vram_probe_mode": vram_probe_mode(),
            "vram_probe_effective": vram_probe_effective(),
            "host_memory": host_mem,
            "gpu_memory": gpu_mem,
            "vram_estimate": vram_est,
            "vram_budget": vram_budget,
            "vram_autotune": _vram_autotune_for_health(),
            "vram_num_ctx_policy": _vram_num_ctx_policy_for_health(),
            "vram_calibration": _vram_calibration_for_health(),
            "waiting": waiting,
            "running": running,
            "fifo_runtime_oldest": self.scheduler.oldest_waiting_fifo_seq(),
            "kv": kv_backend_health(),
            "kv_bind": self._kv_bind_health(),
            "kv_physical": self._kv_physical_health(),
            "kv_scheduler_tick": self._kv_scheduler_tick_health(),
            "kv_native_scheduler_tick": self.loop.last_scheduler_tick,
            "kv_physical_recent": self._kv_physical_recent_health(),
            "kv_decode_steps": self._kv_decode_steps_health(),
            "kv_native_stats": self._kv_native_stats_health(),
            "kv_forward_plans": self._kv_forward_plans_health(),
            "kv_page_bind": self._kv_page_bind_health(),
            "kv_live_physical": self._kv_live_physical_health(),
            "kv_scheduler": kv_scheduler_snapshot(
                self.scheduler,
                self.pools,
                block_size=self.config.block_size,
                slot_snapshot=(
                    self.loop.slot_snapshot()
                    if self.loop.assign_llama_slots
                    else None
                ),
                llama_parallel_slots=self._effective_llama_parallel_slots(),
            ),
            "kv_pools": [
                {
                    "device_id": p.device_id,
                    "num_blocks": p.num_blocks,
                    "block_size": p.block_size,
                    "free": p.num_free,
                    "utilization": round(p.utilization, 4),
                }
                for p in self.pools
            ],
            "llama_server": self._server is not None and self._server.is_running(),
            "llama_backend": self._health_llama_backend(),
            "llama_backend_source": llama_backend_source(self.config),
            "llama_backend_requested": self._requested_llama_backend().value,
            "llama_backend_fallback": self._llama_backend_override is not None,
            "kv_inprocess_n_seq_max": self._health_inprocess_n_seq_max(),
            "loaded_vram_num_ctx": self._loaded_vram_num_ctx,
            "llama_model": str(self.config.llama_model)
            if self.config.llama_model
            else None,
            "tensor_parallel": self.config.tensor_parallel,
            "split_mode": self.config.split_mode,
            "speculative_method": self.config.speculative.method,
            "llama_spec_type": self.config.speculative.method,
        }
        if embed_boot:
            body["embed_boot"] = embed_boot
        if llama_cpp_health is not None:
            body["llama_cpp"] = llama_cpp_health
        return body

    def training_handoff(self) -> InferenceState:
        self.coordinator.training_handoff()
        self.invalidate_health_cache()
        return self.coordinator.state

    def resume_inference(self) -> InferenceState:
        """Allow new inference loads again (e.g. after training-handoff or manual GPU release)."""
        self.coordinator.resume_inference()
        self.invalidate_health_cache()
        return self.coordinator.state

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _gpu_free_for_admission(self) -> int | None:
        """Bottleneck free VRAM across devices used by this runtime config."""
        indices = llama_vram_device_indices(
            self.config.main_gpu,
            self.config.tensor_parallel,
            self.config.device_count,
        )
        free_by_dev = nvidia_free_vram_by_device(indices)
        if not free_by_dev:
            return None
        return min(free_by_dev.values())

    def _effective_vram_free_for_suggest(self) -> int | None:
        from runtime.gpu_vram import effective_vram_free_after_reserve

        return effective_vram_free_after_reserve(
            main_gpu=self.config.main_gpu,
            tensor_parallel=self.config.tensor_parallel,
            device_count=self.config.device_count,
            inference_paused_for_reserve=self.coordinator.training_reserve_active(),
        )

    def _cap_num_ctx_for_vram(
        self,
        gguf: Path,
        num_ctx: int | None,
        *,
        options: dict | None = None,
        priority: InferencePriority | None = None,
    ) -> tuple[int | None, dict[str, Any]]:
        if num_ctx is None or num_ctx <= 0:
            return num_ctx, {}
        eff = self._effective_vram_free_for_suggest()
        if eff is None:
            return num_ctx, {}
        from runtime.vram_suggest import build_suggest_profile, cap_num_ctx_for_vram

        profile = build_suggest_profile(
            None,
            tensor_parallel=self.config.tensor_parallel,
            options=options,
            **self._vram_llama_kwargs(),
        )
        kw = {k: v for k, v in profile.items() if k != "options"}
        return cap_num_ctx_for_vram(
            gguf,
            num_ctx,
            eff,
            options=options,
            priority=priority,
            **kw,
        )

    def _check_admit_policy(
        self,
        options: dict | None = None,
        *,
        priority: InferencePriority | None = None,
        extra_running: int = 0,
        extra_waiting: int = 0,
        skip_generic_vram_gate: bool = False,
    ) -> InferencePriority:
        pri = priority if priority is not None else priority_from_options(options)
        waiting = len(self.scheduler.waiting) + extra_waiting
        running = len(self.scheduler.running) + extra_running
        try:
            self.coordinator.check_admit(
                waiting=waiting,
                running=running,
                gpu_free_bytes=self._gpu_free_for_admission(),
                priority=pri,
                runtime_oldest_fifo=self.scheduler.oldest_waiting_fifo_seq(),
                skip_generic_vram_gate=skip_generic_vram_gate,
            )
        except AdmissionMisconfigured as e:
            raise LlamaServerError(str(e)) from e
        except AdmissionRejected as e:
            raise LlamaServerError(str(e)) from e
        return pri

    def _vram_check_admitting(self, req: Request) -> None:
        """Re-check inference policy and GGUF VRAM budget when dequeuing.

        Why again at dequeue: VRAM can drop while requests waited (training, ggml load).
        """
        self.coordinator.check_admit(
            waiting=len(self.scheduler.waiting),
            running=len(self.scheduler.running) + 1,
            gpu_free_bytes=self._gpu_free_for_admission(),
            priority=req.priority,
            runtime_oldest_fifo=self.scheduler.oldest_waiting_fifo_seq(),
            skip_generic_vram_gate=req.gguf is not None,
        )
        self._vram_precheck_load(req)

    def _vram_precheck_load(self, req: Request) -> None:
        """Fail before load when host RAM or GPU VRAM budget is insufficient."""
        gguf = req.gguf
        if gguf is None and self.config.llama_model:
            gguf = Path(self.config.llama_model)
        if gguf is None:
            return
        try:
            resolved = gguf.resolve()
        except OSError:
            return
        if not resolved.is_file():
            return
        self._vram_precheck_gguf(
            resolved,
            num_ctx=req.num_ctx,
            vram_options=req.vram_options,
            skippable=self._vram_precheck_skippable(req, resolved),
            priority=req.priority,
        )

    def _cancel_batch_pending(
        self, pending: list[tuple[str, Request]]
    ) -> None:
        """Drop queued batch work and any requests left in running after tick failure."""
        for _, req in pending:
            self.scheduler.cancel_waiting(req)
        for req in list(self.scheduler.running):
            self.loop.complete(req)

    def _id_slot_for_request(self, req: Request) -> int:
        """Forward ``kv_slot`` to subprocess ``id_slot`` or in-process ``seq_id``."""
        if req.kv_slot is not None and req.kv_slot >= 0:
            return req.kv_slot
        return -1

    def _assert_kv_bind(self, req: Request, *, at: str) -> None:
        from runtime.kv.bind import assert_kv_capacity

        assert_kv_capacity(req, block_size=self.config.block_size, at=at)

    def _inprocess_session_for_health(self) -> Any | None:
        """Loaded in-process ``LlamaLoadedSession`` (weights in VRAM), if any."""
        if self._resolved_llama_backend() != LlamaBackendKind.INPROCESS:
            return None
        srv = self._server
        if srv is None or not srv.is_running():
            return None
        return getattr(srv, "_session", None)

    def _inprocess_ctx_for_health(self) -> tuple[Any, Any] | None:
        """``(lib, ctx)`` when multi-seq shared context is active."""
        session = self._inprocess_session_for_health()
        if session is None:
            return None
        ctx = getattr(session, "_ctx", None)
        if ctx is None:
            return None
        lib = getattr(session, "_lib", None)
        if lib is None:
            return None
        return lib, ctx

    def _kv_bind_health(self) -> dict[str, Any]:
        from runtime.kv.bind import kv_bind_health
        from runtime.kv.physical import kv_bind_physical_level

        backend = self._health_llama_backend()
        inprocess_loaded = self._inprocess_session_for_health() is not None
        return kv_bind_health(
            llama_backend=backend,
            assign_llama_slots=self.loop.assign_llama_slots,
            parallel_slots=self._effective_llama_parallel_slots(),
            physical_bind_level=kv_bind_physical_level(
                backend, inprocess_weights_loaded=inprocess_loaded
            ),
        )

    def _kv_physical_health(self) -> dict[str, Any] | None:
        from runtime.kv.physical import (
            kv_physical_health,
            kv_physical_health_pa_only,
        )

        session = self._inprocess_session_for_health()
        if session is None:
            return None
        running = list(self.scheduler.running)
        block_size = self.config.block_size
        pair = self._inprocess_ctx_for_health()
        if pair is not None:
            lib, ctx = pair
            return kv_physical_health(
                inprocess_ctx=ctx,
                lib=lib,
                running=running,
                block_size=block_size,
            )
        return kv_physical_health_pa_only(running, block_size=block_size)

    def _kv_scheduler_tick_health(self) -> dict[str, int | str]:
        from runtime.kv.native_tick import scheduler_tick_health

        return scheduler_tick_health(self.loop.last_scheduler_tick)

    def _kv_physical_recent_health(self) -> list[dict[str, Any]]:
        from runtime.kv.physical import recent_alignments

        return recent_alignments()

    def _kv_decode_steps_health(self) -> dict[str, int | str | bool] | None:
        from runtime.kv.native_decode import decode_steps_health

        return decode_steps_health(llama_backend=self._health_llama_backend())

    def _kv_native_stats_health(self) -> dict[str, int] | None:
        from runtime.kv.native_stats import native_kv_stats

        return native_kv_stats()

    def _kv_forward_plans_health(self) -> list[dict[str, Any]]:
        from runtime.kv.forward_plan import kv_forward_plans_for_requests

        reqs = list(self.scheduler.waiting) + list(self.scheduler.running)
        return kv_forward_plans_for_requests(
            reqs, block_size=self.config.block_size
        )

    def _kv_page_bind_health(self) -> dict[str, Any]:
        from runtime.kv.backend import native_available
        from runtime.kv.page_bind import page_bind_health

        return page_bind_health(native_ext_available=native_available())

    def kv_snapshot(self) -> dict[str, Any]:
        """KV-focused subset for ``/internal/kv-snapshot`` (loopback debug)."""
        return {
            "kv": self.health().get("kv"),
            "kv_bind": self._kv_bind_health(),
            "kv_scheduler": self.health().get("kv_scheduler"),
            "kv_physical": self._kv_physical_health(),
            "kv_physical_recent": self._kv_physical_recent_health(),
            "kv_scheduler_tick": self._kv_scheduler_tick_health(),
            "kv_decode_steps": self._kv_decode_steps_health(),
            "kv_native_stats": self._kv_native_stats_health(),
            "kv_forward_plans": self._kv_forward_plans_health(),
            "kv_page_bind": self._kv_page_bind_health(),
            "kv_live_physical": self._kv_live_physical_health(),
        }

    def _kv_decode_steps_before(self) -> int | None:
        from runtime.kv.native_decode import current_decode_steps, decode_hook_enabled

        if not decode_hook_enabled():
            return None
        if self._resolved_llama_backend() != LlamaBackendKind.INPROCESS:
            return None
        return current_decode_steps()

    def _kv_decode_steps_after(self, before: int | None) -> int | None:
        if before is None:
            return None
        from runtime.kv.native_decode import current_decode_steps, decode_steps_delta

        return decode_steps_delta(before, current_decode_steps())

    def _prompt_tokens_for_admit(self, prompt: str, gguf: Path | None) -> list[int]:
        """Token ids for KV reserve; real tokenize when GGUF path is known."""
        admit = gguf
        if admit is None and self.config.llama_model:
            admit = Path(self.config.llama_model)
        if admit is not None and admit.is_file():
            try:
                ids = self.tokenize_gguf_text(admit, prompt, add_special=True)
                if ids:
                    return ids
            except Exception:
                pass
        n = max(1, (len(prompt) + 3) // 4)
        return [0] * n

    def _admit_one(
        self,
        prompt: str,
        n_predict: int,
        *,
        gguf: Path | None = None,
        num_ctx: int | None = None,
        options: dict | None = None,
    ) -> Request:
        admit_gguf = gguf
        if admit_gguf is None and self.config.llama_model:
            admit_gguf = Path(self.config.llama_model)
        priority = self._check_admit_policy(
            options, skip_generic_vram_gate=admit_gguf is not None
        )
        if admit_gguf is not None:
            resolved_ctx, clamp_meta = self.resolve_num_ctx_for_request(
                gguf,
                num_ctx=num_ctx,
                options=options,
                priority=priority,
            )
        else:
            resolved_ctx, clamp_meta = None, {}
        vram_opts = self._vram_options(resolved_ctx, options)
        self._vram_precheck_enqueue(
            admit_gguf,
            num_ctx=resolved_ctx,
            vram_options=vram_opts,
            priority=priority,
        )
        resolved_gguf: Path | None = None
        if gguf is not None:
            try:
                resolved_gguf = gguf.resolve()
            except OSError:
                resolved_gguf = gguf
        req = Request(
            request_id=uuid.uuid4().hex[:12],
            prompt_tokens=self._prompt_tokens_for_admit(prompt, gguf),
            max_tokens=n_predict,
            priority=priority,
            gguf=resolved_gguf,
            num_ctx=resolved_ctx,
            vram_options=vram_opts,
            vram_num_ctx_meta=clamp_meta or None,
        )
        self.scheduler.add_request(req)
        try:
            admitted = self.loop.tick(max_admit=1, vram_check=self._loop_vram_check)
        except LlamaServerError:
            self.scheduler.cancel_waiting(req)
            raise
        if not admitted:
            self.scheduler.cancel_waiting(req)
            raise LlamaServerError("could not admit request (KV pool or pause)")
        return admitted[0]

    def resolve_num_ctx_for_request(
        self,
        gguf: Path | None,
        *,
        num_ctx: int | None = None,
        options: dict | None = None,
        priority: InferencePriority | None = None,
    ) -> tuple[int | None, dict[str, Any]]:
        """Resolve and optionally VRAM-clamp ``num_ctx`` (same path as ``_admit_one``, no queue).

        Why a shared helper: tools render, enqueue precheck, queued ``Request.num_ctx``,
        and ``llama-server`` ``-c`` must agree. Previously HTTP handlers resolved ctx for
        Go render while load used a different (uncapped) value after optional clamp.
        """
        from runtime.gpu_vram import resolve_vram_num_ctx

        opts = options
        admit_gguf = gguf
        if admit_gguf is None and self.config.llama_model:
            admit_gguf = Path(self.config.llama_model)
        pri = priority if priority is not None else priority_from_options(opts)
        if admit_gguf is None:
            ctx = resolve_vram_num_ctx(
                opts,
                None,
                explicit=num_ctx,
                llama_args=self.config.llama_server_args(),
            )
            return ctx, {}
        resolved = resolve_vram_num_ctx(
            opts,
            admit_gguf,
            explicit=num_ctx,
            llama_args=self.config.llama_server_args(),
        )
        if resolved is None:
            return None, {}
        try:
            cap_gguf = admit_gguf.resolve()
        except OSError:
            cap_gguf = admit_gguf
        return self._cap_num_ctx_for_vram(
            cap_gguf, resolved, options=opts, priority=pri
        )

    def _api_vram_num_ctx_from_request(self, req: Request) -> dict[str, Any] | None:
        from runtime.vram_suggest import api_vram_num_ctx_meta

        meta = req.vram_num_ctx_meta or {}
        return api_vram_num_ctx_meta(meta, req.num_ctx)

    def generate(
        self,
        prompt: str,
        n_predict: int = 64,
        *,
        gguf: Path | None = None,
        num_ctx: int | None = None,
        options: dict | None = None,
    ) -> GenerateResult:
        active = self._admit_one(
            prompt, n_predict, gguf=gguf, num_ctx=num_ctx, options=options
        )
        self._assert_kv_bind(active, at="generate")
        vram_opts = active.vram_options or self._vram_options(active.num_ctx, options)
        decode_before = self._kv_decode_steps_before()
        try:
            with self._model_swap.hold(gguf):
                srv = self._ensure_gguf_loaded_unlocked(
                    gguf, num_ctx=active.num_ctx, options=vram_opts
                )
                from runtime.kv.bind import reserved_token_capacity

                raw = srv.completion(
                    prompt,
                    n_predict=n_predict,
                    id_slot=self._id_slot_for_request(active),
                    kv_token_budget=reserved_token_capacity(active),
                    kv_bind_req=active,
                    kv_block_size=self.config.block_size,
                    sampler=sampler_options_from_dict(options),
                )
                content = raw.get("content") or raw.get("response") or ""
                active.state = RequestState.DECODE
                return GenerateResult(
                    content=content,
                    request_id=active.request_id,
                    llama=raw,
                    vram_num_ctx=self._api_vram_num_ctx_from_request(active),
                    kv_decode_steps=self._kv_decode_steps_after(decode_before),
                )
        finally:
            self.loop.complete(active)

    def stream_generate(
        self,
        prompt: str,
        model: str,
        n_predict: int = 64,
        *,
        gguf: Path | None = None,
        num_ctx: int | None = None,
        options: dict | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield Ollama-shaped NDJSON objects for /api/generate streaming."""
        active = self._admit_one(
            prompt, n_predict, gguf=gguf, num_ctx=num_ctx, options=options
        )
        self._assert_kv_bind(active, at="stream_generate")
        vram_opts = active.vram_options or self._vram_options(active.num_ctx, options)
        vram_api = self._api_vram_num_ctx_from_request(active)
        decode_before = self._kv_decode_steps_before()
        created = self._utc_now()
        try:
            with self._model_swap.hold(gguf):
                srv = self._ensure_gguf_loaded_unlocked(
                    gguf, num_ctx=active.num_ctx, options=vram_opts
                )
                saw_stop = False
                first = True
                sampler = sampler_options_from_dict(options)
                from runtime.kv.bind import reserved_token_capacity

                for chunk in srv.completion_stream(
                    prompt,
                    n_predict=n_predict,
                    id_slot=self._id_slot_for_request(active),
                    kv_token_budget=reserved_token_capacity(active),
                    kv_bind_req=active,
                    kv_block_size=self.config.block_size,
                    sampler=sampler,
                ):
                    content = chunk.get("content") or chunk.get("response") or ""
                    stop = bool(chunk.get("stop"))
                    if stop:
                        saw_stop = True
                    out: dict[str, Any] = {
                        "model": model,
                        "created_at": created,
                        "response": content,
                        "done": stop,
                        "done_reason": "stop" if stop else None,
                    }
                    if stop:
                        kv_steps = self._kv_decode_steps_after(decode_before)
                        if kv_steps is not None:
                            out["kv_decode_steps"] = kv_steps
                    if first and vram_api:
                        out["vram_num_ctx"] = vram_api
                        first = False
                    yield out
                if not saw_stop:
                    tail: dict[str, Any] = {
                        "model": model,
                        "created_at": created,
                        "response": "",
                        "done": True,
                        "done_reason": "stop",
                    }
                    if first and vram_api:
                        tail["vram_num_ctx"] = vram_api
                    kv_steps = self._kv_decode_steps_after(decode_before)
                    if kv_steps is not None:
                        tail["kv_decode_steps"] = kv_steps
                    yield tail
        finally:
            self.loop.complete(active)

    def stream_chat(
        self,
        prompt: str,
        model: str,
        n_predict: int = 64,
        *,
        gguf: Path | None = None,
        num_ctx: int | None = None,
        options: dict | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield Ollama-shaped NDJSON objects for /api/chat streaming."""
        for chunk in self.stream_generate(
            prompt, model, n_predict, gguf=gguf, num_ctx=num_ctx, options=options
        ):
            content = chunk.get("response", "")
            done = bool(chunk.get("done"))
            out: dict[str, Any] = {
                "model": model,
                "created_at": chunk.get("created_at", self._utc_now()),
                "message": {"role": "assistant", "content": content},
                "done": done,
                "done_reason": chunk.get("done_reason") if done else None,
            }
            if chunk.get("vram_num_ctx"):
                out["vram_num_ctx"] = chunk["vram_num_ctx"]
            yield out

    def generate_batch(
        self,
        prompts: list[str],
        n_predict: int = 64,
        max_admit: int = 4,
        *,
        options: dict | None = None,
    ) -> list[GenerateResult]:
        """Admit up to ``max_admit`` requests in one tick; decode sequentially."""
        if not prompts:
            return []

        batch_opts = dict(options or {})
        if batch_opts.get("priority") is None:
            batch_opts["priority"] = "batch"
        priority = priority_from_options(batch_opts)
        from runtime.gpu_vram import resolve_vram_num_ctx
        from runtime.server.gguf_path import peek_gguf_path

        batch_gguf = peek_gguf_path(batch_opts)
        if batch_gguf is None and self.config.llama_model:
            batch_gguf = Path(self.config.llama_model)
        batch_num_ctx = (
            resolve_vram_num_ctx(
                batch_opts,
                batch_gguf,
                llama_args=self.config.llama_server_args(),
            )
            if batch_gguf
            else None
        )
        batch_clamp_meta: dict[str, Any] = {}
        if batch_gguf is not None and batch_num_ctx is not None:
            batch_num_ctx, batch_clamp_meta = self._cap_num_ctx_for_vram(
                batch_gguf.resolve(),
                batch_num_ctx,
                options=batch_opts,
                priority=priority,
            )
        vram_opts = self._vram_options(batch_num_ctx, batch_opts)
        resolved_batch_gguf: Path | None = None
        if batch_gguf is not None:
            try:
                resolved_batch_gguf = batch_gguf.resolve()
            except OSError:
                resolved_batch_gguf = batch_gguf
        self._check_admit_policy(
            batch_opts,
            priority=priority,
            extra_waiting=len(prompts),
            skip_generic_vram_gate=batch_gguf is not None,
        )
        self._vram_precheck_enqueue(
            batch_gguf,
            num_ctx=batch_num_ctx,
            vram_options=vram_opts,
            priority=priority,
        )
        pending: list[tuple[str, Request]] = []
        for prompt in prompts:
            req_id = uuid.uuid4().hex[:12]
            req = Request(
                request_id=req_id,
                prompt_tokens=self._prompt_tokens_for_admit(prompt, batch_gguf),
                max_tokens=n_predict,
                priority=priority,
                gguf=resolved_batch_gguf,
                num_ctx=batch_num_ctx,
                vram_options=vram_opts,
            )
            self.scheduler.add_request(req)
            pending.append((prompt, req))

        try:
            admitted = self.loop.tick(
                max_admit=min(max_admit, len(prompts)),
                vram_check=self._loop_vram_check,
            )
        except LlamaServerError:
            self._cancel_batch_pending(pending)
            raise
        admitted_ids = {a.request_id for a in admitted}
        for _, req in pending:
            if req.request_id not in admitted_ids:
                self.scheduler.cancel_waiting(req)
        if not admitted:
            raise LlamaServerError("could not admit batch (KV pool or pause)")

        by_id = {req.request_id: prompt for prompt, req in pending}
        gguf = batch_gguf
        prompts_ordered = [by_id[a.request_id] for a in admitted]
        results: list[GenerateResult] = []
        for active in admitted:
            self._assert_kv_bind(active, at="generate_batch")
        decode_before = self._kv_decode_steps_before()
        try:
            with self._model_swap.hold(gguf):
                srv = self._ensure_gguf_loaded_unlocked(
                    gguf, num_ctx=batch_num_ctx, options=vram_opts
                )
                from runtime.kv.bind import reserved_token_capacity

                raws = srv.completions_parallel(
                    prompts_ordered,
                    n_predict=n_predict,
                    id_slots=[self._id_slot_for_request(a) for a in admitted],
                    kv_token_budgets=[
                        reserved_token_capacity(a) for a in admitted
                    ],
                    kv_bind_reqs=admitted,
                    kv_block_size=self.config.block_size,
                    sampler=sampler_options_from_dict(batch_opts),
                )
                from runtime.vram_suggest import api_vram_num_ctx_meta

                batch_vram = api_vram_num_ctx_meta(batch_clamp_meta, batch_num_ctx)
                batch_kv_decode = (
                    self._kv_decode_steps_after(decode_before)
                    if len(admitted) == 1
                    else None
                )
                for active, raw in zip(admitted, raws, strict=True):
                    content = raw.get("content") or raw.get("response") or ""
                    active.state = RequestState.DECODE
                    results.append(
                        GenerateResult(
                            content=content,
                            request_id=active.request_id,
                            llama=raw,
                            vram_num_ctx=batch_vram,
                            kv_decode_steps=batch_kv_decode,
                        )
                    )
        finally:
            for active in admitted:
                self.loop.complete(active)
        return results
