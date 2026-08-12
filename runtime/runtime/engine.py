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
from runtime.llama_timings import (
    detect_context_overflow,
    merge_cache_tier_details,
    metrics_from_llama_chunk,
)
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


def _c_environ_get(name: str) -> str:
    """Read a process env var from libc getenv (bypasses Python os.environ cache)."""
    try:
        import ctypes

        getenv = ctypes.CDLL(None).getenv
        getenv.argtypes = [ctypes.c_char_p]
        getenv.restype = ctypes.c_char_p
        raw = getenv(name.encode("utf-8"))
        if not raw:
            return ""
        return raw.decode("utf-8", "replace").strip()
    except Exception:
        return ""


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
        # v48/v49/v50: donor-buffer overlay bind (opt-in, ZEROLLAMA_KV_OVERLAY_BIND=1).
        # v48 consumes CPU-host KV buft groups; v49 additionally consumes Metal
        # device buft groups. Both share this same donor id / register-unregister API.
        # v50 auto-wires a host donor on in-process load when AUTO is not disabled;
        # manual register_kv_overlay_donor() still works (sets id before start).
        # WHY buffer keepalive: ctypes donor memory must outlive the consuming
        # llama_context — held on session (auto) or here (manual).
        self._kv_overlay_donor_id: int | None = None
        self._kv_overlay_donor_keepalive: Any | None = None
        self._kv_overlay_donor_ptr: int | None = None
        self._kv_overlay_donor_size: int | None = None
        from runtime.env import configure_runtime_env
        from runtime.kv.live_physical import effective_parallel_slots

        configure_runtime_env(
            llama_backend=self.config.llama_backend,
            n_parallel=effective_parallel_slots(
                self.config.llama_server_args(),
                default=self.config.llama_parallel_slots,
                backend=str(self.config.llama_backend or "subprocess"),
            ),
        )
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
        from runtime.kv.auto_batch import AutoBatchCoordinator
        from runtime.kv.stream_auto_batch import StreamAutoBatchCoordinator

        self._auto_batch = AutoBatchCoordinator(self)
        self._stream_auto_batch = StreamAutoBatchCoordinator(self)
        from runtime.subprocess_slot_state import SubprocessSlotState

        self._subprocess_slots = SubprocessSlotState()
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
        self._subprocess_slots.clear()
        from runtime.decode_graph_policy import bump_all_decode_graph_epochs

        bump_all_decode_graph_epochs(reason="server_stop")
        # v48: unregister the KV overlay donor buffer AFTER the server/session
        # (and thus its llama_context) has been fully stopped above — freeing
        # or reusing donor memory while a context is alive is undefined
        # behavior, same contract as any externally-owned ggml buffer.
        self.unregister_kv_overlay_donor()
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
            self._sync_kv_overlay_donor_from_session()
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
                    self._sync_kv_overlay_donor_from_session()
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
            n_gpu_layers=self.config.n_gpu_layers_default,
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
        """Config llama-server argv with request ``num_ctx`` when it differs from defaults.

        Appends ``--slot-save-path`` via cache_bridge when L3 is enabled. WHY after
        profile/L1 merge: slot dir must reflect final cache-type-k/v and draft model.
        """
        from runtime.llama_args import with_llama_kv_unified, with_llama_num_ctx
        from runtime.cache_bridge import llama_server_cache_argv
        from runtime.env import assert_kv_unified_sizing, kv_unified_enabled
        from runtime.speculative import resolve_method

        base = self.config.llama_server_args()
        ctx = self._vram_num_ctx_for_load(gguf, num_ctx=num_ctx, options=options)
        if ctx is not None and ctx > 0:
            base = with_llama_num_ctx(base, ctx)
        draft: Path | None = None
        spec = self.config.speculative
        if (
            spec.draft_model is not None
            and spec.draft_model.is_file()
            and resolve_method(spec.method).startswith("draft")
        ):
            draft = spec.draft_model
        if "--slot-save-path" not in base:
            base = base + llama_server_cache_argv(
                gguf,
                base,
                draft_model=draft,
                spec_method=spec.method,
                num_ctx=ctx,
            )
        # v53: subprocess parity with in-process v52 unified KV (Radix metadata seq_cp).
        unified = kv_unified_enabled()
        base = with_llama_kv_unified(base, unified)
        # v56: opt-in fail-closed when unified n_ctx is below soft floor.
        if unified:
            from runtime.llama_args import resolve_parallel_slots

            assert_kv_unified_sizing(
                n_ctx=ctx,
                n_parallel=resolve_parallel_slots(
                    base, default=self.config.llama_parallel_slots
                ),
            )
        return base

    def _prefix_cache_policy(
        self,
        gguf: Path | None = None,
        num_ctx: int | None = None,
        *,
        cache_level: str | None = None,
    ):
        from runtime.cache_bridge import apply_cache_level_to_policy
        from runtime.prefix_cache_policy import resolve_prefix_cache_policy

        path = gguf
        if path is None and self.config.llama_model:
            path = Path(self.config.llama_model)
        resolved_ctx = num_ctx if num_ctx is not None else self._loaded_vram_num_ctx
        file_path = path if path is not None and Path(path).is_file() else None
        policy = resolve_prefix_cache_policy(
            gguf=file_path,
            num_ctx=resolved_ctx,
            spec_method=self.config.speculative.method,
        )
        return apply_cache_level_to_policy(policy, cache_level)

    def _prefix_cache_request(self, req: Request, policy: Any) -> tuple[bool, int | None]:
        """Return ``(cache_prompt, resume_pos)`` with one seq_pos read per request.

        WHY subprocess graph invalidate before completion: when policy denies prefix
        resume (SWA block, draft drop-last-block fallback), KV shape changes in
        llama-server — ggml CUDA graphs keyed by topology must be cleared in that
        process via ``POST /cuda-graph/invalidate``, not only epoch-bumped here.
        """
        from runtime.cache_bridge import apply_cache_level_to_policy
        from runtime.prefix_cache_trace import record_prefix_cache_decision
        from runtime.prefix_cache_policy import (
            decode_graph_invalidation_reason,
            prefix_block_pool_snapshot,
            prefix_cache_decision,
        )
        from runtime.worker.factory import LlamaBackendKind

        if getattr(req, "cache_reset", False):
            # WHY under the same prompt_cache_key: clients already have a stable
            # interactive key; a cold: prefix would fragment L3/ps continuity.
            # WHY hard invalidate (epoch + seq_pos/seq_rm): denying cache_prompt
            # alone left residual warm state that looked like a soft resume.
            # WHY skip Radix in admission: cross-slot seed would undo "no reuse."
            slot = self._id_slot_for_request(req)
            subprocess = self._resolved_llama_backend() == LlamaBackendKind.SUBPROCESS
            if slot >= 0:
                from runtime.decode_graph_policy import bump_decode_graph_epoch

                bump_decode_graph_epoch(
                    slot,
                    reason="cache_reset",
                    ctx_ptr=None,
                    base_url=self._subprocess_base_url() if subprocess else None,
                )
                if subprocess:
                    self._subprocess_slots.seed_seq_pos(slot, 0)
                else:
                    self._clear_inprocess_slot_kv(slot)
            record_prefix_cache_decision(
                spec=apply_cache_level_to_policy(
                    policy, getattr(req, "cache_level", None)
                ).to_spec(),
                prompt_cache_key=req.prompt_cache_key,
                seq_pos=0,
                prompt_tokens=req.num_prompt_tokens,
                cache_prompt=False,
                resume_pos=None,
                spec_method=self.config.speculative.method,
                id_slot=slot if slot >= 0 else None,
                deny_reason="cache_reset",
                prefix_block_match=None,
            )
            return False, None

        policy = apply_cache_level_to_policy(policy, getattr(req, "cache_level", None))
        spec = policy.to_spec()
        slot = self._id_slot_for_request(req)
        seq_pos = self._decode_current_pos_for_request(req)
        subprocess = self._resolved_llama_backend() == LlamaBackendKind.SUBPROCESS
        model_hash = self._model_hash_for_cache(req.gguf)
        allow, resume, deny_reason = prefix_cache_decision(
            req.prompt_cache_key,
            policy,
            seq_pos=seq_pos,
            prompt_tokens=req.num_prompt_tokens,
            prompt_token_ids=req.prompt_tokens,
            model_hash=model_hash,
            cache_salt=req.cache_salt,
            subprocess=subprocess,
            load_tier_filter=getattr(req, "load_tier_filter", None),
        )
        block_pool = prefix_block_pool_snapshot(
            prompt_token_ids=req.prompt_tokens,
            model_hash=model_hash,
            cache_salt=req.cache_salt,
            seq_pos=seq_pos,
            resume=resume,
            load_tier_filter=getattr(req, "load_tier_filter", None),
        )
        inv_reason = decode_graph_invalidation_reason(
            allow=allow,
            resume=resume,
            seq_pos=seq_pos,
            slot_pinned=bool(req.slot_pinned),
            deny_reason=deny_reason,
        )
        if inv_reason is not None and subprocess and slot >= 0:
            from runtime.decode_graph_policy import bump_decode_graph_epoch

            bump_decode_graph_epoch(
                slot,
                reason=inv_reason,
                ctx_ptr=None,
                base_url=self._subprocess_base_url(),
            )
        record_prefix_cache_decision(
            spec=spec,
            prompt_cache_key=req.prompt_cache_key,
            seq_pos=seq_pos,
            prompt_tokens=req.num_prompt_tokens,
            cache_prompt=allow,
            resume_pos=resume,
            spec_method=self.config.speculative.method,
            id_slot=slot if slot >= 0 else None,
            deny_reason=deny_reason,
            prefix_block_match=block_pool,
        )
        # Hit count at admit for cache_creation = newly_cached − hit (vLLM #48535).
        if allow and resume is not None and resume > 0:
            req.cached_tokens_at_admit = max(
                int(getattr(req, "cached_tokens_at_admit", 0) or 0),
                int(resume),
            )
        return allow, resume

    def _is_kv_slot_busy(self, slot: int) -> bool:
        return self.loop._slots.is_busy(slot)

    def _prefix_cache_admission(
        self,
        req: Request,
        policy: Any,
    ) -> tuple[bool, int | None]:
        """Policy decision + optional cross-slot Radix KV seed.

        WHY Radix runs here (not in Go): admission needs live slot occupancy,
        ``KVCacheSpec`` window, and either ctypes or subprocess HTTP — all owned
        by the Python engine on the hot decode path.

        WHY Radix runs even when ``cache_prompt`` was denied: SWA window (or
        retention on same-slot resume) may reject the *full* prompt while a
        shorter matched shared prefix still fits — vLLM Marconi + selective
        retention preservation (#47782). Successful seed flips ``allow`` True.

        WHY ``cache_reset`` skips Radix: client asked for no KV reuse under the
        same key this turn; cross-slot seed would undo that contract.
        """
        allow, resume = self._prefix_cache_request(req, policy)
        # cache_reset means no KV reuse this turn — including cross-slot Radix seed.
        if getattr(req, "cache_reset", False):
            return allow, resume
        seq_pos = self._decode_current_pos_for_request(req)
        allow, resume, _trace = self._apply_radix_prefix_share(
            req,
            policy,
            allow=allow,
            resume_pos=resume,
            seq_pos=seq_pos,
        )
        return allow, resume

    def _clear_inprocess_slot_kv(self, slot: int) -> None:
        """Drop all tokens on an in-process seq after ``cache_reset``."""
        if slot < 0:
            return
        pair = self._inprocess_ctx_for_health()
        if pair is None:
            return
        lib, ctx = pair
        try:
            from runtime.worker.libllama_ctypes import _clear_sequence

            _clear_sequence(lib, ctx, int(slot))
        except Exception:
            pass

    def _apply_radix_prefix_share(
        self,
        req: Request,
        policy: Any,
        *,
        allow: bool,
        resume_pos: int | None,
        seq_pos: int | None = None,
    ) -> tuple[bool, int | None, dict[str, Any] | None]:
        """Seed target slot KV from a donor when block pool finds a hash match.

        WHY skip some hybrid layouts: attn+recurrent memory (some LFM2) can abort
        ``seq_cp``; Gemma-style hybrid (full+SWA layers) is allowed when copy fits
        SWA window — see ``radix_seq_copy_policy`` (L3-R5).
        WHY warm catch-up (L3-R2): agent threads extend shared prefix on a warm slot
        while another slot already prefilled further — full seq-copy after verify.
        WHY bump decode graph after copy: CUDA graphs key by topology; seeded KV
        invalidates captured decode graphs on the target slot.
        WHY ``seed_seq_pos`` on subprocess: SWA/draft policy reads slot pos before
        first completion; seq-copy does not update Python-side slot metadata alone.
        WHY skip on ``cache_reset``: client asked for no reuse under the same key;
        Radix seed would undo that.
        """
        from runtime.decode_graph_policy import bump_decode_graph_epoch
        from runtime.kv.radix_prefix_share import find_radix_share_plan, radix_prefix_share_enabled
        from runtime.kv.radix_seq_copy import execute_radix_share_plan
        from runtime.kv.radix_seq_copy_policy import radix_seq_copy_allowed

        if getattr(req, "cache_reset", False):
            return allow, resume_pos, None
        if not radix_prefix_share_enabled():
            return allow, resume_pos, None
        # Draft / disabled base: do not cross-slot seed (same contract as cache_prompt).
        if not getattr(policy, "allow_cache_prompt", True):
            return allow, resume_pos, None
        if not req.slot_pinned or req.kv_slot is None or req.kv_slot < 0:
            return allow, resume_pos, None
        model_hash = self._model_hash_for_cache(req.gguf)
        if not model_hash or not req.prompt_tokens:
            return allow, resume_pos, None

        spec = policy.to_spec()
        plan = find_radix_share_plan(
            req.prompt_tokens,
            target_slot=int(req.kv_slot),
            model_hash=model_hash,
            cache_salt=req.cache_salt,
            seq_pos=seq_pos,
            effective_window=spec.effective_window,
            prefer_session_key=getattr(req, "session_parent", None),
            prefer_session_group=getattr(req, "session_group", None),
        )
        if plan is None:
            # L3-R7: no live donor — try federated slot blob restore.
            return self._apply_blob_restore(
                req,
                model_hash=model_hash,
                seq_pos=seq_pos,
                effective_window=spec.effective_window,
                allow=allow,
                resume_pos=resume_pos,
            )
        copy_ok, skip_reason = radix_seq_copy_allowed(spec, plan)
        if not copy_ok:
            return allow, resume_pos, {"skipped": skip_reason or "seq_copy_denied"}
        if self._is_kv_slot_busy(plan.source_slot):
            return allow, resume_pos, {"skipped": "source_slot_busy"}

        pair = self._inprocess_ctx_for_health()
        copied = False
        if pair is not None:
            copied = execute_radix_share_plan(
                plan, inprocess_lib=pair[0], inprocess_ctx=pair[1]
            )
        else:
            base = self._subprocess_base_url()
            if base:
                copied = execute_radix_share_plan(
                    plan, subprocess_base_url=base
                )
        if not copied:
            # Live seq_cp failed (ghost remote slot?) — fall back to blob.
            blob_out = self._apply_blob_restore(
                req,
                model_hash=model_hash,
                seq_pos=seq_pos,
                effective_window=spec.effective_window,
                allow=allow,
                resume_pos=resume_pos,
            )
            if blob_out[2] and blob_out[2].get("ok"):
                return blob_out
            return allow, resume_pos, {"skipped": "seq_copy_failed"}

        from runtime.worker.libllama_ctypes import _ctx_ptr

        bump_decode_graph_epoch(
            plan.target_slot,
            reason="radix_cross_slot_seed",
            ctx_ptr=_ctx_ptr(pair[1]) if pair is not None else None,
            base_url=self._subprocess_base_url() if pair is None else None,
        )
        trace = {
            "source_slot": plan.source_slot,
            "target_slot": plan.target_slot,
            "copy_tokens": plan.copy_tokens,
            "matched_blocks": plan.matched_blocks,
            "warm_catchup": plan.warm_catchup,
            "target_seq_pos_before": plan.target_seq_pos_before,
        }
        from runtime.kv.radix_prefix_share import (
            approx_kv_bytes_per_token,
            record_radix_copy_cost,
        )
        from runtime.kv.radix_seq_copy import seq_cp_mode

        mode = seq_cp_mode()
        trace["seq_cp_mode"] = mode
        # v52: unified KV → metadata-only share; do not accumulate buffer-copy waste.
        if mode == "buffer_copy":
            bpt = approx_kv_bytes_per_token(
                req.gguf,
                num_ctx=int(self._loaded_vram_num_ctx or 0) or 4096,
            )
            approx_bytes = record_radix_copy_cost(
                copy_tokens=plan.copy_tokens, bytes_per_token=bpt
            )
            if approx_bytes is not None:
                trace["approx_copy_bytes"] = approx_bytes
        else:
            trace["approx_copy_bytes"] = 0
            # Still count tokens for operator visibility of share volume.
            record_radix_copy_cost(copy_tokens=plan.copy_tokens, bytes_per_token=0)
        from runtime.worker.factory import LlamaBackendKind

        if self._resolved_llama_backend() == LlamaBackendKind.SUBPROCESS:
            self._subprocess_slots.seed_seq_pos(plan.target_slot, plan.copy_tokens)
        from runtime.prefix_cache_trace import record_radix_share

        record_radix_share(
            prompt_cache_key=req.prompt_cache_key,
            id_slot=plan.target_slot,
            radix_trace=trace,
            spec_method=self.config.speculative.method,
        )
        return True, plan.copy_tokens, trace

    def _apply_blob_restore(
        self,
        req: Request,
        *,
        model_hash: str,
        seq_pos: int | None,
        effective_window: int | None,
        allow: bool,
        resume_pos: int | None,
    ) -> tuple[bool, int | None, dict[str, Any] | None]:
        """L3-R7: materialize federated slot blob when live Radix donor is absent."""
        from runtime.decode_graph_policy import bump_decode_graph_epoch
        from runtime.kv.radix_blob_restore import (
            execute_blob_restore_plan,
            find_blob_restore_plan,
        )
        from runtime.prefix_cache_trace import record_radix_share
        from runtime.worker.factory import LlamaBackendKind
        from runtime.worker.libllama_ctypes import _ctx_ptr

        blob_plan = find_blob_restore_plan(
            req.prompt_tokens,
            target_slot=int(req.kv_slot),
            model_hash=model_hash,
            cache_salt=req.cache_salt,
            seq_pos=seq_pos,
            effective_window=effective_window,
            load_tier_filter=getattr(req, "load_tier_filter", None),
        )
        if blob_plan is None:
            return allow, resume_pos, None
        pair = self._inprocess_ctx_for_health()
        lib = pair[0] if pair else None
        ctx = pair[1] if pair else None
        trace = execute_blob_restore_plan(
            blob_plan,
            model_hash=model_hash,
            inprocess_lib=lib,
            inprocess_ctx=ctx,
            token_capacity=int(self._loaded_vram_num_ctx or 0) or None,
        )
        if not trace.get("ok"):
            return allow, resume_pos, trace
        restored = int(trace.get("restored_tokens") or blob_plan.restore_tokens)
        # SGLang storage-tier: federated LMCache / radix blob restore.
        from runtime.llama_timings import lmcache_storage_backend_label

        req.cached_tokens_storage = restored
        req.cached_tokens_storage_backend = lmcache_storage_backend_label()
        bump_decode_graph_epoch(
            blob_plan.target_slot,
            reason="radix_blob_restore",
            ctx_ptr=_ctx_ptr(ctx) if ctx is not None else None,
            base_url=self._subprocess_base_url() if pair is None else None,
        )
        if self._resolved_llama_backend() == LlamaBackendKind.SUBPROCESS:
            self._subprocess_slots.seed_seq_pos(blob_plan.target_slot, restored)
        record_radix_share(
            prompt_cache_key=req.prompt_cache_key,
            id_slot=blob_plan.target_slot,
            radix_trace=trace,
            spec_method=self.config.speculative.method,
        )
        return True, restored, trace

    def _cache_prompt_for_request(self, req: Request, policy: Any) -> bool:
        """L3 ``cache_prompt`` with SWA window + draft-spec policy enforcement."""
        allow, _ = self._prefix_cache_admission(req, policy)
        return allow

    def _decode_resume_pos_for_request(self, req: Request, policy: Any) -> int | None:
        """Live seq position for KV resume; ``None`` when prefix cache is blocked."""
        _, resume_pos = self._prefix_cache_admission(req, policy)
        return resume_pos

    def _vram_llama_kwargs(self) -> dict[str, Any]:
        """Flags passed to llama-server (for VRAM estimate parity with subprocess)."""
        from runtime.speculative import resolve_method

        kw: dict[str, Any] = {
            "llama_args": self.config.llama_server_args(),
            "parallel_slots_default": self.config.llama_parallel_slots,
            "llama_backend": self._health_llama_backend(),
            "n_gpu_layers_default": self.config.n_gpu_layers_default,
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
        if gguf is None and self.config.llama_model and self.config.llama_model.is_file():
            gguf = self.config.llama_model
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
            from runtime.infer_trace import infer_trace

            infer_trace(
                "engine.reload",
                reason="path_mismatch_or_cold",
                current=str(current) if current else None,
                resolved=str(resolved),
                server_was_none=self._server is None,
            )
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
            from runtime.infer_trace import infer_trace

            infer_trace(
                "engine.reload",
                reason="ctx_grew",
                resolved=str(resolved),
                needed_ctx=needed_ctx,
                loaded_ctx=loaded_ctx,
            )
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
            from runtime.infer_trace import infer_trace

            infer_trace(
                "engine.reuse",
                resolved=str(resolved),
                needed_ctx=needed_ctx,
                loaded_ctx=loaded_ctx,
            )
            return self._server
        if self._server is not None and not self._server.is_running():
            crash = getattr(self._server, "_exit_code", None)
            if crash not in (None, 0):
                from runtime.infer_trace import infer_trace

                infer_trace(
                    "engine.reload",
                    reason="subprocess_crash",
                    resolved=str(resolved),
                    exit_code=crash,
                )
                self._stop_server()
                self.config.llama_model = resolved
                self._server = self._create_llama_worker(resolved)
                self.coordinator.set_unload_hook(self._stop_server)
                proc_alive = False
        if not self._server.is_running():
            from runtime.infer_trace import infer_trace

            infer_trace(
                "engine.start",
                reason="not_running",
                resolved=str(resolved),
                needed_ctx=needed_ctx,
            )
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

        try:
            host_ram = host_ram_budget_snapshot(resolved)
        except (OSError, TypeError, ValueError):
            host_ram = None
        if host_ram is not None:
            if budget is None:
                budget = {}
            budget["host_ram"] = host_ram
        # WHY on estimate response: /api/can-load surfaces host TP/device layout so
        # Hermes can see multi-GPU without scraping /health separately.
        topology = {
            "device_count": max(1, int(getattr(self.config, "device_count", 1) or 1)),
            "tensor_parallel": max(1, int(getattr(self.config, "tensor_parallel", 1) or 1)),
            "split_mode": str(getattr(self.config, "split_mode", "") or ""),
            "main_gpu": int(getattr(self.config, "main_gpu", 0) or 0),
        }
        ts = getattr(self.config, "tensor_split", None)
        if ts:
            try:
                topology["tensor_split"] = [float(x) for x in ts]
            except (TypeError, ValueError):
                pass
        if est is not None:
            est = dict(est)
            est["topology"] = topology
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
        # WHY ctypes getenv: training may Py_Initialize first, freezing Python's
        # os.environ. Go then setenv(ZEROLLAMA_RUNTIME_EMBED_BOOT) for stale-listener
        # detection; os.environ.get misses it and Go never publishes BaseURL.
        embed_boot = os.environ.get("ZEROLLAMA_RUNTIME_EMBED_BOOT", "").strip()
        if not embed_boot:
            embed_boot = _c_environ_get("ZEROLLAMA_RUNTIME_EMBED_BOOT")
            if embed_boot:
                os.environ["ZEROLLAMA_RUNTIME_EMBED_BOOT"] = embed_boot
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
            "kv_continuous_batch": self._kv_continuous_batch_health(),
            "kv_auto_batch": self._kv_auto_batch_health(),
            "kv_page_bind": self._kv_page_bind_health(),
            "kv_decode_loop": self._kv_decode_loop_health(),
            "kv_resume": self._kv_resume_health(),
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
            "llama_server_status": self._llama_server_status_health(),
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
        if self.config.gpu_profile:
            body["gpu_profile"] = self.config.gpu_profile
        from runtime.llama_fork import fork_health

        body["llama_fork"] = fork_health(
            llama_server_bin=self.config.llama_server_bin
        )
        from runtime.llama_cpp_unified import unified_health

        body["llama_cpp_unified"] = unified_health(
            llama_cpp_root=self.config.llama_cpp_root,
            llama_server_bin=self.config.llama_server_bin,
        )
        from runtime.llama_patch_health import llama_patch_health_summary

        body["llama_patches"] = llama_patch_health_summary()
        from runtime.cache_bridge import cache_health
        from runtime.speculative import resolve_method

        draft: Path | None = None
        spec = self.config.speculative
        if (
            spec.draft_model is not None
            and spec.draft_model.is_file()
            and resolve_method(spec.method).startswith("draft")
        ):
            draft = spec.draft_model
        body["llama_cache"] = cache_health(
            model_path,
            self.config.llama_server_args(),
            draft_model=draft,
            num_ctx=self._loaded_vram_num_ctx,
            spec_method=self.config.speculative.method,
        )
        from runtime.readiness import compute_readiness

        body.update(compute_readiness(body))
        return body

    def _llama_server_status_health(self) -> dict[str, Any] | None:
        srv = self._server
        if srv is None:
            return None
        snap = getattr(srv, "status_snapshot", None)
        if callable(snap):
            return snap()
        running = srv.is_running() if hasattr(srv, "is_running") else False
        return {"running": running, "died": not running, "reachable": None}

    def maybe_auto_resume_inference(self) -> bool:
        """Resume after training handoff when Go mirror shows training idle."""
        import os

        from runtime.gpu.mutex import InferenceState

        raw = os.environ.get("ZEROLLAMA_RUNTIME_AUTO_RESUME", "1").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return False
        if self.coordinator.state == InferenceState.RUNNING:
            return False
        if self.coordinator.go_training_gpu_busy:
            return False
        from runtime.go_coordination import (
            go_coordination_is_fresh,
            go_defer_waiting,
            go_ggml_loads_paused,
            go_training_gpu_blocked,
        )

        if go_coordination_is_fresh():
            if go_training_gpu_blocked():
                return False
            if go_ggml_loads_paused():
                return False
            if go_defer_waiting() > 0:
                return False
        self.resume_inference()
        return True

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

    def _model_hash_for_cache(self, gguf: Path | None = None) -> str | None:
        path = gguf
        if path is None and self.config.llama_model:
            path = Path(self.config.llama_model)
        if path is None or not Path(path).is_file():
            return None
        from runtime.cache_bridge import build_model_hash, cache_type_from_llama_argv

        ck, cv = cache_type_from_llama_argv(self.config.llama_server_args())
        draft = (
            Path(self.config.speculative.draft_model)
            if self.config.speculative.draft_model
            else None
        )
        return build_model_hash(
            target_model_path=path,
            drafter_model_path=draft if draft and draft.is_file() else None,
            cache_type_k=ck,
            cache_type_v=cv,
        )

    def _completion_seq_pos(self, req: Request, result: dict[str, Any]) -> int:
        from runtime.subprocess_slot_state import seq_pos_from_llama_result

        pos = seq_pos_from_llama_result(result)
        if pos is not None:
            return pos
        return max(0, len(req.prompt_tokens) + len(req.generated))

    def _record_subprocess_slot(self, req: Request, result: dict[str, Any]) -> None:
        from runtime.worker.factory import LlamaBackendKind

        if self._resolved_llama_backend() == LlamaBackendKind.SUBPROCESS:
            self._subprocess_slots.record_completion(
                self._id_slot_for_request(req), result
            )
        self._register_prefix_block_pool(req, result)
        created = int(getattr(req, "cache_creation_tokens", 0) or 0)
        if created > 0 and isinstance(result, dict):
            result["cache_creation_tokens"] = created
            result["prompt_eval_cache_creation_count"] = created
            result["created_cache_tokens"] = created

    def _register_prefix_block_pool(self, req: Request, result: dict[str, Any]) -> None:
        from pathlib import Path

        from runtime.cache_bridge import slot_cache_file_path
        from runtime.kv.prefix_block_pool import (
            build_model_scope,
            get_prefix_block_pool,
            prefix_block_pool_enabled,
        )
        from runtime.kv.swa_store_filter import swa_reachable_store_mask
        from runtime.kv_cache_spec import prefix_cache_block_size

        if not prefix_block_pool_enabled():
            return
        if not req.prompt_cache_key or not req.slot_pinned:
            return
        model_hash = self._model_hash_for_cache(req.gguf)
        if not model_hash:
            return
        seq_pos = self._completion_seq_pos(req, result)
        if seq_pos <= 0:
            return
        scope = build_model_scope(model_hash=model_hash, cache_salt=req.cache_salt)
        blob_path: str | None = None
        slot = self._id_slot_for_request(req)
        if slot >= 0:
            blob_path = str(slot_cache_file_path(model_hash, slot, 0))
        pool = get_prefix_block_pool(model_scope=scope)
        # Reuse-race: flush any pending finalize for this slot before overwrite.
        if slot >= 0:
            pool.flush_pending_blob_before_reuse(scope=scope, slot_id=slot)

        store_mask = None
        policy = self._prefix_cache_policy(
            req.gguf,
            req.num_ctx,
            cache_level=getattr(req, "cache_level", None),
        )
        if policy is not None:
            spec = policy.to_spec() if hasattr(policy, "to_spec") else policy
            window = getattr(spec, "effective_window", None)
            if window is None and getattr(spec, "coordinator", None) is not None:
                window = getattr(spec.coordinator, "min_window", None)
            kind = getattr(spec, "kind", None)
            if kind in ("sliding_window", "hybrid") and window:
                bs = prefix_cache_block_size()
                num_blocks = seq_pos // max(1, bs)
                retention = getattr(spec, "retention_interval", None)
                draft = bool(getattr(spec, "drop_last_block_on_resume", False))
                store_mask = swa_reachable_store_mask(
                    num_blocks,
                    block_size=bs,
                    sliding_window=int(window),
                    retention_interval=retention,
                    draft_extra=draft,
                )

        # Auto finalize: publish only when slot blob exists (defer otherwise).
        reg = pool.register_prefix(
            req.prompt_tokens,
            scope=scope,
            seq_pos=seq_pos,
            session_key=req.prompt_cache_key,
            slot_id=slot if slot >= 0 else None,
            blob_path=blob_path,
            session_group=getattr(req, "session_group", None),
            finalize_blob=None,
            store_block_mask=store_mask,
        )
        if (
            not reg.blob_finalized
            and blob_path
            and Path(blob_path).is_file()
            and slot >= 0
        ):
            pool.finalize_slot_blob(scope=scope, slot_id=slot, blob_path=blob_path)

        # vLLM #48535 — creation = newly cached − already hit at admit.
        hit = max(0, int(getattr(req, "cached_tokens_at_admit", 0) or 0))
        created = max(0, min(seq_pos, len(req.prompt_tokens)) - hit)
        req.cache_creation_tokens = created

    def _decode_current_pos_for_request(self, req: Request) -> int | None:
        """Live llama write position for KV resume / SWA cache policy.

        In-process: read ``llama_memory_seq_pos_max`` on the shared ctx.
        Subprocess: last completion timings on the pinned ``id_slot`` (vLLM SWA guard).
        """
        pair = self._inprocess_ctx_for_health()
        if pair is not None:
            from runtime.kv.physical import current_pos_for_seq

            lib, ctx = pair
            return current_pos_for_seq(lib, ctx, self._id_slot_for_request(req))
        if self._resolved_llama_backend() != LlamaBackendKind.SUBPROCESS:
            return None
        slot = self._id_slot_for_request(req)
        pos = self._subprocess_slots.seq_pos(slot)
        if pos is not None:
            return pos
        if slot < 0:
            return None
        srv = self._server
        if srv is not None:
            base_url = getattr(srv, "base_url", None)
            if base_url:
                return self._subprocess_slots.seq_pos_with_fallback(slot, base_url)
        return None

    def _subprocess_base_url(self) -> str | None:
        """llama-server listen URL when subprocess backend is active.

        WHY: ``bump_decode_graph_epoch`` POSTs ``/cuda-graph/invalidate`` here so
        ggml CUDA graph clear runs in the child that owns ``ctx_tgt`` — not in
        this Python process.
        """
        from runtime.worker.factory import LlamaBackendKind

        if self._resolved_llama_backend() != LlamaBackendKind.SUBPROCESS:
            return None
        srv = self._server
        if srv is None:
            return None
        return getattr(srv, "base_url", None)

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
        from runtime.kv.page_bind import page_bind_stats
        from runtime.kv.physical import kv_bind_physical_level

        backend = self._health_llama_backend()
        inprocess_loaded = self._inprocess_session_for_health() is not None
        stats = page_bind_stats()
        physical_bound = bool(stats.get("physical_pages_bound"))
        tensor_bound = bool(stats.get("tensor_pages_bound"))
        level = kv_bind_physical_level(
            backend, inprocess_weights_loaded=inprocess_loaded
        )
        if physical_bound:
            level = "physical"
        elif tensor_bound:
            level = "tensor"
        return kv_bind_health(
            llama_backend=backend,
            assign_llama_slots=self.loop.assign_llama_slots,
            parallel_slots=self._effective_llama_parallel_slots(),
            physical_bind_level=level,
            physical_pages_bound=physical_bound,
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
        from runtime.kv.page_migration_plan import migration_plan_summary
        from runtime.kv.physical import current_pos_by_request_from_physical

        pos_by_id = current_pos_by_request_from_physical(self._kv_physical_health())
        reqs = list(self.scheduler.waiting) + list(self.scheduler.running)
        plans = kv_forward_plans_for_requests(
            reqs,
            block_size=self.config.block_size,
            current_pos_by_request_id=pos_by_id or None,
        )
        probe_by_slot = self._tensor_probe_by_running_kv_slot()
        layers_expected = self._tensor_layers_expected_for_health()
        last_probe_row = None
        for plan in plans:
            if plan.get("state") != "running":
                continue
            slot = plan.get("kv_slot")
            if slot is None:
                continue
            probe = probe_by_slot.get(int(slot))
            if not probe:
                if last_probe_row is None:
                    from runtime.kv.page_bind import page_bind_last_probe_row_for_health

                    last_probe_row = page_bind_last_probe_row_for_health()
                if (
                    last_probe_row is not None
                    and int(last_probe_row["kv_slot"]) == int(slot)
                ):
                    probe = last_probe_row["probe"]
            if not probe:
                continue
            summary = migration_plan_summary(
                probe,
                block_size=self.config.block_size,
                kv_slot=int(slot),
                tensor_layers_expected=layers_expected,
            )
            if summary:
                plan["page_migration_summary"] = summary
        return plans

    def _tensor_layers_expected_for_health(self) -> int | None:
        """Expected full-attention layer count for hybrid models (v36/v40)."""
        try:
            gguf = self._health_gguf_path()
            if gguf is None:
                return None
            from runtime.gguf_estimate import gguf_arch_hints
            from runtime.kv.hybrid_kv_coordinator import build_hybrid_kv_coordinator

            coord = build_hybrid_kv_coordinator(
                gguf_arch_hints(gguf),
                self.config.num_ctx if self.config else None,
            )
            if coord.kind in ("hybrid", "sliding_window"):
                return coord.full_layer_count
        except Exception:
            pass
        return None

    def _tensor_probe_by_running_kv_slot(self) -> dict[int, dict[str, Any]]:
        """Live tensor probes keyed by kv_slot for running requests."""
        from runtime.kv.page_bind import page_bind_tensor_probe_for_ctx
        from runtime.kv.tensor_probe import tensor_probe_available

        out: dict[int, dict[str, Any]] = {}
        if not tensor_probe_available():
            return out
        pair = self._inprocess_ctx_for_health()
        if pair is None:
            return out
        lib, ctx = pair
        seen: set[int] = set()
        for req in self.scheduler.running:
            slot = req.kv_slot
            if slot is None or slot < 0 or slot in seen:
                continue
            seen.add(slot)
            probe = page_bind_tensor_probe_for_ctx(
                lib, ctx, seq_id=slot, kv_slot=slot
            )
            if probe:
                out[slot] = probe
        return out

    def _kv_continuous_batch_health(self) -> dict[str, Any]:
        from runtime.kv.forward_plan import kv_continuous_batch_forward_plan
        from runtime.kv.physical import current_pos_by_request_from_physical

        running = list(self.scheduler.running)
        pos_by_id = current_pos_by_request_from_physical(self._kv_physical_health())
        return kv_continuous_batch_forward_plan(
            running,
            block_size=self.config.block_size,
            current_pos_by_request_id=pos_by_id or None,
            parallel_slots=self._effective_llama_parallel_slots(),
        )

    def _kv_auto_batch_health(self) -> dict[str, Any]:
        from runtime.kv.auto_batch import native_auto_batch_enabled
        from runtime.kv.stream_auto_batch import native_stream_auto_batch_enabled

        out: dict[str, Any] = {
            "non_stream": self._auto_batch.stats(),
            "stream": self._stream_auto_batch.stats(),
        }
        if not native_auto_batch_enabled():
            out["non_stream"].setdefault(
                "note",
                "set ZEROLLAMA_KV_AUTO_BATCH=1 + inprocess multiseq + linked batch decode",
            )
        if not native_stream_auto_batch_enabled():
            out["stream"].setdefault(
                "note",
                "set ZEROLLAMA_KV_AUTO_BATCH_STREAM=1 + inprocess multiseq + linked batch decode",
            )
        return out

    def _kv_page_bind_health(self) -> dict[str, Any]:
        from runtime.kv.backend import native_available
        from runtime.kv.page_bind import (
            page_bind_health,
            page_bind_tensor_probe_for_ctx,
        )

        # WHY only probe running requests: slot 0 with no active bind would
        # return aligned=True trivially and mislead the operator into thinking
        # a live seq has been verified.  Omit probe when nothing is running.
        tensor_probe = None
        probe_kv_slot: int | None = None
        pair = self._inprocess_ctx_for_health()
        if pair is not None:
            lib, ctx = pair
            for req in self.scheduler.running:
                slot = req.kv_slot
                if slot is not None and slot >= 0:
                    probe_kv_slot = slot
                    tensor_probe = page_bind_tensor_probe_for_ctx(
                        lib, ctx, seq_id=slot, kv_slot=slot
                    )
                    break

        # v36: resolve GGUF layer-group coordinator so page_bind_health can emit
        # kv_full_layers / kv_swa_layers / tensor_layers_expected.
        # WHY lazy + best-effort: the coordinator is a read-only GGUF parse; errors
        # must not break /health when the model file is temporarily unavailable.
        kv_coordinator = None
        try:
            gguf = self._health_gguf_path()
            if gguf is not None:
                from runtime.kv.hybrid_kv_coordinator import build_hybrid_kv_coordinator
                from runtime.gguf_estimate import gguf_arch_hints

                arch = gguf_arch_hints(gguf)
                num_ctx = self.config.num_ctx if self.config else None
                kv_coordinator = build_hybrid_kv_coordinator(arch, num_ctx)
        except Exception:
            pass

        return page_bind_health(
            native_ext_available=native_available(),
            tensor_probe=tensor_probe,
            kv_coordinator=kv_coordinator,
            kv_slot=probe_kv_slot,
            block_size=self.config.block_size,
            overlay_bind_donor_id=self._kv_overlay_donor_id,
            overlay_donor_base=self._kv_overlay_donor_ptr,
            overlay_donor_size=self._kv_overlay_donor_size,
            overlay_catalog_ctx=self._overlay_catalog_ctx_args(probe_kv_slot),
        )

    def _overlay_catalog_ctx_args(
        self, kv_slot: int | None
    ) -> tuple[int, int, int] | None:
        """``(ctx_ptr, seq_id, kv_slot)`` for v51 donor page catalog, or None."""
        if kv_slot is None or kv_slot < 0:
            return None
        pair = self._inprocess_ctx_for_health()
        if pair is None:
            return None
        from runtime.worker.libllama_ctypes import _ctx_ptr

        ctx_ptr = _ctx_ptr(pair[1])
        if not ctx_ptr:
            return None
        return int(ctx_ptr), int(kv_slot), int(kv_slot)

    def register_kv_overlay_donor(self, ptr: int, size: int) -> int:
        """Phase 15 v48: register a pre-allocated, correctly-sized host buffer as
        a KV-cache allocation donor. MUST be called before constructing the
        context/model that will consume it (see runtime/kv/overlay_bind.py doc).
        Raises RuntimeError if ZEROLLAMA_KV_OVERLAY_BIND is not set.
        """
        from runtime.kv.overlay_bind import register_donor_buffer

        donor_id = register_donor_buffer(ptr, size)
        self._kv_overlay_donor_id = donor_id
        self._kv_overlay_donor_ptr = int(ptr)
        self._kv_overlay_donor_size = int(size)
        return donor_id

    def unregister_kv_overlay_donor(self) -> None:
        """Unregister the currently tracked donor buffer, if any. Callers must
        ensure the context/model that consumed it has already been unloaded —
        see runtime/kv/overlay_bind.py lifecycle contract."""
        from runtime.kv.overlay_bind import unregister_donor_buffer

        if self._kv_overlay_donor_id is not None:
            unregister_donor_buffer(self._kv_overlay_donor_id)
            self._kv_overlay_donor_id = None
        self._kv_overlay_donor_keepalive = None
        self._kv_overlay_donor_ptr = None
        self._kv_overlay_donor_size = None

    def _sync_kv_overlay_donor_from_session(self) -> None:
        """v50: pick up auto-wired donor id/ptr from in-process session for /health."""
        session = self._inprocess_session_for_health()
        if session is None:
            return
        donor_id = getattr(session, "overlay_donor_id", None)
        if donor_id is None:
            return
        self._kv_overlay_donor_id = int(donor_id)
        # Session owns keepalive; engine mirrors id + geometry for health/catalog.
        handle = getattr(session, "_overlay_donor", None)
        if handle is not None:
            self._kv_overlay_donor_keepalive = getattr(handle, "_keepalive", None)
            self._kv_overlay_donor_ptr = int(getattr(handle, "ptr", 0) or 0) or None
            self._kv_overlay_donor_size = int(getattr(handle, "size", 0) or 0) or None

    def _kv_decode_loop_health(self) -> dict[str, Any]:
        from runtime.kv.native_decode_loop import decode_loop_status

        return decode_loop_status()

    def _kv_resume_health(self) -> dict[str, Any]:
        """In-process KV resume operator probe (Phase 15 v18).

        WHY: v16–v17 resume is session-local on ``LlamaLoadedSession``; without
        this field operators cannot tell whether L3 prefix reuse is armed for
        the loaded in-process model (needs ``llama_parallel_slots > 1``).
        """
        backend = self._health_llama_backend()
        parallel = self._effective_llama_parallel_slots()
        session = self._inprocess_session_for_health()
        owners: dict[str, str] = {}
        if session is not None:
            snap = getattr(session, "resume_owner_snapshot", None)
            if callable(snap):
                owners = {str(slot): owner for slot, owner in snap().items()}
        active = (
            backend == "inprocess"
            and parallel > 1
            and session is not None
            and getattr(session, "_ctx", None) is not None
        )
        note: str | None = None
        if backend != "inprocess":
            note = "subprocess: L3 resume uses llama-server cache_prompt + id_slot"
        elif parallel <= 1:
            note = (
                "single-seq in-process uses per-request ctx; "
                "resume requires llama_parallel_slots>1"
            )
        spec_health = self._prefix_cache_policy(
            None, self._loaded_vram_num_ctx
        ).to_spec().to_health()
        from runtime.kv.prefix_block_pool import build_model_scope, prefix_block_pool_health
        from runtime.kv.radix_prefix_share import radix_share_health
        from runtime.env import (
            kv_unified_enabled,
            kv_unified_operator_note,
            kv_unified_sizing_status,
            kv_unified_source,
        )

        model_hash = self._model_hash_for_cache()
        scope = build_model_scope(model_hash=model_hash) if model_hash else None
        block_pool_health = prefix_block_pool_health(model_scope=scope)
        block_pool_health["radix_share"] = radix_share_health(model_scope=scope)
        session_unified = bool(getattr(session, "kv_unified", False)) if session else False
        unified = session_unified or kv_unified_enabled()
        out = {
            "active": active,
            "llama_parallel_slots": parallel,
            "owners_by_slot": owners,
            "subprocess_slot_seq_pos": self._subprocess_slots.snapshot()
            if backend == "subprocess"
            else {},
            "owner_key_pinned": "cache:{prompt_cache_key}",
            "owner_key_unpinned": "request_id",
            "note": note,
            "prefix_cache_spec": spec_health,
            "prefix_block_pool": block_pool_health,
            "kv_unified": unified,
            "kv_unified_source": kv_unified_source(),
        }
        # v54/v55/v57: advisory note + sizing probe + idle purge stats when unified.
        if unified:
            out["kv_unified_note"] = kv_unified_operator_note()
            sizing = kv_unified_sizing_status(
                n_ctx=self._loaded_vram_num_ctx, n_parallel=parallel
            )
            if sizing is not None:
                out["kv_unified_sizing"] = sizing
            from runtime.kv.idle_slot_purge import idle_slot_purge_health

            out["kv_unified_idle_purge"] = idle_slot_purge_health()
        from runtime.kv.l3_r6_readiness import l3_r6_metadata_readiness

        out["l3_r6_metadata"] = l3_r6_metadata_readiness(
            n_ctx=self._loaded_vram_num_ctx,
            n_parallel=parallel,
            backend=backend,
        )
        return out

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
            "kv_continuous_batch": self._kv_continuous_batch_health(),
            "kv_auto_batch": self._kv_auto_batch_health(),
            "kv_page_bind": self._kv_page_bind_health(),
            "kv_page_migration": self._kv_page_migration_snapshot(),
            "overlay_page_catalog": self._overlay_page_catalog_snapshot(),
            "kv_decode_loop": self._kv_decode_loop_health(),
            "kv_resume": self._kv_resume_health(),
            "kv_live_physical": self._kv_live_physical_health(),
        }

    def _overlay_page_catalog_snapshot(self) -> dict[str, Any] | None:
        """v51: full donor page→offset catalog for loopback debug."""
        if not self._kv_overlay_donor_ptr or not self._kv_overlay_donor_size:
            return None
        from runtime.kv.overlay_bind import donor_buffer_status, overlay_bind_enabled
        from runtime.kv.overlay_page_catalog import build_overlay_page_catalog
        from runtime.kv.page_bind import page_bind_tensor_probe_for_ctx

        if not overlay_bind_enabled() or self._kv_overlay_donor_id is None:
            return None
        status = donor_buffer_status(int(self._kv_overlay_donor_id))
        if not status or not status.get("bound"):
            return {
                "bound": False,
                "note": "donor registered but not consumed by KV alloc",
                "donor_bytes": int(self._kv_overlay_donor_size),
            }
        pair = self._inprocess_ctx_for_health()
        if pair is None:
            return {
                "bound": True,
                "note": "no in-process ctx for page_map",
                "donor_bytes": int(self._kv_overlay_donor_size),
            }
        from runtime.worker.libllama_ctypes import _ctx_ptr

        ctx_ptr = _ctx_ptr(pair[1])
        if not ctx_ptr:
            return None
        kv_slot: int | None = None
        probe: dict[str, Any] | None = None
        for req in self.scheduler.running:
            slot = req.kv_slot
            if slot is not None and slot >= 0:
                kv_slot = int(slot)
                probe = page_bind_tensor_probe_for_ctx(
                    pair[0], pair[1], seq_id=slot, kv_slot=slot
                )
                break
        if kv_slot is None:
            return {
                "bound": True,
                "note": "no running request with kv_slot",
                "donor_bytes": int(self._kv_overlay_donor_size),
            }
        catalog = build_overlay_page_catalog(
            donor_base=int(self._kv_overlay_donor_ptr),
            donor_size=int(self._kv_overlay_donor_size),
            ctx_ptr=int(ctx_ptr),
            seq_id=kv_slot,
            kv_slot=kv_slot,
            block_size=int(self.config.block_size),
            probe=probe,
            max_pages=None,
            include_pages=True,
        )
        if catalog is None:
            return None
        catalog["bound"] = True
        return catalog

    def _page_migration_summary_for_probe(
        self,
        probe: dict[str, Any],
        kv_slot: int | None,
    ) -> dict[str, Any] | None:
        """Lightweight migration summary for snapshot / health (v42)."""
        from runtime.kv.page_bind import page_bind_last_probe_row_for_health
        from runtime.kv.page_migration_plan import migration_plan_summary

        slot = kv_slot
        if slot is None:
            row = page_bind_last_probe_row_for_health()
            if row is not None:
                slot = int(row["kv_slot"])
        if slot is None or slot < 0:
            return None
        return migration_plan_summary(
            probe,
            block_size=self.config.block_size,
            kv_slot=int(slot),
            tensor_layers_expected=self._tensor_layers_expected_for_health(),
        )

    def _kv_page_migration_snapshot(self) -> dict[str, Any] | None:
        """Export v38 copy descriptors for live bound pages (v39 loopback debug).

        WHY on kv_snapshot not /health: migration plans include raw pointers and
        can be large (pages × layers); keep /health lightweight for polling.

        v42 adds ``migration_summary`` on every branch so idle post-decode snapshots
        still report page/layer bind progress without building pages×layers plans.
        """
        from runtime.kv.page_bind import (
            page_bind_last_probe_row_for_health,
            page_bind_last_tensor_probe_for_health,
            page_bind_tensor_probe_for_ctx,
        )
        from runtime.kv.page_migration_plan import (
            build_page_migration_plan,
            migration_include_pointers,
            prepare_migration_plan_for_export,
        )
        from runtime.kv.tensor_probe import tensor_probe_available

        if not tensor_probe_available():
            return None

        pair = self._inprocess_ctx_for_health()

        def _with_summary(payload: dict[str, Any], probe: dict[str, Any], slot: int | None) -> dict[str, Any]:
            summary = self._page_migration_summary_for_probe(probe, slot)
            if summary:
                payload["migration_summary"] = summary
            return payload

        if pair is None:
            row = page_bind_last_probe_row_for_health()
            probe = page_bind_last_tensor_probe_for_health()
            if probe is None:
                return None
            slot = int(row["kv_slot"]) if row else None
            return _with_summary(
                {
                    "active": False,
                    "source": "last_tensor_probe",
                    "note": "no in-process ctx; summary from last decode probe only",
                    "probe": probe,
                },
                probe,
                slot,
            )

        lib, ctx = pair
        probe: dict[str, Any] | None = None
        kv_slot: int | None = None
        seq_id: int | None = None
        for req in self.scheduler.running:
            slot = req.kv_slot
            if slot is not None and slot >= 0:
                kv_slot = slot
                seq_id = slot
                probe = page_bind_tensor_probe_for_ctx(
                    lib, ctx, seq_id=slot, kv_slot=slot
                )
                break

        if probe is None or kv_slot is None or seq_id is None:
            row = page_bind_last_probe_row_for_health()
            probe = page_bind_last_tensor_probe_for_health()
            if probe is None:
                return None
            slot = int(row["kv_slot"]) if row else None
            payload: dict[str, Any] = {
                "active": False,
                "source": "last_tensor_probe",
                "note": "no running request with kv_slot",
                "probe": probe,
            }
            if slot is not None and slot >= 0:
                payload["kv_slot"] = slot
            try:
                ctx_ptr = int(ctx) if isinstance(ctx, int) else int(getattr(ctx, "value", 0) or 0)
            except (TypeError, ValueError):
                ctx_ptr = 0
            if ctx_ptr and slot is not None and slot >= 0:
                plan = build_page_migration_plan(
                    ctx_ptr,
                    slot,
                    slot,
                    block_size=self.config.block_size,
                    probe=probe,
                )
                if plan is not None:
                    payload["plan_available"] = True
                    payload["plan"] = prepare_migration_plan_for_export(plan)
                    payload["pointers_redacted"] = not migration_include_pointers()
                else:
                    payload["plan_available"] = False
            return _with_summary(payload, probe, slot)

        try:
            ctx_ptr = int(ctx) if isinstance(ctx, int) else int(getattr(ctx, "value", 0) or 0)
        except (TypeError, ValueError):
            return None

        plan = build_page_migration_plan(
            ctx_ptr,
            seq_id,
            kv_slot,
            block_size=self.config.block_size,
            probe=probe,
        )
        if plan is None:
            return _with_summary(
                {
                    "active": True,
                    "kv_slot": kv_slot,
                    "source": "live_probe",
                    "plan_available": False,
                    "note": "tensor/physical bind not complete — no writable page map yet",
                    "probe": probe,
                },
                probe,
                kv_slot,
            )
        return _with_summary(
            {
                "active": True,
                "kv_slot": kv_slot,
                "source": "live_probe",
                "plan_available": True,
                "plan": prepare_migration_plan_for_export(plan),
                "pointers_redacted": not migration_include_pointers(),
            },
            probe,
            kv_slot,
        )

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
        from runtime.cache_bridge import (
            cache_pin_from_options,
            extract_cache_level,
            extract_cache_reset,
            extract_kv_load_tiers,
            extract_session_group,
            extract_session_parent,
        )
        from runtime.kv.tier_filter import parse_tier_filter

        # L3: pin llama-server slot before tick so /completion sees stable id_slot.
        prompt_cache_key, kv_slot, slot_pinned, cache_salt = cache_pin_from_options(
            options,
            parallel=self._effective_llama_parallel_slots(),
        )
        req = Request(
            request_id=uuid.uuid4().hex[:12],
            prompt_tokens=self._prompt_tokens_for_admit(prompt, gguf),
            max_tokens=n_predict,
            priority=priority,
            gguf=resolved_gguf,
            num_ctx=resolved_ctx,
            vram_options=vram_opts,
            vram_num_ctx_meta=clamp_meta or None,
            prompt_cache_key=prompt_cache_key,
            cache_salt=cache_salt,
            session_parent=extract_session_parent(options),
            session_group=extract_session_group(options),
            cache_reset=extract_cache_reset(options),
            cache_level=extract_cache_level(options),
            load_tier_filter=parse_tier_filter(extract_kv_load_tiers(options)),
            kv_slot=kv_slot,
            slot_pinned=slot_pinned,
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

    def _generate_one_admitted(
        self,
        prompt: str,
        active: Request,
        *,
        n_predict: int,
        gguf: Path | None,
        options: dict | None,
        prefill_cancel: Any | None = None,
    ) -> GenerateResult:
        """Run single-request decode for an already-admitted request (v32 helper)."""
        self._assert_kv_bind(active, at="generate")
        vram_opts = active.vram_options or self._vram_options(active.num_ctx, options)
        decode_before = self._kv_decode_steps_before()
        with self._model_swap.hold(gguf):
            srv = self._ensure_gguf_loaded_unlocked(
                gguf, num_ctx=active.num_ctx, options=vram_opts
            )
            from runtime.kv.bind import reserved_token_capacity

            policy = self._prefix_cache_policy(gguf, active.num_ctx)
            cache_ok, resume_pos = self._prefix_cache_admission(active, policy)
            raw = srv.completion(
                prompt,
                n_predict=n_predict,
                id_slot=self._id_slot_for_request(active),
                kv_token_budget=reserved_token_capacity(active),
                kv_bind_req=active,
                kv_block_size=self.config.block_size,
                sampler=sampler_options_from_dict(options),
                cache_prompt=cache_ok,
                current_pos=resume_pos,
                prefill_cancel=prefill_cancel,
                format_options=options,
            )
            self._record_subprocess_slot(active, raw)
            content = raw.get("content") or raw.get("response") or ""
            active.state = RequestState.DECODE
            return GenerateResult(
                content=content,
                request_id=active.request_id,
                llama=raw,
                vram_num_ctx=self._api_vram_num_ctx_from_request(active),
                kv_decode_steps=self._kv_decode_steps_after(decode_before),
            )

    def _generate_parallel_admitted(
        self, jobs: list[Any]
    ) -> list[GenerateResult]:
        """Decode N already-admitted requests via C batch path (v32 auto-batch)."""
        from runtime.kv.auto_batch import _PendingJob

        pending = [j for j in jobs if isinstance(j, _PendingJob)]
        if not pending:
            return []
        first = pending[0]
        n_predict = first.n_predict
        gguf = first.gguf
        options = first.options
        for job in pending[1:]:
            if job.n_predict != n_predict:
                raise LlamaServerError("auto-batch n_predict mismatch")
        admitted = [job.request for job in pending]
        prompts = [job.prompt for job in pending]
        for active in admitted:
            self._assert_kv_bind(active, at="auto_batch")
        vram_opts = admitted[0].vram_options or self._vram_options(
            admitted[0].num_ctx, options
        )
        decode_before = self._kv_decode_steps_before()
        with self._model_swap.hold(gguf):
            srv = self._ensure_gguf_loaded_unlocked(
                gguf, num_ctx=admitted[0].num_ctx, options=vram_opts
            )
            from runtime.kv.bind import reserved_token_capacity

            policy = self._prefix_cache_policy(gguf, admitted[0].num_ctx)
            cache_rows = [self._prefix_cache_admission(a, policy) for a in admitted]
            raws = srv.completions_parallel(
                prompts,
                n_predict=n_predict,
                id_slots=[self._id_slot_for_request(a) for a in admitted],
                kv_token_budgets=[reserved_token_capacity(a) for a in admitted],
                kv_bind_reqs=admitted,
                kv_block_size=self.config.block_size,
                sampler=sampler_options_from_dict(options),
                cache_prompts=[row[0] for row in cache_rows],
                current_positions=[row[1] for row in cache_rows],
            )
            if len(raws) != len(admitted):
                raise LlamaServerError(
                    f"auto-batch result count mismatch: {len(admitted)} admitted, {len(raws)} results"
                )
            kv_steps = self._kv_decode_steps_after(decode_before)
            results: list[GenerateResult] = []
            for active, raw in zip(admitted, raws):
                self._record_subprocess_slot(active, raw)
                content = raw.get("content") or raw.get("response") or ""
                active.state = RequestState.DECODE
                results.append(
                    GenerateResult(
                        content=content,
                        request_id=active.request_id,
                        llama=raw,
                        vram_num_ctx=self._api_vram_num_ctx_from_request(active),
                        kv_decode_steps=kv_steps if len(admitted) == 1 else None,
                    )
                )
            return results

    def _stream_parallel_admitted(
        self, jobs: list[Any]
    ) -> Iterator[dict[str, Any]]:
        """Stream decode for N already-admitted requests (v37 stream auto-batch)."""
        from runtime.kv.stream_auto_batch import _PendingStreamJob

        pending = [j for j in jobs if isinstance(j, _PendingStreamJob)]
        if not pending:
            return iter(())
        first = pending[0]
        n_predict = first.n_predict
        gguf = first.gguf
        options = first.options
        for job in pending[1:]:
            if job.n_predict != n_predict:
                raise LlamaServerError("stream auto-batch n_predict mismatch")
        admitted = [job.request for job in pending]
        prompts = [job.prompt for job in pending]
        for active in admitted:
            self._assert_kv_bind(active, at="stream_auto_batch")
        vram_opts = admitted[0].vram_options or self._vram_options(
            admitted[0].num_ctx, options
        )
        decode_before = self._kv_decode_steps_before()
        with self._model_swap.hold(gguf):
            srv = self._ensure_gguf_loaded_unlocked(
                gguf, num_ctx=admitted[0].num_ctx, options=vram_opts
            )
            from runtime.kv.bind import reserved_token_capacity

            policy = self._prefix_cache_policy(gguf, admitted[0].num_ctx)
            cache_rows = [self._prefix_cache_admission(a, policy) for a in admitted]
            req_by_idx = {i: a for i, a in enumerate(admitted)}
            if len(pending) == 1:
                job = pending[0]
                active = job.request
                for chunk in srv.completion_stream(
                    job.prompt,
                    n_predict=n_predict,
                    id_slot=self._id_slot_for_request(active),
                    kv_token_budget=reserved_token_capacity(active),
                    kv_bind_req=active,
                    kv_block_size=self.config.block_size,
                    sampler=sampler_options_from_dict(options),
                    cache_prompt=cache_rows[0][0],
                    current_pos=cache_rows[0][1],
                    format_options=options,
                ):
                    out = dict(chunk)
                    out["request_id"] = active.request_id
                    out.setdefault("seq_idx", 0)
                    if bool(out.get("stop")):
                        self._record_subprocess_slot(active, out)
                        kv_steps = self._kv_decode_steps_after(decode_before)
                        if kv_steps is not None:
                            out["kv_decode_steps"] = kv_steps
                    yield out
                return

            for chunk in srv.completions_parallel_stream(
                prompts,
                n_predict=n_predict,
                id_slots=[self._id_slot_for_request(a) for a in admitted],
                kv_token_budgets=[reserved_token_capacity(a) for a in admitted],
                kv_bind_reqs=admitted,
                kv_block_size=self.config.block_size,
                sampler=sampler_options_from_dict(options),
                cache_prompts=[row[0] for row in cache_rows],
                current_positions=[row[1] for row in cache_rows],
            ):
                seq_idx = int(chunk.get("seq_idx", 0))
                active = req_by_idx.get(seq_idx)
                if active is None:
                    continue
                out = dict(chunk)
                out["request_id"] = active.request_id
                if bool(out.get("stop")):
                    active.state = RequestState.FINISHED
                    self._record_subprocess_slot(active, out)
                    kv_steps = self._kv_decode_steps_after(decode_before)
                    if kv_steps is not None:
                        out["kv_decode_steps"] = kv_steps
                yield out

    def generate(
        self,
        prompt: str,
        n_predict: int = 64,
        *,
        gguf: Path | None = None,
        num_ctx: int | None = None,
        options: dict | None = None,
        prefill_cancel: Any | None = None,
    ) -> GenerateResult:
        from runtime.kv.auto_batch import auto_batch_eligible

        active = self._admit_one(
            prompt, n_predict, gguf=gguf, num_ctx=num_ctx, options=options
        )
        try:
            if auto_batch_eligible(self, gguf=gguf, stream=False):
                return self._auto_batch.submit(
                    prompt=prompt,
                    request=active,
                    n_predict=n_predict,
                    gguf=gguf,
                    num_ctx=active.num_ctx,
                    options=options,
                )
            return self._generate_one_admitted(
                prompt,
                active,
                n_predict=n_predict,
                gguf=gguf,
                options=options,
                prefill_cancel=prefill_cancel,
            )
        finally:
            self.loop.complete(active)

    def _generate_progress_chunk(
        self,
        model: str,
        status: str,
        detail: str,
        *,
        created: str,
        position: int = 0,
        queue_depth: int = 0,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model": model,
            "created_at": created,
            "response": "",
            "done": False,
            "status": status,
            "detail": detail,
        }
        if position > 0:
            out["position"] = position
        if queue_depth > 0:
            out["queue_depth"] = queue_depth
        return out

    def _gguf_needs_load(self, gguf: Path | None) -> bool:
        if gguf is None:
            if self._server is None:
                return True
            return not self._server.is_running()
        try:
            key = str(gguf.resolve())
        except OSError:
            key = str(gguf)
        if self._server is not None:
            try:
                if (
                    str(self._server.model.resolve()) == key
                    and self._server.is_running()
                ):
                    return False
            except OSError:
                pass
        loaded = self._model_swap.stats().get("loaded_gguf")
        return loaded != key

    def stream_generate(
        self,
        prompt: str,
        model: str,
        n_predict: int = 64,
        *,
        gguf: Path | None = None,
        num_ctx: int | None = None,
        options: dict | None = None,
        prefill_cancel: Any | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield Ollama-shaped NDJSON objects for /api/generate streaming."""
        created = self._utc_now()
        yield self._generate_progress_chunk(
            model, "accepted", "request accepted", created=created
        )
        active = self._admit_one(
            prompt, n_predict, gguf=gguf, num_ctx=num_ctx, options=options
        )
        self._assert_kv_bind(active, at="stream_generate")
        vram_opts = active.vram_options or self._vram_options(active.num_ctx, options)
        vram_api = self._api_vram_num_ctx_from_request(active)
        decode_before = self._kv_decode_steps_before()
        from runtime.kv.native_decode_loop import PrefillAbortedError
        from runtime.kv.stream_auto_batch import stream_auto_batch_eligible

        try:
            if (
                prefill_cancel is None
                and stream_auto_batch_eligible(self, gguf=gguf, stream=True)
            ):
                yield self._generate_progress_chunk(
                    model, "generating", "generating response", created=created
                )
                saw_stop = False
                first = True
                for chunk in self._stream_auto_batch.iter_stream(
                    prompt=prompt,
                    request=active,
                    n_predict=n_predict,
                    gguf=gguf,
                    num_ctx=active.num_ctx,
                    options=options,
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
                        kv_steps = chunk.get("kv_decode_steps")
                        if kv_steps is None:
                            kv_steps = self._kv_decode_steps_after(decode_before)
                        if kv_steps is not None:
                            out["kv_decode_steps"] = kv_steps
                        out.update(metrics_from_llama_chunk(chunk))
                        out.update(merge_cache_tier_details(
                            {},
                            host=int(getattr(active, "cached_tokens_host", 0) or 0),
                            storage=int(getattr(active, "cached_tokens_storage", 0) or 0),
                            storage_backend=str(
                                getattr(active, "cached_tokens_storage_backend", "") or ""
                            ),
                            creation=int(getattr(active, "cache_creation_tokens", 0) or 0),
                        ))
                        # Why: runtime proxy skips Go chatPrompt; llama-server
                        # context-shifts silently. Admit-time prompt_tokens is
                        # the pre-shift size clients need on the done chunk.
                        out.update(detect_context_overflow(
                            out, active.num_ctx, len(active.prompt_tokens),
                        ))
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
                return

            with self._model_swap.hold(gguf):
                if self._gguf_needs_load(gguf):
                    yield self._generate_progress_chunk(
                        model,
                        "loading",
                        "loading model into memory",
                        created=created,
                    )
                srv = self._ensure_gguf_loaded_unlocked(
                    gguf, num_ctx=active.num_ctx, options=vram_opts
                )
                yield self._generate_progress_chunk(
                    model, "generating", "generating response", created=created
                )
                saw_stop = False
                first = True
                sampler = sampler_options_from_dict(options)
                from runtime.kv.bind import reserved_token_capacity

                policy = self._prefix_cache_policy(gguf, active.num_ctx)
                cache_ok, resume_pos = self._prefix_cache_admission(active, policy)
                for chunk in srv.completion_stream(
                    prompt,
                    n_predict=n_predict,
                    id_slot=self._id_slot_for_request(active),
                    kv_token_budget=reserved_token_capacity(active),
                    kv_bind_req=active,
                    kv_block_size=self.config.block_size,
                    sampler=sampler,
                    cache_prompt=cache_ok,
                    current_pos=resume_pos,
                    prefill_cancel=prefill_cancel,
                    format_options=options,
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
                        out.update(metrics_from_llama_chunk(chunk))
                        out.update(merge_cache_tier_details(
                            {},
                            host=int(getattr(active, "cached_tokens_host", 0) or 0),
                            storage=int(getattr(active, "cached_tokens_storage", 0) or 0),
                            storage_backend=str(
                                getattr(active, "cached_tokens_storage_backend", "") or ""
                            ),
                            creation=int(getattr(active, "cache_creation_tokens", 0) or 0),
                        ))
                        # Why: same as stream auto-batch path — explicit overflow
                        # for proxies that never ran Go-side truncate.
                        out.update(detect_context_overflow(
                            out, active.num_ctx, len(active.prompt_tokens),
                        ))
                        self._record_subprocess_slot(active, chunk)
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
        except PrefillAbortedError:
            yield {
                "model": model,
                "created_at": created,
                "response": "",
                "done": True,
                "done_reason": "cancelled",
            }
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
        prefill_cancel: Any | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield Ollama-shaped NDJSON objects for /api/chat streaming."""
        for chunk in self.stream_generate(
            prompt, model, n_predict, gguf=gguf, num_ctx=num_ctx, options=options,
            prefill_cancel=prefill_cancel,
        ):
            if chunk.get("status") and not chunk.get("response") and not chunk.get("done"):
                out: dict[str, Any] = {
                    "model": model,
                    "created_at": chunk.get("created_at", self._utc_now()),
                    "message": {"role": "assistant", "content": ""},
                    "done": False,
                    "status": chunk["status"],
                    "detail": chunk.get("detail", ""),
                }
                if chunk.get("position"):
                    out["position"] = chunk["position"]
                if chunk.get("queue_depth"):
                    out["queue_depth"] = chunk["queue_depth"]
                yield out
                continue
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
            if done:
                for key in (
                    "prompt_eval_count",
                    "prompt_eval_cached_count",
                    "cached_prompt_tokens",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                    # Why forward: stream_generate sets these; chat clients must
                    # see the same overflow signal as /api/generate.
                    "prompt_truncated",
                    "original_prompt_tokens",
                ):
                    if key in chunk:
                        out[key] = chunk[key]
            yield out

    def generate_batch(
        self,
        prompts: list[str],
        n_predict: int = 64,
        max_admit: int = 4,
        *,
        options: dict | None = None,
    ) -> list[GenerateResult]:
        """Admit up to ``max_admit`` requests in one tick; decode via C batch path (v27).

        WHY batch API: with ``llama_parallel_slots>1``, merging autoregressive rows
        into one ``run_batch_step`` avoids N separate ``llama_decode`` calls from
        Python. Prefill stays sequential per row (different prompts/resume positions).
        Falls back to per-row ``completion()`` when ``native_batch_decode_available()``
        is false (``ZEROLLAMA_KV_NATIVE_BATCH=0`` or ext not linked).
        """
        if not prompts:
            return []

        batch_opts = dict(options or {})
        if batch_opts.get("priority") is None:
            batch_opts["priority"] = "batch"
        priority = priority_from_options(batch_opts)
        from runtime.cache_bridge import (
            cache_pin_from_options,
            extract_cache_level,
            extract_cache_reset,
            extract_kv_load_tiers,
            extract_session_group,
            extract_session_parent,
        )
        from runtime.gpu_vram import resolve_vram_num_ctx
        from runtime.kv.tier_filter import parse_tier_filter
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
        parallel = self._effective_llama_parallel_slots()
        for idx, prompt in enumerate(prompts):
            req_id = uuid.uuid4().hex[:12]
            # L3 batch: per-row key from prompt_cache_keys[i]; strict index semantics
            # (see cache_bridge.resolve_cache_key_for_batch — no flat-key fallback).
            cache_key, kv_slot, slot_pinned, cache_salt = cache_pin_from_options(
                batch_opts,
                parallel=parallel,
                batch_index=idx,
            )
            req = Request(
                request_id=req_id,
                prompt_tokens=self._prompt_tokens_for_admit(prompt, batch_gguf),
                max_tokens=n_predict,
                priority=priority,
                gguf=resolved_batch_gguf,
                num_ctx=batch_num_ctx,
                vram_options=vram_opts,
                prompt_cache_key=cache_key,
                cache_salt=cache_salt,
                session_parent=extract_session_parent(batch_opts),
                session_group=extract_session_group(batch_opts),
                cache_reset=extract_cache_reset(batch_opts),
                cache_level=extract_cache_level(batch_opts),
                load_tier_filter=parse_tier_filter(extract_kv_load_tiers(batch_opts)),
                kv_slot=kv_slot,
                slot_pinned=slot_pinned,
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

                policy = self._prefix_cache_policy(gguf, batch_num_ctx)
                cache_rows = [self._prefix_cache_admission(a, policy) for a in admitted]
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
                    cache_prompts=[row[0] for row in cache_rows],
                    current_positions=[row[1] for row in cache_rows],
                )
                from runtime.vram_suggest import api_vram_num_ctx_meta

                batch_vram = api_vram_num_ctx_meta(batch_clamp_meta, batch_num_ctx)
                batch_kv_decode = (
                    self._kv_decode_steps_after(decode_before)
                    if len(admitted) == 1
                    else None
                )
                if len(admitted) != len(raws):
                    raise LlamaServerError(
                        f"batch result count mismatch: {len(admitted)} admitted, {len(raws)} results"
                    )
                for active, raw in zip(admitted, raws):
                    self._record_subprocess_slot(active, raw)
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

    def stream_generate_batch(
        self,
        prompts: list[str],
        n_predict: int = 64,
        max_admit: int = 4,
        *,
        options: dict | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Admit up to ``max_admit`` requests; stream via batched decode (v29).

        WHY stream batch: v27 returned full text only at the end; agent and sign-off
        callers need interleaved ``seq_idx``-tagged chunks from the same C
        ``run_batch_step`` hot path. Yields ``{request_id, seq_idx, response, done}``.
        """
        if not prompts:
            return iter(())

        batch_opts = dict(options or {})
        if batch_opts.get("priority") is None:
            batch_opts["priority"] = "batch"
        priority = priority_from_options(batch_opts)
        from runtime.cache_bridge import (
            cache_pin_from_options,
            extract_cache_level,
            extract_cache_reset,
            extract_kv_load_tiers,
            extract_session_group,
            extract_session_parent,
        )
        from runtime.gpu_vram import resolve_vram_num_ctx
        from runtime.kv.tier_filter import parse_tier_filter
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
        parallel = self._effective_llama_parallel_slots()
        for idx, prompt in enumerate(prompts):
            req_id = uuid.uuid4().hex[:12]
            cache_key, kv_slot, slot_pinned, cache_salt = cache_pin_from_options(
                batch_opts,
                parallel=parallel,
                batch_index=idx,
            )
            req = Request(
                request_id=req_id,
                prompt_tokens=self._prompt_tokens_for_admit(prompt, batch_gguf),
                max_tokens=n_predict,
                priority=priority,
                gguf=resolved_batch_gguf,
                num_ctx=batch_num_ctx,
                vram_options=vram_opts,
                prompt_cache_key=cache_key,
                cache_salt=cache_salt,
                session_parent=extract_session_parent(batch_opts),
                session_group=extract_session_group(batch_opts),
                cache_reset=extract_cache_reset(batch_opts),
                cache_level=extract_cache_level(batch_opts),
                load_tier_filter=parse_tier_filter(extract_kv_load_tiers(batch_opts)),
                kv_slot=kv_slot,
                slot_pinned=slot_pinned,
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
        req_by_idx = {i: a for i, a in enumerate(admitted)}
        for active in admitted:
            self._assert_kv_bind(active, at="stream_generate_batch")
        decode_before = self._kv_decode_steps_before()

        def _stream() -> Iterator[dict[str, Any]]:
            from runtime.kv.bind import reserved_token_capacity
            from runtime.vram_suggest import api_vram_num_ctx_meta

            batch_vram = api_vram_num_ctx_meta(batch_clamp_meta, batch_num_ctx)
            policy = self._prefix_cache_policy(gguf, batch_num_ctx)
            cache_rows = [self._prefix_cache_admission(a, policy) for a in admitted]
            try:
                with self._model_swap.hold(gguf):
                    srv = self._ensure_gguf_loaded_unlocked(
                        gguf, num_ctx=batch_num_ctx, options=vram_opts
                    )
                    for chunk in srv.completions_parallel_stream(
                        prompts_ordered,
                        n_predict=n_predict,
                        id_slots=[self._id_slot_for_request(a) for a in admitted],
                        kv_token_budgets=[
                            reserved_token_capacity(a) for a in admitted
                        ],
                        kv_bind_reqs=admitted,
                        kv_block_size=self.config.block_size,
                        sampler=sampler_options_from_dict(batch_opts),
                        cache_prompts=[row[0] for row in cache_rows],
                        current_positions=[row[1] for row in cache_rows],
                    ):
                        seq_idx = int(chunk.get("seq_idx", 0))
                        active = req_by_idx.get(seq_idx)
                        if active is None:
                            continue
                        content = chunk.get("content") or chunk.get("response") or ""
                        stop = bool(chunk.get("stop"))
                        if stop:
                            active.state = RequestState.FINISHED
                            self._record_subprocess_slot(active, chunk)
                        out: dict[str, Any] = {
                            "request_id": active.request_id,
                            "seq_idx": seq_idx,
                            "response": content,
                            "done": stop,
                        }
                        if batch_vram:
                            out["vram_num_ctx"] = batch_vram
                        if stop:
                            kv_steps = self._kv_decode_steps_after(decode_before)
                            if kv_steps is not None:
                                out["kv_decode_steps"] = kv_steps
                        yield out
            finally:
                for active in admitted:
                    self.loop.complete(active)

        return _stream()
