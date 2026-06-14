"""In-process GGUF forward via libllama.so (Phase 14)."""

from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from runtime.llama_args import (
    inprocess_speculative_requested,
    parse_llama_server_args,
)
from runtime.logutil import get_logger
from runtime.worker.libllama_ctypes import LlamaLoadedSession, get_lib, resolve_libllama_path
from runtime.worker.llama_server import LlamaServerError
from runtime.worker.sampler_options import SamplerOptions

logger = get_logger("llama_inprocess")


def _tensor_split_buffer(
    splits: tuple[float, ...] | None,
) -> ctypes.Array[ctypes.c_float] | None:
    if not splits:
        return None
    return (ctypes.c_float * len(splits))(*splits)


@dataclass
class LlamaInprocessWorker:
    """libllama forward in the runtime process (no llama-server subprocess)."""

    model: Path
    n_gpu_layers: int = -1
    main_gpu: int = 0
    lib_path: Path | None = None
    cpp_root: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8082
    parallel_slots: int = 1
    kv_pool_token_cap: int | None = None
    _session: LlamaLoadedSession | None = field(default=None, repr=False)
    _slot_cache_model_hash: str | None = field(default=None, repr=False)
    _tensor_split_buf: ctypes.Array[ctypes.c_float] | None = field(
        default=None, repr=False
    )
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_running(self) -> bool:
        return self._session is not None

    def start(self, extra_args: list[str] | None = None) -> None:
        with self._lock:
            hints = parse_llama_server_args(extra_args or [])
            if inprocess_speculative_requested(hints):
                raise LlamaServerError(
                    "speculative / draft models require subprocess backend "
                    "(ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess)"
                )
            if self._session is not None:
                logger.info(
                    "model %s already loaded in-process (gguf=%s)",
                    self.model.name,
                    self.model.resolve(),
                )
                return
            if not self.model.is_file():
                raise LlamaServerError(f"model not found: {self.model}")
            n_gpu = self.n_gpu_layers
            if hints.n_gpu_layers is not None:
                n_gpu = hints.n_gpu_layers
            resolve_libllama_path(self.lib_path, self.cpp_root)
            get_lib(self.lib_path, self.cpp_root)
            self._tensor_split_buf = _tensor_split_buffer(hints.tensor_split)
            n_seq = max(1, self.parallel_slots)
            from runtime.cache_bridge import (
                build_model_hash,
                cache_type_from_llama_argv,
                llama_cache_enabled,
            )

            slot_hash: str | None = None
            if llama_cache_enabled():
                ck, cv = cache_type_from_llama_argv(extra_args or [])
                slot_hash = build_model_hash(
                    target_model_path=self.model,
                    cache_type_k=ck,
                    cache_type_v=cv,
                )
                from runtime.cache_bridge import (
                    inprocess_disk_cache_enabled,
                    prepare_slot_cache_dir,
                )

                if inprocess_disk_cache_enabled() and slot_hash:
                    prepare_slot_cache_dir(slot_hash, evict=True)
            self._slot_cache_model_hash = slot_hash
            self._session = LlamaLoadedSession(
                self.model,
                n_gpu_layers=n_gpu,
                num_ctx=hints.num_ctx,
                n_seq_max=n_seq,
                lib_path=self.lib_path,
                cpp_root=self.cpp_root,
                load_hints=hints,
                default_main_gpu=self.main_gpu,
                tensor_split_buf=self._tensor_split_buf,
                kv_pool_token_cap=self.kv_pool_token_cap,
                slot_cache_model_hash=slot_hash,
            )
            logger.info(
                "model %s ready in-process (gguf=%s, n_gpu_layers=%s, n_ctx=%s, "
                "n_seq_max=%s, main_gpu=%s, split_mode=%s)",
                self.model.name,
                self.model.resolve(),
                n_gpu,
                hints.num_ctx,
                n_seq,
                hints.main_gpu if hints.main_gpu is not None else self.main_gpu,
                hints.split_mode or "layer",
            )

    def stop(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.close()
                self._session = None
            self._tensor_split_buf = None

    def _require_session(self) -> LlamaLoadedSession:
        if self._session is None:
            raise LlamaServerError("in-process llama is not loaded")
        return self._session

    def completion(
        self,
        prompt: str,
        n_predict: int | None = None,
        id_slot: int = -1,
        *,
        kv_token_budget: int | None = None,
        kv_bind_req: Any | None = None,
        kv_block_size: int = 16,
        sampler: SamplerOptions | None = None,
        cache_prompt: bool | None = None,
        current_pos: int | None = None,
    ) -> dict[str, Any]:
        # cache_prompt: RAM resume via pinned slot (v17); disk save when enabled (L3).
        _ = cache_prompt
        n_gen = 64 if n_predict is None or n_predict <= 0 else n_predict
        with self._lock:
            text = self._require_session().complete(
                prompt,
                n_predict=n_gen,
                stream=False,
                sampler=sampler,
                seq_id=id_slot,
                kv_token_budget=kv_token_budget,
                kv_bind_req=kv_bind_req,
                kv_block_size=kv_block_size,
                current_pos=current_pos,
            )
        assert isinstance(text, str)
        return {"content": text, "response": text, "stop": True}

    def completion_stream(
        self,
        prompt: str,
        n_predict: int | None = None,
        id_slot: int = -1,
        *,
        kv_token_budget: int | None = None,
        kv_bind_req: Any | None = None,
        kv_block_size: int = 16,
        sampler: SamplerOptions | None = None,
        cache_prompt: bool | None = None,
        current_pos: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        # cache_prompt: RAM resume via pinned slot (v17); disk save when enabled (L3).
        _ = cache_prompt
        n_gen = 64 if n_predict is None or n_predict <= 0 else n_predict

        def _gen() -> Iterator[dict[str, Any]]:
            with self._lock:
                stream = self._require_session().complete(
                    prompt,
                    n_predict=n_gen,
                    stream=True,
                    sampler=sampler,
                    seq_id=id_slot,
                    kv_token_budget=kv_token_budget,
                    kv_bind_req=kv_bind_req,
                    kv_block_size=kv_block_size,
                    current_pos=current_pos,
                )
                yield from stream

        return _gen()

    def completions_parallel(
        self,
        prompts: list[str],
        n_predict: int | None = None,
        *,
        id_slots: list[int] | None = None,
        kv_token_budgets: list[int] | None = None,
        kv_bind_reqs: list[Any] | None = None,
        kv_block_size: int = 16,
        sampler: SamplerOptions | None = None,
        cache_prompts: list[bool] | None = None,
        current_positions: list[int | None] | None = None,
    ) -> list[dict[str, Any]]:
        del cache_prompts
        if not prompts:
            return []
        n_gen = 64 if n_predict is None or n_predict <= 0 else n_predict
        slots = id_slots if id_slots is not None else [-1] * len(prompts)
        budgets = kv_token_budgets
        bind_reqs = kv_bind_reqs
        pos_list = current_positions

        def _slot(idx: int) -> int:
            return slots[idx] if idx < len(slots) else -1

        def _budget(idx: int) -> int | None:
            if budgets is None or idx >= len(budgets):
                return None
            b = budgets[idx]
            return b if b > 0 else None

        def _bind_req(idx: int) -> Any | None:
            if bind_reqs is None or idx >= len(bind_reqs):
                return None
            return bind_reqs[idx]

        def _current_pos(idx: int) -> int | None:
            if pos_list is None or idx >= len(pos_list):
                return None
            return pos_list[idx]

        if len(prompts) == 1:
            return [
                self.completion(
                    prompts[0],
                    n_predict=n_gen,
                    id_slot=_slot(0),
                    kv_token_budget=_budget(0),
                    kv_bind_req=_bind_req(0),
                    kv_block_size=kv_block_size,
                    sampler=sampler,
                    current_pos=_current_pos(0),
                )
            ]

        from runtime.kv.native_decode_loop import native_batch_decode_available

        if native_batch_decode_available():
            with self._lock:
                session = self._require_session()
                if session.n_seq_max > 1:
                    try:
                        texts = session.complete_parallel(
                            prompts,
                            n_predict=n_gen,
                            seq_ids=[_slot(i) for i in range(len(prompts))],
                            kv_token_budgets=[
                                _budget(i) for i in range(len(prompts))
                            ],
                            kv_bind_reqs=[_bind_req(i) for i in range(len(prompts))],
                            kv_block_size=kv_block_size,
                            current_positions=[
                                _current_pos(i) for i in range(len(prompts))
                            ],
                            sampler=sampler,
                        )
                        return [
                            {"content": t, "response": t, "stop": True} for t in texts
                        ]
                    except LlamaServerError as e:
                        logger.warning(
                            "parallel batch decode unavailable, falling back: %s",
                            e,
                        )

        return [
            self.completion(
                p,
                n_predict=n_gen,
                id_slot=_slot(i),
                kv_token_budget=_budget(i),
                kv_bind_req=_bind_req(i),
                kv_block_size=kv_block_size,
                sampler=sampler,
                current_pos=_current_pos(i),
            )
            for i, p in enumerate(prompts)
        ]

    def completions_parallel_stream(
        self,
        prompts: list[str],
        n_predict: int | None = None,
        *,
        id_slots: list[int] | None = None,
        kv_token_budgets: list[int] | None = None,
        kv_bind_reqs: list[Any] | None = None,
        kv_block_size: int = 16,
        sampler: SamplerOptions | None = None,
        cache_prompts: list[bool] | None = None,
        current_positions: list[int | None] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream N completions; batched decode when native batch is available (v29)."""
        del cache_prompts
        if not prompts:
            return iter(())
        n_gen = 64 if n_predict is None or n_predict <= 0 else n_predict
        slots = id_slots if id_slots is not None else [-1] * len(prompts)
        budgets = kv_token_budgets
        bind_reqs = kv_bind_reqs
        pos_list = current_positions

        def _slot(idx: int) -> int:
            return slots[idx] if idx < len(slots) else -1

        def _budget(idx: int) -> int | None:
            if budgets is None or idx >= len(budgets):
                return None
            b = budgets[idx]
            return b if b > 0 else None

        def _bind_req(idx: int) -> Any | None:
            if bind_reqs is None or idx >= len(bind_reqs):
                return None
            return bind_reqs[idx]

        def _current_pos(idx: int) -> int | None:
            if pos_list is None or idx >= len(pos_list):
                return None
            return pos_list[idx]

        if len(prompts) == 1:
            def _one() -> Iterator[dict[str, Any]]:
                for chunk in self.completion_stream(
                    prompts[0],
                    n_predict=n_gen,
                    id_slot=_slot(0),
                    kv_token_budget=_budget(0),
                    kv_bind_req=_bind_req(0),
                    kv_block_size=kv_block_size,
                    sampler=sampler,
                    current_pos=_current_pos(0),
                ):
                    out = dict(chunk)
                    out.setdefault("seq_idx", 0)
                    out.setdefault("seq_id", _slot(0))
                    yield out

            return _one()

        from runtime.kv.native_decode_loop import native_batch_decode_available

        if native_batch_decode_available():
            def _batch() -> Iterator[dict[str, Any]]:
                with self._lock:
                    session = self._require_session()
                    if session.n_seq_max > 1:
                        try:
                            yield from session.complete_parallel_stream(
                                prompts,
                                n_predict=n_gen,
                                seq_ids=[_slot(i) for i in range(len(prompts))],
                                kv_token_budgets=[
                                    _budget(i) for i in range(len(prompts))
                                ],
                                kv_bind_reqs=[
                                    _bind_req(i) for i in range(len(prompts))
                                ],
                                kv_block_size=kv_block_size,
                                current_positions=[
                                    _current_pos(i) for i in range(len(prompts))
                                ],
                                sampler=sampler,
                            )
                            return
                        except LlamaServerError as e:
                            logger.warning(
                                "parallel batch stream unavailable, falling back: %s",
                                e,
                            )

                for idx, prompt in enumerate(prompts):
                    for chunk in self.completion_stream(
                        prompt,
                        n_predict=n_gen,
                        id_slot=_slot(idx),
                        kv_token_budget=_budget(idx),
                        kv_bind_req=_bind_req(idx),
                        kv_block_size=kv_block_size,
                        sampler=sampler,
                        current_pos=_current_pos(idx),
                    ):
                        out = dict(chunk)
                        out.setdefault("seq_idx", idx)
                        out.setdefault("seq_id", _slot(idx))
                        yield out

            return _batch()

        def _sequential() -> Iterator[dict[str, Any]]:
            for idx, prompt in enumerate(prompts):
                for chunk in self.completion_stream(
                    prompt,
                    n_predict=n_gen,
                    id_slot=_slot(idx),
                    kv_token_budget=_budget(idx),
                    kv_bind_req=_bind_req(idx),
                    kv_block_size=kv_block_size,
                    sampler=sampler,
                    current_pos=_current_pos(idx),
                ):
                    out = dict(chunk)
                    out.setdefault("seq_idx", idx)
                    out.setdefault("seq_id", _slot(idx))
                    yield out

        return _sequential()
