"""In-process forward via llama-cpp-python wheel (Phase 14 optional backend)."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from runtime.llama_args import (
    LlamaServerArgHints,
    inprocess_speculative_requested,
    parse_llama_server_args,
    split_mode_to_llama_cpp_int,
)
from runtime.logutil import get_logger
from runtime.worker.llama_server import LlamaServerError
from runtime.worker.sampler_options import SamplerOptions, sampler_to_llama_cpp_kwargs

logger = get_logger("llama_cpp_python")


def llama_cpp_n_gpu_layers(n_gpu_layers: int, hints_n_gpu_layers: int | None) -> int:
    """Resolve GPU layer count for llama-cpp-python.

    The pip wheel can abort on ``create_completion`` with GPU offload on some
  hosts/builds (``free(): invalid pointer``) while ctypes ``libllama.so`` works.
    Default **CPU** (0 layers) unless ``-ngl`` / hints or
    ``ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS`` is set.
    """
    if hints_n_gpu_layers is not None:
        return max(0, hints_n_gpu_layers)
    raw = os.environ.get("ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS", "").strip()
    if raw:
        try:
            n = int(raw)
        except ValueError as e:
            raise LlamaServerError(
                f"invalid ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS={raw!r} (integer layers)"
            ) from e
        if n < 0:
            logger.warning(
                "ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS=%s: wheel GPU may abort on this "
                "host; using CPU (0). Use ctypes inprocess for GPU sign-off.",
                raw,
            )
            return 0
        return n
    if n_gpu_layers < 0:
        return 0
    return max(0, n_gpu_layers)


def llama_cpp_wheel_health(
    worker: Any | None = None,
    *,
    default_n_gpu_layers: int = -1,
) -> dict[str, Any]:
    """Operator-facing wheel GPU offload state for ``/health``."""
    env_raw = os.environ.get("ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS", "").strip()
    loaded_layers = (
        getattr(worker, "_loaded_n_gpu_layers", None) if worker is not None else None
    )
    if loaded_layers is not None:
        effective = int(loaded_layers)
    else:
        effective = llama_cpp_n_gpu_layers(default_n_gpu_layers, None)
    return {
        "n_gpu_layers": effective,
        "loaded": loaded_layers is not None,
        "gpu_mode": "gpu" if effective > 0 else "cpu",
        "env_n_gpu_layers": env_raw or None,
    }


def _import_llama():
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise LlamaServerError(
            "llama-cpp-python is not installed; "
            "pip install llama-cpp-python (CUDA wheel) or use "
            "ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess|subprocess"
        ) from e
    return Llama


def build_llama_cpp_load_kwargs(
    model_path: Path,
    hints: LlamaServerArgHints,
    *,
    n_gpu_layers: int,
    main_gpu: int,
    num_ctx: int | None = None,
    vocab_only: bool = False,
) -> dict[str, Any]:
    """Shared Llama() kwargs for full load and vocab-only tokenize."""
    if vocab_only:
        n_gpu = 0
    else:
        n_gpu = llama_cpp_n_gpu_layers(n_gpu_layers, hints.n_gpu_layers)
    if vocab_only:
        ctx = 512
    elif num_ctx is not None and num_ctx > 0:
        ctx = num_ctx
    elif hints.num_ctx is not None and hints.num_ctx > 0:
        ctx = hints.num_ctx
    else:
        ctx = 4096
    kwargs: dict[str, Any] = {
        "model_path": str(model_path.resolve()),
        "n_gpu_layers": n_gpu,
        "n_ctx": ctx,
        "verbose": False,
        "vocab_only": vocab_only,
        "split_mode": split_mode_to_llama_cpp_int(hints.split_mode),
    }
    if hints.main_gpu is not None:
        kwargs["main_gpu"] = hints.main_gpu
    elif main_gpu:
        kwargs["main_gpu"] = main_gpu
    if hints.tensor_split and not vocab_only:
        kwargs["tensor_split"] = list(hints.tensor_split)
    return kwargs


class LlamaCppVocabSession:
    """Vocab-only load via llama-cpp-python (render tokenize without libllama.so)."""

    def __init__(self, model_path: Path, *, main_gpu: int = 0) -> None:
        self.model_path = model_path.resolve()
        Llama = _import_llama()
        hints = LlamaServerArgHints()
        self._llama = Llama(
            **build_llama_cpp_load_kwargs(
                self.model_path,
                hints,
                n_gpu_layers=0,
                main_gpu=main_gpu,
                vocab_only=True,
            )
        )

    def tokenize_text(self, text: str, *, add_special: bool = True) -> list[int]:
        return self._llama.tokenize(
            text.encode("utf-8"), add_bos=add_special, special=False
        )

    def close(self) -> None:
        llama = self._llama
        self._llama = None
        if llama is not None:
            try:
                llama.close()
            except Exception:
                pass


@dataclass
class LlamaCppPythonWorker:
    """GGUF forward using the llama-cpp-python package (no local libllama.so build)."""

    model: Path
    n_gpu_layers: int = -1
    main_gpu: int = 0
    host: str = "127.0.0.1"
    port: int = 8082
    _llama: Any = field(default=None, repr=False)
    _n_ctx: int | None = field(default=None, repr=False)
    _loaded_n_gpu_layers: int | None = field(default=None, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_running(self) -> bool:
        return self._llama is not None

    def start(self, extra_args: list[str] | None = None) -> None:
        with self._lock:
            hints = parse_llama_server_args(extra_args or [])
            if inprocess_speculative_requested(hints):
                raise LlamaServerError(
                    "speculative / draft models require subprocess backend "
                    "(ZEROLLAMA_RUNTIME_LLAMA_BACKEND=subprocess)"
                )
            if self._llama is not None:
                logger.info(
                    "model %s already loaded (llama-cpp-python, gguf=%s)",
                    self.model.name,
                    self.model.resolve(),
                )
                return
            if not self.model.is_file():
                raise LlamaServerError(f"model not found: {self.model}")
            Llama = _import_llama()
            kwargs = build_llama_cpp_load_kwargs(
                self.model,
                hints,
                n_gpu_layers=self.n_gpu_layers,
                main_gpu=self.main_gpu,
            )
            self._llama = Llama(**kwargs)
            self._n_ctx = int(kwargs["n_ctx"])
            self._loaded_n_gpu_layers = int(kwargs["n_gpu_layers"])
            if kwargs["n_gpu_layers"] == 0 and (
                hints.n_gpu_layers is None and self.n_gpu_layers < 0
            ):
                logger.warning(
                    "llama-cpp-python using CPU (n_gpu_layers=0); set -ngl on "
                    "load args or ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS for GPU offload"
                )
            logger.info(
                "model %s ready (llama-cpp-python, gguf=%s, n_gpu_layers=%s, "
                "n_ctx=%s, split_mode=%s)",
                self.model.name,
                self.model.resolve(),
                kwargs["n_gpu_layers"],
                self._n_ctx,
                hints.split_mode or "layer",
            )

    def stop(self) -> None:
        with self._lock:
            llama = self._llama
            self._llama = None
            self._n_ctx = None
            self._loaded_n_gpu_layers = None
            if llama is not None:
                try:
                    llama.close()
                except Exception:
                    pass

    def _require_llama(self) -> Any:
        if self._llama is None:
            raise LlamaServerError("llama-cpp-python model is not loaded")
        return self._llama

    def tokenize_text(self, text: str, *, add_special: bool = True) -> list[int]:
        llama = self._require_llama()
        return llama.tokenize(text.encode("utf-8"), add_bos=add_special, special=False)

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
    ) -> dict[str, Any]:
        # L3 engine passes cache_prompt for subprocess; wheel backend has no slot bridge yet.
        del id_slot, kv_token_budget, kv_bind_req, kv_block_size, cache_prompt
        n_gen = 64 if n_predict is None or n_predict <= 0 else n_predict
        with self._lock:
            out = self._require_llama().create_completion(
                prompt,
                max_tokens=n_gen,
                stream=False,
                **sampler_to_llama_cpp_kwargs(sampler),
            )
        text = out["choices"][0].get("text") or ""
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
    ) -> Iterator[dict[str, Any]]:
        # L3 engine passes cache_prompt for subprocess; wheel backend has no slot bridge yet.
        del id_slot, kv_token_budget, kv_bind_req, kv_block_size, cache_prompt
        n_gen = 64 if n_predict is None or n_predict <= 0 else n_predict

        def _gen() -> Iterator[dict[str, Any]]:
            with self._lock:
                stream = self._require_llama().create_completion(
                    prompt,
                    max_tokens=n_gen,
                    stream=True,
                    **sampler_to_llama_cpp_kwargs(sampler),
                )
                for chunk in stream:
                    piece = chunk["choices"][0].get("text") or ""
                    if piece:
                        yield {"content": piece, "response": piece, "stop": False}
                yield {"content": "", "response": "", "stop": True}

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
    ) -> list[dict[str, Any]]:
        del id_slots, kv_token_budgets, kv_bind_reqs, kv_block_size
        if not prompts:
            return []
        return [
            self.completion(p, n_predict=n_predict, sampler=sampler) for p in prompts
        ]
