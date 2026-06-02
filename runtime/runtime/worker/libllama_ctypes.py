"""ctypes bindings to pinned libllama.so (Phase 14).

WHY ctypes against the same tree as llama-server: no second vendored llama.cpp via pip;
operators already build ``build/bin/libllama.so`` with ``scripts/build_llama_server.sh``.
"""

from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path
from typing import Any, Iterator

from runtime.worker.llama_server import LlamaServerError
from runtime.worker.sampler_options import SamplerOptions

_lib_lock = threading.Lock()
_lib: ctypes.CDLL | None = None
_backend_init = False

LLAMA_TOKEN = ctypes.c_int32

# llama.h llama_split_mode
_SPLIT_MODE_ENUM = {
    "none": 0,
    "layer": 1,
    "row": 2,
    "tensor": 3,
}


class LlamaModelParams(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("_pad_main", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_direct_io", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
    ]


class LlamaContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("_pad_tail", ctypes.c_byte * 2),
        ("samplers", ctypes.c_void_p),
        ("n_samplers", ctypes.c_size_t),
    ]


class LlamaSamplerChainParams(ctypes.Structure):
    _fields_ = [("no_perf", ctypes.c_bool)]


class LlamaBatch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(LLAMA_TOKEN)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(ctypes.c_int32)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


def resolve_libllama_path(explicit: Path | None = None, cpp_root: Path | None = None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if p.is_file():
            return p
        raise LlamaServerError(f"LLAMA_CPP_LIB not found: {p}")
    env = os.environ.get("LLAMA_CPP_LIB", "").strip()
    if env:
        return resolve_libllama_path(Path(env))
    root = cpp_root
    if root is None:
        raw = os.environ.get("LLAMA_CPP_ROOT", "").strip()
        if raw:
            root = Path(raw)
        else:
            # runtime/runtime/worker -> zerollama repo root is parents[3]; llama.cpp is sibling.
            repo = Path(__file__).resolve().parents[3]
            root = repo.parent / "llama.cpp"
    for candidate in (
        root / "build" / "bin" / "libllama.so",
        root / "build" / "lib" / "libllama.so",
    ):
        if candidate.is_file():
            return candidate.resolve()
    raise LlamaServerError(
        f"libllama.so not found under {root}; set LLAMA_CPP_LIB or build llama.cpp"
    )


def _prepend_ld_library_path(libdir: Path) -> None:
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = str(libdir)
    if cur.startswith(prefix + ":") or cur == prefix:
        return
    os.environ["LD_LIBRARY_PATH"] = (
        f"{prefix}:{cur}" if cur else prefix
    )


def get_lib(
    lib_path: Path | None = None, cpp_root: Path | None = None
) -> ctypes.CDLL:
    global _lib, _backend_init
    with _lib_lock:
        if _lib is not None:
            return _lib
        path = resolve_libllama_path(lib_path, cpp_root)
        _prepend_ld_library_path(path.parent)
        _lib = ctypes.CDLL(str(path))
        _bind(_lib)
        if not _backend_init:
            _lib.llama_backend_init()
            if hasattr(_lib, "ggml_backend_load_all"):
                _lib.ggml_backend_load_all()
            _backend_init = True
        return _lib


def _bind(lib: ctypes.CDLL) -> None:
    lib.llama_backend_init.argtypes = []
    lib.llama_backend_init.restype = None

    lib.llama_model_default_params.argtypes = []
    lib.llama_model_default_params.restype = LlamaModelParams

    lib.llama_context_default_params.argtypes = []
    lib.llama_context_default_params.restype = LlamaContextParams

    lib.llama_sampler_chain_default_params.argtypes = []
    lib.llama_sampler_chain_default_params.restype = LlamaSamplerChainParams

    lib.llama_model_load_from_file.argtypes = [
        ctypes.c_char_p,
        LlamaModelParams,
    ]
    lib.llama_model_load_from_file.restype = ctypes.c_void_p

    lib.llama_init_from_model.argtypes = [ctypes.c_void_p, LlamaContextParams]
    lib.llama_init_from_model.restype = ctypes.c_void_p

    lib.llama_model_free.argtypes = [ctypes.c_void_p]
    lib.llama_model_free.restype = None
    lib.llama_free.argtypes = [ctypes.c_void_p]
    lib.llama_free.restype = None
    lib.llama_sampler_free.argtypes = [ctypes.c_void_p]
    lib.llama_sampler_free.restype = None

    lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
    lib.llama_model_get_vocab.restype = ctypes.c_void_p

    lib.llama_tokenize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(LLAMA_TOKEN),
        ctypes.c_int32,
        ctypes.c_bool,
        ctypes.c_bool,
    ]
    lib.llama_tokenize.restype = ctypes.c_int32

    lib.llama_token_to_piece.argtypes = [
        ctypes.c_void_p,
        LLAMA_TOKEN,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_bool,
    ]
    lib.llama_token_to_piece.restype = ctypes.c_int32

    lib.llama_batch_get_one.argtypes = [
        ctypes.POINTER(LLAMA_TOKEN),
        ctypes.c_int32,
    ]
    lib.llama_batch_get_one.restype = LlamaBatch

    lib.llama_batch_init.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.llama_batch_init.restype = LlamaBatch
    lib.llama_batch_free.argtypes = [LlamaBatch]
    lib.llama_batch_free.restype = None

    lib.llama_get_memory.argtypes = [ctypes.c_void_p]
    lib.llama_get_memory.restype = ctypes.c_void_p
    lib.llama_memory_seq_rm.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.llama_memory_seq_rm.restype = ctypes.c_bool
    lib.llama_memory_seq_pos_min.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.llama_memory_seq_pos_min.restype = ctypes.c_int32
    lib.llama_memory_seq_pos_max.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.llama_memory_seq_pos_max.restype = ctypes.c_int32

    lib.llama_decode.argtypes = [ctypes.c_void_p, LlamaBatch]
    lib.llama_decode.restype = ctypes.c_int32
    lib.llama_encode.argtypes = [ctypes.c_void_p, LlamaBatch]
    lib.llama_encode.restype = ctypes.c_int32

    lib.llama_model_has_encoder.argtypes = [ctypes.c_void_p]
    lib.llama_model_has_encoder.restype = ctypes.c_bool
    lib.llama_model_decoder_start_token.argtypes = [ctypes.c_void_p]
    lib.llama_model_decoder_start_token.restype = LLAMA_TOKEN
    lib.llama_vocab_bos.argtypes = [ctypes.c_void_p]
    lib.llama_vocab_bos.restype = LLAMA_TOKEN
    lib.llama_vocab_is_eog.argtypes = [ctypes.c_void_p, LLAMA_TOKEN]
    lib.llama_vocab_is_eog.restype = ctypes.c_bool

    lib.llama_sampler_chain_init.argtypes = [LlamaSamplerChainParams]
    lib.llama_sampler_chain_init.restype = ctypes.c_void_p
    lib.llama_sampler_chain_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.llama_sampler_chain_add.restype = None
    lib.llama_sampler_chain_n.argtypes = [ctypes.c_void_p]
    lib.llama_sampler_chain_n.restype = ctypes.c_int
    lib.llama_sampler_init_greedy.argtypes = []
    lib.llama_sampler_init_greedy.restype = ctypes.c_void_p
    lib.llama_sampler_init_dist.argtypes = [ctypes.c_uint32]
    lib.llama_sampler_init_dist.restype = ctypes.c_void_p
    lib.llama_sampler_init_top_k.argtypes = [ctypes.c_int32]
    lib.llama_sampler_init_top_k.restype = ctypes.c_void_p
    lib.llama_sampler_init_top_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
    lib.llama_sampler_init_top_p.restype = ctypes.c_void_p
    lib.llama_sampler_init_min_p.argtypes = [ctypes.c_float, ctypes.c_size_t]
    lib.llama_sampler_init_min_p.restype = ctypes.c_void_p
    lib.llama_sampler_init_typical.argtypes = [ctypes.c_float, ctypes.c_size_t]
    lib.llama_sampler_init_typical.restype = ctypes.c_void_p
    lib.llama_sampler_init_temp.argtypes = [ctypes.c_float]
    lib.llama_sampler_init_temp.restype = ctypes.c_void_p
    if hasattr(lib, "llama_sampler_init_penalties"):
        lib.llama_sampler_init_penalties.argtypes = [
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.llama_sampler_init_penalties.restype = ctypes.c_void_p
    lib.llama_sampler_sample.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    lib.llama_sampler_sample.restype = LLAMA_TOKEN

    lib.llama_n_ctx.argtypes = [ctypes.c_void_p]
    lib.llama_n_ctx.restype = ctypes.c_uint32


def tokenize(vocab: ctypes.c_void_p, text: str, *, add_special: bool = True) -> list[int]:
    lib = get_lib()
    raw = text.encode("utf-8")
    n = lib.llama_tokenize(
        vocab, raw, len(raw), None, 0, add_special, True
    )
    if n >= 0:
        raise LlamaServerError("llama_tokenize size probe failed")
    n_tokens = -n
    buf = (LLAMA_TOKEN * n_tokens)()
    got = lib.llama_tokenize(
        vocab, raw, len(raw), buf, n_tokens, add_special, True
    )
    if got < 0:
        raise LlamaServerError("llama_tokenize failed")
    return [int(buf[i]) for i in range(got)]


def token_to_piece(vocab: ctypes.c_void_p, token: int) -> str:
    lib = get_lib()
    buf = ctypes.create_string_buffer(256)
    n = lib.llama_token_to_piece(vocab, token, buf, len(buf), 0, True)
    if n < 0:
        return ""
    return buf.raw[:n].decode("utf-8", errors="replace")


def apply_load_hints_to_model_params(
    mparams: LlamaModelParams,
    hints: Any,
    *,
    n_gpu_layers: int,
    default_main_gpu: int = 0,
    tensor_split_buf: ctypes.Array[ctypes.c_float] | None = None,
) -> None:
    """Map parsed llama-server argv hints onto ctypes model params."""
    from runtime.llama_args import LlamaServerArgHints

    if not isinstance(hints, LlamaServerArgHints):
        hints = LlamaServerArgHints()
    mparams.n_gpu_layers = (
        hints.n_gpu_layers if hints.n_gpu_layers is not None else n_gpu_layers
    )
    mparams.main_gpu = (
        hints.main_gpu if hints.main_gpu is not None else default_main_gpu
    )
    sm = (hints.split_mode or "layer").strip().lower()
    mparams.split_mode = _SPLIT_MODE_ENUM.get(sm, _SPLIT_MODE_ENUM["layer"])
    if tensor_split_buf is not None:
        mparams.tensor_split = ctypes.cast(
            tensor_split_buf, ctypes.POINTER(ctypes.c_float)
        )


class LlamaVocabSession:
    """Tokenizer only (``vocab_only`` load) for render truncation without full weights."""

    def __init__(
        self,
        model_path: Path,
        *,
        lib_path: Path | None = None,
        cpp_root: Path | None = None,
        load_hints: Any | None = None,
        default_main_gpu: int = 0,
        tensor_split_buf: ctypes.Array[ctypes.c_float] | None = None,
    ) -> None:
        self.model_path = model_path.resolve()
        self._lib = get_lib(lib_path, cpp_root)
        mparams = self._lib.llama_model_default_params()
        mparams.vocab_only = True
        mparams.n_gpu_layers = 0
        apply_load_hints_to_model_params(
            mparams,
            load_hints,
            n_gpu_layers=0,
            default_main_gpu=default_main_gpu,
            tensor_split_buf=tensor_split_buf,
        )
        self._model = self._lib.llama_model_load_from_file(
            str(self.model_path).encode(), mparams
        )
        if not self._model:
            raise LlamaServerError(f"failed to load vocab for: {model_path}")
        self._vocab = self._lib.llama_model_get_vocab(self._model)

    def close(self) -> None:
        if self._model:
            self._lib.llama_model_free(self._model)
            self._model = None

    def tokenize_text(self, text: str, *, add_special: bool = True) -> list[int]:
        if not self._model:
            raise LlamaServerError("vocab session is closed")
        return tokenize(self._vocab, text, add_special=add_special)


def build_sampler_chain(
    lib: ctypes.CDLL, sampler: SamplerOptions | None
) -> ctypes.c_void_p:
    """Build a per-completion sampler chain (greedy when ``sampler`` is None)."""
    sparams = lib.llama_sampler_chain_default_params()
    smpl = lib.llama_sampler_chain_init(sparams)
    if not smpl:
        raise LlamaServerError("sampler chain init failed")

    def _add(s: ctypes.c_void_p | int | None) -> None:
        if s:
            lib.llama_sampler_chain_add(smpl, s)

    if sampler is None or sampler.greedy_only:
        _add(lib.llama_sampler_init_greedy())
        return smpl

    if hasattr(lib, "llama_sampler_init_penalties"):
        _add(
            lib.llama_sampler_init_penalties(
                ctypes.c_int32(sampler.repeat_last_n),
                ctypes.c_float(sampler.repeat_penalty),
                ctypes.c_float(sampler.frequency_penalty),
                ctypes.c_float(sampler.presence_penalty),
            )
        )
    if sampler.top_k > 0:
        _add(lib.llama_sampler_init_top_k(ctypes.c_int32(sampler.top_k)))
    if sampler.typical_p < 1.0:
        _add(
            lib.llama_sampler_init_typical(
                ctypes.c_float(sampler.typical_p), ctypes.c_size_t(1)
            )
        )
    _add(
        lib.llama_sampler_init_top_p(
            ctypes.c_float(sampler.top_p), ctypes.c_size_t(1)
        )
    )
    if sampler.min_p > 0.0:
        _add(
            lib.llama_sampler_init_min_p(
                ctypes.c_float(sampler.min_p), ctypes.c_size_t(1)
            )
        )
    _add(lib.llama_sampler_init_temp(ctypes.c_float(sampler.temperature)))
    _add(lib.llama_sampler_init_dist(ctypes.c_uint32(sampler.dist_seed)))
    return smpl


def _normalize_seq_id(seq_id: int, n_seq_max: int) -> int:
    sid = 0 if seq_id < 0 else int(seq_id)
    if sid < 0 or sid >= n_seq_max:
        raise LlamaServerError(f"seq_id {seq_id} out of range (n_seq_max={n_seq_max})")
    return sid


def _clear_sequence(lib: ctypes.CDLL, ctx: ctypes.c_void_p, seq_id: int) -> None:
    mem = lib.llama_get_memory(ctx)
    if not mem:
        return
    lib.llama_memory_seq_rm(mem, ctypes.c_int32(seq_id), ctypes.c_int32(-1), ctypes.c_int32(-1))


def sequence_kv_usage(
    lib: ctypes.CDLL, ctx: ctypes.c_void_p, seq_id: int
) -> tuple[int, int, int] | None:
    """Return ``(seq_id, pos_min, pos_max)`` or None if memory is unavailable."""
    mem = lib.llama_get_memory(ctx)
    if not mem:
        return None
    sid = ctypes.c_int32(seq_id)
    pmin = int(lib.llama_memory_seq_pos_min(mem, sid))
    pmax = int(lib.llama_memory_seq_pos_max(mem, sid))
    return (seq_id, pmin, pmax)


def _batch_from_tokens(
    lib: ctypes.CDLL,
    tokens: list[int],
    *,
    seq_id: int,
    n_seq_max: int,
    logits_last: bool,
) -> LlamaBatch:
    """Build a heap batch with an explicit sequence id (Phase 15 in-process KV)."""
    n = len(tokens)
    if n == 0:
        raise LlamaServerError("empty token batch")
    batch = lib.llama_batch_init(ctypes.c_int32(n), ctypes.c_int32(0), ctypes.c_int32(n_seq_max))
    batch.n_tokens = n
    for i, tok in enumerate(tokens):
        batch.token[i] = LLAMA_TOKEN(int(tok))
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = ctypes.c_int32(seq_id)
        batch.logits[i] = 1 if (not logits_last or i == n - 1) else 0
    return batch


class LlamaLoadedSession:
    """Model in VRAM; one shared context when ``n_seq_max > 1`` (multi-seq KV)."""

    def __init__(
        self,
        model_path: Path,
        *,
        n_gpu_layers: int = -1,
        num_ctx: int | None = None,
        n_seq_max: int = 1,
        lib_path: Path | None = None,
        cpp_root: Path | None = None,
        load_hints: Any | None = None,
        default_main_gpu: int = 0,
        tensor_split_buf: ctypes.Array[ctypes.c_float] | None = None,
        kv_pool_token_cap: int | None = None,
    ) -> None:
        self.model_path = model_path.resolve()
        self.n_gpu_layers = n_gpu_layers
        self.num_ctx = num_ctx
        self.n_seq_max = max(1, int(n_seq_max))
        self.kv_pool_token_cap = (
            int(kv_pool_token_cap) if kv_pool_token_cap and kv_pool_token_cap > 0 else None
        )
        self.lib_path = lib_path
        self.cpp_root = cpp_root
        self._lib = get_lib(lib_path, cpp_root)
        self._ctx: ctypes.c_void_p | None = None
        mparams = self._lib.llama_model_default_params()
        apply_load_hints_to_model_params(
            mparams,
            load_hints,
            n_gpu_layers=n_gpu_layers,
            default_main_gpu=default_main_gpu,
            tensor_split_buf=tensor_split_buf,
        )
        self._model = self._lib.llama_model_load_from_file(
            str(self.model_path).encode(), mparams
        )
        if not self._model:
            raise LlamaServerError(f"failed to load model: {model_path}")
        self._vocab = self._lib.llama_model_get_vocab(self._model)
        if self.n_seq_max > 1:
            self._init_shared_context(n_prompt_budget=512)

    def _init_shared_context(self, *, n_prompt_budget: int) -> None:
        if self._ctx is not None:
            return
        cparams = self._lib.llama_context_default_params()
        need = self.num_ctx if self.num_ctx and self.num_ctx > 0 else 4096
        if self.kv_pool_token_cap is not None:
            need = min(need, self.kv_pool_token_cap)
        cparams.n_ctx = max(int(need), n_prompt_budget + 64)
        cparams.n_seq_max = self.n_seq_max
        cparams.n_batch = min(cparams.n_ctx, max(n_prompt_budget, 512))
        self._ctx = self._lib.llama_init_from_model(self._model, cparams)
        if not self._ctx:
            raise LlamaServerError("llama_init_from_model failed (multi-seq)")

    def tokenize_text(self, text: str, *, add_special: bool = True) -> list[int]:
        if not self._model:
            raise LlamaServerError("model session is closed")
        return tokenize(self._vocab, text, add_special=add_special)

    def close(self) -> None:
        if self._ctx:
            self._lib.llama_free(self._ctx)
            self._ctx = None
        if self._model:
            self._lib.llama_model_free(self._model)
            self._model = None

    def _physical_check_after_decode(
        self,
        ctx: ctypes.c_void_p,
        seq_id: int,
        *,
        kv_bind_req: Any | None,
        kv_block_size: int,
    ) -> None:
        if kv_bind_req is None:
            return
        from runtime.kv.physical import usage_from_libllama, verify_after_decode

        usage = usage_from_libllama(self._lib, ctx, seq_id)
        verify_after_decode(
            kv_bind_req, usage, block_size=kv_block_size, at="inprocess_complete"
        )

    def complete(
        self,
        prompt: str,
        *,
        n_predict: int,
        stream: bool = False,
        sampler: SamplerOptions | None = None,
        seq_id: int = -1,
        kv_token_budget: int | None = None,
        kv_bind_req: Any | None = None,
        kv_block_size: int = 16,
    ) -> str | Iterator[dict[str, Any]]:
        tokens = self.tokenize_text(prompt, add_special=True)
        if not tokens:
            raise LlamaServerError("empty prompt after tokenize")
        n_prompt = len(tokens)
        smpl = build_sampler_chain(self._lib, sampler)

        if self.n_seq_max > 1 and self._ctx:
            sid = _normalize_seq_id(seq_id, self.n_seq_max)
            need_ctx = n_prompt + max(n_predict, 1)
            if kv_token_budget is not None and need_ctx > kv_token_budget:
                raise LlamaServerError(
                    f"prompt+generation ({need_ctx} tokens) exceeds PA KV reserve "
                    f"({kv_token_budget} tokens)"
                )
            if need_ctx > int(self._lib.llama_n_ctx(self._ctx)):
                raise LlamaServerError(
                    f"prompt+generation ({need_ctx} tokens) exceeds n_ctx "
                    f"({self._lib.llama_n_ctx(self._ctx)})"
                )
            _clear_sequence(self._lib, self._ctx, sid)

            def _release_smpl() -> None:
                self._lib.llama_sampler_free(smpl)

            if not stream:
                try:
                    text = _decode_non_stream(
                        self._lib,
                        self._model,
                        self._ctx,
                        self._vocab,
                        smpl,
                        tokens,
                        n_predict=n_predict,
                        seq_id=sid,
                        n_seq_max=self.n_seq_max,
                    )
                    self._physical_check_after_decode(
                        self._ctx,
                        sid,
                        kv_bind_req=kv_bind_req,
                        kv_block_size=kv_block_size,
                    )
                    return text
                finally:
                    _release_smpl()

            def _stream_gen() -> Iterator[dict[str, Any]]:
                try:
                    yield from _decode_stream(
                        self._lib,
                        self._model,
                        self._ctx,
                        self._vocab,
                        smpl,
                        tokens,
                        n_predict=n_predict,
                        seq_id=sid,
                        n_seq_max=self.n_seq_max,
                    )
                finally:
                    self._physical_check_after_decode(
                        self._ctx,
                        sid,
                        kv_bind_req=kv_bind_req,
                        kv_block_size=kv_block_size,
                    )
                    _release_smpl()

            return _stream_gen()

        cparams = self._lib.llama_context_default_params()
        need_ctx = (
            self.num_ctx
            if self.num_ctx and self.num_ctx > 0
            else n_prompt + max(n_predict, 1)
        )
        need_ctx = max(need_ctx, n_prompt + max(n_predict, 1))
        if kv_token_budget is not None and need_ctx > kv_token_budget:
            raise LlamaServerError(
                f"prompt+generation ({need_ctx} tokens) exceeds PA KV reserve "
                f"({kv_token_budget} tokens)"
            )
        cparams.n_ctx = need_ctx
        cparams.n_batch = min(cparams.n_ctx, max(n_prompt, 512))
        ctx = self._lib.llama_init_from_model(self._model, cparams)
        if not ctx:
            raise LlamaServerError("llama_init_from_model failed")
        sid = _normalize_seq_id(seq_id, 1)

        def _release() -> None:
            self._lib.llama_sampler_free(smpl)
            self._lib.llama_free(ctx)

        if not stream:
            try:
                text = _decode_non_stream(
                    self._lib,
                    self._model,
                    ctx,
                    self._vocab,
                    smpl,
                    tokens,
                    n_predict=n_predict,
                    seq_id=sid,
                    n_seq_max=1,
                )
                self._physical_check_after_decode(
                    ctx,
                    sid,
                    kv_bind_req=kv_bind_req,
                    kv_block_size=kv_block_size,
                )
                return text
            finally:
                _release()

        def _stream_gen() -> Iterator[dict[str, Any]]:
            try:
                yield from _decode_stream(
                    self._lib,
                    self._model,
                    ctx,
                    self._vocab,
                    smpl,
                    tokens,
                    n_predict=n_predict,
                    seq_id=sid,
                    n_seq_max=1,
                )
            finally:
                self._physical_check_after_decode(
                    ctx,
                    sid,
                    kv_bind_req=kv_bind_req,
                    kv_block_size=kv_block_size,
                )
                _release()

        return _stream_gen()


def generate_text(
    model_path: Path,
    prompt: str,
    *,
    n_predict: int,
    n_gpu_layers: int = -1,
    num_ctx: int | None = None,
    lib_path: Path | None = None,
    cpp_root: Path | None = None,
    stream: bool = False,
) -> str | Iterator[dict[str, Any]]:
    """One-shot load+decode (tests); production uses ``LlamaLoadedSession``."""
    session = LlamaLoadedSession(
        model_path,
        n_gpu_layers=n_gpu_layers,
        num_ctx=num_ctx,
        lib_path=lib_path,
        cpp_root=cpp_root,
    )
    try:
        return session.complete(prompt, n_predict=n_predict, stream=stream)
    finally:
        session.close()


def _decode_non_stream(
    lib: ctypes.CDLL,
    model: ctypes.c_void_p,
    ctx: ctypes.c_void_p,
    vocab: ctypes.c_void_p,
    smpl: ctypes.c_void_p,
    prompt_tokens: list[int],
    *,
    n_predict: int,
    seq_id: int = 0,
    n_seq_max: int = 1,
) -> str:
    pieces: list[str] = []
    for chunk in _decode_stream(
        lib,
        model,
        ctx,
        vocab,
        smpl,
        prompt_tokens,
        n_predict=n_predict,
        seq_id=seq_id,
        n_seq_max=n_seq_max,
    ):
        if chunk.get("content"):
            pieces.append(str(chunk["content"]))
    return "".join(pieces)


def _decode_stream(
    lib: ctypes.CDLL,
    model: ctypes.c_void_p,
    ctx: ctypes.c_void_p,
    vocab: ctypes.c_void_p,
    smpl: ctypes.c_void_p,
    prompt_tokens: list[int],
    *,
    n_predict: int,
    seq_id: int = 0,
    n_seq_max: int = 1,
) -> Iterator[dict[str, Any]]:
    from runtime.kv.native_decode import record_decode_step

    n_prompt = len(prompt_tokens)
    limit = max(0, n_predict)
    use_seq_batch = n_seq_max > 1

    def _make_batch(tokens: list[int], *, logits_last: bool) -> LlamaBatch:
        if use_seq_batch:
            return _batch_from_tokens(
                lib,
                tokens,
                seq_id=seq_id,
                n_seq_max=n_seq_max,
                logits_last=logits_last,
            )
        arr = (LLAMA_TOKEN * len(tokens))(*tokens)
        return lib.llama_batch_get_one(arr, len(tokens))

    batch = _make_batch(prompt_tokens, logits_last=False)
    batch_owned = use_seq_batch

    if lib.llama_model_has_encoder(model):
        if lib.llama_encode(ctx, batch) != 0:
            if batch_owned:
                lib.llama_batch_free(batch)
            raise LlamaServerError("llama_encode failed")
        record_decode_step(1)
        start = lib.llama_model_decoder_start_token(model)
        if int(start) == -1:
            start = lib.llama_vocab_bos(vocab)
        if batch_owned:
            lib.llama_batch_free(batch)
        batch = _make_batch([int(start)], logits_last=True)
        batch_owned = use_seq_batch

    n_pos = 0

    def _emit_piece(piece: str, *, stop: bool) -> dict[str, Any]:
        return {"content": piece, "response": piece, "stop": stop}

    try:
        while n_pos + batch.n_tokens < n_prompt + limit:
            if lib.llama_decode(ctx, batch) != 0:
                raise LlamaServerError("llama_decode failed")
            record_decode_step(1)
            n_pos += batch.n_tokens
            new_id = int(lib.llama_sampler_sample(smpl, ctx, -1))
            if lib.llama_vocab_is_eog(vocab, new_id):
                yield _emit_piece("", stop=True)
                return
            piece = token_to_piece(vocab, new_id)
            yield _emit_piece(piece, stop=False)
            if batch_owned:
                lib.llama_batch_free(batch)
            batch = _make_batch([new_id], logits_last=True)
            batch_owned = use_seq_batch
        yield _emit_piece("", stop=True)
    finally:
        if batch_owned:
            lib.llama_batch_free(batch)
