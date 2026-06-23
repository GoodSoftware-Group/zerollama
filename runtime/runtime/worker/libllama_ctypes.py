"""ctypes bindings to pinned libllama.so (Phase 14).

WHY ctypes against the same tree as llama-server: no second vendored llama.cpp via pip;
operators already build ``build/bin/libllama.so`` with ``scripts/build_llama_server.sh``.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from runtime.cache_bridge import slot_resume_owner_key
from runtime.worker.llama_server import LlamaServerError
from runtime.worker.sampler_options import SamplerOptions

_llama_log = logging.getLogger(__name__)
_lib: ctypes.CDLL | None = None
_lib_lock = threading.Lock()
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
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    for candidate in (
        root / "build" / "bin" / f"libllama{suffix}",
        root / "build" / "lib" / f"libllama{suffix}",
        # Linux layout fallback when probing a non-darwin host path.
        root / "build" / "bin" / "libllama.so",
        root / "build" / "lib" / "libllama.so",
    ):
        if candidate.is_file():
            return candidate.resolve()
    raise LlamaServerError(
        f"libllama{suffix} not found under {root}; set LLAMA_CPP_LIB or build llama.cpp"
    )


def _prepend_ld_library_path(libdir: Path) -> None:
    key = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    cur = os.environ.get(key, "")
    prefix = str(libdir)
    if cur.startswith(prefix + ":") or cur == prefix:
        return
    os.environ[key] = f"{prefix}:{cur}" if cur else prefix


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

    if hasattr(lib, "llama_state_seq_save_file"):
        lib.llama_state_seq_save_file.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.POINTER(LLAMA_TOKEN),
            ctypes.c_size_t,
        ]
        lib.llama_state_seq_save_file.restype = ctypes.c_size_t
    if hasattr(lib, "llama_state_seq_load_file"):
        lib.llama_state_seq_load_file.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.POINTER(LLAMA_TOKEN),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.llama_state_seq_load_file.restype = ctypes.c_size_t


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


def _ctx_ptr(ctx: ctypes.c_void_p | None) -> int | None:
    """Extract integer address from ctypes ctx for native invalidate calls.

    WHY: ``bump_decode_graph_epoch`` passes this to ``invalidate_cuda_graphs`` so
    ggml clears captured CUDA graphs when KV slots are cleared — epoch alone does
    not reach ggml's internal graph map.
    """
    if not ctx:
        return None
    val = int(ctypes.cast(ctx, ctypes.c_void_p).value or 0)
    return val if val else None


def _save_slot_cache_disk(
    lib: ctypes.CDLL,
    ctx: ctypes.c_void_p,
    *,
    seq_id: int,
    model_hash: str,
) -> int:
    """Persist sequence KV to ``slot_<id>_0.bin`` (L3 in-process disk parity).

    WHY use live pos_max rather than the caller's prompt_tokens: the sequence KV
    accumulates both prompt AND generated tokens over multi-turn exchanges.  Saving
    only the input prompt for the *current* turn under-reports the token count stored
    in the blob, breaking prefix-match on the next restore.  We derive the true count
    from the live KV via ``llama_memory_seq_pos_max`` (pos_max + 1 == n_tokens).
    """
    from runtime.cache_bridge import (
        inprocess_disk_cache_enabled,
        prepare_slot_cache_dir,
        slot_cache_file_path,
    )

    # Env gate only; ``LlamaLoadedSession.slot_cache_disk_persist`` enforces policy.
    if not inprocess_disk_cache_enabled() or not model_hash:
        return 0
    if not hasattr(lib, "llama_state_seq_save_file"):
        return 0
    usage = sequence_kv_usage(lib, ctx, seq_id)
    if usage is None:
        return 0
    _seq_id, _pmin, pmax = usage
    if pmax < 0:
        return 0
    n = pmax + 1
    path = slot_cache_file_path(model_hash, seq_id)
    prepare_slot_cache_dir(model_hash)
    buf = (LLAMA_TOKEN * n)()
    written = int(
        lib.llama_state_seq_save_file(
            ctx,
            str(path).encode(),
            ctypes.c_int32(seq_id),
            buf,
            ctypes.c_size_t(n),
        )
    )
    return written


def _try_restore_slot_cache_disk(
    lib: ctypes.CDLL,
    ctx: ctypes.c_void_p,
    *,
    seq_id: int,
    model_hash: str,
    token_capacity: int,
) -> int:
    """Load ``slot_<id>_0.bin`` when present; returns restored token count or 0."""
    from runtime.cache_bridge import inprocess_disk_cache_enabled, slot_cache_file_path

    # Env gate only; ``LlamaLoadedSession.slot_cache_disk_persist`` enforces policy.
    if not inprocess_disk_cache_enabled() or not model_hash:
        return 0
    if not hasattr(lib, "llama_state_seq_load_file"):
        return 0
    path = slot_cache_file_path(model_hash, seq_id)
    if not path.is_file():
        return 0
    cap = max(1, int(token_capacity))
    out_buf = (LLAMA_TOKEN * cap)()
    n_out = ctypes.c_size_t(0)
    nread = int(
        lib.llama_state_seq_load_file(
            ctx,
            str(path).encode(),
            ctypes.c_int32(seq_id),
            out_buf,
            ctypes.c_size_t(cap),
            ctypes.byref(n_out),
        )
    )
    if nread == 0 or int(n_out.value) == 0:
        return 0
    return int(n_out.value)


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
    pos_start: int = 0,
) -> LlamaBatch:
    """Build a heap batch with explicit seq_id and positions.

    ``llama_batch_get_one`` stores a pointer to caller-owned token memory (UAF).
    ``llama_batch_init`` allocates ``pos[]`` but leaves it uninitialized; llama.cpp
    only auto-fills positions when ``batch.pos`` is NULL, so we set ``pos_start + i``.
    """
    n = len(tokens)
    if n == 0:
        raise LlamaServerError("empty token batch")
    batch = lib.llama_batch_init(ctypes.c_int32(n), ctypes.c_int32(0), ctypes.c_int32(n_seq_max))
    batch.n_tokens = n
    for i, tok in enumerate(tokens):
        batch.token[i] = LLAMA_TOKEN(int(tok))
        batch.pos[i] = pos_start + i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = ctypes.c_int32(seq_id)
        batch.logits[i] = 1 if (not logits_last or i == n - 1) else 0
    return batch


class LlamaLoadedSession:
    """Model in VRAM; one shared context when ``n_seq_max > 1`` (multi-seq KV).

    Phase 15 v16b adds ``_seq_last_owner`` so ``complete()`` can skip
    ``_clear_sequence`` only when the *same owner* last wrote a given slot.
    WHY needed: ``decode_pos > 0`` (v16) is a necessary but not sufficient
    condition — a different session may have written that slot after the first
    one completed.  v17 uses ``prompt_cache_key`` for L3 pinned turns (new
    ``request_id`` every HTTP call).
    """

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
        slot_cache_model_hash: str | None = None,
        slot_cache_disk_persist: bool = True,
        kv_cache_spec: Any | None = None,
    ) -> None:
        self.model_path = model_path.resolve()
        self.n_gpu_layers = n_gpu_layers
        self.num_ctx = num_ctx
        self.n_seq_max = max(1, int(n_seq_max))
        self.kv_pool_token_cap = (
            int(kv_pool_token_cap) if kv_pool_token_cap and kv_pool_token_cap > 0 else None
        )
        self.slot_cache_model_hash = slot_cache_model_hash
        self.slot_cache_disk_persist = slot_cache_disk_persist
        self.kv_cache_spec = kv_cache_spec
        self.lib_path = lib_path
        self.cpp_root = cpp_root
        self._lib = get_lib(lib_path, cpp_root)
        self._ctx: ctypes.c_void_p | None = None
        # WHY _seq_last_owner: guards the skip-clear path in complete().
        # We only skip _clear_sequence when the same owner last wrote this
        # slot — otherwise a new session would resume into stale KV from a
        # prior (different) owner.  Keyed by normalised seq_id (int).
        # v17: owner is prompt_cache_key for L3 pinned sessions, else request_id.
        # Reset in close() so teardown/reload never inherits stale owners.
        self._seq_last_owner: dict[int, str] = {}
        # WHY: libllama + Metal are not safe for concurrent decode on the same
        # model (even with per-request contexts when n_seq_max==1).  Go proxy
        # broker + runtime smokes can overlap two /api/generate calls.
        self._infer_lock = threading.RLock()
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
        # WHY max(..., n_seq_max): continuous batch decode feeds one row per slot.
        cparams.n_batch = min(
            cparams.n_ctx, max(n_prompt_budget, self.n_seq_max, 512)
        )
        self._ctx = self._lib.llama_init_from_model(self._model, cparams)
        if not self._ctx:
            raise LlamaServerError("llama_init_from_model failed (multi-seq)")

    def tokenize_text(self, text: str, *, add_special: bool = True) -> list[int]:
        if not self._model:
            raise LlamaServerError("model session is closed")
        return tokenize(self._vocab, text, add_special=add_special)

    def resume_owner_snapshot(self) -> dict[int, str]:
        """Per-slot resume owner keys for operator /health (Phase 15 v18).

        WHY expose: L3 multi-turn resume is invisible without inspecting
        ``_seq_last_owner``; operators debugging prefix reuse need slot→owner map.
        """
        return dict(self._seq_last_owner)

    def close(self) -> None:
        with self._infer_lock:
            # WHY bump_all on close: model unload invalidates every slot's KV; global
            # epoch ensures future capture keys miss even for never-bumped slots.
            from runtime.decode_graph_policy import bump_all_decode_graph_epochs

            bump_all_decode_graph_epochs(
                reason="session_close",
                ctx_ptr=_ctx_ptr(self._ctx),
            )
            if self._ctx:
                self._lib.llama_free(self._ctx)
                self._ctx = None
            if self._model:
                self._lib.llama_model_free(self._model)
                self._model = None
            # WHY clear owners on teardown: a future in-place reload must not match
            # resume against KV from a prior model/context on the same slot index.
            self._seq_last_owner.clear()

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
        from runtime.kv.physical import physical_strict_enabled, usage_from_libllama, verify_after_decode

        usage = usage_from_libllama(self._lib, ctx, seq_id)
        verify_after_decode(
            kv_bind_req, usage, block_size=kv_block_size, at="inprocess_complete"
        )
        self._tensor_probe_after_decode(
            ctx, seq_id, kv_bind_req, usage,
            strict=physical_strict_enabled(),
        )

    def _tensor_probe_after_decode(
        self,
        ctx: ctypes.c_void_p,
        seq_id: int,
        kv_bind_req: Any | None,
        usage: Any | None,
        *,
        strict: bool = False,
    ) -> None:
        """v19 accounting bind: PA page table vs live llama seq cells.

        WHY no-op when ext not linked: ``run_tensor_probe`` returns ``None``
        when ``ZEROLLAMA_KV_DECODE_LOOP`` was not set at build time; the
        method exits silently so non-linked operators are not affected.
        """
        if kv_bind_req is None or usage is None:
            return
        slot = getattr(kv_bind_req, "kv_slot", None)
        if slot is None or slot < 0:
            return
        from runtime.kv.tensor_probe import run_tensor_probe

        try:
            ctx_ptr = int(ctx.value) if ctx is not None else 0
        except (TypeError, ValueError, AttributeError):
            return
        probe = run_tensor_probe(ctx_ptr, seq_id, int(slot))
        if not probe:
            return
        if probe.get("tensor_pages_bound"):
            _llama_log.debug(
                "KV tensor bind ok (%s kv_slot=%s stream=%s)",
                seq_id,
                slot,
                probe.get("kv_stream"),
            )
            return
        if probe.get("aligned"):
            blocker = probe.get("blocker") or ""
            # cell_map_gap: cell map failed for a live page (sparse cells, defrag, etc.)
            # kv_tensor_not_materialized: backend has no host-accessible tensor data
            # unsupported_memory_type: hybrid/iSWA/recurrent — ext does not support
            if blocker in ("cell_map_gap", "kv_tensor_not_materialized", "unsupported_memory_type"):
                msg = (
                    f"KV tensor bind ({seq_id=} kv_slot={slot}): "
                    f"accounting ok but bind incomplete (blocker={blocker!r})"
                )
                if strict:
                    raise LlamaServerError(msg)
                _llama_log.warning("%s", msg)
            return
        msg = (
            f"KV tensor probe ({seq_id=} kv_slot={slot}): llama cells "
            f"{probe.get('llama_token_cells')} exceed PA reserve "
            f"({probe.get('pa_pages_registered')} pages × "
            f"{probe.get('pa_block_size')} tokens)"
        )
        if strict:
            raise LlamaServerError(msg)
        _llama_log.warning("%s", msg)

    def _resolve_decode_current_pos(
        self,
        ctx: ctypes.c_void_p,
        seq_id: int,
        current_pos: int | None,
    ) -> int:
        """Return next llama write position; ``None`` reads live seq state (v16 resume).

        WHY no-op for single-seq path: when ``n_seq_max == 1`` the engine passes
        ``current_pos=None`` (no shared ctx).  This method is then called on a
        freshly-created per-request context whose ``pos_max`` is ``-1``, so it
        returns ``0``.  That is correct — single-seq sessions never resume.
        """
        if current_pos is not None:
            return max(0, int(current_pos))
        from runtime.kv.physical import current_pos_for_seq

        return current_pos_for_seq(self._lib, ctx, seq_id)

    def _prepare_seq_for_decode(
        self,
        ctx: ctypes.c_void_p,
        sid: int,
        *,
        n_prompt: int,
        n_predict: int,
        kv_token_budget: int | None,
        kv_bind_req: Any | None,
        current_pos: int | None,
        cache_prompt: bool | None = None,
    ) -> int:
        """Clear or resume a multi-seq slot; return the llama write position (v16–v17).

        WHY decode-graph bumps here: any path that clears KV (policy block, owner
        mismatch, cache_prompt=false) must bump epoch and call ggml CUDA graph
        invalidate when ``ctx`` is live — stale graphs after prefix reuse are unsafe.
        """
        from runtime.infer_trace import infer_trace

        if cache_prompt is False:
            from runtime.decode_graph_policy import bump_decode_graph_epoch

            self._seq_last_owner.pop(sid, None)
            bump_decode_graph_epoch(
                sid,
                reason="cache_prompt_disabled",
                ctx_ptr=_ctx_ptr(ctx),
            )
            infer_trace(
                "complete.clear",
                seq_id=sid,
                stale_decode_pos=0,
                incoming_owner="cache_prompt_disabled",
            )
            _clear_sequence(self._lib, ctx, sid)
            return 0

        decode_pos = self._resolve_decode_current_pos(ctx, sid, current_pos)
        need_ctx = max(
            n_prompt + max(n_predict, 1),
            decode_pos + max(n_predict, 1),
        )
        if kv_token_budget is not None and need_ctx > kv_token_budget:
            raise LlamaServerError(
                f"prompt+generation ({need_ctx} tokens) exceeds PA KV reserve "
                f"({kv_token_budget} tokens)"
            )
        if need_ctx > int(self._lib.llama_n_ctx(ctx)):
            raise LlamaServerError(
                f"prompt+generation ({need_ctx} tokens) exceeds n_ctx "
                f"({self._lib.llama_n_ctx(ctx)})"
            )
        incoming_owner = slot_resume_owner_key(kv_bind_req)
        is_resume = (
            decode_pos > 0
            and incoming_owner is not None
            and self._seq_last_owner.get(sid) == incoming_owner
        )
        if (
            not is_resume
            and incoming_owner is not None
            and getattr(kv_bind_req, "slot_pinned", False)
            and self.slot_cache_model_hash
            and self.slot_cache_disk_persist
        ):
            n_ctx_cap = int(self._lib.llama_n_ctx(ctx))
            restored = _try_restore_slot_cache_disk(
                self._lib,
                ctx,
                seq_id=sid,
                model_hash=self.slot_cache_model_hash,
                token_capacity=n_ctx_cap,
            )
            if restored > 0:
                decode_pos = self._resolve_decode_current_pos(ctx, sid, None)
                if decode_pos > 0:
                    is_resume = True
                    self._seq_last_owner[sid] = incoming_owner
                    infer_trace(
                        "complete.disk_restore",
                        seq_id=sid,
                        restored_tokens=restored,
                        decode_pos=decode_pos,
                    )
        if is_resume:
            spec = getattr(self, "kv_cache_spec", None)
            cache_key = (
                getattr(kv_bind_req, "prompt_cache_key", None) if kv_bind_req else None
            )
            if spec is not None and cache_key:
                from runtime.kv.spec_bind import resume_allowed_by_spec

                if not resume_allowed_by_spec(
                    spec,
                    prompt_cache_key=cache_key,
                    seq_pos=decode_pos,
                    n_prompt=n_prompt,
                    cache_prompt=cache_prompt,
                ):
                    from runtime.decode_graph_policy import bump_decode_graph_epoch

                    self._seq_last_owner.pop(sid, None)
                    bump_decode_graph_epoch(
                        sid,
                        reason="spec_bind_swa_block",
                        ctx_ptr=_ctx_ptr(ctx),
                    )
                    infer_trace(
                        "complete.clear",
                        seq_id=sid,
                        stale_decode_pos=decode_pos,
                        incoming_owner="spec_bind_swa_block",
                    )
                    _clear_sequence(self._lib, ctx, sid)
                    decode_pos = 0
                    return decode_pos
        if not is_resume:
            from runtime.decode_graph_policy import bump_decode_graph_epoch

            self._seq_last_owner.pop(sid, None)
            bump_decode_graph_epoch(
                sid,
                reason="slot_clear",
                ctx_ptr=_ctx_ptr(ctx),
            )
            infer_trace(
                "complete.clear",
                seq_id=sid,
                stale_decode_pos=decode_pos,
                incoming_owner=str(incoming_owner)[:32] if incoming_owner else None,
            )
            _clear_sequence(self._lib, ctx, sid)
            decode_pos = 0
        return decode_pos

    def _parallel_jobs_and_smpls(
        self,
        prompts: list[str],
        *,
        n_predict: int,
        seq_ids: list[int],
        kv_token_budgets: list[int | None] | None,
        kv_bind_reqs: list[Any] | None,
        current_positions: list[int | None] | None,
        cache_prompts: list[bool | None] | None = None,
        sampler: SamplerOptions | None,
    ) -> tuple[list[_ParallelDecodeJob], list[ctypes.c_void_p]]:
        """Build parallel decode jobs + one sampler per row (v27/v29)."""
        ctx = self._ctx
        if ctx is None:
            raise LlamaServerError("shared ctx not initialized")
        smpls = [build_sampler_chain(self._lib, sampler) for _ in prompts]
        jobs: list[_ParallelDecodeJob] = []
        try:
            for idx, prompt in enumerate(prompts):
                tokens = self.tokenize_text(prompt, add_special=True)
                if not tokens:
                    raise LlamaServerError("empty prompt after tokenize")
                sid = _normalize_seq_id(
                    seq_ids[idx] if idx < len(seq_ids) else idx,
                    self.n_seq_max,
                )
                budget = None
                if kv_token_budgets is not None and idx < len(kv_token_budgets):
                    b = kv_token_budgets[idx]
                    budget = b if b is not None and b > 0 else None
                bind_req = (
                    kv_bind_reqs[idx]
                    if kv_bind_reqs is not None and idx < len(kv_bind_reqs)
                    else None
                )
                cur_pos = (
                    current_positions[idx]
                    if current_positions is not None and idx < len(current_positions)
                    else None
                )
                cache_ok = (
                    cache_prompts[idx]
                    if cache_prompts is not None and idx < len(cache_prompts)
                    else None
                )
                decode_pos = self._prepare_seq_for_decode(
                    ctx,
                    sid,
                    n_prompt=len(tokens),
                    n_predict=n_predict,
                    kv_token_budget=budget,
                    kv_bind_req=bind_req,
                    current_pos=cur_pos,
                    cache_prompt=cache_ok,
                )
                jobs.append(
                    _ParallelDecodeJob(
                        prompt_tokens=tokens,
                        seq_id=sid,
                        kv_slot=sid,
                        decode_pos=decode_pos,
                        n_predict=n_predict,
                        kv_bind_req=bind_req,
                    )
                )
        except Exception:
            for smpl in smpls:
                self._lib.llama_sampler_free(smpl)
            raise
        return jobs, smpls

    def _finalize_parallel_jobs(
        self,
        jobs: list[_ParallelDecodeJob],
        *,
        kv_block_size: int,
    ) -> None:
        from runtime.infer_trace import infer_trace

        ctx = self._ctx
        if ctx is None:
            return
        for job in jobs:
            owner = slot_resume_owner_key(job.kv_bind_req)
            if owner is not None:
                self._seq_last_owner[job.seq_id] = owner
            self._physical_check_after_decode(
                ctx,
                job.seq_id,
                kv_bind_req=job.kv_bind_req,
                kv_block_size=kv_block_size,
            )
            if (
                job.kv_bind_req is not None
                and getattr(job.kv_bind_req, "slot_pinned", False)
                and self.slot_cache_model_hash
                and self.slot_cache_disk_persist
            ):
                saved = _save_slot_cache_disk(
                    self._lib,
                    ctx,
                    seq_id=job.seq_id,
                    model_hash=self.slot_cache_model_hash,
                )
                if saved:
                    infer_trace(
                        "complete.disk_save",
                        seq_id=job.seq_id,
                        nbytes=saved,
                    )

    def complete_parallel(
        self,
        prompts: list[str],
        *,
        n_predict: int,
        seq_ids: list[int],
        kv_token_budgets: list[int | None] | None = None,
        kv_bind_reqs: list[Any] | None = None,
        kv_block_size: int = 16,
        current_positions: list[int | None] | None = None,
        cache_prompts: list[bool | None] | None = None,
        sampler: SamplerOptions | None = None,
    ) -> list[str]:
        """Decode N prompts on one shared ctx via C continuous batch steps (v27).

        WHY separate from ``complete``: ``generate_batch`` admits N requests with
        distinct PA slots; v26 ``run_batch_step`` merges their decode rows into one
        ``llama_decode``.  Prefill stays sequential (different prompts/resume pos);
        autoregressive steps use the batched C path when linked.
        """
        if not prompts:
            return []
        if self.n_seq_max <= 1 or self._ctx is None:
            raise LlamaServerError("complete_parallel requires n_seq_max > 1")
        from runtime.kv.native_decode_loop import native_batch_decode_available

        if not native_batch_decode_available():
            raise LlamaServerError("native batch decode not available")

        with self._infer_lock:
            from runtime.infer_trace import infer_trace

            infer_trace(
                "complete.parallel.enter",
                n_prompts=len(prompts),
                n_seq_max=self.n_seq_max,
            )
            smpls: list[ctypes.c_void_p] = []
            jobs: list[_ParallelDecodeJob] = []
            try:
                jobs, smpls = self._parallel_jobs_and_smpls(
                    prompts,
                    n_predict=n_predict,
                    seq_ids=seq_ids,
                    kv_token_budgets=kv_token_budgets,
                    kv_bind_reqs=kv_bind_reqs,
                    current_positions=current_positions,
                    cache_prompts=cache_prompts,
                    sampler=sampler,
                )
                ctx = self._ctx
                assert ctx is not None
                texts = _decode_parallel_non_stream(
                    self._lib,
                    self._model,
                    ctx,
                    self._vocab,
                    smpls,
                    jobs,
                    kv_block_size=kv_block_size,
                )
                self._finalize_parallel_jobs(jobs, kv_block_size=kv_block_size)
                return texts
            finally:
                for smpl in smpls:
                    self._lib.llama_sampler_free(smpl)

    def complete_parallel_stream(
        self,
        prompts: list[str],
        *,
        n_predict: int,
        seq_ids: list[int],
        kv_token_budgets: list[int | None] | None = None,
        kv_bind_reqs: list[Any] | None = None,
        kv_block_size: int = 16,
        current_positions: list[int | None] | None = None,
        sampler: SamplerOptions | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream N prompts via batched decode steps (v29).

        WHY stream parallel decode: same prefill + ``run_batch_step`` path as
        ``complete_parallel`` (v27) but yields ``seq_idx``-tagged chunks for
        interleaved multi-request token delivery. One ``stop=True`` sentinel per
        sequence when it finishes.
        """
        if not prompts:
            return iter(())
        if self.n_seq_max <= 1 or self._ctx is None:
            raise LlamaServerError("complete_parallel_stream requires n_seq_max > 1")
        from runtime.kv.native_decode_loop import native_batch_decode_available

        if not native_batch_decode_available():
            raise LlamaServerError("native batch decode not available")

        def _locked_stream() -> Iterator[dict[str, Any]]:
            from runtime.infer_trace import infer_trace

            infer_trace(
                "complete.parallel.stream.enter",
                n_prompts=len(prompts),
                n_seq_max=self.n_seq_max,
            )
            smpls: list[ctypes.c_void_p] = []
            jobs: list[_ParallelDecodeJob] = []
            try:
                with self._infer_lock:
                    jobs, smpls = self._parallel_jobs_and_smpls(
                        prompts,
                        n_predict=n_predict,
                        seq_ids=seq_ids,
                        kv_token_budgets=kv_token_budgets,
                        kv_bind_reqs=kv_bind_reqs,
                        current_positions=current_positions,
                        sampler=sampler,
                    )
                    ctx = self._ctx
                    assert ctx is not None
                    try:
                        yield from _decode_parallel_stream(
                            self._lib,
                            self._model,
                            ctx,
                            self._vocab,
                            smpls,
                            jobs,
                            kv_block_size=kv_block_size,
                        )
                    finally:
                        self._finalize_parallel_jobs(
                            jobs, kv_block_size=kv_block_size
                        )
            finally:
                for smpl in smpls:
                    self._lib.llama_sampler_free(smpl)

        return _locked_stream()

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
        current_pos: int | None = None,
        cache_prompt: bool | None = None,
        prefill_cancel: Any | None = None,
    ) -> str | Iterator[dict[str, Any]]:
        if not stream:
            with self._infer_lock:
                return self._complete_locked(
                    prompt,
                    n_predict=n_predict,
                    stream=False,
                    sampler=sampler,
                    seq_id=seq_id,
                    kv_token_budget=kv_token_budget,
                    kv_bind_req=kv_bind_req,
                    kv_block_size=kv_block_size,
                    current_pos=current_pos,
                    cache_prompt=cache_prompt,
                    prefill_cancel=prefill_cancel,
                )

        def _locked_stream() -> Iterator[dict[str, Any]]:
            with self._infer_lock:
                inner = self._complete_locked(
                    prompt,
                    n_predict=n_predict,
                    stream=True,
                    sampler=sampler,
                    seq_id=seq_id,
                    kv_token_budget=kv_token_budget,
                    kv_bind_req=kv_bind_req,
                    kv_block_size=kv_block_size,
                    current_pos=current_pos,
                    cache_prompt=cache_prompt,
                    prefill_cancel=prefill_cancel,
                )
                yield from inner

        return _locked_stream()

    def _complete_locked(
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
        current_pos: int | None = None,
        cache_prompt: bool | None = None,
        prefill_cancel: Any | None = None,
    ) -> str | Iterator[dict[str, Any]]:
        from runtime.infer_trace import infer_trace

        ctx_int = int(ctypes.cast(self._ctx, ctypes.c_void_p).value or 0) if self._ctx else 0
        infer_trace(
            "complete.enter",
            seq_id=seq_id,
            n_predict=n_predict,
            stream=stream,
            n_prompt=len(prompt) if prompt else 0,
            ctx=hex(ctx_int) if ctx_int else None,
            n_seq_max=self.n_seq_max,
        )
        tokens = self.tokenize_text(prompt, add_special=True)
        if not tokens:
            raise LlamaServerError("empty prompt after tokenize")
        n_prompt = len(tokens)
        smpl = build_sampler_chain(self._lib, sampler)

        if self.n_seq_max > 1 and self._ctx:
            sid = _normalize_seq_id(seq_id, self.n_seq_max)
            decode_pos = self._prepare_seq_for_decode(
                self._ctx,
                sid,
                n_prompt=n_prompt,
                n_predict=n_predict,
                kv_token_budget=kv_token_budget,
                kv_bind_req=kv_bind_req,
                current_pos=current_pos,
                cache_prompt=cache_prompt,
            )
            incoming_owner = slot_resume_owner_key(kv_bind_req)

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
                        kv_slot=sid,
                        kv_block_size=kv_block_size,
                        current_pos=decode_pos,
                        prefill_cancel=prefill_cancel,
                    )
                    if incoming_owner is not None:
                        # Record owner so next turn (possibly new request_id) can resume.
                        self._seq_last_owner[sid] = incoming_owner
                    if (
                        getattr(kv_bind_req, "slot_pinned", False)
                        and self.slot_cache_model_hash
                        and self.slot_cache_disk_persist
                    ):
                        saved = _save_slot_cache_disk(
                            self._lib,
                            self._ctx,
                            seq_id=sid,
                            model_hash=self.slot_cache_model_hash,
                        )
                        if saved:
                            infer_trace(
                                "complete.disk_save",
                                seq_id=sid,
                                nbytes=saved,
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
                        kv_slot=sid,
                        kv_block_size=kv_block_size,
                        current_pos=decode_pos,
                        prefill_cancel=prefill_cancel,
                    )
                finally:
                    # Record owner in finally so a partial stream still
                    # allows the next turn to resume from the position
                    # reached (rather than clearing a partially written slot).
                    if incoming_owner is not None:
                        self._seq_last_owner[sid] = incoming_owner
                    if (
                        getattr(kv_bind_req, "slot_pinned", False)
                        and self.slot_cache_model_hash
                        and self.slot_cache_disk_persist
                    ):
                        _save_slot_cache_disk(
                            self._lib,
                            self._ctx,
                            seq_id=sid,
                            model_hash=self.slot_cache_model_hash,
                        )
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
        decode_pos = self._resolve_decode_current_pos(ctx, sid, current_pos)

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
                    kv_slot=sid,
                    kv_block_size=kv_block_size,
                    current_pos=decode_pos,
                    prefill_cancel=prefill_cancel,
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
                    kv_slot=sid,
                    kv_block_size=kv_block_size,
                    current_pos=decode_pos,
                    prefill_cancel=prefill_cancel,
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
    kv_slot: int | None = None,
    kv_block_size: int | None = None,
    current_pos: int = 0,
    prefill_cancel: Any | None = None,
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
        kv_slot=kv_slot,
        kv_block_size=kv_block_size,
        current_pos=current_pos,
        prefill_cancel=prefill_cancel,
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
    kv_slot: int | None = None,
    kv_block_size: int | None = None,
    current_pos: int = 0,
    prefill_cancel: Any | None = None,
) -> Iterator[dict[str, Any]]:
    """Token streaming decode with optional Phase 15 v8–v15 native decode loop.

    v8: C batch metadata + page-bind validation; llama_decode via ctypes.
    v13: when ZEROLLAMA_KV_DECODE_LOOP ext is built, prefill + decode steps
         run entirely in C (kv_decode_loop_run_prefill / run_step).
    v14: GIL release, ``pos_start`` resume prefill, page-bind validation.
    v15: ``current_pos`` for mid-request resume; C ``llama_sampler_sample`` when linked.

    Why kv_slot/kv_block_size: scheduler registers PA page tables per kv_slot
    at admit; long prompts may prefill in page-aligned chunks; overrun raises
    LlamaServerError.
    """
    from runtime.kv.native_decode import record_decode_step
    from runtime.kv.native_decode_loop import PrefillAbortedError
    from runtime.kv.native_decode_loop import run_prefill as _native_prefill
    from runtime.kv.native_decode_loop import run_sample as _native_sample
    from runtime.kv.native_decode_loop import run_step as _native_step

    def _raise_if_cancelled() -> None:
        if prefill_cancel is not None and prefill_cancel.is_cancelled():
            raise PrefillAbortedError("KV prefill aborted (client disconnect)")

    n_prompt = len(prompt_tokens)
    limit = max(0, n_predict)
    bind_slot = kv_slot if kv_slot is not None else seq_id
    start_pos = max(0, int(current_pos))
    ctx_int = ctypes.cast(ctx, ctypes.c_void_p).value or 0
    native_decode = os.environ.get("ZEROLLAMA_KV_NATIVE_DECODE", "1") != "0"
    use_native_step = bool(
        ctx_int and not lib.llama_model_has_encoder(model) and native_decode
    )
    smpl_int = int(ctypes.cast(smpl, ctypes.c_void_p).value or 0)
    use_native_sample = bool(
        use_native_step
        and smpl_int
        and os.environ.get("ZEROLLAMA_KV_NATIVE_SAMPLE", "1") != "0"
    )
    from runtime.infer_trace import infer_trace
    from runtime.decode_graph_policy import decode_graph_epoch, graph_capture_key

    infer_trace(
        "decode.begin",
        seq_id=seq_id,
        n_prompt=n_prompt,
        n_predict=limit,
        start_pos=start_pos,
        ctx=hex(ctx_int) if ctx_int else None,
        kv_slot=bind_slot,
        kv_block_size=kv_block_size,
        native_decode=native_decode,
        native_step=use_native_step,
        native_sample=use_native_sample,
        decode_graph_epoch=decode_graph_epoch(bind_slot),
        graph_capture_key=graph_capture_key(bind_slot),
    )

    def _make_batch(
        tokens: list[int], *, logits_last: bool, pos_start: int
    ) -> LlamaBatch:
        from runtime.kv.native_decode_batch import (
            build_batch_from_tokens,
            native_decode_batch_available,
        )
        from runtime.kv.page_bind import validate_token_positions

        if native_decode_batch_available():
            return build_batch_from_tokens(
                lib,
                tokens,
                seq_id=seq_id,
                n_seq_max=n_seq_max,
                logits_last=logits_last,
                pos_start=pos_start,
                kv_slot=bind_slot,
            )
        if bind_slot is not None:
            validate_token_positions(bind_slot, pos_start, len(tokens))
        return _batch_from_tokens(
            lib,
            tokens,
            seq_id=seq_id,
            n_seq_max=n_seq_max,
            logits_last=logits_last,
            pos_start=pos_start,
        )

    def _prefill_prompt() -> tuple[LlamaBatch | None, int]:
        """Return (batch, n_pos). batch=None when prefill was handled entirely in C.

        WHY logits_last on final prefill token only: intermediate chunks skip logits;
        the last chunk (C or ctypes) sets logits_last so the first sample after
        prefill has valid output.  v9 ``decode_prefill`` export mirrors this.

        WHY v15 ``start_pos``: ``decode_work.current_pos`` may be mid-prefill or
        mid-decode; remaining prompt slices use ``pos_start=start_pos``.
        """
        ctx_int = ctypes.cast(ctx, ctypes.c_void_p).value or 0

        if start_pos >= n_prompt:
            return None, start_pos

        _raise_if_cancelled()
        prefill_tokens = (
            prompt_tokens[start_pos:] if start_pos > 0 else prompt_tokens
        )
        pos_start = start_pos

        # v13+ native path: page-aligned chunking + llama_decode entirely in C.
        if (
            ctx_int
            and not lib.llama_model_has_encoder(model)
            and native_decode
        ):
            infer_trace(
                "decode.prefill",
                path="native",
                pos_start=pos_start,
                n_tokens=len(prefill_tokens),
                decode_graph_epoch=decode_graph_epoch(bind_slot),
            )
            steps = _native_prefill(
                ctx_int,
                prefill_tokens,
                seq_id=seq_id,
                block_size=kv_block_size or 0,
                pos_start=pos_start,
                kv_slot=bind_slot,
            )
            if steps is not None:
                record_decode_step(steps)
                return None, n_prompt

        infer_trace(
            "decode.prefill",
            path="ctypes",
            pos_start=pos_start,
            n_tokens=len(prefill_tokens),
            chunked=bool(kv_block_size and kv_block_size > 0),
            decode_graph_epoch=decode_graph_epoch(bind_slot),
        )
        # ctypes multi-chunk prefill — v23: same chunker as kv_decode_prefill_plan export.
        # WHY prompt_tokens + pos_start (not pre-sliced prefill_tokens): iter_prefill_execute_chunks
        # slices internally so the returned chunk_pos values are correct absolute positions.
        from runtime.kv.decode_plan import iter_prefill_execute_chunks

        if kv_block_size and kv_block_size > 0:
            exec_chunks = iter_prefill_execute_chunks(
                prompt_tokens,
                block_size=kv_block_size,
                pos_start=pos_start,
            )
            if len(exec_chunks) > 1:
                for chunk_tokens, chunk_pos, logits_last in exec_chunks:
                    _raise_if_cancelled()
                    chunk_batch = _make_batch(
                        chunk_tokens,
                        logits_last=logits_last,
                        pos_start=chunk_pos,
                    )
                    if lib.llama_decode(ctx, chunk_batch) != 0:
                        lib.llama_batch_free(chunk_batch)
                        raise LlamaServerError("llama_decode failed")
                    record_decode_step(1)
                    lib.llama_batch_free(chunk_batch)
                return None, n_prompt
        # Single chunk (short prompt or native batch unavailable): use pre-sliced tokens.
        return _make_batch(
            prefill_tokens, logits_last=True, pos_start=pos_start
        ), pos_start

    batch, n_pos = _prefill_prompt()
    batch_owned = batch is not None
    prefill_chunked = batch is None

    if lib.llama_model_has_encoder(model):
        if batch is None:
            batch = _make_batch(prompt_tokens, logits_last=False, pos_start=0)
            batch_owned = True
            prefill_chunked = False
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
        batch = _make_batch([int(start)], logits_last=True, pos_start=0)
        batch_owned = True
        n_pos = 0
        prefill_chunked = False

    smpl_int = int(ctypes.cast(smpl, ctypes.c_void_p).value or 0)

    def _emit_piece(piece: str, *, stop: bool) -> dict[str, Any]:
        return {"content": piece, "response": piece, "stop": stop}

    _sample_path_logged = False

    def _sample_token() -> int | None:
        nonlocal _sample_path_logged
        path = "native" if use_native_sample else "ctypes"
        if not _sample_path_logged:
            infer_trace("decode.sample", path=path, pos=n_pos)
            _sample_path_logged = True
        if use_native_sample:
            tid = _native_sample(smpl_int, ctx_int)
            if tid is not None:
                new_id = int(tid)
            else:
                new_id = int(lib.llama_sampler_sample(smpl, ctx, -1))
        else:
            new_id = int(lib.llama_sampler_sample(smpl, ctx, -1))
        if lib.llama_vocab_is_eog(vocab, new_id):
            return None
        return new_id

    def _native_step_and_sample(token: int, pos: int) -> int | None:
        if use_native_sample:
            step_out = _native_step(
                ctx_int,
                token,
                seq_id=seq_id,
                current_pos=pos,
                kv_slot=bind_slot,
                smpl_ptr=smpl_int,
            )
            if not isinstance(step_out, tuple):
                # smpl_ptr was set; run_step must return (steps, token).
                # Non-tuple means the ext returned unexpectedly — do not
                # silently fall through and decode the same position again.
                raise LlamaServerError(
                    "native decode step with smpl_ptr returned non-tuple; "
                    "possible ABI mismatch between Python and linked libllama"
                )
            record_decode_step(step_out[0])
            new_id = int(step_out[1])
            if lib.llama_vocab_is_eog(vocab, new_id):
                return None
            return new_id
        steps = _native_step(
            ctx_int, token, seq_id=seq_id, current_pos=pos, kv_slot=bind_slot
        )
        if steps is not None:
            record_decode_step(steps)
            return _sample_token()
        return None

    try:
        target = n_pos + limit
        if prefill_chunked and n_pos >= n_prompt and limit > 0:
            # Prefill ran in C — sample first decode token from the last prefill logits.
            new_id = _sample_token()
            if new_id is None:
                yield _emit_piece("", stop=True)
                return
            yield _emit_piece(token_to_piece(vocab, new_id), stop=False)
            if use_native_step:
                while n_pos < target:
                    _raise_if_cancelled()
                    next_id = _native_step_and_sample(new_id, n_pos)
                    n_pos += 1
                    if next_id is None:
                        yield _emit_piece("", stop=True)
                        return
                    if n_pos >= target:
                        break
                    yield _emit_piece(token_to_piece(vocab, next_id), stop=False)
                    new_id = next_id
                yield _emit_piece("", stop=True)
                return
            batch = _make_batch([new_id], logits_last=True, pos_start=n_pos)
            batch_owned = True
            prefill_chunked = False

        while n_pos < target:
            _raise_if_cancelled()
            if batch is None:
                raise LlamaServerError("decode state: missing batch")
            if lib.llama_decode(ctx, batch) != 0:
                raise LlamaServerError("llama_decode failed")
            record_decode_step(1)
            n_pos += batch.n_tokens
            if n_pos >= target:
                break
            new_id = _sample_token()
            if new_id is None:
                yield _emit_piece("", stop=True)
                return
            yield _emit_piece(token_to_piece(vocab, new_id), stop=False)
            if batch_owned:
                lib.llama_batch_free(batch)
            batch = _make_batch([new_id], logits_last=True, pos_start=n_pos)
            batch_owned = True
        yield _emit_piece("", stop=True)
    finally:
        if batch_owned and batch is not None:
            lib.llama_batch_free(batch)


@dataclass
class _ParallelDecodeJob:
    prompt_tokens: list[int]
    seq_id: int
    kv_slot: int
    decode_pos: int
    n_predict: int
    kv_bind_req: Any | None = None


@dataclass
class _ParallelSeqState:
    job: _ParallelDecodeJob
    n_pos: int
    generated: int
    pieces: list[str]
    job_idx: int = 0
    feed_token: int | None = None
    done: bool = False


def _parallel_stream_chunk(
    seq_idx: int,
    seq_id: int,
    piece: str,
    *,
    stop: bool = False,
) -> dict[str, Any]:
    return {
        "seq_idx": seq_idx,
        "seq_id": seq_id,
        "content": piece,
        "response": piece,
        "stop": stop,
    }


def _decode_parallel_stream(
    lib: ctypes.CDLL,
    model: ctypes.c_void_p,
    ctx: ctypes.c_void_p,
    vocab: ctypes.c_void_p,
    smpls: list[ctypes.c_void_p],
    jobs: list[_ParallelDecodeJob],
    *,
    kv_block_size: int,
) -> Iterator[dict[str, Any]]:
    """Multi-sequence decode stream: sequential prefill, batched decode steps (v29).

    Yields ``_parallel_stream_chunk`` dicts tagged with ``seq_idx`` / ``seq_id``.
    One ``stop=True`` sentinel per sequence when it finishes (including EOG / limit).
    """
    from runtime.infer_trace import infer_trace
    from runtime.kv.native_decode import record_decode_step
    from runtime.kv.native_decode_loop import run_batch_step as _native_batch_step
    from runtime.kv.native_decode_loop import run_prefill as _native_prefill
    from runtime.kv.native_decode_loop import run_sample as _native_sample

    ctx_int = int(ctypes.cast(ctx, ctypes.c_void_p).value or 0)
    native_decode = os.environ.get("ZEROLLAMA_KV_NATIVE_DECODE", "1") != "0"
    use_native = bool(
        ctx_int and not lib.llama_model_has_encoder(model) and native_decode
    )
    _first_smpl_int = (
        int(ctypes.cast(smpls[0], ctypes.c_void_p).value or 0) if smpls else 0
    )
    use_native_sample = bool(
        use_native
        and _first_smpl_int
        and os.environ.get("ZEROLLAMA_KV_NATIVE_SAMPLE", "1") != "0"
    )
    infer_trace(
        "decode.parallel.stream.begin",
        n_jobs=len(jobs),
        native=use_native,
        native_sample=use_native_sample,
    )
    if not use_native:
        raise LlamaServerError("parallel decode requires native decode loop")

    def _smpl_int(job_idx: int) -> int:
        if job_idx < len(smpls):
            return int(ctypes.cast(smpls[job_idx], ctypes.c_void_p).value or 0)
        return 0

    def _sample_one_for(job_idx: int, batch_idx: int = -1) -> int:
        si = _smpl_int(job_idx)
        # WHY: use_native_sample reads the last logit row (single-row assumption).
        # After run_batch_step the matrix has N rows; only use native path for the
        # post-prefill sample (batch_idx == -1) where there is exactly one row.
        if use_native_sample and si and batch_idx == -1:
            tid = _native_sample(si, ctx_int)
            if tid is not None:
                return int(tid)
        smpl = smpls[job_idx] if job_idx < len(smpls) else smpls[0]
        return int(lib.llama_sampler_sample(smpl, ctx, batch_idx))

    def _emit_token(st: _ParallelSeqState, new_id: int) -> Iterator[dict[str, Any]]:
        if lib.llama_vocab_is_eog(vocab, new_id):
            st.done = True
            yield _parallel_stream_chunk(
                st.job_idx, st.job.seq_id, "", stop=True
            )
            return
        piece = token_to_piece(vocab, new_id)
        st.pieces.append(piece)
        st.generated += 1
        yield _parallel_stream_chunk(st.job_idx, st.job.seq_id, piece, stop=False)
        if st.generated >= st.job.n_predict:
            st.done = True
            yield _parallel_stream_chunk(
                st.job_idx, st.job.seq_id, "", stop=True
            )
        else:
            st.feed_token = int(new_id)

    states: list[_ParallelSeqState] = []
    for job_idx, job in enumerate(jobs):
        n_prompt = len(job.prompt_tokens)
        start_pos = max(0, int(job.decode_pos))
        limit = max(0, int(job.n_predict))
        bind_slot = job.kv_slot

        if use_native and start_pos < n_prompt:
            infer_trace(
                "decode.parallel.prefill",
                seq_id=job.seq_id,
                pos_start=start_pos,
                n_tokens=n_prompt - start_pos,
            )
            steps = _native_prefill(
                ctx_int,
                job.prompt_tokens,
                seq_id=job.seq_id,
                block_size=kv_block_size or 0,
                pos_start=start_pos,
                kv_slot=bind_slot,
            )
            if steps is None:
                raise LlamaServerError("native prefill unavailable in parallel decode")
            record_decode_step(steps)

        n_pos = n_prompt if start_pos < n_prompt else start_pos
        st = _ParallelSeqState(
            job=job, n_pos=n_pos, generated=0, pieces=[], job_idx=job_idx
        )

        if limit <= 0:
            st.done = True
            states.append(st)
            yield _parallel_stream_chunk(job_idx, job.seq_id, "", stop=True)
            continue

        new_id = _sample_one_for(job_idx)
        if lib.llama_vocab_is_eog(vocab, new_id):
            st.done = True
            states.append(st)
            yield _parallel_stream_chunk(job_idx, job.seq_id, "", stop=True)
            continue

        piece = token_to_piece(vocab, new_id)
        st.pieces.append(piece)
        st.generated = 1
        states.append(st)
        yield _parallel_stream_chunk(job_idx, job.seq_id, piece, stop=False)
        if st.generated >= limit:
            st.done = True
            yield _parallel_stream_chunk(job_idx, job.seq_id, "", stop=True)
        else:
            st.feed_token = new_id

    while True:
        active = [s for s in states if not s.done and s.feed_token is not None]
        if not active:
            break

        tokens = [int(s.feed_token) for s in active]
        seq_ids = [s.job.seq_id for s in active]
        positions = [s.n_pos for s in active]
        infer_trace(
            "decode.parallel.batch_step",
            n_rows=len(active),
            seq_ids=seq_ids,
        )

        active_smpls = [_smpl_int(st.job_idx) for st in active]
        batch_out = None
        # WHY smpl_ptrs (v30): after a multi-row batch step the logit matrix has N
        # rows. ctypes batch_idx sampling works but stays in Python; per-row C
        # samplers keep decode+sample in one GIL-released call with correct indices
        # and isolated accept state (v27 audit — no shared sampler chain).
        if use_native_sample and active_smpls and all(active_smpls):
            batch_out = _native_batch_step(
                ctx_int,
                tokens,
                seq_ids,
                positions,
                smpl_ptrs=active_smpls,
            )
        if batch_out is None:
            batch_out = _native_batch_step(ctx_int, tokens, seq_ids, positions)
        if batch_out is None:
            raise LlamaServerError("native batch step unavailable")
        if isinstance(batch_out, tuple):
            steps, sampled = batch_out
            record_decode_step(int(steps))
            for st, new_id in zip(active, sampled):
                st.n_pos += 1
                st.feed_token = None
                yield from _emit_token(st, int(new_id))
            continue

        record_decode_step(int(batch_out))
        for batch_idx, st in enumerate(active):
            st.n_pos += 1
            st.feed_token = None
            new_id = _sample_one_for(st.job_idx, batch_idx)
            yield from _emit_token(st, new_id)


def _decode_parallel_non_stream(
    lib: ctypes.CDLL,
    model: ctypes.c_void_p,
    ctx: ctypes.c_void_p,
    vocab: ctypes.c_void_p,
    smpls: list[ctypes.c_void_p],
    jobs: list[_ParallelDecodeJob],
    *,
    kv_block_size: int,
) -> list[str]:
    """Collect ``_decode_parallel_stream`` into per-sequence text (v27)."""
    texts = [""] * len(jobs)
    for chunk in _decode_parallel_stream(
        lib, model, ctx, vocab, smpls, jobs, kv_block_size=kv_block_size
    ):
        if chunk.get("stop"):
            continue
        idx = int(chunk["seq_idx"])
        texts[idx] += str(chunk.get("content") or "")
    return texts
