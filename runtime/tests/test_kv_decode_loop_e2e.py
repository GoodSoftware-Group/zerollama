"""Phase 15 v14 — linked-build E2E decode loop parity (optional GPU)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from runtime.kv.native_decode_loop import (
    decode_loop_status,
    greedy_decode_tokens,
    native_decode_loop_available,
    run_prefill,
)
from runtime.worker.libllama_ctypes import (
    LlamaLoadedSession,
    build_sampler_chain,
    get_lib,
    token_to_piece,
)
from runtime.worker.sampler_options import SamplerOptions


def _linked_e2e_ready() -> bool:
    if not os.environ.get("RUN_E2E_DECODE_LOOP", "").strip():
        return False
    gguf = os.environ.get("LLAMA_MODEL", "").strip()
    if not gguf or not Path(gguf).is_file():
        return False
    return native_decode_loop_available()


def _stream_text(chunks) -> str:
    parts: list[str] = []
    for chunk in chunks:
        if chunk.get("stop"):
            break
        parts.append(str(chunk.get("content") or ""))
    return "".join(parts)


@pytest.mark.skipif(not _linked_e2e_ready(), reason="RUN_E2E_DECODE_LOOP=1, LLAMA_MODEL, linked ext")
def test_decode_loop_status_sampling_in_c():
    st = decode_loop_status()
    assert st.get("sampling_in_c") is True


@pytest.mark.skipif(not _linked_e2e_ready(), reason="RUN_E2E_DECODE_LOOP=1, LLAMA_MODEL, linked ext")
def test_decode_loop_status_gil_released():
    st = decode_loop_status()
    assert st["available"] is True
    assert st.get("gil_released") is True


@pytest.mark.skipif(not _linked_e2e_ready(), reason="RUN_E2E_DECODE_LOOP=1, LLAMA_MODEL, linked ext")
def test_native_greedy_matches_ctypes_stream():
    """C prefill + C steps must match full ctypes decode when native is disabled."""
    import ctypes
    from runtime.worker.libllama_ctypes import _decode_stream

    gguf = Path(os.environ["LLAMA_MODEL"]).resolve()
    lib_path = Path(os.environ["LLAMA_CPP_LIB"]) if os.environ.get("LLAMA_CPP_LIB") else None
    cpp_root = Path(os.environ["LLAMA_CPP_ROOT"]) if os.environ.get("LLAMA_CPP_ROOT") else None
    lib = get_lib(lib_path, cpp_root)

    prompt = "Hello"
    n_predict = 4
    block_size = 16

    session = LlamaLoadedSession(
        gguf,
        n_gpu_layers=int(os.environ.get("LLAMA_N_GPU_LAYERS", "-1")),
        num_ctx=512,
        lib_path=lib_path,
        cpp_root=cpp_root,
    )
    try:
        tokens = session.tokenize_text(prompt, add_special=True)
        smpl_opts = SamplerOptions(temperature=0.0)

        cparams = lib.llama_context_default_params()
        cparams.n_ctx = 512
        cparams.n_batch = min(512, max(len(tokens), 64))

        # ctypes baseline — disable native fast path so we compare against pure ctypes.
        smpl_ctypes = build_sampler_chain(lib, smpl_opts)
        ctx = lib.llama_init_from_model(session._model, cparams)
        assert ctx
        try:
            with patch("runtime.kv.native_decode_loop.run_prefill", return_value=None):
                with patch("runtime.kv.native_decode_loop.run_step", return_value=None):
                    with patch("runtime.kv.native_decode_loop.run_sample", return_value=None):
                        ctypes_text = _stream_text(
                            _decode_stream(
                                lib,
                                session._model,
                                ctx,
                                session._vocab,
                                smpl_ctypes,
                                tokens,
                                n_predict=n_predict,
                                seq_id=0,
                                n_seq_max=1,
                                kv_slot=0,
                                kv_block_size=block_size,
                            )
                        )
        finally:
            lib.llama_sampler_free(smpl_ctypes)
            lib.llama_free(ctx)

        smpl_native = build_sampler_chain(lib, smpl_opts)
        ctx2 = lib.llama_init_from_model(session._model, cparams)
        assert ctx2
        ctx_int = int(ctypes.cast(ctx2, ctypes.c_void_p).value or 0)
        try:
            native_ids = greedy_decode_tokens(
                ctx_int,
                lib,
                ctx2,
                session._vocab,
                smpl_native,
                tokens,
                n_predict=n_predict,
                seq_id=0,
                block_size=block_size,
                kv_slot=0,
            )
            native_text = "".join(
                token_to_piece(session._vocab, tid) for tid in native_ids
            )
        finally:
            lib.llama_sampler_free(smpl_native)
            lib.llama_free(ctx2)

        assert native_text == ctypes_text, (native_text, ctypes_text)
    finally:
        session.close()


@pytest.mark.skipif(not _linked_e2e_ready(), reason="RUN_E2E_DECODE_LOOP=1, LLAMA_MODEL, linked ext")
def test_native_prefill_pos_start_resume():
    """Prefill first half via C, second half from pos_start — must not error."""
    import ctypes

    gguf = Path(os.environ["LLAMA_MODEL"]).resolve()
    session = LlamaLoadedSession(gguf, num_ctx=512)
    try:
        tokens = session.tokenize_text("One two three four five", add_special=True)
        if len(tokens) < 4:
            pytest.skip("prompt too short after tokenize")
        mid = len(tokens) // 2
        first, rest = tokens[:mid], tokens[mid:]

        lib = session._lib
        cparams = lib.llama_context_default_params()
        cparams.n_ctx = 512
        cparams.n_batch = 512
        ctx = lib.llama_init_from_model(session._model, cparams)
        assert ctx
        ctx_int = int(ctypes.cast(ctx, ctypes.c_void_p).value or 0)
        try:
            s1 = run_prefill(ctx_int, first, seq_id=0, block_size=16, pos_start=0)
            assert s1 is not None and s1 >= 1
            s2 = run_prefill(ctx_int, rest, seq_id=0, block_size=16, pos_start=mid)
            assert s2 is not None and s2 >= 1
        finally:
            lib.llama_free(ctx)
    finally:
        session.close()
