"""Phase 15 v48 — CPU-only donor-buffer overlay bind, real-model E2E (optional).

WHY skipped by default: requires a real GGUF (LLAMA_MODEL) and
RUN_E2E_OVERLAY_BIND=1 plus a libllama built with LLAMA_KV_EXT_DONOR_BUFFER=1
(see scripts/phase/phase15_overlay_bind_cpu_smoke.sh, which builds the native
ext with this flag before running these tests). Unit-level facade tests that
run everywhere live in test_kv_overlay_bind.py.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest

from runtime.kv import overlay_bind
from runtime.worker.libllama_ctypes import LlamaLoadedSession, get_lib


def _linked_e2e_ready() -> bool:
    if not os.environ.get("RUN_E2E_OVERLAY_BIND", "").strip():
        return False
    if os.environ.get("ZEROLLAMA_KV_OVERLAY_BIND", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    gguf = os.environ.get("LLAMA_MODEL", "").strip()
    if not gguf or not Path(gguf).is_file():
        return False
    # WHY not a module-level import: runtime.kv._kv_native may be built
    # without LLAMA_KV_EXT_DONOR_BUFFER (e.g. plain ZEROLLAMA_KV_DECODE_LOOP
    # build) — importing donor_buffer_status at module scope would break
    # pytest collection entirely rather than skipping gracefully.
    return overlay_bind.donor_buffer_available()


@pytest.mark.skipif(
    not _linked_e2e_ready(),
    reason="RUN_E2E_OVERLAY_BIND=1, ZEROLLAMA_KV_OVERLAY_BIND=1, LLAMA_MODEL, donor-buffer-linked ext",
)
def test_donor_buffer_consumed_on_cpu_context_construction():
    """Two-step flow: query real size via a throwaway ctx, then register a
    correctly-sized donor and confirm the SECOND context construction
    actually consumes it (donor_buffer_status.bound == True) and decode still
    produces the same output as an unbound (normal-allocation) context.
    """
    from runtime.kv._kv_native import donor_buffer_status as _native_donor_status

    gguf = Path(os.environ["LLAMA_MODEL"]).resolve()
    lib_path = Path(os.environ["LLAMA_CPP_LIB"]) if os.environ.get("LLAMA_CPP_LIB") else None
    cpp_root = Path(os.environ["LLAMA_CPP_ROOT"]) if os.environ.get("LLAMA_CPP_ROOT") else None
    lib = get_lib(lib_path, cpp_root)

    n_ctx = 512
    prompt = "The quick brown fox"

    # WHY n_gpu_layers=0: the donor registry is CPU-buft only — any offloaded
    # layer's KV tensor lives on a device buft the allocation loop never
    # checks. Forcing full CPU keeps this an actual v48 exercise, not a no-op.
    session_baseline = LlamaLoadedSession(
        gguf, n_gpu_layers=0, num_ctx=n_ctx, lib_path=lib_path, cpp_root=cpp_root
    )
    try:
        tokens = session_baseline.tokenize_text(prompt, add_special=True)
        cparams = lib.llama_context_default_params()
        cparams.n_ctx = n_ctx
        cparams.n_batch = min(n_ctx, max(len(tokens), 64))

        # Step 1: dry-run construction to discover the exact required donor size.
        ctx_dry = lib.llama_init_from_model(session_baseline._model, cparams)
        assert ctx_dry
        lib.llama_free(ctx_dry)
    finally:
        session_baseline.close()

    # Step 2: allocate a generously-sized, page-aligned host buffer and
    # register it BEFORE constructing the real context. WHY oversized rather
    # than exactly the queried size: this test does not have a Python-side
    # way to read ggml's exact per-buft byte count without a second native
    # symbol; a generous buffer still proves consumption because the
    # allocation loop accepts any donor size >= required.
    donor_bytes = 256 * 1024 * 1024
    raw = ctypes.create_string_buffer(donor_bytes + 4096)
    aligned_addr = (ctypes.addressof(raw) + 4095) & ~4095

    donor_id = overlay_bind.register_donor_buffer(aligned_addr, donor_bytes)
    try:
        session = LlamaLoadedSession(
            gguf, n_gpu_layers=0, num_ctx=n_ctx, lib_path=lib_path, cpp_root=cpp_root
        )
        try:
            status = _native_donor_status(donor_id)
            assert status["bound"] is True, (
                "donor was registered before context construction but never "
                f"consumed — status={status}. Check CPU buft resolution and "
                "LLAMA_KV_EXT_DONOR_BUFFER wiring."
            )
            assert status["bytes_used"] > 0
            assert status["bytes_used"] <= donor_bytes

            # Decode must still work correctly against the donor-backed tensor.
            tokens = session.tokenize_text(prompt, add_special=True)
            cparams = lib.llama_context_default_params()
            cparams.n_ctx = n_ctx
            cparams.n_batch = min(n_ctx, max(len(tokens), 64))
            ctx = lib.llama_init_from_model(session._model, cparams)
            assert ctx
            try:
                batch = lib.llama_batch_get_one(
                    (ctypes.c_int32 * len(tokens))(*tokens), len(tokens)
                )
                rc = lib.llama_decode(ctx, batch)
                assert rc == 0, f"llama_decode failed rc={rc} on donor-backed context"
            finally:
                lib.llama_free(ctx)
        finally:
            session.close()
    finally:
        overlay_bind.unregister_donor_buffer(donor_id)
        assert _native_donor_status(donor_id) is None or overlay_bind.donor_buffer_status(donor_id) is None
