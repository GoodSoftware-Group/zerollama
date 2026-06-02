"""In-process sampler chain construction (requires libllama.so)."""

from __future__ import annotations

import pytest

from runtime.worker.libllama_ctypes import build_sampler_chain, get_lib, resolve_libllama_path
from runtime.worker.sampler_options import SamplerOptions


def test_build_sampler_chain_greedy():
    try:
        lib = get_lib()
    except Exception as e:
        pytest.skip(f"libllama unavailable: {e}")
    smpl = build_sampler_chain(lib, None)
    try:
        assert smpl
        n = lib.llama_sampler_chain_n(smpl)
        assert int(n) >= 1
    finally:
        lib.llama_sampler_free(smpl)


def test_build_sampler_chain_stochastic():
    try:
        lib = get_lib()
    except Exception as e:
        pytest.skip(f"libllama unavailable: {e}")
    opts = SamplerOptions(temperature=0.8, top_k=40, top_p=0.9)
    smpl = build_sampler_chain(lib, opts)
    try:
        assert smpl
        n = int(lib.llama_sampler_chain_n(smpl))
        assert n >= 3
    finally:
        lib.llama_sampler_free(smpl)


def test_resolve_libllama_for_chain():
    try:
        resolve_libllama_path(cpp_root=__import__("pathlib").Path("/root/llama.cpp"))
    except Exception:
        pytest.skip("no libllama build")
