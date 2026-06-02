"""Health exposes KV backend status (Phase 15)."""

from __future__ import annotations

import importlib
import importlib.util

import pytest

from runtime.kv import backend as backend_mod
from runtime.kv._py_block_pool import BlockPool as PyBlockPool
from runtime.kv.backend import reset_kv_backend_cache

_NATIVE_BUILT = importlib.util.find_spec("runtime.kv._kv_native") is not None


def _fresh_backend(monkeypatch, **env: str | None) -> None:
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_KV_NATIVE", raising=False)
    for key, val in env.items():
        if val is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, val)
    reset_kv_backend_cache()


def test_kv_backend_default_python(monkeypatch):
    _fresh_backend(monkeypatch)
    assert backend_mod.kv_backend_name() == "python"
    h = backend_mod.kv_backend_health()
    assert h["backend"] == "python"
    assert h["native_requested"] is False
    assert "note" not in h


def test_kv_native_requested_without_extension(monkeypatch):
    _fresh_backend(monkeypatch, ZEROLLAMA_RUNTIME_KV_NATIVE="1")
    monkeypatch.setattr(backend_mod, "_NATIVE", None)
    monkeypatch.setattr(backend_mod, "_NATIVE_TRIED", True)
    monkeypatch.setattr(backend_mod, "_BACKEND", None)
    monkeypatch.setattr(backend_mod, "_WARNED_NATIVE_UNAVAILABLE", False)

    assert backend_mod.kv_backend_name() == "python"
    h = backend_mod.kv_backend_health()
    assert h["backend"] == "python"
    assert h["native_requested"] is True
    assert h["native_available"] is False
    assert "note" in h
    pool = backend_mod.create_block_pool(num_blocks=4, block_size=16)
    assert isinstance(pool, PyBlockPool)


@pytest.mark.skipif(not _NATIVE_BUILT, reason="native extension not built")
def test_kv_native_health_when_built(monkeypatch):
    _fresh_backend(monkeypatch, ZEROLLAMA_RUNTIME_KV_NATIVE="1")
    from runtime.kv._kv_native import BlockPool as NativeBlockPool

    h = backend_mod.kv_backend_health()
    assert h["backend"] == "native"
    assert h["native_requested"] is True
    assert h["native_available"] is True
    assert "note" not in h
    pool = backend_mod.create_block_pool(num_blocks=4, block_size=16)
    assert isinstance(pool, NativeBlockPool)


@pytest.mark.skipif(not _NATIVE_BUILT, reason="native extension not built")
def test_block_pool_lazy_respects_env(monkeypatch):
    _fresh_backend(monkeypatch, ZEROLLAMA_RUNTIME_KV_NATIVE="1")
    from runtime.kv._kv_native import BlockPool as NativeBlockPool

    import runtime.kv.block_pool as bp_mod

    importlib.reload(bp_mod)
    assert bp_mod.BlockPool is NativeBlockPool
