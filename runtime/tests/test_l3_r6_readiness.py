"""Phase 15 — L3-R6 metadata readiness + L3-R6b COW."""

from __future__ import annotations

import os

import pytest

from runtime.kv.l3_r6_readiness import l3_r6_metadata_readiness


def _clear_cow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZEROLLAMA_KV_COW", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_COW_TENSORS", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_COW_PAGES", raising=False)


def test_l3_r6_incomplete_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED", raising=False)
    monkeypatch.delenv("ZEROLLAMA_RADIX_PREFIX_SHARE", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED_WITH_RADIX", raising=False)
    _clear_cow(monkeypatch)
    from runtime.env import configure_l3_settings, reset_runtime_env_for_tests

    reset_runtime_env_for_tests()
    configure_l3_settings({})
    out = l3_r6_metadata_readiness(n_ctx=8192, n_parallel=4)
    assert out["complete"] is False
    assert "true_cow_metadata_cells_env_off" in out["deferred"]
    assert "nixl_rdma_blobs" in out["deferred"]


def test_l3_r6_complete_with_unified(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED", "1")
    monkeypatch.delenv("ZEROLLAMA_RADIX_PREFIX_SHARE", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED_STRICT", raising=False)
    _clear_cow(monkeypatch)
    from runtime.env import configure_l3_settings, reset_runtime_env_for_tests

    reset_runtime_env_for_tests()
    configure_l3_settings({})
    out = l3_r6_metadata_readiness(
        n_ctx=8192, n_parallel=4, backend="inprocess"
    )
    assert out["complete"] is True


def test_l3_r6_radix_requires_metadata_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_RADIX_PREFIX_SHARE", "1")
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED", "0")
    _clear_cow(monkeypatch)
    from runtime.env import configure_l3_settings, reset_runtime_env_for_tests

    reset_runtime_env_for_tests()
    configure_l3_settings({})
    out = l3_r6_metadata_readiness(n_ctx=8192, n_parallel=4)
    assert out["complete"] is False
    assert out["checks"]["seq_cp_mode_metadata"] is False


def test_l3_r6_strict_undersize_blocks_complete(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED", "1")
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED_STRICT", "1")
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED_MIN_TOKENS_PER_SLOT", raising=False)
    _clear_cow(monkeypatch)
    from runtime.env import configure_l3_settings, reset_runtime_env_for_tests

    reset_runtime_env_for_tests()
    configure_l3_settings({})
    out = l3_r6_metadata_readiness(n_ctx=1024, n_parallel=4)
    assert out["complete"] is False


def test_l3_r6b_cells_only_partial(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED", "1")
    monkeypatch.setenv("ZEROLLAMA_KV_COW", "1")
    monkeypatch.delenv("ZEROLLAMA_KV_COW_TENSORS", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_COW_PAGES", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED_STRICT", raising=False)
    from runtime.env import configure_l3_settings, reset_runtime_env_for_tests

    reset_runtime_env_for_tests()
    configure_l3_settings({})
    out = l3_r6_metadata_readiness(
        n_ctx=8192, n_parallel=4, backend="inprocess"
    )
    assert out["l3_r6b"] == "partial_cells"
    assert "tensor_deep_copy_cow_env_off" in out["deferred"]


def test_l3_r6b_tensors_without_pages(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED", "1")
    monkeypatch.setenv("ZEROLLAMA_KV_COW", "1")
    monkeypatch.setenv("ZEROLLAMA_KV_COW_TENSORS", "1")
    monkeypatch.delenv("ZEROLLAMA_KV_COW_PAGES", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED_STRICT", raising=False)
    from runtime.env import configure_l3_settings, reset_runtime_env_for_tests

    reset_runtime_env_for_tests()
    configure_l3_settings({})
    out = l3_r6_metadata_readiness(
        n_ctx=8192, n_parallel=4, backend="inprocess"
    )
    assert out["l3_r6b"] == "done_full_tensor"
    assert "page_granular_cow_optimization" in out["deferred"]


def test_l3_r6b_pages_done(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED", "1")
    monkeypatch.setenv("ZEROLLAMA_KV_COW", "1")
    monkeypatch.setenv("ZEROLLAMA_KV_COW_TENSORS", "1")
    monkeypatch.setenv("ZEROLLAMA_KV_COW_PAGES", "1")
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED_STRICT", raising=False)
    from runtime.env import configure_l3_settings, reset_runtime_env_for_tests

    reset_runtime_env_for_tests()
    configure_l3_settings({})
    out = l3_r6_metadata_readiness(
        n_ctx=8192, n_parallel=4, backend="inprocess"
    )
    assert out["l3_r6b"] == "done"
    assert out["kv_cow_pages"] is True
    assert "page_granular_cow_optimization" not in out["deferred"]


def test_l3_r6b_yaml_pages_sync(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED", "1")
    _clear_cow(monkeypatch)
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED_STRICT", raising=False)
    from runtime.env import (
        configure_l3_settings,
        kv_cow_pages_enabled,
        reset_runtime_env_for_tests,
    )

    reset_runtime_env_for_tests()
    configure_l3_settings(
        {"kv_cow": True, "kv_cow_tensors": True, "kv_cow_pages": True}
    )
    assert kv_cow_pages_enabled() is True
    assert os.environ.get("ZEROLLAMA_KV_COW_PAGES") == "1"
    assert os.environ.get("ZEROLLAMA_KV_COW_TENSORS") == "1"
    out = l3_r6_metadata_readiness(
        n_ctx=8192, n_parallel=4, backend="inprocess"
    )
    assert out["l3_r6b"] == "done"
    assert out["kv_cow_pages_source"] == "yaml"
