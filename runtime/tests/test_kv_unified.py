"""Phase 15 v52 — kv_unified + hardened in-process seq_cp."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from runtime.kv import overlay_bind
from runtime.kv.radix_seq_copy import copy_sequence_prefix_inprocess, seq_cp_mode


def test_kv_unified_disabled_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED", raising=False)
    monkeypatch.delenv("ZEROLLAMA_RADIX_PREFIX_SHARE", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED_WITH_RADIX", raising=False)
    from runtime.env import (
        configure_l3_settings,
        kv_unified_enabled,
        kv_unified_source,
        reset_runtime_env_for_tests,
    )

    reset_runtime_env_for_tests()
    configure_l3_settings({})
    assert kv_unified_enabled() is False
    assert kv_unified_source() == "off"
    assert seq_cp_mode() == "buffer_copy"


def test_kv_unified_radix_couple(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED_WITH_RADIX", raising=False)
    monkeypatch.setenv("ZEROLLAMA_RADIX_PREFIX_SHARE", "1")
    from runtime.env import (
        configure_l3_settings,
        kv_unified_enabled,
        kv_unified_source,
        reset_runtime_env_for_tests,
    )

    reset_runtime_env_for_tests()
    configure_l3_settings({})
    assert kv_unified_enabled() is True
    assert kv_unified_source() == "radix_couple"
    assert seq_cp_mode() == "metadata"

    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED_WITH_RADIX", "0")
    assert kv_unified_enabled() is False
    assert kv_unified_source() == "off"

    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED_WITH_RADIX", raising=False)
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED", "0")
    assert kv_unified_enabled() is False
    assert kv_unified_source() == "off"


def test_kv_unified_yaml_enables_without_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED", raising=False)
    from runtime.env import (
        configure_l3_settings,
        kv_unified_enabled,
        kv_unified_operator_note,
        reset_runtime_env_for_tests,
    )

    reset_runtime_env_for_tests()
    configure_l3_settings({"kv_unified": True})
    assert kv_unified_enabled() is True
    assert seq_cp_mode() == "metadata"
    note = kv_unified_operator_note()
    assert note is not None
    assert "shared cell pool" in note
    from runtime.env import kv_unified_source

    assert kv_unified_source() == "yaml"


def test_kv_unified_enabled_truthy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED", "1")
    from runtime.env import kv_unified_enabled

    assert kv_unified_enabled() is True
    assert seq_cp_mode() == "metadata"


def test_kv_unified_sizing_ok_and_risk(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED", "1")
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED_MIN_TOKENS_PER_SLOT", raising=False)
    from runtime.env import kv_unified_sizing_status, reset_runtime_env_for_tests

    reset_runtime_env_for_tests()
    unknown = kv_unified_sizing_status(n_ctx=None, n_parallel=4)
    assert unknown is not None
    assert unknown["unknown"] is True
    assert unknown["ok"] is None
    assert unknown["recommended_min_ctx"] == 4 * 512

    ok = kv_unified_sizing_status(n_ctx=8192, n_parallel=4)
    assert ok is not None and ok["ok"] is True

    risk = kv_unified_sizing_status(n_ctx=1024, n_parallel=4)
    assert risk is not None and risk["ok"] is False
    assert "raise -c" in str(risk.get("note", ""))


def test_kv_unified_sizing_none_when_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED", raising=False)
    from runtime.env import (
        kv_unified_sizing_status,
        reset_runtime_env_for_tests,
    )

    reset_runtime_env_for_tests()
    assert kv_unified_sizing_status(n_ctx=4096, n_parallel=4) is None


def test_kv_unified_strict_raises_when_undersized(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED", "1")
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED_STRICT", "1")
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED_MIN_TOKENS_PER_SLOT", raising=False)
    from runtime.env import (
        KvUnifiedSizingError,
        assert_kv_unified_sizing,
        kv_unified_sizing_status,
        reset_runtime_env_for_tests,
    )

    reset_runtime_env_for_tests()
    status = kv_unified_sizing_status(n_ctx=1024, n_parallel=4)
    assert status is not None and status["strict"] is True and status["ok"] is False
    with pytest.raises(KvUnifiedSizingError, match="raise -c"):
        assert_kv_unified_sizing(n_ctx=1024, n_parallel=4)
    assert_kv_unified_sizing(n_ctx=8192, n_parallel=4)  # ok path


def test_kv_unified_strict_noop_when_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED", "1")
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED_STRICT", raising=False)
    from runtime.env import assert_kv_unified_sizing, reset_runtime_env_for_tests

    reset_runtime_env_for_tests()
    # Advisory risk only — no raise.
    assert_kv_unified_sizing(n_ctx=1024, n_parallel=4)


def test_estimate_donor_streams_one_when_unified(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_KV_OVERLAY_DONOR_BYTES", raising=False)
    monkeypatch.setattr(
        "runtime.gguf_estimate.estimate_kv_cache_bytes",
        lambda *a, **k: 50_000_000,
    )
    monkeypatch.setattr("runtime.gguf_estimate.gguf_arch_hints", lambda p: object())
    from pathlib import Path

    size_u, _ = overlay_bind.estimate_overlay_donor_bytes(
        Path("/tmp/fake.gguf"), num_ctx=4096, n_seq_max=8, kv_unified=True
    )
    size_m, _ = overlay_bind.estimate_overlay_donor_bytes(
        Path("/tmp/fake.gguf"), num_ctx=4096, n_seq_max=8, kv_unified=False
    )
    assert size_u < size_m


def test_copy_sequence_prefix_inprocess_full_range_then_trim():
    lib = MagicMock()
    mem = MagicMock()
    lib.llama_get_memory.return_value = mem
    ctx = MagicMock()
    assert copy_sequence_prefix_inprocess(
        lib, ctx, source_slot=1, target_slot=3, pos_end=256
    )
    # rm clear, cp full, rm trim
    assert lib.llama_memory_seq_rm.call_count == 2
    assert lib.llama_memory_seq_cp.call_count == 1
    cp_args = lib.llama_memory_seq_cp.call_args[0]
    # mem, src, dst, p0=-1, p1=-1
    assert int(cp_args[3].value) == -1
    assert int(cp_args[4].value) == -1
    trim_args = lib.llama_memory_seq_rm.call_args_list[1][0]
    assert int(trim_args[2].value) == 256
    assert int(trim_args[3].value) == -1
