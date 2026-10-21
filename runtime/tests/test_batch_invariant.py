"""DI5 — ZEROLLAMA_BATCH_INVARIANT eval hygiene."""

from __future__ import annotations

import pytest

from runtime.batch_invariant import (
    apply_batch_invariant_options,
    batch_invariant_denies_auto_batch,
    batch_invariant_denies_prefix_cache,
    batch_invariant_enabled,
    batch_invariant_health,
    batch_invariant_serializes_generate_batch,
    seq_seed_default,
)


def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_BATCH_INVARIANT", raising=False)
    assert batch_invariant_enabled() is False
    assert apply_batch_invariant_options({"temperature": 0.7}) == {"temperature": 0.7}
    assert batch_invariant_denies_prefix_cache() is False
    assert batch_invariant_denies_auto_batch() is False
    assert batch_invariant_serializes_generate_batch() is False


def test_enabled_forces_greedy_over_client_temp(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_BATCH_INVARIANT", "1")
    monkeypatch.setenv("ZEROLLAMA_SEQ_SEED", "42")
    assert batch_invariant_enabled() is True
    assert seq_seed_default() == 42
    # Client/Modelfile temperature must not win — that was the audit gap.
    out = apply_batch_invariant_options({"temperature": 0.8, "top_k": 40, "top_p": 0.9})
    assert out["temperature"] == 0.0
    assert out["top_k"] == 1
    assert out["top_p"] == 1.0
    assert out["seed"] == 42


def test_seed_client_nonneg_wins_minus_one_uses_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_BATCH_INVARIANT", "1")
    monkeypatch.setenv("ZEROLLAMA_SEQ_SEED", "99")
    assert apply_batch_invariant_options({"seed": 7})["seed"] == 7
    assert apply_batch_invariant_options({"seed": -1})["seed"] == 99
    assert apply_batch_invariant_options(None)["seed"] == 99


def test_denies_prefix_auto_batch_and_serializes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_BATCH_INVARIANT", "1")
    assert batch_invariant_denies_prefix_cache() is True
    assert batch_invariant_denies_auto_batch() is True
    assert batch_invariant_serializes_generate_batch() is True
    from runtime.env import radix_prefix_share_enabled

    monkeypatch.setenv("ZEROLLAMA_RADIX_PREFIX_SHARE", "1")
    assert radix_prefix_share_enabled() is False


def test_health_eval_hygiene(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_BATCH_INVARIANT", "1")
    h = batch_invariant_health()
    assert h["enabled"] is True
    assert h["mode"] == "eval_hygiene"
    assert h["forces_greedy"] is True
    assert h["serializes_generate_batch"] is True
    assert "eval hygiene" in h["note"]
