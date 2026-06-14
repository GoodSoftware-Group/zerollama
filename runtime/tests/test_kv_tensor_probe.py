"""Tests for Phase 15 v19 tensor bind scaffold."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from runtime.kv.backend import native_available
from runtime.kv.page_bind import page_bind_health
from runtime.kv.tensor_probe import export_page_table, tensor_probe_available


@pytest.mark.skipif(not native_available(), reason="native ext not built")
def test_page_bind_table_export():
    from runtime.kv._kv_native import page_bind_clear, page_bind_set

    page_bind_clear(4)
    page_bind_set(4, 16, [10, 20])
    rows = export_page_table(4)
    assert len(rows) == 2
    assert rows[0] == {
        "page": 0,
        "block_id": 10,
        "token_start": 0,
        "token_end": 15,
    }
    assert rows[1]["block_id"] == 20
    assert rows[1]["token_start"] == 16
    page_bind_clear(4)


def test_page_bind_health_bound_when_tensor_pages_bound(monkeypatch):
    monkeypatch.setattr("runtime.kv.page_bind._native_page_bind_available", lambda: True)
    probe = {
        "memory_non_null": 1,
        "aligned": 1,
        "cell_pages_bound": 1,
        "tensor_pages_bound": 1,
        "blocker": "",
    }
    h = page_bind_health(native_ext_available=True, tensor_probe=probe)
    assert h["status"] == "bound"
    assert h["bind_level"] == "tensor"
    assert h["tensor_pages_bound"] is True
    assert h["tensor_bind_ready"] is True
    assert h["blocker"] == ""


def test_page_bind_health_cell_index_partial(monkeypatch):
    monkeypatch.setattr("runtime.kv.page_bind._native_page_bind_available", lambda: True)
    probe = {
        "memory_non_null": 1,
        "aligned": 1,
        "cell_pages_bound": 1,
        "tensor_pages_bound": 0,
        "blocker": "kv_tensor_not_materialized",
    }
    h = page_bind_health(native_ext_available=True, tensor_probe=probe)
    assert h["status"] == "partial"
    assert h["bind_level"] == "cell_index"
    assert h["cell_pages_bound"] is True


def test_page_bind_health_includes_v19_fields(monkeypatch):
    monkeypatch.setattr("runtime.kv.page_bind._native_page_bind_available", lambda: True)
    probe = {
        "memory_non_null": 1,
        "aligned": 1,
        "tensor_bind_ready": False,
        "blocker": "no_public_kv_page_handle_api",
    }
    h = page_bind_health(native_ext_available=True, tensor_probe=probe)
    assert h["tensor_bind_ready"] is False
    assert h["blocker"] == "no_public_kv_page_handle_api"
    assert h["tensor_probe"] == probe
    assert h["accounting_aligned"] is True


def test_page_bind_health_bound_not_overridden_by_misaligned(monkeypatch):
    """tensor_bound wins over aligned=0 — bound status must not flip to misaligned."""
    monkeypatch.setattr("runtime.kv.page_bind._native_page_bind_available", lambda: True)
    probe = {
        "memory_non_null": 1,
        "aligned": 0,
        "cell_pages_bound": 1,
        "tensor_pages_bound": 1,
        "blocker": "",
    }
    h = page_bind_health(native_ext_available=True, tensor_probe=probe)
    assert h["status"] == "bound"
    assert h["tensor_pages_bound"] is True


def test_page_bind_health_misaligned_when_probe_not_aligned(monkeypatch):
    monkeypatch.setattr("runtime.kv.page_bind._native_page_bind_available", lambda: True)
    probe = {"memory_non_null": 1, "aligned": 0, "llama_cells": 64, "pa_pages": 2}
    h = page_bind_health(native_ext_available=True, tensor_probe=probe)
    assert h["status"] == "misaligned"
    assert h["accounting_aligned"] is False
    assert "exceed PA page reserve" in h["reason"]


def test_page_table_native_parity():
    from runtime.kv.tensor_probe import page_table_native_parity

    logical = [
        {"page": 0, "block_id": 1, "token_start": 0, "token_end": 15},
        {"page": 1, "block_id": 2, "token_start": 16, "token_end": 31},
    ]
    assert page_table_native_parity(logical, list(logical)) is True
    assert page_table_native_parity(logical, logical[:1]) is False
    bad = [dict(logical[0]), dict(logical[1], block_id=99)]
    assert page_table_native_parity(logical, bad) is False


def test_page_bind_health_blocker_from_probe_when_cell_bound(monkeypatch):
    """cell_bound but not tensor_bound → blocker comes from probe, not the fallback string."""
    monkeypatch.setattr("runtime.kv.page_bind._native_page_bind_available", lambda: True)
    probe = {
        "memory_non_null": 1,
        "aligned": 1,
        "cell_pages_bound": 1,
        "tensor_pages_bound": 0,
        "blocker": "pa_cap_exceeded",
    }
    h = page_bind_health(native_ext_available=True, tensor_probe=probe)
    assert h["blocker"] == "pa_cap_exceeded"
    assert "llama_kv_ext_not_linked" not in h["blocker"]


def test_page_bind_health_blocker_fallback_when_no_probe(monkeypatch):
    """No probe → blocker is the ext-not-linked fallback."""
    monkeypatch.setattr("runtime.kv.page_bind._native_page_bind_available", lambda: True)
    h = page_bind_health(native_ext_available=True, tensor_probe=None)
    assert h["blocker"] == "llama_kv_ext_not_linked_or_no_decode"


def test_tensor_probe_available_false_without_linked_build():
    if tensor_probe_available():
        pytest.skip("linked build present")
    from runtime.kv.tensor_probe import run_tensor_probe

    assert run_tensor_probe(1, 0, 0) is None


def test_engine_page_bind_health_probes_running_slot():
    """Probe is called for the first running request's slot, not a fallback slot 0."""
    from runtime.engine import InferenceEngine
    from runtime.scheduler.scheduler import Request

    req = MagicMock(spec=Request)
    req.kv_slot = 3

    eng = MagicMock()
    eng.scheduler = MagicMock(running=[req])
    eng._inprocess_ctx_for_health.return_value = (MagicMock(), MagicMock())
    with patch("runtime.kv.backend.native_available", return_value=True):
        with patch(
            "runtime.kv.page_bind.page_bind_tensor_probe_for_ctx",
            return_value={"memory_non_null": 1, "aligned": 1},
        ) as mock_probe:
            with patch("runtime.kv.page_bind._native_page_bind_available", return_value=True):
                out = InferenceEngine._kv_page_bind_health(eng)
    mock_probe.assert_called_once()
    _, kwargs = mock_probe.call_args
    assert kwargs.get("kv_slot") == 3
    assert out.get("tensor_probe") is not None


def test_engine_page_bind_health_no_probe_when_no_running_requests():
    """No running request → tensor_probe omitted from health (not misleading slot 0)."""
    from runtime.engine import InferenceEngine

    eng = MagicMock()
    eng.scheduler = MagicMock(running=[])
    eng._inprocess_ctx_for_health.return_value = (MagicMock(), MagicMock())
    with patch("runtime.kv.backend.native_available", return_value=True):
        with patch(
            "runtime.kv.page_bind.page_bind_tensor_probe_for_ctx",
        ) as mock_probe:
            with patch("runtime.kv.page_bind._native_page_bind_available", return_value=True):
                out = InferenceEngine._kv_page_bind_health(eng)
    mock_probe.assert_not_called()
    assert out.get("tensor_probe") is None
