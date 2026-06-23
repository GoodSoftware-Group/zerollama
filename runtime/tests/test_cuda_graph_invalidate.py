"""CUDA graph invalidation wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from runtime.kv.cuda_graph_invalidate import (
    cuda_graph_invalidate_enabled,
    invalidate_cuda_graphs,
)
from runtime.decode_graph_policy import bump_decode_graph_epoch, reset_decode_graph_epochs


def test_invalidate_disabled_by_env(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_DECODE_GRAPH_INVALIDATE", "0")
    out = invalidate_cuda_graphs(123, reason="test")
    assert out["path"] == "disabled"


def test_bump_epoch_calls_invalidate():
    reset_decode_graph_epochs()
    with patch(
        "runtime.kv.cuda_graph_invalidate.invalidate_cuda_graphs",
        return_value={"ok": True, "backends_cleared": 1, "path": "native"},
    ) as inv:
        bump_decode_graph_epoch(2, reason="slot_clear", ctx_ptr=0xABC)
    inv.assert_called_once_with(0xABC, reason="slot_clear")


def test_invalidate_no_ctx():
    assert cuda_graph_invalidate_enabled() is True
    out = invalidate_cuda_graphs(None, reason="x")
    assert out["ok"] is False
    assert out["reason"] == "x"


def test_invalidate_native_path():
    mock_mod = MagicMock()
    mock_mod.invalidate_cuda_graphs.return_value = {
        "ok": True,
        "backends_cleared": 2,
    }
    with patch.dict(
        "sys.modules",
        {"runtime.kv._kv_native": mock_mod},
    ):
        out = invalidate_cuda_graphs(42, reason="slot_clear")
    assert out["path"] == "native"
    assert out["backends_cleared"] == 2
