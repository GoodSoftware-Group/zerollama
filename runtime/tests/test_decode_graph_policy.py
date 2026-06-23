"""Breakable decode graph epoch scaffold."""

from __future__ import annotations

from unittest.mock import MagicMock

from runtime.decode_graph_policy import (
    bump_all_decode_graph_epochs,
    bump_decode_graph_epoch,
    decode_graph_epoch,
    decode_graph_health,
    graph_capture_key,
    reset_decode_graph_epochs,
)
from runtime.worker.libllama_ctypes import LlamaLoadedSession


def test_bump_decode_graph_epoch_per_slot():
    reset_decode_graph_epochs()
    assert decode_graph_epoch(2) == 0
    assert bump_decode_graph_epoch(2, reason="test") == 1
    assert decode_graph_epoch(2) == 1
    assert decode_graph_epoch(3) == 0


def test_graph_capture_key_changes_on_bump():
    reset_decode_graph_epochs()
    assert graph_capture_key(2) == "2:0:0"
    bump_decode_graph_epoch(2, reason="slot_clear")
    assert graph_capture_key(2) == "2:1:1"


def test_graph_capture_key_global_invalidation_without_slot_bump():
    reset_decode_graph_epochs()
    assert graph_capture_key(9) == "9:0:0"
    bump_all_decode_graph_epochs(reason="model_swap")
    assert graph_capture_key(9) == "9:0:1"


def test_decode_graph_health_scaffold():
    reset_decode_graph_epochs()
    bump_decode_graph_epoch(1, reason="slot_clear")
    h = decode_graph_health()
    assert h["capture_ready"] is False
    assert h["capture_key_format"] == "slot_id:slot_epoch:global_epoch"
    assert h["slot_epochs"]["1"] == 1
    assert "invalidation" in h["note"]


def test_bump_all_decode_graph_epochs():
    reset_decode_graph_epochs()
    bump_decode_graph_epoch(1, reason="a")
    bump_decode_graph_epoch(2, reason="b")
    g = bump_all_decode_graph_epochs(reason="model_swap")
    assert g >= 2
    assert decode_graph_epoch(1) == 2
    assert decode_graph_epoch(2) == 2


def test_session_close_bumps_global_epoch():
    reset_decode_graph_epochs()
    session = MagicMock(spec=LlamaLoadedSession)
    session._ctx = None
    session._model = None
    session._seq_last_owner = {}
    session._infer_lock = __import__("threading").RLock()
    LlamaLoadedSession.close(session)
    assert decode_graph_epoch(-1) >= 1
