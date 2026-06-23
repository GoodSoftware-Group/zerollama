"""Tests for prefill cancel token (Phase 15 v31 engine wiring)."""

from __future__ import annotations

from unittest.mock import patch

from runtime.kv.prefill_cancel import PrefillCancelToken


def test_prefill_cancel_token_sets_abort():
    token = PrefillCancelToken()
    with patch("runtime.kv.prefill_cancel.prefill_abort_set") as mock_set:
        token.cancel()
        mock_set.assert_called_once()
    assert token.is_cancelled()


def test_prefill_cancel_token_reset_clears_abort():
    token = PrefillCancelToken()
    with patch("runtime.kv.prefill_cancel.prefill_abort_clear") as mock_clear:
        token.cancel()
        token.reset()
        mock_clear.assert_called_once()
    assert not token.is_cancelled()
