"""Tests for opt-in infer_trace helper."""

from __future__ import annotations

import logging

import pytest

from runtime.infer_trace import infer_trace, infer_trace_enabled


def test_infer_trace_disabled_by_default(monkeypatch, caplog):
    monkeypatch.delenv("ZEROLLAMA_INFER_TRACE", raising=False)
    assert infer_trace_enabled() is False
    with caplog.at_level(logging.INFO, logger="zerollama-runtime"):
        infer_trace("test.event", foo=1)
    assert "infer_trace" not in caplog.text


def test_infer_trace_logs_when_enabled(monkeypatch, caplog):
    monkeypatch.setenv("ZEROLLAMA_INFER_TRACE", "1")
    assert infer_trace_enabled() is True
    with caplog.at_level(logging.INFO, logger="zerollama-runtime"):
        infer_trace("test.event", foo="bar")
    assert "infer_trace test.event" in caplog.text
    assert "foo='bar'" in caplog.text
