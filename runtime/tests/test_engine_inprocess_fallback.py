"""In-process → subprocess fallback when Metal ctypes load fails (darwin auto)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import threading

import pytest

from runtime.worker.factory import LlamaBackendKind
from runtime.worker.llama_server import LlamaServerError


def test_inprocess_fallback_enabled_auto_darwin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from runtime.engine import InferenceEngine

    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_INPROCESS_FALLBACK", raising=False)
    eng = InferenceEngine.__new__(InferenceEngine)
    eng.config = MagicMock()
    eng.config.llama_server_bin = tmp_path / "llama-server"
    eng.config.llama_server_bin.write_text("x")
    assert eng._inprocess_fallback_enabled() is True


def test_inprocess_fallback_disabled_by_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from runtime.engine import InferenceEngine

    monkeypatch.setenv("ZEROLLAMA_RUNTIME_INPROCESS_FALLBACK", "0")
    eng = InferenceEngine.__new__(InferenceEngine)
    eng.config = MagicMock()
    eng.config.llama_server_bin = tmp_path / "llama-server"
    eng.config.llama_server_bin.write_text("x")
    assert eng._inprocess_fallback_enabled() is False


def test_inprocess_fallback_disabled_off_darwin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from runtime.engine import InferenceEngine

    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_INPROCESS_FALLBACK", raising=False)
    eng = InferenceEngine.__new__(InferenceEngine)
    eng.config = MagicMock()
    eng.config.llama_server_bin = tmp_path / "llama-server"
    eng.config.llama_server_bin.write_text("x")
    assert eng._inprocess_fallback_enabled() is False


def test_start_server_with_fallback_switches_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from runtime.engine import InferenceEngine

    model = tmp_path / "m.gguf"
    model.write_text("x")
    bin_p = tmp_path / "llama-server"
    bin_p.write_text("x")

    failing = MagicMock()
    failing.model = model
    failing.start.side_effect = LlamaServerError("ctypes load failed")

    subprocess_worker = MagicMock()
    subprocess_worker.model = model

    eng = InferenceEngine.__new__(InferenceEngine)
    eng.config = MagicMock()
    eng.config.llama_server_bin = bin_p
    eng.config.host = "127.0.0.1"
    eng.config.port = 8081
    eng.config.llama_cpp_lib = None
    eng.config.llama_cpp_root = None
    eng.config.main_gpu = 0
    eng._server = failing
    eng._llama_backend_override = None
    eng._health_cache = {"llama_backend": "inprocess"}
    eng._health_cache_at = 999.0
    eng._health_cache_lock = threading.Lock()

    monkeypatch.setattr(
        eng,
        "_requested_llama_backend",
        lambda: LlamaBackendKind.INPROCESS,
    )
    monkeypatch.setattr(eng, "_inprocess_fallback_enabled", lambda: True)
    monkeypatch.setattr(
        "runtime.engine.create_llama_worker",
        lambda **kw: subprocess_worker,
    )

    eng._start_server_with_fallback(extra_args=["-ngl", "99"])

    assert eng._llama_backend_override == LlamaBackendKind.SUBPROCESS
    assert eng._server is subprocess_worker
    subprocess_worker.start.assert_called_once()
    failing.stop.assert_called_once()
    assert eng._health_cache is None
