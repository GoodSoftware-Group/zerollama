"""Health caching and shared-interpreter VRAM probe defaults."""

from __future__ import annotations

import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.gpu_vram import (
    _shared_auto_without_smi,
    nvidia_free_vram_bytes,
    shared_interpreter_embedded,
    vram_probe_effective,
    vram_probe_mode,
    warn_shared_interpreter_no_smi_once,
)


def test_shared_interpreter_detects_training_native(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_SHARED_PYTHON", raising=False)
    fake = type(sys)("fake")
    monkeypatch.setitem(sys.modules, "ollama_training_native", fake)
    try:
        assert shared_interpreter_embedded() is True
    finally:
        sys.modules.pop("ollama_training_native", None)


def test_vram_probe_prefers_smi_when_shared(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_PROBE", "auto")
    monkeypatch.setitem(sys.modules, "ollama_training_native", object())
    try:
        with patch("runtime.gpu_vram.nvidia_smi_available", return_value=True):
            assert vram_probe_mode() == "smi"
    finally:
        sys.modules.pop("ollama_training_native", None)


@pytest.fixture
def engine(cfg_root, tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=gguf,
        num_blocks=64,
        block_size=16,
        device_count=1,
    )
    return InferenceEngine(cfg)


def test_health_body_exposes_probe_mode(engine, monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_SHARED_PYTHON", raising=False)
    sys.modules.pop("ollama_training_native", None)
    import runtime.gpu_vram as gv

    monkeypatch.setattr(gv, "_active_vram_probe", None)
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_PROBE", "smi")
    with patch(
        "runtime.engine.nvidia_free_vram_by_device",
        return_value={0: 8 * 1024**3},
    ):
        with patch(
            "runtime.go_coordination.go_coordination_health",
            return_value={},
        ):
            with patch.object(
                engine.coordinator,
                "policy_snapshot",
                return_value={},
            ):
                with patch.object(engine, "vram_estimate_and_budget", return_value=(None, None)):
                    out = engine._health_body()
    assert out.get("vram_probe_mode") == "smi"
    assert out.get("vram_probe_effective") == "smi"
    assert "shared_interpreter" in out


def test_health_uses_short_ttl_cache(engine):
    calls: list[int] = []

    def _body():
        calls.append(1)
        return {"status": "ok", "n": len(calls)}

    with patch.object(engine, "_health_body", side_effect=_body):
        a = engine.health()
        b = engine.health()
    assert a == b
    assert len(calls) == 1


def test_vram_probe_effective_honors_explicit_smi_over_last_probe(monkeypatch):
    import runtime.gpu_vram as gv

    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_PROBE", "smi")
    monkeypatch.setattr(gv, "_active_vram_probe", "nvml")
    assert vram_probe_effective() == "smi"


def test_vram_probe_effective_skipped_without_smi(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_PROBE", "auto")
    monkeypatch.setitem(sys.modules, "ollama_training_native", object())
    try:
        with patch("runtime.gpu_vram.nvidia_smi_available", return_value=False):
            assert vram_probe_effective() == "skipped"
    finally:
        sys.modules.pop("ollama_training_native", None)


def test_shared_auto_without_smi_skips_nvml_probe(monkeypatch):
    import runtime.gpu_vram as gv

    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_PROBE", "auto")
    monkeypatch.setitem(sys.modules, "ollama_training_native", object())
    with gv._smi_lock:
        gv._smi_cache.clear()
    monkeypatch.setattr(gv, "_active_vram_probe", None)
    try:
        with patch("runtime.gpu_vram.nvidia_smi_available", return_value=False):
            assert _shared_auto_without_smi() is True
            with patch("runtime.gpu_vram._query_nvml_free_vram_bytes") as nvml:
                assert nvidia_free_vram_bytes(0, fresh=True) is None
                nvml.assert_not_called()
    finally:
        sys.modules.pop("ollama_training_native", None)


def test_warn_shared_no_smi_once(monkeypatch, caplog):
    import logging

    import runtime.gpu_vram as gv

    monkeypatch.setattr(gv, "_shared_no_smi_warned", False)
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_PROBE", "auto")
    monkeypatch.setitem(sys.modules, "ollama_training_native", object())
    with patch("runtime.gpu_vram.nvidia_smi_available", return_value=False):
        with caplog.at_level(logging.WARNING, logger="gpu_vram"):
            warn_shared_interpreter_no_smi_once()
            warn_shared_interpreter_no_smi_once()
    assert sum(1 for r in caplog.records if "nvidia-smi" in r.message) == 1


def test_health_cache_invalidated_on_handoff(engine):
    seq: list[int] = []

    def _body():
        seq.append(1)
        return {"status": "ok", "n": len(seq)}

    with patch.object(engine, "_health_body", side_effect=_body):
        assert engine.health()["n"] == 1
        engine.training_handoff()
        assert engine.health()["n"] == 2
    assert len(seq) == 2


def test_health_single_flight(engine):
    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def _slow_body():
        calls.append(1)
        started.set()
        assert release.wait(timeout=5)
        return {"status": "ok", "seq": len(calls)}

    with patch.object(engine, "_health_body", side_effect=_slow_body):
        t0 = threading.Thread(target=engine.health)
        t1 = threading.Thread(target=engine.health)
        t0.start()
        t1.start()
        assert started.wait(timeout=5)
        time.sleep(0.05)
        release.set()
        t0.join(timeout=5)
        t1.join(timeout=5)
    assert len(calls) == 1
