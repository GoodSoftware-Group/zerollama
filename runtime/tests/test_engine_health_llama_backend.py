"""Health exposes configured llama forward backend (Phase 14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.worker.factory import LlamaBackendKind


@pytest.fixture
def engine(cfg_root, tmp_path: Path):
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=tmp_path / "llama-server",
        llama_model=None,
        num_blocks=8,
        block_size=16,
        device_count=1,
        llama_backend="inprocess",
        llama_backend_from_file=True,
    )
    cfg.llama_server_bin.write_text("#!/bin/sh\ntrue\n")
    cfg.llama_server_bin.chmod(0o755)
    return InferenceEngine(cfg)


def test_health_llama_backend_from_config(engine: InferenceEngine):
    body = engine.health()
    assert body["llama_backend"] == "inprocess"
    assert body["llama_backend_source"] == "config"
    assert body["llama_backend_requested"] == "inprocess"
    assert body["llama_backend_fallback"] is False
    assert "llama_cpp" not in body
    assert body["llama_patches"]["status"] == "pass"
    assert body["llama_patches"]["required_patches_ok"] is True


def test_health_llama_backend_fallback_flag(engine: InferenceEngine):
    engine._llama_backend_override = LlamaBackendKind.SUBPROCESS
    body = engine.health()
    assert body["llama_backend"] == "subprocess"
    assert body["llama_backend_requested"] == "inprocess"
    assert body["llama_backend_fallback"] is True


def test_health_llama_backend_default_subprocess(
    cfg_root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", raising=False)
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=tmp_path / "llama-server",
        llama_model=None,
        num_blocks=8,
        block_size=16,
        device_count=1,
    )
    cfg.llama_server_bin.write_text("#!/bin/sh\ntrue\n")
    cfg.llama_server_bin.chmod(0o755)
    body = InferenceEngine(cfg).health()
    assert body["llama_backend"] == "subprocess"
    assert body["llama_backend_source"] == "default"
    assert "llama_cpp" not in body


def test_health_llama_backend_env_wins(
    engine: InferenceEngine, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "llama-cpp-python")
    body = engine.health()
    assert body["llama_backend"] == "llama-cpp-python"
    assert body["llama_backend_source"] == "env"
    assert body["llama_cpp"]["gpu_mode"] == "cpu"
    assert body["llama_cpp"]["n_gpu_layers"] == 0
    assert body["llama_cpp"]["loaded"] is False


def test_health_embed_boot_when_env_set(
    cfg_root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_EMBED_BOOT", "test-boot-token")
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", raising=False)
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=tmp_path / "llama-server",
        llama_model=None,
        num_blocks=8,
        block_size=16,
        device_count=1,
    )
    cfg.llama_server_bin.write_text("#!/bin/sh\ntrue\n")
    cfg.llama_server_bin.chmod(0o755)
    body = InferenceEngine(cfg).health()
    assert body.get("embed_boot") == "test-boot-token"
