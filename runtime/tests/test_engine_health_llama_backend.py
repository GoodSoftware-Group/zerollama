"""Health exposes configured llama forward backend (Phase 14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine


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
    )
    cfg.llama_server_bin.write_text("#!/bin/sh\ntrue\n")
    cfg.llama_server_bin.chmod(0o755)
    return InferenceEngine(cfg)


def test_health_llama_backend_from_config(engine: InferenceEngine):
    body = engine.health()
    assert body["llama_backend"] == "inprocess"


def test_health_llama_backend_env_wins(
    engine: InferenceEngine, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "llama-cpp-python")
    body = engine.health()
    assert body["llama_backend"] == "llama-cpp-python"
