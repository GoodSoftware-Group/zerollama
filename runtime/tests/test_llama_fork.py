from pathlib import Path
from unittest.mock import MagicMock

import pytest

from runtime.gpu_profiles import flags_from_gpu_config, llama_argv_from_profile_flags
from runtime.llama_fork import (
    fork_health,
    fork_detection_source,
    llama_fork_enabled,
    llama_fork_env_override,
    normalize_cache_type,
    probe_fork_llama_server,
)


def test_normalize_cache_type_aliases():
    assert normalize_cache_type("turbo3_0") == "tbq3_0"
    assert normalize_cache_type("qjl1_256") == "qjl1_256"


def test_llama_fork_env_override():
    assert llama_fork_env_override() is None


def test_probe_fork_detects_ctx_checkpoints(monkeypatch, tmp_path):
    from runtime.llama_fork import clear_fork_probe_cache

    fake = tmp_path / "llama-server"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    clear_fork_probe_cache()
    monkeypatch.setattr(
        "runtime.llama_fork.subprocess.run",
        lambda *a, **k: MagicMock(
            returncode=0,
            stdout="--ctx-checkpoints N",
            stderr="",
        ),
    )
    assert probe_fork_llama_server(str(fake)) is True


def test_probe_fork_stock_help(monkeypatch, tmp_path):
    from runtime.llama_fork import clear_fork_probe_cache

    fake = tmp_path / "llama-server"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    clear_fork_probe_cache()
    monkeypatch.setattr(
        "runtime.llama_fork.subprocess.run",
        lambda *a, **k: MagicMock(
            returncode=0,
            stdout="--cache-type-k TYPE  q4_0, q8_0, f16",
            stderr="",
        ),
    )
    assert probe_fork_llama_server(str(fake)) is False


def test_llama_fork_enabled_force_env(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_LLAMA_FORK", "1")
    assert llama_fork_enabled() is True


def test_llama_fork_env_off_overrides_probe(monkeypatch, tmp_path):
    fake = tmp_path / "llama-server"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("ZEROLLAMA_LLAMA_FORK", "stock")
    monkeypatch.setattr(
        "runtime.llama_fork.probe_fork_llama_server",
        lambda _bin: True,
    )
    assert llama_fork_enabled(llama_server_bin=fake) is False
    assert fork_detection_source(llama_server_bin=fake) == "env_off"


def test_flags_from_gpu_config_stock_path():
    cfg = {
        "llama_server_flags": {"cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        "_eliza_fork_llama_server_flags": {"cache_type_k": "qjl1_256"},
    }
    flags, fb = flags_from_gpu_config(cfg, fork_enabled=False)
    assert flags["cache_type_k"] == "q8_0"
    assert fb is False


def test_flags_from_gpu_config_fork_merge():
    cfg = {
        "llama_server_flags": {
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "n_parallel": 8,
        },
        "_eliza_fork_llama_server_flags": {
            "cache_type_k": "qjl1_256",
            "cache_type_v": "q4_polar",
        },
        "_fork_only_llama_server_flags": {
            "ctx_checkpoints": 8,
            "ctx_checkpoint_interval": 8192,
        },
    }
    flags, fb = flags_from_gpu_config(cfg, fork_enabled=True)
    assert flags["cache_type_k"] == "qjl1_256"
    assert flags["cache_type_v"] == "q4_polar"
    assert flags["ctx_checkpoints"] == 8
    assert fb is False


def test_llama_argv_emits_fork_checkpoints():
    args = llama_argv_from_profile_flags(
        {"ctx_checkpoints": 4, "ctx_checkpoint_interval": 8192, "batch_size": 512}
    )
    assert "--ctx-checkpoints" in args
    assert "4" in args
    assert "--ctx-checkpoint-interval" in args


def test_runtime_config_fork_profile(monkeypatch, tmp_path):
    from runtime.config import RuntimeConfig

    fake_bin = tmp_path / "llama-server"
    fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_bin.chmod(0o755)
    monkeypatch.setenv("ZEROLLAMA_GPU_PROFILE", "1")
    monkeypatch.setenv("ZEROLLAMA_LLAMA_FORK", "1")
    monkeypatch.setenv("LLAMA_SERVER_BIN", str(fake_bin))
    monkeypatch.setattr(
        "runtime.gpu_profiles.detect_nvidia_gpu_name",
        lambda device_index=0: "NVIDIA GeForce RTX 4090",
    )
    monkeypatch.setattr(
        "runtime.gpu_profiles.detect_gpu_total_vram_gb",
        lambda device_index=0: 24.0,
    )
    path = Path(__file__).resolve().parents[1] / "configs" / "single_gpu.yaml"
    cfg = RuntimeConfig.from_file(path)
    assert cfg.gpu_profile is not None
    assert cfg.gpu_profile.get("llama_fork") is True
    args = cfg.llama_server_args()
    assert "--cache-type-k" in args
    idx = args.index("--cache-type-k")
    assert args[idx + 1] == "qjl1_256"
    assert "--ctx-checkpoints" in args


def test_fork_health():
    h = fork_health()
    assert "enabled" in h
    assert "pin_ref" in h
