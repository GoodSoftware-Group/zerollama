"""Tests for YAML vram: defaults applied at startup."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from runtime.vram_yaml_defaults import apply_vram_defaults_from_config


@pytest.fixture
def yaml_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "single_gpu.yaml"
    cfg.write_text(
        """
device_count: 1
vram:
  min_free: 1GiB
  training_reserve: 2GiB
  estimate_factor_autotune: auto
  clamp_num_ctx: "0"
""",
        encoding="utf-8",
    )
    return cfg


def test_apply_sets_unset_env(monkeypatch: pytest.MonkeyPatch, yaml_config: Path):
    for key in (
        "ZEROLLAMA_RUNTIME_VRAM_MIN_FREE",
        "ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE",
        "ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX",
    ):
        monkeypatch.delenv(key, raising=False)
    import runtime.vram_yaml_defaults as mod

    mod._APPLIED = False
    mod._APPLY_RESULT = None
    result = apply_vram_defaults_from_config(yaml_config, force=True)
    assert "ZEROLLAMA_RUNTIME_VRAM_MIN_FREE" in result["applied"]
    assert os.environ["ZEROLLAMA_RUNTIME_VRAM_MIN_FREE"] == "1GiB"
    assert os.environ["ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX"] == "0"
    for key in result["applied"]:
        os.environ.pop(key, None)


def test_apply_skips_existing_env(monkeypatch: pytest.MonkeyPatch, yaml_config: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_MIN_FREE", "512MiB")
    import runtime.vram_yaml_defaults as mod

    mod._APPLIED = False
    mod._APPLY_RESULT = None
    result = apply_vram_defaults_from_config(yaml_config, force=True)
    assert "ZEROLLAMA_RUNTIME_VRAM_MIN_FREE" in result["skipped"]
    assert os.environ["ZEROLLAMA_RUNTIME_VRAM_MIN_FREE"] == "512MiB"


def test_apply_apple_silicon_repo_defaults(monkeypatch: pytest.MonkeyPatch):
    configs = Path(__file__).resolve().parents[1] / "configs" / "apple_silicon.yaml"
    assert configs.is_file()
    for key in (
        "ZEROLLAMA_RUNTIME_VRAM_MIN_FREE",
        "ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE",
    ):
        monkeypatch.delenv(key, raising=False)
    import runtime.vram_yaml_defaults as mod

    mod._APPLIED = False
    mod._APPLY_RESULT = None
    result = apply_vram_defaults_from_config(configs, force=True)
    assert "ZEROLLAMA_RUNTIME_VRAM_MIN_FREE" in result["applied"]
    assert os.environ["ZEROLLAMA_RUNTIME_VRAM_MIN_FREE"] == "512MiB"
    assert os.environ["ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE"] == "1GiB"
    for key in result["applied"]:
        os.environ.pop(key, None)


def test_apply_serve_llama_fork_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = tmp_path / "dual_vram.yaml"
    cfg.write_text(
        """
device_count: 2
serve:
  llama_fork: 1
  llama_fork_profile: vram
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("ZEROLLAMA_LLAMA_FORK", raising=False)
    monkeypatch.delenv("ZEROLLAMA_LLAMA_FORK_PROFILE", raising=False)
    import runtime.vram_yaml_defaults as mod

    mod._APPLIED = False
    mod._APPLY_RESULT = None
    result = apply_vram_defaults_from_config(cfg, force=True)
    assert "ZEROLLAMA_LLAMA_FORK" in result["applied"]
    assert "ZEROLLAMA_LLAMA_FORK_PROFILE" in result["applied"]
    assert os.environ["ZEROLLAMA_LLAMA_FORK"] == "1"
    assert os.environ["ZEROLLAMA_LLAMA_FORK_PROFILE"] == "vram"
    for key in result["applied"]:
        os.environ.pop(key, None)


def test_dual_4090_vram_repo_config_exists():
    path = Path(__file__).resolve().parents[1] / "configs" / "dual_4090_vram.yaml"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "llama_fork_profile: vram" in text
    assert "llama_fork: 1" in text
