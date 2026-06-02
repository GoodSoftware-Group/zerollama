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
