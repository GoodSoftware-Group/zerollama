"""Autoconfig YAML selection tests."""

from pathlib import Path

import pytest

from runtime.autoconfig import autoconfig_health, resolved_config_path


def test_l3_profile_selects_agent_yaml(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_CONFIG", raising=False)
    monkeypatch.setenv("ZEROLLAMA_L3_PROFILE", "agent")
    path = resolved_config_path()
    assert path.name == "l3_agent_subprocess.yaml"
    health = autoconfig_health()
    assert health.get("l3_profile") == "agent"
    assert health.get("pick") == "l3_agent"
