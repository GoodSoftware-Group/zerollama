"""Metal-unified VRAM probe on macOS."""

from __future__ import annotations

import pytest


def test_metal_unified_probe_on_darwin(monkeypatch: pytest.MonkeyPatch):
    import runtime.gpu_vram as gv

    monkeypatch.setattr(gv.sys, "platform", "darwin")
    monkeypatch.setattr(gv, "nvidia_smi_available", lambda: False)
    monkeypatch.setattr(gv, "nvml_available", lambda: False)
    # Block both nvidia probe paths so the test falls through to metal-unified
    monkeypatch.setattr(gv, "_query_nvml_free_vram_bytes", lambda _idx: None)
    monkeypatch.setattr(gv, "_query_nvidia_smi_free_vram_bytes", lambda _idx: None)
    monkeypatch.setattr(gv, "_shared_auto_without_smi", lambda: False)
    monkeypatch.setattr(
        gv,
        "_host_unified_free_vram_bytes",
        lambda: 8 * 1024**3,
    )
    assert gv.gpu_vram_check_enabled() is True
    val, name = gv._query_free_vram_bytes(0)
    assert val == 8 * 1024**3
    assert name == "metal-unified"


def test_autoconfig_picks_apple_silicon_on_darwin(monkeypatch: pytest.MonkeyPatch):
    import runtime.autoconfig as ac

    monkeypatch.setattr(ac.sys, "platform", "darwin")
    monkeypatch.setattr(ac, "auto_config_enabled", lambda: True)
    path = ac.resolve_default_config_path()
    assert path.name == "apple_silicon.yaml"
