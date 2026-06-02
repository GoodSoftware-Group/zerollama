import os

from runtime.vram_env_apply import (
    apply_exported_vram_env,
    apply_export_path,
    vram_apply_exported_env_enabled,
)
from runtime.vram_factor_export import export_last_calibration


def test_apply_disabled_by_default(monkeypatch, tmp_path):
    import runtime.vram_env_apply as mod

    mod._APPLIED = False
    mod._APPLY_RESULT = None
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_VRAM_APPLY_EXPORTED_ENV", raising=False)
    assert not vram_apply_exported_env_enabled()
    r = apply_exported_vram_env()
    assert r["applied"] is False
    assert r["reason"] == "disabled"
    r2 = apply_exported_vram_env()
    assert r2["reason"] == "disabled"


def test_apply_loads_factor(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_EXPORT", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_APPLY_EXPORTED_ENV", "1")
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR", raising=False)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    export_last_calibration(1.35, model=model)
    r = apply_exported_vram_env(force=True)
    assert r["applied"] is True
    assert os.environ.get("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR") == "1.35"
    assert apply_export_path().is_file()


def test_apply_skips_when_env_set(monkeypatch, tmp_path):
    import runtime.vram_env_apply as mod

    mod._APPLIED = False
    mod._APPLY_RESULT = None
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_APPLY_EXPORTED_ENV", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR", "2.0")
    r = apply_exported_vram_env()
    assert r["applied"] is False
    assert r["reason"] == "env_already_set"
