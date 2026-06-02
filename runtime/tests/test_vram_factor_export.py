import json

from runtime.vram_autotune_persist import autotune_state_path, clear_persisted_autotune
from runtime.vram_autotune_persist import clear_persisted_autotune
from runtime.vram_factor_export import (
    clear_export_files,
    export_catalog_path,
    export_factor_catalog,
    export_last_calibration,
    export_last_factor_path,
    export_status,
    vram_factor_export_enabled,
)


def test_export_last_calibration(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_EXPORT", "1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    assert export_last_calibration(1.2, model=model, observed_bytes=1200, estimated_raw_bytes=1000)
    path = export_last_factor_path()
    text = path.read_text()
    assert "ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR=1.2" in text
    assert str(model.resolve()) in text
    st = export_status()
    assert st["last_export_exists"] is True


def test_export_catalog_from_persist(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_EXPORT", "1")
    m_a = tmp_path / "a.gguf"
    m_b = tmp_path / "b.gguf"
    m_a.write_bytes(b"a")
    m_b.write_bytes(b"b")
    from runtime.vram_autotune_persist import save_persisted_autotune

    save_persisted_autotune(1.1, model=m_a)
    save_persisted_autotune(1.9, model=m_b)
    assert export_factor_catalog()
    text = export_catalog_path().read_text()
    assert "1.1" in text
    assert "1.9" in text
    assert str(m_a.resolve()) in text


def test_clear_persisted_removes_exports(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_EXPORT", "1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    export_last_calibration(1.1, model=model)
    assert export_last_factor_path().is_file()
    clear_persisted_autotune()
    assert not export_last_factor_path().exists()
    assert not export_catalog_path().exists()


def test_export_off(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_EXPORT", "0")
    assert not vram_factor_export_enabled()
    assert not export_last_calibration(1.0, model=tmp_path / "x.gguf")
