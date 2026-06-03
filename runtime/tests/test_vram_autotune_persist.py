import json

import runtime.gpu_vram as gpu_vram
from runtime.gpu_vram import (
    effective_vram_estimate_factor,
    session_vram_estimate_factor,
    set_session_vram_estimate_factor,
)
from runtime.vram_autotune_persist import (
    autotune_state_path,
    clear_persisted_autotune,
    load_persisted_autotune,
    model_autotune_key,
    save_persisted_autotune,
)


def _reset_autotune(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_STATE_DIR", str(tmp_path))
    gpu_vram._session_autotune_factor = None
    gpu_vram._session_autotune_model = None
    clear_persisted_autotune()


def test_persist_save_and_hydrate(monkeypatch, tmp_path):
    _reset_autotune(monkeypatch, tmp_path)
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR", "1.0")

    model = tmp_path / "a.gguf"
    model.write_bytes(b"x")
    assert save_persisted_autotune(1.35, model=model)
    path = autotune_state_path()
    data = json.loads(path.read_text())
    assert data["version"] == 2
    key = model_autotune_key(model)
    assert data["models"][key]["estimate_factor"] == 1.35

    gpu_vram._session_autotune_factor = None
    gpu_vram._session_autotune_model = None
    assert load_persisted_autotune(model) == 1.35
    assert effective_vram_estimate_factor(gguf=model) == 1.35


def test_per_model_factors_differ(monkeypatch, tmp_path):
    _reset_autotune(monkeypatch, tmp_path)
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR", "1.0")

    m_a = tmp_path / "a.gguf"
    m_b = tmp_path / "b.gguf"
    m_a.write_bytes(b"a")
    m_b.write_bytes(b"b")
    save_persisted_autotune(1.1, model=m_a)
    save_persisted_autotune(1.9, model=m_b)

    gpu_vram._session_autotune_factor = None
    gpu_vram._session_autotune_model = None
    assert effective_vram_estimate_factor(gguf=m_a) == 1.1
    assert effective_vram_estimate_factor(gguf=m_b) == 1.9


def test_v1_migration(monkeypatch, tmp_path):
    _reset_autotune(monkeypatch, tmp_path)
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST", "1")
    model = tmp_path / "legacy.gguf"
    model.write_bytes(b"x")
    path = autotune_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "estimate_factor": 1.25,
                "model": str(model),
            }
        ),
        encoding="utf-8",
    )
    assert load_persisted_autotune(model) == 1.25


def test_persist_off_skips_write(monkeypatch, tmp_path):
    _reset_autotune(monkeypatch, tmp_path)
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST", "0")
    assert save_persisted_autotune(1.2) is False
    assert not autotune_state_path().exists()


def test_set_session_persists(monkeypatch, tmp_path):
    _reset_autotune(monkeypatch, tmp_path)
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST", "1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"m")
    set_session_vram_estimate_factor(1.42, model=model)
    assert autotune_state_path().is_file()
    gpu_vram._session_autotune_factor = None
    gpu_vram._session_autotune_model = None
    assert session_vram_estimate_factor(model=model) == 1.42
    set_session_vram_estimate_factor(None)


def test_invalid_persist_file_ignored(monkeypatch, tmp_path):
    _reset_autotune(monkeypatch, tmp_path)
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST", "1")
    autotune_state_path().parent.mkdir(parents=True, exist_ok=True)
    autotune_state_path().write_text("not json", encoding="utf-8")
    assert load_persisted_autotune() is None


def test_model_in_persist_catalog(monkeypatch, tmp_path):
    _reset_autotune(monkeypatch, tmp_path)
    model = tmp_path / "a.gguf"
    model.write_bytes(b"a")
    other = tmp_path / "b.gguf"
    other.write_bytes(b"b")
    from runtime.vram_autotune_persist import model_in_persist_catalog

    assert not model_in_persist_catalog(model)
    save_persisted_autotune(1.1, model=model)
    assert model_in_persist_catalog(model)
    assert not model_in_persist_catalog(other)


def test_persist_catalog(monkeypatch, tmp_path):
    _reset_autotune(monkeypatch, tmp_path)
    m_a = tmp_path / "a.gguf"
    m_b = tmp_path / "b.gguf"
    m_a.write_bytes(b"a")
    m_b.write_bytes(b"b")
    save_persisted_autotune(1.1, model=m_a)
    save_persisted_autotune(1.9, model=m_b)

    from runtime.vram_autotune_persist import persist_catalog, persist_status

    catalog, truncated = persist_catalog()
    assert len(catalog) == 2
    assert truncated is False
    assert {row["basename"] for row in catalog} == {"a.gguf", "b.gguf"}
    assert sum(1 for row in catalog if row.get("last")) == 1

    st = persist_status()
    assert len(st["catalog"]) == 2
    assert st["model_count"] == 2


def test_persist_catalog_truncated(monkeypatch, tmp_path):
    _reset_autotune(monkeypatch, tmp_path)
    for i in range(70):
        p = tmp_path / f"m{i}.gguf"
        p.write_bytes(b"x")
        save_persisted_autotune(1.0 + i * 0.01, model=p)
    from runtime.vram_autotune_persist import persist_catalog, persist_status

    catalog, truncated = persist_catalog(max_entries=64)
    assert len(catalog) == 64
    assert truncated is True
    st = persist_status()
    assert st["catalog_truncated"] is True


def test_try_model_autotune_key_returns_none_on_resolve_error(monkeypatch):
    from runtime.vram_autotune_persist import try_model_autotune_key
    import runtime.vram_autotune_persist as mod

    def _boom(_model):
        raise OSError("resolve failed")

    monkeypatch.setattr(mod, "model_autotune_key", _boom)
    assert try_model_autotune_key("/any/model.gguf") is None


def test_factor_source_env_when_path_unresolvable(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE", "1")
    import runtime.gpu_vram as gv

    monkeypatch.setattr(gv, "_try_model_autotune_key", lambda _m: None)
    from runtime.gpu_vram import vram_estimate_factor_source

    assert vram_estimate_factor_source(gguf="/any/model.gguf") == "env"


def test_future_version_ignored(monkeypatch, tmp_path):
    _reset_autotune(monkeypatch, tmp_path)
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST", "1")
    autotune_state_path().write_text(
        json.dumps({"version": 99, "models": {}}),
        encoding="utf-8",
    )
    assert load_persisted_autotune() is None
