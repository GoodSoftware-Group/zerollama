import pytest

from runtime.gpu_vram import (
    effective_vram_estimate_factor,
    session_vram_estimate_factor,
    set_session_vram_estimate_factor,
    vram_estimate_autotune_enabled,
    vram_estimate_autotune_status,
    vram_estimate_factor_source,
)
from runtime.go_coordination import cross_queue_pressure_score, update_go_coordination
from runtime.vram_calibration import record_vram_load_sample


def test_autotune_applies_session_factor(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST", "0")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    set_session_vram_estimate_factor(1.25, model=model)
    assert effective_vram_estimate_factor(gguf=model) == 1.25
    set_session_vram_estimate_factor(None)


def test_autotune_off_uses_env(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR", "1.5")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE", "0")
    set_session_vram_estimate_factor(1.1)
    assert effective_vram_estimate_factor() == 1.5
    set_session_vram_estimate_factor(None)


def test_autotune_status_requires_calibrate(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE", "0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST", "1")
    st = vram_estimate_autotune_status()
    assert st["enabled"] is False
    assert st["probe_calibrate_required"] is False


def test_autotune_factor_source_session_and_catalog(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST", "1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    other = tmp_path / "other.gguf"
    other.write_bytes(b"y")

    assert vram_estimate_factor_source(gguf=model) == "env"

    from runtime.vram_autotune_persist import save_persisted_autotune

    save_persisted_autotune(1.2, model=model)
    assert vram_estimate_factor_source(gguf=model) == "catalog"
    assert vram_estimate_factor_source(gguf=other) == "env"

    set_session_vram_estimate_factor(1.25, model=model)
    assert vram_estimate_factor_source(gguf=model) == "session"
    set_session_vram_estimate_factor(None)


def test_record_sets_session_autotune(monkeypatch, tmp_path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_AUTOTUNE_PERSIST", "0")
    p = tmp_path / "m.gguf"
    p.write_bytes(b"x")
    record_vram_load_sample(
        model_path=p,
        device_index=0,
        estimated_raw_bytes=1_000,
        estimated_effective_bytes=1_000,
        free_before=5_000,
        free_after=3_000,
        probe="nvml",
    )
    assert session_vram_estimate_factor() == 2.0
    set_session_vram_estimate_factor(None)


def test_cross_queue_pressure_score():
    update_go_coordination(
        {"defer_waiting": 2, "sched_pending": 1, "sched_active": 1}
    )
    score = cross_queue_pressure_score(runtime_waiting=3, runtime_running=1)
    assert score == 8  # runtime 4 + defer 2 + ggml 2
