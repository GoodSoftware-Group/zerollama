import pytest

from runtime.gpu import admission as adm
from runtime.gpu.admission import (
    AdmissionMisconfigured,
    AdmissionRejected,
    VRAM_MIN_FREE_DEFAULT_BYTES,
    admission_vram_gate_enabled,
    admission_vram_gate_mode,
    check_vram_admission,
    configured_min_free_vram_bytes,
    configured_training_vram_reserve_bytes,
    parse_size_bytes,
    training_vram_reserve_bytes,
)
from runtime.gpu.priority import InferencePriority, parse_inference_priority


def test_parse_size_bytes():
    assert parse_size_bytes("1GiB") == 1024**3
    assert parse_size_bytes("512MiB") == 512 * 1024**2


def test_check_vram_admission_rejects(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    with pytest.raises(AdmissionRejected, match="below admission minimum"):
        check_vram_admission(VRAM_MIN_FREE_DEFAULT_BYTES - 1, backlog=0)


def test_check_vram_admission_training_reserve(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    free = int(1.5 * 1024**3)
    with pytest.raises(AdmissionRejected, match="reserve"):
        check_vram_admission(free, backlog=1, inference_paused=True)


def test_training_reserve_when_busy():
    assert training_vram_reserve_bytes(inference_paused=False) == 0
    assert training_vram_reserve_bytes(inference_paused=True) == (
        configured_training_vram_reserve_bytes()
    )


def test_vram_min_free_from_env(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_MIN_FREE", "512MiB")
    assert configured_min_free_vram_bytes() == 512 * 1024**2


def test_training_reserve_from_env(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE", "3GiB")
    assert configured_training_vram_reserve_bytes() == 3 * 1024**3
    assert training_vram_reserve_bytes(inference_paused=True) == 3 * 1024**3


def test_gate_requires_probe_when_checks_on(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    with pytest.raises(AdmissionMisconfigured, match="unavailable"):
        check_vram_admission(None, backlog=0)


def test_admission_vram_gate_follows_check_gpu_vram(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "0")
    assert admission_vram_gate_mode() == "off"
    assert admission_vram_gate_enabled() is False


def test_admission_vram_gate_on_when_inference_policy_off(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_INFERENCE_POLICY", "off")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    assert admission_vram_gate_enabled() is True


def test_high_bypasses_vram_gate(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    check_vram_admission(1024, backlog=0, priority=InferencePriority.HIGH)
