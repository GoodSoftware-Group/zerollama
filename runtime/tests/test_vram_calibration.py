from __future__ import annotations

from pathlib import Path

from runtime.vram_calibration import (
    record_vram_load_sample,
    vram_calibration_health,
    vram_probe_calibrate_enabled,
)


def test_vram_probe_calibrate_auto(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE", "auto")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    monkeypatch.setattr(
        "runtime.gpu_vram.nvml_available",
        lambda: True,
    )
    assert vram_probe_calibrate_enabled()


def test_record_vram_load_sample_suggested_factor(tmp_path: Path):
    p = tmp_path / "m.gguf"
    record_vram_load_sample(
        model_path=p,
        device_index=0,
        estimated_raw_bytes=10_000_000_000,
        estimated_effective_bytes=15_000_000_000,
        free_before=20_000_000_000,
        free_after=8_000_000_000,
        probe="nvml",
    )
    snap = vram_calibration_health()
    assert snap is not None
    assert snap["observed_bytes"] == 12_000_000_000
    assert snap["suggested_estimate_factor"] == 1.2
    assert snap["estimated_effective_bytes"] == 12_000_000_000
    assert snap["estimated_precheck_bytes"] == 15_000_000_000
    assert snap["model"] == str(p)
    assert "suggested_factor_note" in snap


def test_record_vram_load_sample_tp_warning(tmp_path: Path):
    p = tmp_path / "m.gguf"
    record_vram_load_sample(
        model_path=p,
        device_index=0,
        estimated_raw_bytes=1_000,
        estimated_effective_bytes=1_000,
        free_before=5_000,
        free_after=4_000,
        probe="nvml",
        tensor_parallel=2,
    )
    snap = vram_calibration_health()
    assert snap is not None
    assert snap["scope_warning"]


def test_nvidia_free_vram_fresh_bypasses_cache(monkeypatch):
    import runtime.gpu_vram as gv

    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_PROBE", "nvml")
    with gv._smi_lock:
        gv._smi_cache.clear()
    calls: list[bool] = []

    def fake_query(device_index: int) -> tuple[int | None, str | None]:
        calls.append(True)
        return 100, "nvml"

    monkeypatch.setattr("runtime.gpu_vram._query_free_vram_bytes", fake_query)
    monkeypatch.setattr("runtime.gpu_vram.nvml_available", lambda: True)
    from runtime.gpu_vram import nvidia_free_vram_bytes

    nvidia_free_vram_bytes(0)
    nvidia_free_vram_bytes(0)
    nvidia_free_vram_bytes(0, fresh=True)
    assert len(calls) == 2
