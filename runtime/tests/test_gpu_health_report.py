"""Tests for operator /health formatting (scripts/gpu_health_report.sh)."""

from runtime.gpu_health_report import format_gpu_health_tuning_report

SAMPLE_HEALTH = {
    "status": "ok",
    "llama_server": True,
    "inference_state": "running",
    "autoconfig": {"pick": "single_gpu", "config_path": "/cfg/single_gpu.yaml"},
    "vram_budget": {
        "admission_fits": True,
        "fits_with_margin": True,
        "suggested_max_num_ctx": 8192,
    },
    "vram_calibration": {
        "model": "/models/m.gguf",
        "suggested_estimate_factor": 1.15,
        "observed_bytes": 8_000_000_000,
        "estimated_raw_bytes": 7_000_000_000,
        "active_estimate_factor": 1.1,
        "age_s": 12.5,
    },
    "vram_autotune": {
        "enabled": True,
        "pending_first_calibration": False,
        "session_model": "/models/m.gguf",
        "session_factor": 1.15,
        "effective_factor": 1.15,
        "env_factor": 1.0,
        "probe_calibrate_required": False,
        "persist": {"last_model": "m.gguf", "persisted_factor": 1.15},
    },
    "vram_num_ctx_policy": {"env": "0", "clamp_enabled": False},
    "admission": {
        "vram_min_free_configured": 1_073_741_824,
        "vram_training_reserve_configured": 2_147_483_648,
    },
}


def test_format_gpu_health_includes_calibration_and_autotune_fields():
    out = format_gpu_health_tuning_report(SAMPLE_HEALTH)
    assert "vram_calibration.observed_bytes:" in out
    assert "vram_calibration.suggested_estimate_factor: 1.15" in out
    assert "vram_autotune.session_model:" in out
    assert "vram_autotune.effective_factor: 1.15" in out
    assert "vram_autotune.persist.last_model:" in out
    assert "probe_required_bytes" not in out
    assert "ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR=1.15" in out


def test_format_gpu_health_warns_when_factor_out_of_range():
    h = {"status": "ok", "vram_calibration": {"suggested_estimate_factor": 9.0}}
    out = format_gpu_health_tuning_report(h)
    assert "out of clamp range" in out
    assert "export ZEROLLAMA" not in out


def test_format_gpu_health_suggests_clamp_when_budget_tight():
    h = dict(SAMPLE_HEALTH)
    h["vram_budget"] = {"suggested_max_num_ctx": 4096}
    h["vram_num_ctx_policy"] = {"env": "0", "clamp_enabled": False}
    out = format_gpu_health_tuning_report(h)
    assert "VRAM_CLAMP_NUM_CTX=auto" in out
    assert "num_ctx <= 4096" in out
