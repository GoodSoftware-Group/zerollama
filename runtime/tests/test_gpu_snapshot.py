"""Tests for Phase 13 snapshot recommendations."""

from __future__ import annotations

from runtime.gpu_snapshot import format_snapshot_recommendations, recommend_from_snapshot

_SNAPSHOT_5080 = {
    "gguf": "/root/Llama-OuteTTS-1.0-1B-Q8_0.gguf",
    "num_ctx_probe": 4096,
    "autoconfig": {"pick": "single_gpu", "config_path": "/cfg/single_gpu.yaml"},
    "vram_budget": {
        "fits_with_margin": True,
        "suggested_max_num_ctx": 131072,
    },
    "vram_calibration": {
        "suggested_estimate_factor": 1.1988948308733205,
        "observed_bytes": 2056257536,
    },
    "vram_autotune": {
        "enabled": True,
        "effective_factor": 1.1988948308733205,
        "session_model": "/root/Llama-OuteTTS-1.0-1B-Q8_0.gguf",
        "persist": {
            "persisted_factor": 1.1988948308733205,
            "catalog": [
                {
                    "model": "/root/Llama-OuteTTS-1.0-1B-Q8_0.gguf",
                    "basename": "Llama-OuteTTS-1.0-1B-Q8_0.gguf",
                    "estimate_factor": 1.1988948308733205,
                    "last": True,
                },
            ],
        },
    },
    "admission": {
        "vram_min_free_configured": 1073741824,
        "vram_training_reserve_configured": 2147483648,
    },
}


def test_recommend_prefers_autotune_over_global_factor():
    lines = recommend_from_snapshot(_SNAPSHOT_5080)
    text = "\n".join(lines)
    assert "autotune catalog:" in text
    assert "persist wins" in text
    assert "headroom configured" in text
    assert "export ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR" not in text


def test_recommend_lists_autotune_catalog():
    lines = recommend_from_snapshot(_SNAPSHOT_5080)
    text = "\n".join(lines)
    assert "Llama-OuteTTS-1.0-1B-Q8_0.gguf" in text


def test_recommend_warns_when_probe_gguf_not_in_catalog(tmp_path):
    other = tmp_path / "other.gguf"
    other.write_bytes(b"x")
    snap = dict(_SNAPSHOT_5080)
    snap["gguf"] = str(other)
    text = "\n".join(recommend_from_snapshot(snap))
    assert "not in catalog" in text
    assert "other.gguf" in text


def test_recommend_export_when_no_autotune():
    snap = dict(_SNAPSHOT_5080)
    snap["vram_autotune"] = {"enabled": False}
    lines = recommend_from_snapshot(snap)
    assert any("export ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR=1.19889" in ln for ln in lines)


def test_format_includes_harmony_skip_hint():
    out = format_snapshot_recommendations(_SNAPSHOT_5080)
    assert "phase12_golden_ci.sh" in out
    assert "single_gpu.yaml" in out


def test_warn_when_either_budget_fits_false():
    snap = dict(_SNAPSHOT_5080)
    snap["vram_budget"] = {"fits_with_margin": True}
    snap["vram_estimate_budget"] = {"fits_with_margin": False}
    text = "\n".join(recommend_from_snapshot(snap))
    assert "fits_with_margin=false" in text
