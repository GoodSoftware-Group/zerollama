"""Unit tests for compute_readiness (no FastAPI)."""

from __future__ import annotations

from runtime.readiness import compute_readiness


def test_ready_warns_missing_nvfp4_markers():
    body = {
        "accepts_new_loads": True,
        "llama_patches": {
            "status": "pass",
            "cuda_weight_formats": {
                "libggml_cuda": "/tmp/libggml-cuda.so",
                "nvfp4": False,
                "mxfp4": True,
                "skipped": False,
            },
        },
    }
    out = compute_readiness(body)
    assert out["ready"] is True
    assert any("NVFP4" in w for w in out["ready_warnings"])
    assert not any("MXFP4" in w for w in out["ready_warnings"])
