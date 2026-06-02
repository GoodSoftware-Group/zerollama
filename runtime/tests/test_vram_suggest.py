from pathlib import Path
from unittest.mock import patch

import pytest

from runtime.autoconfig import autoconfig_health, detect_gpu_total_vram_bytes
from runtime.gpu_vram import vram_budget_health
from runtime.gpu_vram import check_gguf_vram_budget
from runtime.vram_suggest import (
    build_suggest_profile,
    cap_num_ctx_for_vram,
    format_suggest_num_ctx_hint,
    suggest_max_num_ctx,
    vram_num_ctx_clamp_enabled,
)
from runtime.worker.llama_server import LlamaServerError


def test_suggest_max_num_ctx_binary_search(monkeypatch, tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")

    def fake_estimate(path, *, num_ctx=None, **kwargs):
        del path, kwargs
        ctx = num_ctx or 4096
        return max(1024, (ctx // 1024) * 1024 * 1024)

    with patch(
        "runtime.gpu_vram.estimate_gguf_vram_bytes",
        side_effect=fake_estimate,
    ):
        with patch(
            "runtime.gguf_estimate.gguf_arch_hints",
        ) as hints:
            hints.return_value.scalar = {"context_length": 8192}
            got = suggest_max_num_ctx(
                gguf,
                5 * 1024**3,
                margin=1.0,
                min_free_bytes=0,
            )
    assert got is not None
    assert got <= 8192
    assert got >= 512


def test_suggest_returns_none_when_too_tight(monkeypatch, tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch(
        "runtime.gpu_vram.estimate_gguf_vram_bytes",
        return_value=10 * 1024**3,
    ):
        with patch(
            "runtime.gguf_estimate.gguf_arch_hints",
        ) as hints:
            hints.return_value.scalar = {"context_length": 4096}
            assert suggest_max_num_ctx(
                gguf, 1024, margin=1.0, min_free_bytes=0
            ) is None


def test_suggest_respects_min_free_floor(tmp_path: Path):
    """Admission uses max(estimate, min_free); suggestion must match."""
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    min_free = 1024**3

    with patch(
        "runtime.gpu_vram.estimate_gguf_vram_bytes",
        return_value=1024,
    ):
        with patch(
            "runtime.gguf_estimate.gguf_arch_hints",
        ) as hints:
            hints.return_value.scalar = {"context_length": 8192}
            assert (
                suggest_max_num_ctx(
                    gguf,
                    int(0.99 * 1024**3),
                    margin=1.0,
                    min_free_bytes=min_free,
                )
                is None
            )
            got = suggest_max_num_ctx(
                gguf,
                int(1.1 * 1024**3),
                margin=1.0,
                min_free_bytes=min_free,
            )
    assert got is not None


def test_build_suggest_profile_from_estimate():
    prof = build_suggest_profile(
        {
            "n_gpu_layers": 32,
            "parallel_slots": 2,
            "draft_model": "/no/such/draft.gguf",
        },
        tensor_parallel=1,
        llama_args=["-ngl", "40"],
    )
    assert prof["n_gpu_layers_default"] == 32
    assert prof["parallel_slots_default"] == 2
    assert prof["llama_args"] == ["-ngl", "40"]
    assert "draft_gguf" not in prof or prof.get("draft_gguf") is None


def test_vram_budget_passes_suggest_profile_to_suggest(tmp_path: Path):
    est = {
        "gguf": str(tmp_path / "m.gguf"),
        "required_per_gpu_bytes": 8 * 1024**3,
        "num_ctx": 32768,
        "n_gpu_layers": 24,
        "parallel_slots": 2,
    }
    (tmp_path / "m.gguf").write_bytes(b"x")
    captured: dict = {}

    def capture_suggest(gguf, eff, **kwargs):
        captured.update(kwargs)
        return 8192

    with patch(
        "runtime.vram_suggest.suggest_max_num_ctx",
        side_effect=capture_suggest,
    ):
        budget = vram_budget_health(
            est,
            gpu_free_bottleneck=16 * 1024**3,
            inference_paused_for_reserve=False,
            suggest_profile={"llama_args": ["-ngl", "24"], "tensor_parallel": 1},
        )
    assert budget is not None
    assert budget.get("suggested_max_num_ctx") == 8192
    assert captured.get("n_gpu_layers_default") == 24
    assert captured.get("parallel_slots_default") == 2
    assert captured.get("llama_args") == ["-ngl", "24"]


def test_vram_budget_includes_suggested_num_ctx(tmp_path: Path):
    est = {
        "gguf": str(tmp_path / "m.gguf"),
        "required_per_gpu_bytes": 8 * 1024**3,
        "num_ctx": 32768,
    }
    (tmp_path / "m.gguf").write_bytes(b"x")
    with patch(
        "runtime.vram_suggest.suggest_max_num_ctx",
        return_value=8192,
    ):
        budget = vram_budget_health(
            est,
            gpu_free_bottleneck=16 * 1024**3,
            inference_paused_for_reserve=False,
        )
    assert budget is not None
    assert budget.get("suggested_max_num_ctx") == 8192
    assert budget.get("num_ctx_over_budget") is True


def test_autoconfig_health_pick(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CONFIG", "/x/single_gpu.yaml")
    monkeypatch.setenv("ZEROLLAMA_AUTO_CONFIG", "1")
    with patch("runtime.autoconfig.detect_visible_gpu_count", return_value=1):
        with patch("runtime.autoconfig.detect_gpu_total_vram_bytes", return_value=None):
            h = autoconfig_health()
    assert h["pick"] == "single_gpu"
    assert "single_gpu.yaml" in str(h["config_path"])


def test_format_suggest_num_ctx_hint_with_request(tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch(
        "runtime.vram_suggest.suggest_max_num_ctx",
        return_value=4096,
    ):
        hint = format_suggest_num_ctx_hint(
            gguf,
            8 * 1024**3,
            num_ctx=32768,
            min_free_bytes=0,
        )
    assert "4096" in hint
    assert "32768" in hint


def test_check_gguf_reject_includes_suggest_hint(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch(
        "runtime.gpu_vram.nvidia_free_vram_by_device",
        return_value={0: 2 * 1024**3},
    ):
        with patch(
            "runtime.gpu_vram.estimate_gguf_vram_bytes",
            return_value=8 * 1024**3,
        ):
            with patch(
                "runtime.vram_suggest.format_suggest_num_ctx_hint",
                return_value="; try num_ctx<=4096 (requested 8192)",
            ):
                with pytest.raises(LlamaServerError, match="num_ctx<=4096"):
                    check_gguf_vram_budget(
                        gguf,
                        margin=1.0,
                        num_ctx=8192,
                    )


def test_cap_num_ctx_clamps_when_enabled(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX", "1")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch(
        "runtime.vram_suggest.suggest_max_num_ctx",
        return_value=4096,
    ):
        ctx, meta = cap_num_ctx_for_vram(
            gguf,
            32768,
            16 * 1024**3,
            min_free_bytes=0,
        )
    assert ctx == 4096
    assert meta.get("num_ctx_clamped") is True
    assert meta.get("num_ctx_clamped_from") == 32768


def test_cap_num_ctx_off_by_default(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX", raising=False)
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    ctx, meta = cap_num_ctx_for_vram(gguf, 32768, 16 * 1024**3)
    assert ctx == 32768
    assert not meta.get("num_ctx_clamped")


def test_vram_num_ctx_clamp_default_env_off(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX", raising=False)
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    assert not vram_num_ctx_clamp_enabled()


def test_resolve_num_ctx_for_request_clamps_before_admit(monkeypatch, cfg_root, tmp_path):
    from runtime.config import RuntimeConfig
    from runtime.engine import InferenceEngine

    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=gguf,
        num_blocks=64,
        block_size=16,
        device_count=1,
    )
    eng = InferenceEngine(cfg)
    with patch.object(eng, "_effective_vram_free_for_suggest", return_value=16 * 1024**3):
        with patch(
            "runtime.vram_suggest.suggest_max_num_ctx",
            return_value=4096,
        ):
            ctx, meta = eng.resolve_num_ctx_for_request(
                gguf, options={"num_ctx": 32768}
            )
    assert ctx == 4096
    assert meta.get("num_ctx_clamped") is True


def test_api_vram_num_ctx_meta_when_clamped():
    from runtime.vram_suggest import api_vram_num_ctx_meta

    meta = {
        "num_ctx_clamped": True,
        "num_ctx_clamped_from": 32768,
        "suggested_max_num_ctx": 4096,
    }
    api = api_vram_num_ctx_meta(meta, 4096)
    assert api is not None
    assert api["num_ctx"] == 4096
    assert api["num_ctx_clamped_from"] == 32768
    assert api_vram_num_ctx_meta({}, 4096) is None


def test_vram_num_ctx_clamp_auto_follows_gpu_check(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX", "auto")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    assert vram_num_ctx_clamp_enabled()
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "0")
    assert not vram_num_ctx_clamp_enabled()


def test_detect_gpu_total_vram_bytes_parses_mib():
    from runtime.autoconfig import clear_autoconfig_probe_cache

    clear_autoconfig_probe_cache()

    class Proc:
        returncode = 0
        stdout = "16384 MiB\n"

    with patch("runtime.autoconfig.subprocess.run", return_value=Proc()):
        b = detect_gpu_total_vram_bytes(0)
    assert b == 16384 * 1024 * 1024
