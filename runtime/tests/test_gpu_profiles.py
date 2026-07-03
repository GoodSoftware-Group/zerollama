from pathlib import Path

import pytest

from runtime.gpu_profiles import (
    darwin_unified_memory_gb,
    flags_from_gpu_config,
    llama_argv_from_profile_flags,
    load_gpu_config,
    profile_emit_options,
    profile_n_gpu_layers,
    sanitize_llama_flags,
    select_by_apple_memory_bucket,
    select_by_vram_bucket,
)
from runtime.config import RuntimeConfig


def test_sanitize_fork_cache_types():
    flags, fb = sanitize_llama_flags(
        {
            "cache_type_k": "qjl1_256",
            "cache_type_v": "q4_polar",
            "n_parallel": 8,
            "ctx_checkpoints": 8,
            "ctx_checkpoint_interval": 8192,
        }
    )
    assert flags["cache_type_k"] == "q8_0"
    assert flags["cache_type_v"] == "q8_0"
    assert "ctx_checkpoints" not in flags
    assert fb is True


def test_sanitize_tbq_canonical_names():
    """tbq3_0 / tbq4_0 canonical names must also be blocked on stock path."""
    flags, fb = sanitize_llama_flags(
        {"cache_type_k": "tbq3_0", "cache_type_v": "tbq4_0"}
    )
    assert flags["cache_type_k"] == "q8_0"
    assert flags["cache_type_v"] == "q8_0"
    assert fb is True


def test_sanitize_tbq_alias_names():
    """turbo* aliases normalize to tbq* then fall back on stock path."""
    flags, fb = sanitize_llama_flags(
        {"cache_type_k": "turbo4_0", "cache_type_v": "turbo3_0"}
    )
    assert flags["cache_type_k"] == "q8_0"
    assert flags["cache_type_v"] == "q8_0"
    assert fb is True


def test_fork_only_flags_stripped_on_stock_sanitize():
    flags, _fb = sanitize_llama_flags(
        {
            "ctx_checkpoints": 8,
            "ctx_checkpoint_interval": 8192,
            "batch_size": 512,
        }
    )
    assert "ctx_checkpoints" not in flags
    args = llama_argv_from_profile_flags(flags)
    assert "--ctx-checkpoints" not in args


def test_llama_argv_from_profile_flags():
    args = llama_argv_from_profile_flags(
        {
            "ctx_size": 32768,
            "batch_size": 2048,
            "ubatch_size": 512,
            "flash_attn": True,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "mlock": True,
        }
    )
    assert "-c" in args and "32768" in args
    assert "-b" in args and "2048" in args
    assert "-ub" in args and "512" in args
    assert "-fa" in args
    assert "--cache-type-k" in args
    assert "--mlock" in args


def test_llama_argv_respects_emit_options():
    flags = {"ctx_size": 32768, "mlock": True, "batch_size": 512}
    args = llama_argv_from_profile_flags(flags, emit={"ctx_size": False, "mlock": False})
    assert "-c" not in args
    assert "--mlock" not in args
    assert "-b" in args


def test_profile_emit_options_env():
    opts = profile_emit_options({"llama_profile": {"apply_ctx_size": True}})
    assert opts["ctx_size"] is True


def test_profile_n_gpu_layers_maps_all_layers():
    assert profile_n_gpu_layers({"n_gpu_layers": 999}) == -1
    assert profile_n_gpu_layers({"n_gpu_layers": 80}) == 80


def test_vram_bucket_5080():
    cfg, label, scale = select_by_vram_bucket(16.0)
    assert cfg is not None
    assert cfg["id"] == "rtx-5080"
    assert label == "small"
    assert scale == 1.0


def test_vram_bucket_scales_parallel():
    cfg, label, scale = select_by_vram_bucket(32.0)
    assert cfg is not None
    assert cfg["id"] == "rtx-5090"
    assert label == "mid-plus"
    assert scale == 0.5


def test_apple_memory_bucket_128g():
    cfg, label, _scale = select_by_apple_memory_bucket(128.0)
    assert cfg is not None
    assert cfg["id"] == "apple-silicon-128g"
    assert label == "128g"
    assert cfg["llama_server_flags"]["n_parallel"] == 8


def test_apple_memory_bucket_16g():
    cfg, label, _scale = select_by_apple_memory_bucket(16.0)
    assert cfg is not None
    assert cfg["id"] == "apple-silicon-16g"
    assert label == "16g"


def test_load_4090_profile():
    cfg = load_gpu_config("rtx-4090")
    assert cfg["llama_server_flags"]["n_parallel"] == 8
    assert cfg["llama_server_flags"]["mlock"] is False
    assert "ctx_checkpoints" not in cfg["llama_server_flags"]
    assert cfg["_fork_only_llama_server_flags"]["ctx_checkpoints"] == 8


def test_flags_from_gpu_config_stock_sanitize():
    cfg = {
        "llama_server_flags": {
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "n_parallel": 8,
        },
        "_eliza_fork_llama_server_flags": {
            "cache_type_k": "qjl1_256",
            "cache_type_v": "q4_polar",
        },
        "_fork_only_llama_server_flags": {
            "ctx_checkpoints": 8,
        },
    }
    flags, fb = flags_from_gpu_config(cfg, fork_enabled=False)
    assert flags["cache_type_k"] == "q8_0"
    assert "ctx_checkpoints" not in flags
    assert fb is False


def test_runtime_config_fork_env_off_overrides_probe(monkeypatch, tmp_path):
    fake_bin = tmp_path / "llama-server"
    fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_bin.chmod(0o755)
    monkeypatch.setenv("ZEROLLAMA_LLAMA_FORK", "off")
    monkeypatch.setenv("LLAMA_SERVER_BIN", str(fake_bin))
    monkeypatch.setenv("ZEROLLAMA_GPU_PROFILE", "1")
    monkeypatch.setattr(
        "runtime.llama_fork.probe_fork_llama_server",
        lambda _bin: True,
    )
    monkeypatch.setattr(
        "runtime.gpu_profiles.detect_nvidia_gpu_name",
        lambda device_index=0: "NVIDIA GeForce RTX 4090",
    )
    monkeypatch.setattr(
        "runtime.gpu_profiles.detect_gpu_total_vram_gb",
        lambda device_index=0: 24.0,
    )
    path = Path(__file__).resolve().parents[1] / "configs" / "single_gpu.yaml"
    cfg = RuntimeConfig.from_file(path)
    assert cfg.gpu_profile is not None
    assert cfg.gpu_profile.get("llama_fork") is False
    args = cfg.llama_server_args()
    assert "--ctx-checkpoints" not in args
    idx = args.index("--cache-type-k")
    assert args[idx + 1] == "q8_0"


def test_runtime_config_applies_profile_by_name(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_GPU_PROFILE", "1")
    monkeypatch.setattr(
        "runtime.gpu_profiles.detect_nvidia_gpu_name",
        lambda device_index=0: "NVIDIA GeForce RTX 4090",
    )
    monkeypatch.setattr(
        "runtime.gpu_profiles.detect_gpu_total_vram_gb",
        lambda device_index=0: 24.0,
    )
    path = Path(__file__).resolve().parents[1] / "configs" / "single_gpu.yaml"
    cfg = RuntimeConfig.from_file(path)
    assert cfg.gpu_profile is not None
    assert cfg.gpu_profile["id"] == "rtx-4090"
    assert cfg.llama_parallel_slots == 8
    args = cfg.llama_server_args()
    assert "-fa" in args
    assert "--cache-type-k" in args
    assert cfg.speculative.draft_n_max == 24
    assert "--mlock" not in args


def test_runtime_config_skips_profile_ctx_via_env(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_GPU_PROFILE", "1")
    monkeypatch.setenv("ZEROLLAMA_GPU_PROFILE_CTX", "0")
    monkeypatch.setattr(
        "runtime.gpu_profiles.detect_nvidia_gpu_name",
        lambda device_index=0: "NVIDIA GeForce RTX 4090",
    )
    monkeypatch.setattr(
        "runtime.gpu_profiles.detect_gpu_total_vram_gb",
        lambda device_index=0: 24.0,
    )
    path = Path(__file__).resolve().parents[1] / "configs" / "single_gpu.yaml"
    cfg = RuntimeConfig.from_file(path)
    args = cfg.llama_server_args()
    assert "-c" not in args
    assert cfg.gpu_profile is not None
    assert cfg.gpu_profile["emit_ctx_size"] is False


def test_runtime_config_applies_apple_profile(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_GPU_PROFILE", "1")
    monkeypatch.setattr(
        "runtime.gpu_profiles.darwin_unified_memory_gb",
        lambda: 128.0,
    )
    path = Path(__file__).resolve().parents[1] / "configs" / "apple_silicon.yaml"
    cfg = RuntimeConfig.from_file(path)
    assert cfg.gpu_profile is not None
    assert cfg.gpu_profile["id"] == "apple-silicon-128g"
    assert cfg.gpu_profile["bucket_label"] == "128g"
    assert cfg.gpu_profile.get("kv_num_blocks") == 8192
    assert cfg.num_blocks == 8192
    assert cfg.llama_parallel_slots == 8
    args = cfg.llama_server_args()
    assert "-fa" in args
    assert "-c" in args and "131072" in args


def test_runtime_config_applies_apple_profile_on_host():
    if darwin_unified_memory_gb() is None:
        pytest.skip("darwin unified memory probe unavailable")
    path = Path(__file__).resolve().parents[1] / "configs" / "apple_silicon.yaml"
    cfg = RuntimeConfig.from_file(path)
    assert cfg.gpu_profile is not None
    assert cfg.gpu_profile["id"].startswith("apple-silicon-")
    assert cfg.gpu_profile.get("unified_memory_gb") is not None


def test_gpu_profile_disabled(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_GPU_PROFILE", "0")
    path = Path(__file__).resolve().parents[1] / "configs" / "single_gpu.yaml"
    cfg = RuntimeConfig.from_file(path)
    assert cfg.gpu_profile is None
    assert cfg.llama_parallel_slots == 1


def test_arc_a380_profile_json():
    cfg = load_gpu_config("arc-a380")
    assert cfg["vram_gb"] == 6
    flags, _ = sanitize_llama_flags(cfg["llama_server_flags"])
    assert flags["n_parallel"] == 1
    assert flags["ctx_size"] == 4096
    assert flags.get("flash_attn") is False


def test_forced_gpu_profile_id_arc_a380(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_GPU_PROFILE", "1")
    monkeypatch.setenv("ZEROLLAMA_GPU_PROFILE_ID", "arc-a380")
    path = Path(__file__).resolve().parents[1] / "configs" / "arc_a380.yaml"
    cfg = RuntimeConfig.from_file(path)
    assert cfg.gpu_profile is not None
    assert cfg.gpu_profile["id"] == "arc-a380"
    assert cfg.gpu_profile["source"] == "env"


def test_match_arc_a380_by_name():
    from runtime.gpu_profiles import match_gpu_config_by_name

    matched = match_gpu_config_by_name("Intel(R) Arc(tm) A380 Graphics (DG2)")
    assert matched is not None
    assert matched["id"] == "arc-a380"
