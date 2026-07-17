"""Centralized runtime env defaults."""

import sys

import pytest

from runtime.env import (
    configure_l3_settings,
    configure_runtime_env,
    env_tri_state,
    llama_cache_disk_default,
    llama_cache_disk_enabled,
    lmcache_tier_enabled,
    prefix_block_pool_enabled,
    prefix_cache_block_size,
    radix_prefix_share_enabled,
    reset_runtime_env_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_hints():
    reset_runtime_env_for_tests()
    yield
    reset_runtime_env_for_tests()


def test_disk_cache_explicit_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_LLAMA_CACHE_DISK", "0")
    assert llama_cache_disk_enabled() is False
    monkeypatch.setenv("ZEROLLAMA_LLAMA_CACHE_DISK", "1")
    assert llama_cache_disk_enabled() is True


def test_disk_cache_platform_default_when_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CACHE_DISK", raising=False)
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "subprocess")
    expected = sys.platform != "darwin"
    assert llama_cache_disk_default(backend="subprocess") is expected
    assert llama_cache_disk_enabled() is expected


def test_disk_cache_darwin_default_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_LLAMA_CACHE_DISK", raising=False)
    monkeypatch.setattr("runtime.env.sys.platform", "darwin")
    assert llama_cache_disk_enabled(backend="subprocess") is False


def test_prefix_block_pool_auto_multislot(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_PREFIX_BLOCK_POOL", raising=False)
    monkeypatch.delenv("ZEROLLAMA_RADIX_PREFIX_SHARE", raising=False)
    monkeypatch.delenv("ZEROLLAMA_LMCACHE_URI", raising=False)
    monkeypatch.delenv("ZEROLLAMA_LMCACHE_TIER", raising=False)
    configure_runtime_env(n_parallel=4)
    assert prefix_block_pool_enabled() is True
    configure_runtime_env(n_parallel=1)
    assert prefix_block_pool_enabled() is False


def test_prefix_block_pool_explicit_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_PREFIX_BLOCK_POOL", "0")
    configure_runtime_env(n_parallel=8)
    assert prefix_block_pool_enabled() is False


def test_lmcache_uri_implies_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_LMCACHE_TIER", raising=False)
    monkeypatch.setenv("ZEROLLAMA_LMCACHE_URI", "file:///tmp/lmcache-test")
    assert lmcache_tier_enabled() is True


def test_lmcache_tier_legacy_alias(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_LMCACHE_URI", raising=False)
    monkeypatch.setenv("ZEROLLAMA_LMCACHE_TIER", "1")
    assert lmcache_tier_enabled() is True


def test_env_tri_state():
    assert env_tri_state("ZEROLLAMA_NOT_SET_VAR_XYZ") is None


def test_l3_yaml_radix_without_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_RADIX_PREFIX_SHARE", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED_WITH_RADIX", raising=False)
    from runtime.env import kv_unified_enabled, kv_unified_source, reset_runtime_env_for_tests

    reset_runtime_env_for_tests()
    configure_l3_settings({"radix_share": True, "block_size": 128})
    assert radix_prefix_share_enabled() is True
    assert prefix_cache_block_size() == 128
    # v58: radix YAML couples unified by default
    assert kv_unified_enabled() is True
    assert kv_unified_source() == "radix_couple"


def test_l3_yaml_kv_unified_without_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_KV_UNIFIED", raising=False)
    from runtime.env import kv_unified_enabled, reset_runtime_env_for_tests

    reset_runtime_env_for_tests()
    configure_l3_settings({"kv_unified": True})
    assert kv_unified_enabled() is True


def test_l3_yaml_kv_unified_env_wins(monkeypatch: pytest.MonkeyPatch):
    from runtime.env import kv_unified_enabled, reset_runtime_env_for_tests

    reset_runtime_env_for_tests()
    configure_l3_settings({"kv_unified": True})
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED", "0")
    assert kv_unified_enabled() is False
    monkeypatch.setenv("ZEROLLAMA_KV_UNIFIED", "1")
    configure_l3_settings({"kv_unified": False})
    assert kv_unified_enabled() is True


def test_l3_yaml_kv_cow_syncs_environ(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_KV_COW", raising=False)
    from runtime.env import kv_cow_enabled, kv_cow_source, reset_runtime_env_for_tests
    import os

    reset_runtime_env_for_tests()
    configure_l3_settings({"kv_cow": True})
    assert kv_cow_enabled() is True
    assert kv_cow_source() == "yaml"
    assert os.environ.get("ZEROLLAMA_KV_COW") == "1"


def test_l3_yaml_kv_cow_env_kill_switch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_KV_COW", "0")
    from runtime.env import kv_cow_enabled, reset_runtime_env_for_tests

    reset_runtime_env_for_tests()
    configure_l3_settings({"kv_cow": True})
    assert kv_cow_enabled() is False


def test_l3_yaml_kv_cow_tensors_sync(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_KV_COW", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_COW_TENSORS", raising=False)
    from runtime.env import (
        kv_cow_enabled,
        kv_cow_tensors_enabled,
        reset_runtime_env_for_tests,
    )
    import os

    reset_runtime_env_for_tests()
    configure_l3_settings({"kv_cow": True, "kv_cow_tensors": True})
    assert kv_cow_enabled() is True
    assert kv_cow_tensors_enabled() is True
    assert os.environ.get("ZEROLLAMA_KV_COW_TENSORS") == "1"


def test_l3_yaml_kv_cow_pages_sync(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_KV_COW", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_COW_TENSORS", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_COW_PAGES", raising=False)
    from runtime.env import kv_cow_pages_enabled, reset_runtime_env_for_tests
    import os

    reset_runtime_env_for_tests()
    configure_l3_settings({"kv_cow_pages": True})
    assert kv_cow_pages_enabled() is True
    assert os.environ.get("ZEROLLAMA_KV_COW_PAGES") == "1"
    assert os.environ.get("ZEROLLAMA_KV_COW_TENSORS") == "1"
    assert os.environ.get("ZEROLLAMA_KV_COW") == "1"


def test_l3_profile_resolves_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_CONFIG", raising=False)
    monkeypatch.setenv("ZEROLLAMA_L3_PROFILE", "agent")
    from runtime.env import resolve_l3_profile_config_path

    path = resolve_l3_profile_config_path()
    assert path is not None
    assert path.name == "l3_agent_subprocess.yaml"


def test_debug_l3_enables_prefix_trace(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_PREFIX_CACHE_TRACE", raising=False)
    monkeypatch.setenv("ZEROLLAMA_DEBUG", "l3")
    from runtime.env import prefix_cache_trace_enabled

    assert prefix_cache_trace_enabled() is True


def test_infer_trace_debug_tag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_INFER_TRACE", raising=False)
    monkeypatch.setenv("ZEROLLAMA_DEBUG", "infer")
    from runtime.env import infer_trace_enabled

    assert infer_trace_enabled() is True


def test_kv_env_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_KV_NATIVE_DECODE", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_AUTO_BATCH", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_AUTO_BATCH_STREAM", raising=False)
    from runtime.env import (
        kv_auto_batch_enabled,
        kv_auto_batch_stream_enabled,
        kv_native_decode_enabled,
        kv_env_health,
    )

    assert kv_native_decode_enabled() is True
    assert kv_auto_batch_enabled() is False
    assert kv_auto_batch_stream_enabled() is False
    assert kv_env_health()["auto_batch_ms"] == 5


def test_vram_env_helpers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_MARGIN", "1.25")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_FACTOR", "1.2")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_RAM_OVERHEAD", "1.15")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_WEIGHT_TENSOR", "auto")
    from runtime.env import (
        inference_first_policy_enabled,
        runtime_shared_python_embedded,
        vram_check_gpu_explicit,
        vram_env_health,
        vram_kv_factor,
        vram_margin,
        vram_probe_calibrate_enabled,
        vram_ram_overhead,
        vram_scratch_factor,
        vram_weight_tensor_mode,
        vram_weight_tensor_use_per_tensor,
    )

    assert vram_margin() == 1.25
    assert vram_check_gpu_explicit() is False
    assert vram_kv_factor(kv_exact=True) == 1.2
    assert vram_scratch_factor() == 1.05
    assert vram_ram_overhead() == 1.15
    assert vram_weight_tensor_mode() == "auto"
    assert vram_weight_tensor_use_per_tensor(8) is True
    assert vram_weight_tensor_use_per_tensor(None) is False
    assert inference_first_policy_enabled() is True
    assert vram_probe_calibrate_enabled() is False
    health = vram_env_health()
    assert health["margin"] == 1.25
    assert health["weight_tensor_mode"] == "auto"
    assert health["ram_overhead"] == 1.15
    assert health["probe_calibrate_enabled"] is False
    assert isinstance(runtime_shared_python_embedded(), bool)
