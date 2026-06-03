from pathlib import Path
from unittest.mock import patch

import pytest

from runtime.gpu_vram import (
    active_vram_probe,
    check_gguf_vram_budget,
    estimate_gguf_vram_bytes,
    format_vram_reject_kv_hint,
    vram_budget_health,
    gguf_layer_kv_scale,
    gpu_vram_check_enabled,
    llama_vram_device_indices,
    nvidia_free_vram_bytes,
    resolve_num_ctx,
    resolve_vram_num_ctx,
    vram_probe_mode,
    vram_ctx_scale,
)
from runtime.worker.llama_server import LlamaServerError


def test_gpu_vram_check_disabled(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "0")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x" * 1024)
    check_gguf_vram_budget(gguf)


def test_check_gguf_vram_budget_reserve_on_all_tp_devices(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE", "3GiB")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    reserve = 3 * 1024**3
    with patch(
        "runtime.gpu_vram.nvidia_free_vram_by_device",
        return_value={0: 10 * 1024**3, 1: 4 * 1024**3},
    ):
        with patch(
            "runtime.gpu_vram.estimate_gguf_vram_bytes",
            return_value=3 * 1024**3,
        ):
            with pytest.raises(LlamaServerError, match="training reserve"):
                check_gguf_vram_budget(
                    gguf,
                    margin=1.0,
                    tensor_parallel=2,
                    device_count=2,
                    training_reserve_active=True,
                )


def test_check_gguf_vram_budget_subtracts_training_reserve(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE", "4GiB")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")

    # 10 GiB free, 7 GiB required → ok without reserve; fails with 4 GiB reserve (6 GiB effective).
    with patch(
        "runtime.gpu_vram.nvidia_free_vram_by_device", return_value={0: 10 * 1024**3}
    ):
        with patch(
            "runtime.gpu_vram.estimate_gguf_vram_bytes",
            return_value=7 * 1024**3,
        ):
            check_gguf_vram_budget(gguf, margin=1.0, training_reserve_active=False)
            with pytest.raises(LlamaServerError, match="training reserve"):
                check_gguf_vram_budget(gguf, margin=1.0, training_reserve_active=True)


def test_check_gguf_vram_budget_fail_closed_when_checks_on_no_probe(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch("runtime.gpu_vram.nvidia_free_vram_by_device", return_value={}):
        with pytest.raises(LlamaServerError, match="VRAM checks are enabled"):
            check_gguf_vram_budget(gguf)


def test_check_gguf_vram_budget_enforces_admission_min_free(monkeypatch, tmp_path: Path):
    from runtime.gpu.priority import InferencePriority

    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"x")
    with patch(
        "runtime.gpu_vram.nvidia_free_vram_by_device",
        return_value={0: 512 * 1024**2},
    ):
        with patch(
            "runtime.gpu_vram.estimate_gguf_vram_bytes",
            return_value=100 * 1024**2,
        ):
            with pytest.raises(LlamaServerError, match="GPU memory"):
                check_gguf_vram_budget(
                    gguf, margin=1.0, priority=InferencePriority.NORMAL
                )


def test_check_gguf_vram_budget_rejects(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    gguf = tmp_path / "huge.gguf"
    gguf.write_bytes(b"x" * (200 * 1024 * 1024))

    with patch("runtime.gpu_vram.nvidia_free_vram_bytes", return_value=1024):
        with patch(
            "runtime.gpu_vram.estimate_gguf_vram_bytes",
            return_value=10 * 1024 * 1024 * 1024,
        ):
            with pytest.raises(LlamaServerError, match="GPU memory"):
                check_gguf_vram_budget(gguf, margin=1.0)


def test_format_vram_reject_kv_hint():
    assert "kv_cache" in format_vram_reject_kv_hint(
        {"kv_cache_bytes": 1024, "kv_bytes_per_slot": 8}
    )
    assert format_vram_reject_kv_hint({}) == ""


def test_vram_budget_health_fits(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_MARGIN", "1.0")
    est = {"required_per_gpu_bytes": 4 * 1024**3, "path": "exact_kv", "gguf": "/m.gguf"}
    bud = vram_budget_health(est, gpu_free_bottleneck=8 * 1024**3)
    assert bud is not None and bud["fits"] is True
    assert bud["fits_with_margin"] is True
    assert bud["headroom_bytes"] == 4 * 1024**3
    assert bud["model_gguf"] == "/m.gguf"


def test_vram_budget_health_margin(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_MARGIN", "1.5")
    est = {"required_per_gpu_bytes": 4 * 1024**3}
    bud = vram_budget_health(est, gpu_free_bottleneck=5 * 1024**3)
    assert bud is not None and bud["fits"] is True
    assert bud["fits_with_margin"] is False
    assert bud["required_with_margin_bytes"] == 6 * 1024**3


def test_vram_budget_health_short():
    est = {"required_per_gpu_bytes": 10 * 1024**3}
    bud = vram_budget_health(est, gpu_free_bottleneck=2 * 1024**3)
    assert bud is not None and bud["fits"] is False


def test_vram_budget_health_admission_gate(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    est = {"required_per_gpu_bytes": 1 * 1024**3}
    bud = vram_budget_health(est, gpu_free_bottleneck=8 * 1024**3)
    assert bud is not None
    assert "admission_fits" in bud
    assert bud["admission_fits"] is True


def test_check_gguf_vram_budget_reject_includes_kv_hint(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")

    with patch("runtime.gpu_vram.nvidia_free_vram_bytes", return_value=1024):
        with patch(
            "runtime.gpu_vram.estimate_gguf_vram_bytes",
            return_value=10 * 1024 * 1024 * 1024,
        ):
            with patch(
                "runtime.gpu_vram.describe_vram_estimate",
                return_value={"kv_cache_bytes": 512 * 1024 * 1024, "kv_bytes_per_slot": 8},
            ):
                with pytest.raises(LlamaServerError, match=r"kv_cache"):
                    check_gguf_vram_budget(gguf, margin=1.0)


def test_estimate_gguf_vram_bytes_uses_kv_factor(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_FACTOR", "2.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_EXACT", "0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_SCRATCH_FACTOR", "1.0")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x" * 1000)
    with patch("runtime.host_memory.estimate_gguf_ram_bytes", return_value=100):
        assert estimate_gguf_vram_bytes(gguf) == 200
        assert estimate_gguf_vram_bytes(gguf, tensor_parallel=2) == 100


def test_estimate_gguf_vram_bytes_estimate_factor(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE", "0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_EXACT", "0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_FACTOR", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_SCRATCH_FACTOR", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR", "1.25")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x" * 1000)
    with patch("runtime.host_memory.estimate_gguf_ram_bytes", return_value=100):
        assert estimate_gguf_vram_bytes(gguf) == 125


def test_estimate_factor_not_doubled_on_speculative_draft(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE", "0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_EXACT", "0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_FACTOR", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_SCRATCH_FACTOR", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR", "2.0")
    main = tmp_path / "main.gguf"
    draft = tmp_path / "draft.gguf"
    main.write_bytes(b"x" * 100)
    draft.write_bytes(b"x" * 100)

    def _ram(path: Path) -> int:
        return 1000 if path == main else 500

    with patch("runtime.host_memory.estimate_gguf_ram_bytes", side_effect=_ram):
        with patch(
            "runtime.gpu_vram._resolve_draft_gguf",
            return_value=draft,
        ):
            total = estimate_gguf_vram_bytes(main, llama_args=["--model-draft", str(draft)])
    # (1000 + 500) * 2.0 = 3000, not 1000*2 + 500*4
    assert total == 3000


def test_estimate_gguf_vram_bytes_exact_kv_path(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_EXACT", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_FACTOR", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_SCRATCH_FACTOR", "1.0")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x" * 1000)
    from runtime.gguf_estimate import GgufArchHints

    arch = GgufArchHints(
        scalar={"block_count": 32, "head_count_kv": 8, "key_length": 128}
    )
    with patch("runtime.host_memory.estimate_gguf_ram_bytes", return_value=1_000_000_000):
        with patch("runtime.gguf_estimate.gguf_arch_hints", return_value=arch):
            with patch("runtime.gpu_vram.resolve_vram_num_ctx", return_value=4096):
                with patch(
                    "runtime.gguf_estimate.estimate_kv_cache_bytes",
                    return_value=500_000_000,
                ):
                    assert estimate_gguf_vram_bytes(gguf) == 1_500_000_000


def test_tensor_parallel_uses_bottleneck_gpu(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch(
        "runtime.gpu_vram.nvidia_free_vram_by_device",
        return_value={0: 8 * 1024**3, 1: 1024},
    ):
        with patch(
            "runtime.gpu_vram.estimate_gguf_vram_bytes",
            return_value=4 * 1024**3,
        ):
            with pytest.raises(LlamaServerError, match="GPU 1"):
                check_gguf_vram_budget(
                    gguf, tensor_parallel=2, device_count=2, margin=1.0
                )


def test_llama_vram_device_indices_main_gpu_offset():
    assert llama_vram_device_indices(1, 1, 2) == [1]
    assert llama_vram_device_indices(1, 2, 3) == [1, 2]


def test_check_fails_closed_when_smi_partial(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch(
        "runtime.gpu_vram.nvidia_free_vram_by_device",
        return_value={0: 1024},
    ):
        with pytest.raises(LlamaServerError, match="VRAM checks are enabled"):
            check_gguf_vram_budget(gguf, tensor_parallel=2, device_count=2)


def test_vram_ctx_scale(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_NUM_CTX", "8192")
    assert vram_ctx_scale() == 2.0


def test_vram_ctx_scale_request_num_ctx_overrides_env(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_NUM_CTX", "4096")
    assert vram_ctx_scale(8192) == 2.0
    assert resolve_num_ctx({"num_ctx": 2048}) == 2048


def test_estimate_vram_uses_request_num_ctx(monkeypatch, tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch("runtime.host_memory.estimate_gguf_ram_bytes", return_value=1000):
        a = estimate_gguf_vram_bytes(gguf, num_ctx=4096)
        b = estimate_gguf_vram_bytes(gguf, num_ctx=8192)
    assert b > a


def test_nvidia_smi_cache(monkeypatch):
    import runtime.gpu_vram as gv

    gv._smi_cache.clear()
    calls = {"n": 0}

    def fake_query(device_index: int):
        calls["n"] += 1
        return 1024

    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_PROBE", "nvidia-smi")
    monkeypatch.setattr("runtime.gpu_vram.nvidia_smi_available", lambda: True)
    monkeypatch.setattr("runtime.gpu_vram._query_nvidia_smi_free_vram_bytes", fake_query)
    assert nvidia_free_vram_bytes(0) == 1024
    assert nvidia_free_vram_bytes(0) == 1024
    assert calls["n"] == 1
    assert active_vram_probe() == "nvidia-smi"


def test_vram_probe_auto_prefers_nvml(monkeypatch):
    import runtime.gpu_vram as gv

    gv._smi_cache.clear()
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_PROBE", "auto")
    monkeypatch.setattr("runtime.gpu_vram._query_nvml_free_vram_bytes", lambda i: 2048)
    monkeypatch.setattr(
        "runtime.gpu_vram._query_nvidia_smi_free_vram_bytes",
        lambda i: (_ for _ in ()).throw(AssertionError("smi should not run")),
    )
    assert nvidia_free_vram_bytes(0) == 2048
    assert active_vram_probe() == "nvml"


def test_gguf_layer_kv_scale(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_LAYER_BASE", "32")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch("runtime.gguf_estimate.gguf_model_hints", return_value={"block_count": 64}):
        assert gguf_layer_kv_scale(gguf) == 2.0
    with patch("runtime.gguf_estimate.gguf_model_hints", return_value={}):
        assert gguf_layer_kv_scale(gguf) == 1.0


def test_vram_probe_mode_env(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_PROBE", "nvml")
    assert vram_probe_mode() == "nvml"


def test_resolve_vram_num_ctx_from_gguf_hints(tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch(
        "runtime.gguf_estimate.gguf_model_hints",
        return_value={"context_length": 4096, "block_count": 32},
    ):
        assert resolve_vram_num_ctx(None, gguf) == 4096
        assert vram_ctx_scale(gguf=gguf) == 1.0


def test_resolve_vram_num_ctx_request_over_gguf(tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch(
        "runtime.gguf_estimate.gguf_model_hints",
        return_value={"context_length": 4096},
    ):
        assert resolve_vram_num_ctx({"num_ctx": 8192}, gguf) == 8192


def test_nvml_init_retries_after_cooldown(monkeypatch):
    import sys
    import time

    import runtime.gpu_vram as gv

    gv._reset_nvml()
    gv._pynvml_mod = None
    gv._nvml_init_failed_at = time.monotonic() - 60
    monkeypatch.setattr(gv, "_NVML_INIT_RETRY_COOLDOWN_S", 30.0)

    mod = type("m", (), {})()
    mod.nvmlInit = lambda: None
    mod.nvmlShutdown = lambda: None
    sys.modules["pynvml"] = mod
    try:
        got = gv._pynvml()
        assert got is mod
    finally:
        gv._reset_nvml()
        sys.modules.pop("pynvml", None)


def test_nvml_unified_fallback(monkeypatch):
    import runtime.gpu_vram as gv

    gv._smi_cache.clear()
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_PROBE", "nvml")

    class NotSupported(Exception):
        pass

    fake_nvml = type(
        "nvml",
        (),
        {
            "nvmlInit": staticmethod(lambda: None),
            "nvmlShutdown": staticmethod(lambda: None),
            "nvmlDeviceGetHandleByIndex": staticmethod(lambda i: i),
            "nvmlDeviceGetMemoryInfo": staticmethod(
                lambda h: (_ for _ in ()).throw(NotSupported("not supported"))
            ),
            "NVML_ERROR_NOT_SUPPORTED": 3,
            "NVMLError": NotSupported,
        },
    )()

    monkeypatch.setattr(gv, "_pynvml", lambda: fake_nvml)
    monkeypatch.setattr(gv, "_host_unified_free_vram_bytes", lambda: 8 * 1024**3)
    assert gv._query_nvml_free_vram_bytes(0) == 8 * 1024**3
    assert active_vram_probe() == "host-unified"


def test_gpu_vram_check_enabled_env(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "1")
    assert gpu_vram_check_enabled() is True
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CHECK_GPU_VRAM", "0")
    assert gpu_vram_check_enabled() is False


def test_resolve_vram_num_ctx_from_llama_c(monkeypatch, tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch(
        "runtime.gguf_estimate.gguf_model_hints",
        return_value={"context_length": 32768},
    ):
        assert (
            resolve_vram_num_ctx(None, gguf, llama_args=["-c", "8192"]) == 8192
        )


def test_estimate_vram_live_physical_yaml_inprocess_backend(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_EXACT", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_FACTOR", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_SCRATCH_FACTOR", "1.0")
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", raising=False)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    from runtime.gguf_estimate import GgufArchHints

    arch = GgufArchHints(
        scalar={"block_count": 32, "head_count_kv": 8, "key_length": 128}
    )
    args = ["-sm", "layer", "-mg", "0", "-np", "1"]
    with patch("runtime.host_memory.estimate_gguf_ram_bytes", return_value=100):
        with patch("runtime.gguf_estimate.gguf_arch_hints", return_value=arch):
            with patch(
                "runtime.gguf_estimate.estimate_kv_cache_bytes",
                return_value=1000,
            ):
                one = estimate_gguf_vram_bytes(
                    gguf,
                    llama_args=args,
                    parallel_slots_default=1,
                    llama_backend="subprocess",
                )
                two = estimate_gguf_vram_bytes(
                    gguf,
                    llama_args=args,
                    parallel_slots_default=1,
                    llama_backend="inprocess",
                )
    assert two > one
    assert one == 100 + 1000
    assert two == 100 + 2000


def test_estimate_vram_parallel_slots(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_EXACT", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_FACTOR", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_SCRATCH_FACTOR", "1.0")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    from runtime.gguf_estimate import GgufArchHints

    arch = GgufArchHints(
        scalar={"block_count": 32, "head_count_kv": 8, "key_length": 128}
    )
    with patch("runtime.host_memory.estimate_gguf_ram_bytes", return_value=100):
        with patch("runtime.gguf_estimate.gguf_arch_hints", return_value=arch):
            with patch(
                "runtime.gguf_estimate.estimate_kv_cache_bytes",
                return_value=1000,
            ):
                one = estimate_gguf_vram_bytes(gguf, llama_args=["-np", "1"])
                two = estimate_gguf_vram_bytes(
                    gguf, llama_args=["-np", "2"], parallel_slots_default=1
                )
    assert two > one
    assert two == 100 + 2000


def test_heuristic_parallel_slots_scales_kv_not_weights(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_EXACT", "0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_FACTOR", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_SCRATCH_FACTOR", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_LAYER_SCALE", "0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_NUM_CTX", "8192")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch("runtime.host_memory.estimate_gguf_ram_bytes", return_value=1000):
        one = estimate_gguf_vram_bytes(gguf, llama_args=["-np", "1"])
        four = estimate_gguf_vram_bytes(gguf, llama_args=["-np", "4"])
    assert four > one
    assert one == 2000
    assert four == 5000


def test_estimate_vram_includes_draft_model(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_EXACT", "0")
    main = tmp_path / "main.gguf"
    draft = tmp_path / "draft.gguf"
    main.write_bytes(b"x" * 1000)
    draft.write_bytes(b"x" * 500)
    with patch("runtime.host_memory.estimate_gguf_ram_bytes", side_effect=lambda p: 1000 if "main" in str(p) else 400):
        solo = estimate_gguf_vram_bytes(main, llama_args=["-np", "1"])
        both = estimate_gguf_vram_bytes(
            main,
            llama_args=["--model-draft", str(draft)],
            draft_gguf=draft,
        )
    assert both > solo


def test_estimate_vram_partial_ngl(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_EXACT", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_FACTOR", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_SCRATCH_FACTOR", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_RAM_OVERHEAD", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_WEIGHT_TENSOR", "0")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    from runtime.gguf_estimate import GgufArchHints, estimate_kv_cache_bytes

    arch = GgufArchHints(
        scalar={"block_count": 32, "head_count_kv": 8, "key_length": 128}
    )
    with patch("runtime.host_memory.estimate_gguf_ram_bytes", return_value=1000):
        with patch("runtime.gguf_estimate.gguf_arch_hints", return_value=arch):
            full = estimate_gguf_vram_bytes(gguf, llama_args=["-ngl", "-1"])
            half = estimate_gguf_vram_bytes(gguf, llama_args=["-ngl", "16"])
    kv_half = estimate_kv_cache_bytes(arch, 4096, n_gpu_layers=16, elem_bytes=2)
    kv_full = estimate_kv_cache_bytes(arch, 4096, elem_bytes=2)
    assert kv_half == kv_full // 2
    assert half < full
    assert half == 500 + kv_half


def test_estimate_vram_ngl_zero_skips_kv(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_EXACT", "1")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_KV_FACTOR", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_SCRATCH_FACTOR", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_RAM_OVERHEAD", "1.0")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_VRAM_WEIGHT_TENSOR", "0")
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    from runtime.gguf_estimate import GgufArchHints

    arch = GgufArchHints(scalar={"block_count": 8, "head_count_kv": 4, "key_length": 64})
    with patch("runtime.host_memory.estimate_gguf_ram_bytes", return_value=800):
        with patch("runtime.gguf_estimate.gguf_arch_hints", return_value=arch):
            cpu = estimate_gguf_vram_bytes(gguf, llama_args=["-ngl", "0"])
            gpu = estimate_gguf_vram_bytes(gguf, llama_args=["-ngl", "-1"])
    assert cpu == 0
    assert gpu > 800
