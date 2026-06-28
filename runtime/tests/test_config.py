from pathlib import Path

import pytest

from runtime.config import RuntimeConfig


def test_load_dual_4090_yaml(monkeypatch):
    for key in (
        "ZEROLLAMA_DEVICE_COUNT",
        "ZEROLLAMA_TENSOR_PARALLEL",
        "ZEROLLAMA_KV_NUM_BLOCKS",
        "ZEROLLAMA_KV_BLOCK_SIZE",
        "LLAMA_MODEL",
        "LLAMA_SERVER_BIN",
    ):
        monkeypatch.delenv(key, raising=False)
    path = Path(__file__).resolve().parents[1] / "configs" / "dual_4090.yaml"
    cfg = RuntimeConfig.from_file(path)
    assert cfg.device_count == 2
    assert cfg.tensor_parallel == 2
    assert cfg.num_blocks == 8192
    assert cfg.block_size == 16
    assert cfg.active_kv_pools() == 2
    assert cfg.llama_backend == "subprocess"
    args = cfg.llama_server_args()
    assert "-sm" in args and "tensor" in args
    ts_idx = args.index("-ts")
    assert args[ts_idx + 1] in ("1,1", "1.0,1.0")


def test_load_l3_agent_yaml(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_GPU_PROFILE", "0")
    for key in ("ZEROLLAMA_RADIX_PREFIX_SHARE", "ZEROLLAMA_PREFIX_CACHE_BLOCK_SIZE"):
        monkeypatch.delenv(key, raising=False)
    path = Path(__file__).resolve().parents[1] / "configs" / "l3_agent_subprocess.yaml"
    cfg = RuntimeConfig.from_file(path)
    from runtime.env import l3_settings, radix_prefix_share_enabled

    assert cfg.llama_parallel_slots == 4
    assert l3_settings().radix_share is True
    assert l3_settings().block_size == 512
    assert radix_prefix_share_enabled() is True
