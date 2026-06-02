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
    args = cfg.llama_server_args()
    assert "-sm" in args and "tensor" in args
    ts_idx = args.index("-ts")
    assert args[ts_idx + 1] in ("1,1", "1.0,1.0")
