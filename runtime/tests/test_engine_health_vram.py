from pathlib import Path
from unittest.mock import patch

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine


def test_health_vram_budget_from_config_model(cfg_root, tmp_path: Path):
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
    est = {
        "required_per_gpu_bytes": 6 * 1024**3,
        "path": "exact_kv",
        "kv_cache_bytes": 2 * 1024**3,
    }
    with patch("runtime.engine.nvidia_free_vram_by_device", return_value={0: 8 * 1024**3}):
        with patch("runtime.engine.describe_vram_estimate", return_value=est):
            body = eng.health()
    assert body["llama_server"] is False
    assert body["llama_backend"] == "subprocess"
    assert body["vram_estimate"] == est
    assert body["vram_budget"]["fits"] is True
    assert body["vram_budget"]["kv_cache_bytes"] == 2 * 1024**3
    assert body["admission"]["vram_load_fits"] is True
