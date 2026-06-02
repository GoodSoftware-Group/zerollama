from pathlib import Path
from unittest.mock import MagicMock

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine


def test_vocab_sessions_cleared_on_stop_server(tmp_path: Path):
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=18080,
        llama_cpp_root=tmp_path,
        llama_server_bin=tmp_path / "llama-server",
        llama_model=None,
        num_blocks=8,
        block_size=16,
        device_count=1,
        main_gpu=0,
    )
    eng = InferenceEngine(cfg)
    mock = MagicMock()
    eng._vocab_sessions["/fake.gguf"] = mock
    eng._stop_server()
    mock.close.assert_called_once()
    assert eng._vocab_sessions == {}
