from unittest.mock import MagicMock, patch

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.scheduler.scheduler import Request


def test_stream_generate_yields_ollama_chunks(cfg_root):
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=None,
        num_blocks=64,
        block_size=16,
        device_count=1,
    )
    eng = InferenceEngine(cfg)
    mock_srv = MagicMock()
    mock_srv.completion_stream.return_value = [
        {"content": "a", "stop": False},
        {"content": "", "stop": True},
    ]
    fake_req = Request("r1", [1], 8, num_ctx=512)
    eng.scheduler.add_request(fake_req)
    eng.scheduler.waiting.clear()
    fake_req.block_table.ensure_capacity(512)  # type: ignore[union-attr]
    with patch.object(eng, "_admit_one", return_value=fake_req):
        with patch.object(eng, "_ensure_gguf_loaded_unlocked", return_value=mock_srv):
            chunks = list(eng.stream_generate("hi", "m", n_predict=8))
    assert chunks[0]["status"] == "accepted"
    token_chunks = [c for c in chunks if c.get("response")]
    assert token_chunks[0]["response"] == "a"
    assert token_chunks[0]["done"] is False
    assert chunks[-1]["done"] is True
