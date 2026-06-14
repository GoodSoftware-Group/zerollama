from unittest.mock import MagicMock, patch

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.worker.llama_server import LlamaServerError


def test_generate_batch_passes_cache_prompts(cfg_root):
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=None,
        num_blocks=256,
        block_size=16,
        device_count=1,
        tensor_parallel=1,
    )
    eng = InferenceEngine(cfg)
    mock_srv = MagicMock()
    mock_srv.completions_parallel.return_value = [
        {"content": "a"},
        {"content": "b"},
    ]
    with (
        patch.object(eng, "_ensure_server", return_value=mock_srv),
        patch.object(eng, "_gpu_free_for_admission", return_value=8 * 1024**3),
    ):
        eng.generate_batch(
            ["one", "two"],
            n_predict=8,
            max_admit=2,
            options={
                "prompt_cache_keys": ["key-a", "key-b"],
            },
        )
    kwargs = mock_srv.completions_parallel.call_args.kwargs
    assert kwargs.get("cache_prompts") == [True, True]


def test_generate_batch_admits_multiple(cfg_root):
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=None,
        num_blocks=256,
        block_size=16,
        device_count=1,
        tensor_parallel=1,
    )
    eng = InferenceEngine(cfg)
    mock_srv = MagicMock()
    mock_srv.completions_parallel.return_value = [
        {"content": "a"},
        {"content": "b"},
    ]
    with (
        patch.object(eng, "_ensure_server", return_value=mock_srv),
        patch.object(eng, "_gpu_free_for_admission", return_value=8 * 1024**3),
    ):
        out = eng.generate_batch(["one", "two"], n_predict=8, max_admit=2)
    assert [r.content for r in out] == ["a", "b"]
    assert len(eng.scheduler.running) == 0
    mock_srv.completions_parallel.assert_called_once()


def test_generate_batch_rejects_when_paused(cfg_root):
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
    eng.training_handoff()
    try:
        eng.generate_batch(["x"])
        raise AssertionError("expected pause")
    except LlamaServerError as e:
        assert "paused" in str(e).lower()
