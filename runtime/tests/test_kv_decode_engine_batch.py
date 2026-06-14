"""Phase 15 v27 — engine wiring for C continuous batch decode."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.worker.libllama_ctypes import (
    LlamaLoadedSession,
    _ParallelDecodeJob,
    _decode_parallel_non_stream,
    _decode_parallel_stream,
)


def test_native_batch_decode_available_env(monkeypatch):
    from runtime.kv.native_decode_loop import native_batch_decode_available

    monkeypatch.setenv("ZEROLLAMA_KV_NATIVE_BATCH", "0")
    assert native_batch_decode_available() is False


def test_completions_parallel_uses_complete_parallel(cfg_root):
    from runtime.worker.llama_inprocess import LlamaInprocessWorker

    worker = LlamaInprocessWorker(
        model=cfg_root / "dummy.gguf",
        parallel_slots=4,
    )
    mock_session = MagicMock()
    mock_session.n_seq_max = 4
    mock_session.complete_parallel.return_value = ["aa", "bb"]
    worker._session = mock_session

    with patch(
        "runtime.kv.native_decode_loop.native_batch_decode_available",
        return_value=True,
    ):
        out = worker.completions_parallel(
            ["one", "two"],
            n_predict=8,
            id_slots=[0, 1],
        )

    assert [d["content"] for d in out] == ["aa", "bb"]
    mock_session.complete_parallel.assert_called_once()
    kwargs = mock_session.complete_parallel.call_args.kwargs
    assert kwargs["seq_ids"] == [0, 1]
    assert kwargs["n_predict"] == 8


def test_completions_parallel_falls_back_when_batch_disabled(cfg_root):
    from runtime.worker.llama_inprocess import LlamaInprocessWorker

    worker = LlamaInprocessWorker(
        model=cfg_root / "dummy.gguf",
        parallel_slots=4,
    )
    worker.completion = MagicMock(
        side_effect=[
            {"content": "x", "response": "x", "stop": True},
            {"content": "y", "response": "y", "stop": True},
        ]
    )

    with patch(
        "runtime.kv.native_decode_loop.native_batch_decode_available",
        return_value=False,
    ):
        out = worker.completions_parallel(["a", "b"], n_predict=4, id_slots=[0, 1])

    assert [d["content"] for d in out] == ["x", "y"]
    assert worker.completion.call_count == 2


def test_decode_parallel_non_stream_calls_batch_step(monkeypatch):
    import ctypes

    calls: list[int] = []

    def _fake_prefill(*args, **kwargs):
        return 1

    def _fake_sample(_smpl, _ctx):
        return 10

    def _fake_batch_step(_ctx, tokens, seq_ids, positions, *, smpl_ptr=0, smpl_ptrs=None):
        calls.append(len(tokens))
        if smpl_ptrs:
            return 1, [11 + i for i in range(len(tokens))]
        if smpl_ptr:
            return 1, [11, 12][: len(tokens)]
        return 1

    monkeypatch.setenv("ZEROLLAMA_KV_NATIVE_DECODE", "1")
    monkeypatch.setenv("ZEROLLAMA_KV_NATIVE_SAMPLE", "1")
    with (
        patch(
            "runtime.kv.native_decode_loop.run_prefill",
            side_effect=_fake_prefill,
        ),
        patch(
            "runtime.kv.native_decode_loop.run_sample",
            side_effect=_fake_sample,
        ),
        patch(
            "runtime.kv.native_decode_loop.run_batch_step",
            side_effect=_fake_batch_step,
        ),
        patch("runtime.kv.native_decode.record_decode_step"),
        patch("runtime.infer_trace.infer_trace"),
    ):
        lib = MagicMock()
        lib.llama_model_has_encoder.return_value = False
        lib.llama_vocab_is_eog.return_value = False
        vocab = MagicMock()
        ctx = ctypes.c_void_p(1)
        # Pass one sampler per job (new v27 contract)
        smpls = [ctypes.c_void_p(2), ctypes.c_void_p(3)]

        jobs = [
            _ParallelDecodeJob(
                prompt_tokens=[1, 2],
                seq_id=0,
                kv_slot=0,
                decode_pos=0,
                n_predict=2,
            ),
            _ParallelDecodeJob(
                prompt_tokens=[3, 4],
                seq_id=1,
                kv_slot=1,
                decode_pos=0,
                n_predict=2,
            ),
        ]
        with patch(
            "runtime.worker.libllama_ctypes.token_to_piece",
            side_effect=lambda _v, t: chr(ord("a") + t - 10),
        ):
            texts = _decode_parallel_non_stream(
                lib,
                MagicMock(),
                ctx,
                vocab,
                smpls,
                jobs,
                kv_block_size=16,
            )

    assert len(texts) == 2
    assert calls, "expected at least one batched decode step"
    assert max(calls) == 2


def test_decode_parallel_stream_passes_smpl_ptrs(monkeypatch):
    import ctypes

    smpl_ptr_calls: list[list[int]] = []

    def _fake_prefill(*args, **kwargs):
        return 1

    def _fake_sample(_smpl, _ctx):
        return 10

    def _fake_batch_step(
        _ctx, tokens, seq_ids, positions, *, smpl_ptr=0, smpl_ptrs=None
    ):
        if smpl_ptrs is not None:
            smpl_ptr_calls.append(list(smpl_ptrs))
            return 1, [11 + i for i in range(len(tokens))]
        return 1

    monkeypatch.setenv("ZEROLLAMA_KV_NATIVE_DECODE", "1")
    monkeypatch.setenv("ZEROLLAMA_KV_NATIVE_SAMPLE", "1")
    with (
        patch(
            "runtime.kv.native_decode_loop.run_prefill",
            side_effect=_fake_prefill,
        ),
        patch(
            "runtime.kv.native_decode_loop.run_sample",
            side_effect=_fake_sample,
        ),
        patch(
            "runtime.kv.native_decode_loop.run_batch_step",
            side_effect=_fake_batch_step,
        ),
        patch("runtime.kv.native_decode.record_decode_step"),
        patch("runtime.infer_trace.infer_trace"),
    ):
        lib = MagicMock()
        lib.llama_model_has_encoder.return_value = False
        lib.llama_vocab_is_eog.return_value = False
        vocab = MagicMock()
        ctx = ctypes.c_void_p(1)
        smpls = [ctypes.c_void_p(2), ctypes.c_void_p(3)]
        jobs = [
            _ParallelDecodeJob(
                prompt_tokens=[1, 2],
                seq_id=0,
                kv_slot=0,
                decode_pos=0,
                n_predict=2,
            ),
            _ParallelDecodeJob(
                prompt_tokens=[3, 4],
                seq_id=1,
                kv_slot=1,
                decode_pos=0,
                n_predict=2,
            ),
        ]
        with patch(
            "runtime.worker.libllama_ctypes.token_to_piece",
            side_effect=lambda _v, t: chr(ord("a") + t - 10),
        ):
            list(
                _decode_parallel_stream(
                    lib,
                    MagicMock(),
                    ctx,
                    vocab,
                    smpls,
                    jobs,
                    kv_block_size=16,
                )
            )

    assert smpl_ptr_calls, "expected batch step with per-row smpl_ptrs"
    assert smpl_ptr_calls[0] == [2, 3]


def test_run_batch_step_forwards_smpl_ptrs(monkeypatch):
    import sys
    from runtime.kv.native_decode_loop import run_batch_step

    mock_batch = MagicMock(return_value=(1, [5, 6]))
    fake_kv = MagicMock()
    fake_kv.decode_loop_batch_step = mock_batch
    monkeypatch.setitem(sys.modules, "runtime.kv._kv_native", fake_kv)
    monkeypatch.setattr(
        "runtime.kv.native_decode_loop.native_decode_loop_available",
        lambda: True,
    )
    out = run_batch_step(99, [1, 2], [0, 1], [10, 11], smpl_ptrs=[7, 8])
    assert out == (1, [5, 6])
    mock_batch.assert_called_once_with(99, [1, 2], [0, 1], [10, 11], [7, 8])


def test_generate_batch_uses_parallel_path(cfg_root):
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
        patch.object(eng, "_effective_llama_parallel_slots", return_value=4),
    ):
        out = eng.generate_batch(["one", "two"], n_predict=8, max_admit=2)
    assert [r.content for r in out] == ["a", "b"]
    mock_srv.completions_parallel.assert_called_once()


def test_completions_parallel_stream_uses_complete_parallel_stream(cfg_root):
    from runtime.worker.llama_inprocess import LlamaInprocessWorker

    worker = LlamaInprocessWorker(
        model=cfg_root / "dummy.gguf",
        parallel_slots=4,
    )
    mock_session = MagicMock()
    mock_session.n_seq_max = 4

    def _stream():
        yield {"seq_idx": 0, "seq_id": 0, "content": "a", "response": "a", "stop": False}
        yield {"seq_idx": 0, "seq_id": 0, "content": "", "response": "", "stop": True}
        yield {"seq_idx": 1, "seq_id": 1, "content": "b", "response": "b", "stop": False}
        yield {"seq_idx": 1, "seq_id": 1, "content": "", "response": "", "stop": True}

    mock_session.complete_parallel_stream.return_value = _stream()
    worker._session = mock_session

    with patch(
        "runtime.kv.native_decode_loop.native_batch_decode_available",
        return_value=True,
    ):
        chunks = list(
            worker.completions_parallel_stream(
                ["one", "two"],
                n_predict=8,
                id_slots=[0, 1],
            )
        )

    assert [c["content"] for c in chunks if not c.get("stop")] == ["a", "b"]
    mock_session.complete_parallel_stream.assert_called_once()


def test_decode_parallel_stream_yields_tagged_chunks(monkeypatch):
    import ctypes

    def _fake_prefill(*args, **kwargs):
        return 1

    def _fake_sample(_smpl, _ctx):
        return 10

    def _fake_batch_step(_ctx, tokens, seq_ids, positions, *, smpl_ptr=0, smpl_ptrs=None):
        return 1

    monkeypatch.setenv("ZEROLLAMA_KV_NATIVE_DECODE", "1")
    monkeypatch.setenv("ZEROLLAMA_KV_NATIVE_SAMPLE", "1")
    with (
        patch(
            "runtime.kv.native_decode_loop.run_prefill",
            side_effect=_fake_prefill,
        ),
        patch(
            "runtime.kv.native_decode_loop.run_sample",
            side_effect=_fake_sample,
        ),
        patch(
            "runtime.kv.native_decode_loop.run_batch_step",
            side_effect=_fake_batch_step,
        ),
        patch("runtime.kv.native_decode.record_decode_step"),
        patch("runtime.infer_trace.infer_trace"),
    ):
        lib = MagicMock()
        lib.llama_model_has_encoder.return_value = False
        lib.llama_vocab_is_eog.return_value = False
        vocab = MagicMock()
        ctx = ctypes.c_void_p(1)
        smpls = [ctypes.c_void_p(2), ctypes.c_void_p(3)]
        jobs = [
            _ParallelDecodeJob(
                prompt_tokens=[1, 2],
                seq_id=0,
                kv_slot=0,
                decode_pos=0,
                n_predict=1,
            ),
            _ParallelDecodeJob(
                prompt_tokens=[3, 4],
                seq_id=1,
                kv_slot=1,
                decode_pos=0,
                n_predict=1,
            ),
        ]
        with patch(
            "runtime.worker.libllama_ctypes.token_to_piece",
            side_effect=lambda _v, t: chr(ord("a") + t - 10),
        ):
            chunks = list(
                _decode_parallel_stream(
                    lib,
                    MagicMock(),
                    ctx,
                    vocab,
                    smpls,
                    jobs,
                    kv_block_size=16,
                )
            )

    tokens = [c for c in chunks if not c.get("stop")]
    assert len(tokens) == 2
    assert {c["seq_idx"] for c in tokens} == {0, 1}


def test_stream_generate_batch_uses_parallel_stream(cfg_root):
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

    def _stream():
        yield {
            "seq_idx": 0,
            "content": "a",
            "response": "a",
            "stop": False,
        }
        yield {
            "seq_idx": 0,
            "content": "",
            "response": "",
            "stop": True,
        }
        yield {
            "seq_idx": 1,
            "content": "b",
            "response": "b",
            "stop": False,
        }
        yield {
            "seq_idx": 1,
            "content": "",
            "response": "",
            "stop": True,
        }

    mock_srv.completions_parallel_stream.return_value = _stream()
    with (
        patch.object(eng, "_ensure_server", return_value=mock_srv),
        patch.object(eng, "_gpu_free_for_admission", return_value=8 * 1024**3),
        patch.object(eng, "_effective_llama_parallel_slots", return_value=4),
    ):
        chunks = list(
            eng.stream_generate_batch(["one", "two"], n_predict=8, max_admit=2)
        )

    assert [c["response"] for c in chunks if not c.get("done")] == ["a", "b"]
    assert all("request_id" in c for c in chunks)
    mock_srv.completions_parallel_stream.assert_called_once()
