"""Phase 15 v15 — _decode_stream current_pos + native prefill resume wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from runtime.worker.libllama_ctypes import _decode_stream


def _mock_lib(*, has_encoder: bool = False) -> MagicMock:
    lib = MagicMock()
    lib.llama_model_has_encoder.return_value = has_encoder
    lib.llama_vocab_is_eog.return_value = False
    lib.llama_sampler_sample.return_value = 42
    return lib


@pytest.mark.parametrize("current_pos", [0, 10])
def test_decode_stream_native_prefill_pos_start(current_pos: int):
    """Native prefill receives remaining tokens and pos_start from current_pos."""
    prompt = list(range(20))
    lib = _mock_lib()
    ctx = MagicMock()
    model = MagicMock()
    vocab = MagicMock()
    smpl = MagicMock()

    with patch("runtime.kv.native_decode_loop.run_prefill") as mock_prefill:
        mock_prefill.return_value = 2
        with patch("runtime.kv.native_decode_loop.run_sample", return_value=99):
            with patch("runtime.kv.native_decode_loop.run_step") as mock_step:
                mock_step.return_value = (1, 100)
                with patch(
                    "runtime.worker.libllama_ctypes.token_to_piece",
                    return_value="x",
                ):
                    with patch(
                        "runtime.worker.libllama_ctypes.ctypes"
                    ) as mock_ctypes:
                        mock_ctypes.cast.return_value.value = 0x1000
                        mock_ctypes.c_void_p = MagicMock()
                        chunks = list(
                            _decode_stream(
                                lib,
                                model,
                                ctx,
                                vocab,
                                smpl,
                                prompt,
                                n_predict=1,
                                current_pos=current_pos,
                            )
                        )

    if current_pos == 0:
        mock_prefill.assert_called_once()
        _args, kwargs = mock_prefill.call_args
        assert _args[1] == prompt
        assert kwargs.get("pos_start") == 0
    else:
        mock_prefill.assert_called_once()
        _args, kwargs = mock_prefill.call_args
        assert _args[1] == prompt[current_pos:]
        assert kwargs.get("pos_start") == current_pos
    assert any(c.get("content") for c in chunks)


def test_decode_stream_skips_prefill_when_current_pos_past_prompt():
    """When prefill is complete, native prefill must not run."""
    prompt = list(range(20))
    lib = _mock_lib()
    ctx = MagicMock()
    model = MagicMock()
    vocab = MagicMock()
    smpl = MagicMock()

    with patch("runtime.kv.native_decode_loop.run_prefill") as mock_prefill:
        with patch("runtime.kv.native_decode_loop.run_sample", return_value=7):
            with patch("runtime.kv.native_decode_loop.run_step", return_value=(1, 8)):
                with patch(
                    "runtime.worker.libllama_ctypes.token_to_piece",
                    return_value="t",
                ):
                    with patch(
                        "runtime.worker.libllama_ctypes.ctypes"
                    ) as mock_ctypes:
                        mock_ctypes.cast.return_value.value = 0x2000
                        mock_ctypes.c_void_p = MagicMock()
                        list(
                            _decode_stream(
                                lib,
                                model,
                                ctx,
                                vocab,
                                smpl,
                                prompt,
                                n_predict=1,
                                current_pos=25,
                            )
                        )

    mock_prefill.assert_not_called()


def test_native_step_and_sample_non_tuple_raises():
    """If run_step returns an int instead of (steps, token) when smpl_ptr is set,
    _decode_stream must raise LlamaServerError, not silently decode twice."""
    from runtime.worker.libllama_ctypes import LlamaServerError

    prompt = list(range(5))
    lib = _mock_lib()

    with patch("runtime.kv.native_decode_loop.run_prefill", return_value=1):
        with patch("runtime.kv.native_decode_loop.run_sample", return_value=10):
            # Simulate ABI mismatch: run_step returns plain int instead of tuple.
            with patch("runtime.kv.native_decode_loop.run_step", return_value=1):
                with patch(
                    "runtime.worker.libllama_ctypes.token_to_piece",
                    return_value="x",
                ):
                    with patch(
                        "runtime.worker.libllama_ctypes.ctypes"
                    ) as mock_ctypes:
                        mock_ctypes.cast.return_value.value = 0x3000
                        mock_ctypes.c_void_p = MagicMock()
                        with pytest.raises(LlamaServerError, match="non-tuple"):
                            list(
                                _decode_stream(
                                    lib,
                                    MagicMock(),
                                    MagicMock(),
                                    MagicMock(),
                                    MagicMock(),
                                    prompt,
                                    n_predict=2,
                                )
                            )
