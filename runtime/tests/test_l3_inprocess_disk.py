"""L3 in-process disk cache parity (llama_state_seq_*)."""

from __future__ import annotations

import ctypes
from unittest.mock import MagicMock, patch

import pytest

from runtime.cache_bridge import build_model_hash, slot_cache_file_path
from runtime.worker.libllama_ctypes import (
    LlamaLoadedSession,
    _save_slot_cache_disk,
    _try_restore_slot_cache_disk,
)


def test_save_slot_cache_disk_writes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_LLAMA_CACHE_ROOT", str(tmp_path))
    model_hash = build_model_hash(target_model_path="/m.gguf")
    lib = MagicMock()
    lib.llama_get_memory.return_value = ctypes.c_void_p(0x42)
    # pos_min=0, pos_max=2 → n=3 tokens
    lib.llama_memory_seq_pos_min.return_value = 0
    lib.llama_memory_seq_pos_max.return_value = 2
    lib.llama_state_seq_save_file.return_value = 128
    ctx = ctypes.c_void_p(1)
    n = _save_slot_cache_disk(
        lib,
        ctx,
        seq_id=2,
        model_hash=model_hash,
    )
    assert n == 128
    lib.llama_state_seq_save_file.assert_called_once()
    path = slot_cache_file_path(model_hash, 2)
    assert path.parent.is_dir()


def test_try_restore_slot_cache_disk_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_LLAMA_CACHE_ROOT", str(tmp_path))
    model_hash = build_model_hash(target_model_path="/m.gguf")
    lib = MagicMock()
    lib.llama_state_seq_load_file.return_value = 64
    ctx = ctypes.c_void_p(1)
    assert (
        _try_restore_slot_cache_disk(
            lib,
            ctx,
            seq_id=1,
            model_hash=model_hash,
            token_capacity=512,
        )
        == 0
    )
    lib.llama_state_seq_load_file.assert_not_called()


def test_complete_disk_restore_skips_clear(monkeypatch, tmp_path):
    """Pinned turn 2 restores from disk when RAM owner map is cold."""
    import threading

    monkeypatch.setenv("ZEROLLAMA_LLAMA_CACHE_ROOT", str(tmp_path))
    model_hash = build_model_hash(target_model_path="/m.gguf")
    slot_path = slot_cache_file_path(model_hash, 1)
    slot_path.parent.mkdir(parents=True, exist_ok=True)
    slot_path.write_bytes(b"fake-slot")

    session = MagicMock(spec=LlamaLoadedSession)
    session.n_seq_max = 2
    session._ctx = ctypes.c_void_p(0x1000)
    session._model = MagicMock()
    session._vocab = MagicMock()
    session._lib = MagicMock()
    session._lib.llama_n_ctx.return_value = 4096
    session.tokenize_text.return_value = [1, 2, 3]
    session._seq_last_owner = {}
    session._infer_lock = threading.RLock()
    session.slot_cache_model_hash = model_hash
    session.slot_cache_disk_persist = True
    session._resolve_decode_current_pos = (
        LlamaLoadedSession._resolve_decode_current_pos.__get__(session)
    )
    session._complete_locked = (
        LlamaLoadedSession._complete_locked.__get__(session)
    )
    session._prepare_seq_for_decode = (
        LlamaLoadedSession._prepare_seq_for_decode.__get__(session)
    )

    req = MagicMock()
    req.slot_pinned = True
    req.prompt_cache_key = "agent-1"
    req.request_id = "turn-2"

    with patch(
        "runtime.worker.libllama_ctypes.build_sampler_chain",
        return_value=MagicMock(),
    ):
        with patch(
            "runtime.worker.libllama_ctypes._normalize_seq_id", return_value=1
        ):
            with patch(
                "runtime.worker.libllama_ctypes._clear_sequence"
            ) as mock_clear:
                with patch(
                    "runtime.worker.libllama_ctypes._decode_non_stream",
                    return_value="ok",
                ):
                    with patch.object(session, "_physical_check_after_decode"):
                        with patch.object(session._lib, "llama_sampler_free"):
                            with patch(
                                "runtime.worker.libllama_ctypes._try_restore_slot_cache_disk",
                                return_value=42,
                            ) as mock_restore:
                                with patch.object(
                                    session,
                                    "_resolve_decode_current_pos",
                                    side_effect=[0, 20],
                                ):
                                    out = LlamaLoadedSession.complete(
                                        session,
                                        "hi",
                                        n_predict=1,
                                        current_pos=0,
                                        kv_bind_req=req,
                                    )

    assert out == "ok"
    mock_restore.assert_called_once()
    mock_clear.assert_not_called()
    assert session._seq_last_owner[1] == "cache:agent-1"


def test_complete_cache_prompt_false_skips_disk_restore(monkeypatch, tmp_path):
    """Policy block must not restore stale slot blobs from disk."""
    import threading

    monkeypatch.setenv("ZEROLLAMA_LLAMA_CACHE_ROOT", str(tmp_path))
    model_hash = build_model_hash(target_model_path="/m.gguf")
    slot_path = slot_cache_file_path(model_hash, 1)
    slot_path.parent.mkdir(parents=True, exist_ok=True)
    slot_path.write_bytes(b"fake-slot")

    session = MagicMock(spec=LlamaLoadedSession)
    session.n_seq_max = 2
    session._ctx = ctypes.c_void_p(0x1000)
    session._model = MagicMock()
    session._vocab = MagicMock()
    session._lib = MagicMock()
    session._lib.llama_n_ctx.return_value = 4096
    session.tokenize_text.return_value = [1, 2, 3]
    session._seq_last_owner = {}
    session._infer_lock = threading.RLock()
    session.slot_cache_model_hash = model_hash
    session.slot_cache_disk_persist = True
    session._resolve_decode_current_pos = (
        LlamaLoadedSession._resolve_decode_current_pos.__get__(session)
    )
    session._complete_locked = (
        LlamaLoadedSession._complete_locked.__get__(session)
    )
    session._prepare_seq_for_decode = (
        LlamaLoadedSession._prepare_seq_for_decode.__get__(session)
    )

    req = MagicMock()
    req.slot_pinned = True
    req.prompt_cache_key = "agent-1"
    req.request_id = "turn-2"

    with patch(
        "runtime.worker.libllama_ctypes.build_sampler_chain",
        return_value=MagicMock(),
    ):
        with patch(
            "runtime.worker.libllama_ctypes._normalize_seq_id", return_value=1
        ):
            with patch(
                "runtime.worker.libllama_ctypes._clear_sequence"
            ) as mock_clear:
                with patch(
                    "runtime.worker.libllama_ctypes._decode_non_stream",
                    return_value="ok",
                ):
                    with patch.object(session, "_physical_check_after_decode"):
                        with patch.object(session._lib, "llama_sampler_free"):
                            with patch(
                                "runtime.worker.libllama_ctypes._try_restore_slot_cache_disk",
                                return_value=42,
                            ) as mock_restore:
                                out = LlamaLoadedSession.complete(
                                    session,
                                    "hi",
                                    n_predict=1,
                                    current_pos=0,
                                    kv_bind_req=req,
                                    cache_prompt=False,
                                )

    assert out == "ok"
    mock_restore.assert_not_called()
    mock_clear.assert_called_once()
