"""Phase 15 v16–v17 — engine current_pos wiring + resume prefill."""

from __future__ import annotations

import ctypes
import threading
from unittest.mock import MagicMock, patch

import pytest

from runtime.cache_bridge import slot_resume_owner_key
from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.kv.physical import current_pos_for_seq
from runtime.worker.libllama_ctypes import LlamaLoadedSession


@pytest.fixture
def engine(cfg_root, tmp_path):
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
    return InferenceEngine(cfg)


def test_current_pos_for_seq_empty():
    with patch(
        "runtime.kv.physical.usage_from_libllama", return_value=None
    ):
        assert current_pos_for_seq(MagicMock(), MagicMock(), 0) == 0


def test_current_pos_for_seq_from_pos_max():
    from runtime.kv.physical import SequenceKvUsage

    usage = SequenceKvUsage(seq_id=0, pos_min=0, pos_max=19)
    with patch("runtime.kv.physical.usage_from_libllama", return_value=usage):
        assert current_pos_for_seq(MagicMock(), MagicMock(), 0) == 20


def test_slot_resume_owner_key_pinned_uses_cache_key():
    req = MagicMock()
    req.slot_pinned = True
    req.prompt_cache_key = "sess-abc"
    req.request_id = "turn-2"
    assert slot_resume_owner_key(req) == "cache:sess-abc"


def test_slot_resume_owner_key_unpinned_uses_request_id():
    req = MagicMock()
    req.slot_pinned = False
    req.prompt_cache_key = "ignored"
    req.request_id = "req-xyz"
    assert slot_resume_owner_key(req) == "req-xyz"


def test_slot_resume_owner_key_pinned_without_cache_key_falls_back():
    """Defensive: pinned without key should not happen from admit; uses request_id."""
    req = MagicMock()
    req.slot_pinned = True
    req.prompt_cache_key = None
    req.request_id = "req-fallback"
    assert slot_resume_owner_key(req) == "req-fallback"


def test_close_clears_seq_last_owner():
    session = _make_session_stub()
    session._seq_last_owner = {0: "cache:sess", 1: "req-x"}
    session._ctx = None
    session._model = None

    LlamaLoadedSession.close(session)

    assert session._seq_last_owner == {}


def _make_session_stub() -> MagicMock:
    session = MagicMock(spec=LlamaLoadedSession)
    session.n_seq_max = 2
    # WHY c_void_p: _complete_locked ctypes.cast(self._ctx, ...) segfaults on MagicMock (Py 3.9).
    session._ctx = ctypes.c_void_p(0x1000)
    session._model = MagicMock()
    session._vocab = MagicMock()
    session._lib = MagicMock()
    session._lib.llama_n_ctx.return_value = 4096
    session.tokenize_text.return_value = [1, 2, 3]
    session._seq_last_owner = {}
    session._infer_lock = threading.RLock()
    session.slot_cache_model_hash = None
    session._resolve_decode_current_pos = (
        LlamaLoadedSession._resolve_decode_current_pos.__get__(session)
    )
    session._prepare_seq_for_decode = (
        LlamaLoadedSession._prepare_seq_for_decode.__get__(session)
    )
    session._complete_locked = (
        LlamaLoadedSession._complete_locked.__get__(session)
    )
    return session


def _run_complete(session, kv_bind_req=None, current_pos=0, *, decode_side_effect=None):
    decode_mock = patch(
        "runtime.worker.libllama_ctypes._decode_non_stream",
        side_effect=decode_side_effect,
        return_value="ok" if decode_side_effect is None else None,
    )
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
                with decode_mock:
                    with patch.object(session, "_physical_check_after_decode"):
                        with patch.object(session._lib, "llama_sampler_free"):
                            out = LlamaLoadedSession.complete(
                                session,
                                "hi",
                                n_predict=1,
                                current_pos=current_pos,
                                kv_bind_req=kv_bind_req,
                            )
    return out, mock_clear


def test_complete_skips_clear_same_request_id():
    """Same unpinned request resumes: decode_pos > 0 + matching owner → no clear."""
    session = _make_session_stub()
    req = MagicMock()
    req.slot_pinned = False
    req.prompt_cache_key = None
    req.request_id = "req-abc"
    session._seq_last_owner[1] = "req-abc"

    out, mock_clear = _run_complete(session, kv_bind_req=req, current_pos=10)

    assert out == "ok"
    mock_clear.assert_not_called()


def test_complete_skips_clear_l3_second_turn():
    """L3 pinned: new request_id but same prompt_cache_key → no clear (v17)."""
    session = _make_session_stub()
    req = MagicMock()
    req.slot_pinned = True
    req.prompt_cache_key = "sess-agent-1"
    req.request_id = "turn-2-new-id"
    session._seq_last_owner[1] = "cache:sess-agent-1"

    out, mock_clear = _run_complete(session, kv_bind_req=req, current_pos=42)

    assert out == "ok"
    mock_clear.assert_not_called()


def test_complete_clears_sequence_different_pinned_session():
    """Different L3 session on same slot → clear even if decode_pos > 0."""
    session = _make_session_stub()
    session._seq_last_owner[1] = "cache:sess-a"

    req = MagicMock()
    req.slot_pinned = True
    req.prompt_cache_key = "sess-b"
    req.request_id = "turn-1"

    _out, mock_clear = _run_complete(session, kv_bind_req=req, current_pos=10)

    mock_clear.assert_called_once()


def test_complete_clears_sequence_different_request_id():
    """Different unpinned request takes same slot: must clear."""
    session = _make_session_stub()
    session._seq_last_owner[1] = "req-old"

    new_req = MagicMock()
    new_req.slot_pinned = False
    new_req.prompt_cache_key = None
    new_req.request_id = "req-new"

    _out, mock_clear = _run_complete(session, kv_bind_req=new_req, current_pos=10)

    mock_clear.assert_called_once()


def test_complete_clears_sequence_no_req_id_with_decode_pos():
    """decode_pos > 0 but no kv_bind_req → conservative clear."""
    session = _make_session_stub()
    session._seq_last_owner[1] = "req-abc"

    _out, mock_clear = _run_complete(session, kv_bind_req=None, current_pos=10)

    mock_clear.assert_called_once()


def test_complete_clears_sequence_when_current_pos_zero():
    """decode_pos == 0 always clears regardless of owner."""
    session = _make_session_stub()
    req = MagicMock()
    req.slot_pinned = False
    req.request_id = "req-abc"
    session._seq_last_owner[1] = "req-abc"

    _out, mock_clear = _run_complete(session, kv_bind_req=req, current_pos=0)

    mock_clear.assert_called_once()


def test_complete_resets_decode_pos_after_clear():
    """v22: stale decode_pos must not reach _decode_non_stream after _clear_sequence."""
    session = _make_session_stub()
    session._seq_last_owner[1] = "req-old"

    new_req = MagicMock()
    new_req.slot_pinned = False
    new_req.prompt_cache_key = None
    new_req.request_id = "req-new"

    decode_positions: list[int] = []

    def _capture_decode(*_args, **kwargs):
        decode_positions.append(int(kwargs.get("current_pos", -1)))
        return "ok"

    _out, mock_clear = _run_complete(
        session,
        kv_bind_req=new_req,
        current_pos=10,
        decode_side_effect=_capture_decode,
    )

    mock_clear.assert_called_once()
    assert decode_positions == [0]


def test_engine_decode_current_pos_for_request():
    from runtime.engine import InferenceEngine

    eng = MagicMock(spec=InferenceEngine)
    fake_lib = MagicMock()
    fake_ctx = MagicMock()
    eng._inprocess_ctx_for_health.return_value = (fake_lib, fake_ctx)
    eng._id_slot_for_request.return_value = 3

    with patch(
        "runtime.kv.physical.current_pos_for_seq", return_value=15
    ) as mock_pos:
        pos = InferenceEngine._decode_current_pos_for_request(eng, MagicMock())

    assert pos == 15
    mock_pos.assert_called_once_with(fake_lib, fake_ctx, 3)


def test_engine_decode_current_pos_none_when_subprocess():
    from runtime.engine import InferenceEngine

    eng = MagicMock(spec=InferenceEngine)
    eng._inprocess_ctx_for_health.return_value = None
    assert InferenceEngine._decode_current_pos_for_request(eng, MagicMock()) is None


def test_kv_resume_health_inactive_without_inprocess():
    from runtime.engine import InferenceEngine

    eng = MagicMock(spec=InferenceEngine)
    eng._health_llama_backend.return_value = "subprocess"
    eng._effective_llama_parallel_slots.return_value = 2
    eng._inprocess_session_for_health.return_value = None

    out = InferenceEngine._kv_resume_health(eng)

    assert out["active"] is False
    assert "subprocess" in (out.get("note") or "")


def test_kv_resume_health_shows_owners():
    from runtime.engine import InferenceEngine

    session = MagicMock()
    session._ctx = MagicMock()
    session.resume_owner_snapshot.return_value = {1: "cache:sess-a"}

    eng = MagicMock(spec=InferenceEngine)
    eng._health_llama_backend.return_value = "inprocess"
    eng._effective_llama_parallel_slots.return_value = 2
    eng._inprocess_session_for_health.return_value = session

    out = InferenceEngine._kv_resume_health(eng)

    assert out["active"] is True
    assert out["owners_by_slot"] == {"1": "cache:sess-a"}


def test_generate_l3_second_turn_passes_current_pos(engine, monkeypatch):
    """Engine forwards live current_pos on L3 turn 2 (v18 wiring)."""
    from contextlib import contextmanager
    from runtime.gpu.inference_policy import InferencePriority

    monkeypatch.setattr(engine, "_vram_precheck_enqueue", lambda *a, **k: None)
    monkeypatch.setattr(
        engine, "_check_admit_policy", lambda opts, **k: InferencePriority.NORMAL
    )

    @contextmanager
    def _hold(_gguf):
        yield

    monkeypatch.setattr(engine._model_swap, "hold", _hold)

    positions: list[int | None] = []
    mock_srv = MagicMock()
    mock_srv.is_running.return_value = True

    def _completion(*_args, current_pos=None, **_kwargs):
        positions.append(current_pos)
        return {"content": "ok", "response": "ok"}

    mock_srv.completion = _completion

    with patch.object(
        engine, "resolve_num_ctx_for_request", return_value=(512, {})
    ):
        with patch.object(engine, "_ensure_gguf_loaded_unlocked", return_value=mock_srv):
            with patch.object(
                engine,
                "_decode_current_pos_for_request",
                side_effect=[None, 55],
            ):
                engine.generate(
                    "turn one",
                    4,
                    options={"prompt_cache_key": "agent-sess"},
                )
                engine.generate(
                    "turn one plus more",
                    4,
                    options={"prompt_cache_key": "agent-sess"},
                )

    assert len(positions) == 2
    assert positions[1] == 55
