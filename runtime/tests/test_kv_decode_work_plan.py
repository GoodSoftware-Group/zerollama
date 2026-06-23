"""Phase 15 v11–v13 — unified decode work plan + native decode loop."""

from __future__ import annotations

from runtime.kv.decode_plan import kv_decode_work_plan
from runtime.kv.native_decode_loop import (
    PrefillAbortedError,
    decode_loop_status,
    greedy_decode_tokens,
    native_decode_loop_available,
    prefill_abort_clear,
    prefill_abort_set,
    run_prefill,
    run_step,
)
from runtime.kv.physical import current_pos_by_request_from_physical


def test_decode_work_plan_admit_phase():
    out = kv_decode_work_plan(list(range(20)), block_size=16, max_tokens=8)
    assert out["phase"] == "admit"
    assert out["current_pos"] == 0
    assert out["prefill"]["n_prefill_batches"] >= 1
    assert "decode" not in out  # decode sub-object absent for admit phase


def test_decode_work_plan_empty_prompt_prefill_complete():
    """Empty prompt → prefill_complete True; phase decode when current_pos given."""
    out_admit = kv_decode_work_plan([], block_size=16, max_tokens=4)
    assert out_admit["phase"] == "admit"
    assert out_admit["prefill"]["prefill_complete"] is True

    out_live = kv_decode_work_plan([], block_size=16, max_tokens=4, current_pos=0)
    assert out_live["phase"] == "decode"
    assert out_live["prefill"]["prefill_complete"] is True


def test_decode_work_plan_prefill_phase():
    out = kv_decode_work_plan(
        list(range(20)),
        block_size=16,
        max_tokens=8,
        current_pos=10,
    )
    assert out["phase"] == "prefill"
    assert out["decode"]["pending_prefill"] is True


def test_decode_work_plan_decode_phase():
    out = kv_decode_work_plan(
        list(range(20)),
        block_size=16,
        max_tokens=8,
        current_pos=20,
    )
    assert out["phase"] == "decode"
    assert out["prefill"]["prefill_complete"] is True
    assert out["decode"]["n_decode_batches_remaining"] == 8


def test_decode_work_plan_done_phase():
    out = kv_decode_work_plan(
        list(range(20)),
        block_size=16,
        max_tokens=5,
        current_pos=25,
    )
    assert out["phase"] == "done"
    assert out["decode"]["n_decode_batches_remaining"] == 0


def test_current_pos_by_request_from_physical():
    physical = {
        "running": [
            {
                "request_id": "r1",
                "llama_tracked": True,
                "llama_pos_max": 19,
            },
            {
                "request_id": "r2",
                "llama_tracked": False,
            },
        ]
    }
    assert current_pos_by_request_from_physical(physical) == {"r1": 20}
    assert current_pos_by_request_from_physical(None) == {}


def test_decode_loop_status_not_linked_by_default():
    status = decode_loop_status()
    assert status["link"] in ("ctypes", "native")
    assert "available" in status
    assert "reason" in status
    if status["link"] == "native" and status["available"]:
        assert int(status.get("llama_max_devices", 0)) >= 1
        if status.get("gil_released") is not None:
            assert status.get("gil_released") is True
        if status.get("sampling_in_c") is not None:
            assert status.get("sampling_in_c") is True
    else:
        assert native_decode_loop_available() is False
        assert "gil_released" not in status or status.get("gil_released") is not True


def test_run_prefill_returns_none_when_not_linked():
    """run_prefill returns None without a live libllama ctx — safe no-op."""
    if native_decode_loop_available():
        return  # skip: can only test the false branch without a real ctx
    result = run_prefill(0, [1, 2, 3], seq_id=0, block_size=0)
    assert result is None


def test_run_step_returns_none_when_not_linked():
    """run_step returns None without a live libllama ctx — safe no-op."""
    if native_decode_loop_available():
        return
    result = run_step(0, 42, seq_id=0, current_pos=5)
    assert result is None


def test_prefill_abort_helpers_no_op_when_not_linked():
    """prefill_abort_set/clear must not raise even when the ext is not built."""
    if native_decode_loop_available():
        return  # skip: only test the ImportError-swallowed path
    prefill_abort_set()   # should silently swallow ImportError
    prefill_abort_clear()  # same


def _patched_run_prefill_with_error(err: Exception) -> object:
    """Helper: patch the inner ``decode_loop_prefill`` import inside run_prefill."""
    import sys
    from unittest.mock import MagicMock, patch

    # Inject a fake _kv_native module so the ``from runtime.kv._kv_native import …``
    # inside run_prefill resolves without loading the real (unlinked) .so.
    fake_native = MagicMock()
    fake_native.decode_loop_prefill = MagicMock(side_effect=err)
    fake_native.decode_loop_abort_clear = MagicMock()

    with patch.dict(sys.modules, {"runtime.kv._kv_native": fake_native}):
        with patch(
            "runtime.kv.native_decode_loop.native_decode_loop_available",
            return_value=True,
        ):
            try:
                run_prefill(0, [1, 2, 3], seq_id=0, block_size=16)
                return None  # no exception
            except Exception as exc:
                return exc


def test_run_prefill_raises_prefill_aborted_error_on_minus_three():
    """run_prefill must raise PrefillAbortedError when C raises 'KV prefill aborted'."""
    abort_err = ValueError("KV prefill aborted: cancel flag set between chunks")
    exc = _patched_run_prefill_with_error(abort_err)
    assert isinstance(exc, PrefillAbortedError), f"expected PrefillAbortedError, got {exc!r}"


def test_run_prefill_page_bind_error_still_raises_on_minus_two():
    """KV page bind errors must still raise (not be swallowed by abort check)."""
    bind_err = ValueError("KV page bind: token position out of range for kv_slot")
    exc = _patched_run_prefill_with_error(bind_err)
    assert exc is not None, "expected an exception"
    assert "page bind" in str(exc).lower(), f"unexpected exception: {exc!r}"


def test_greedy_decode_tokens_zero_predict_returns_empty():
    """n_predict=0 must not sample — prefill only."""
    from unittest.mock import MagicMock, patch

    lib = MagicMock()
    with patch(
        "runtime.kv.native_decode_loop.native_decode_loop_available", return_value=True
    ):
        with patch("runtime.kv.native_decode_loop.run_prefill", return_value=1):
            out = greedy_decode_tokens(
                123,
                lib,
                MagicMock(),
                MagicMock(),
                MagicMock(),
                [1, 2, 3],
                n_predict=0,
            )
    assert out == []
    lib.llama_sampler_sample.assert_not_called()
