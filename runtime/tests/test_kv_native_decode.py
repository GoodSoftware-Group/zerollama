"""Phase 15 v6: native decode step hook."""

import pytest

from runtime.kv.native_decode import (
    decode_hook_enabled,
    decode_steps_health,
    record_decode_step,
    reset_decode_steps_for_tests,
)


def test_decode_hook_enabled_by_default(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_KV_DECODE_HOOK", raising=False)
    assert decode_hook_enabled() is True
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_KV_DECODE_HOOK", "0")
    assert decode_hook_enabled() is False


def test_decode_steps_health_subprocess_inactive():
    h = decode_steps_health(llama_backend="subprocess")
    assert h is not None
    assert h["active"] is False
    assert "subprocess" in str(h["reason"])


def test_python_decode_step_fallback(monkeypatch):
    reset_decode_steps_for_tests()
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_KV_DECODE_HOOK", "1")
    monkeypatch.setattr(
        "runtime.kv.native_decode.native_decode_available", lambda: False
    )
    assert record_decode_step(1) == 1
    assert record_decode_step(2) == 3
    h = decode_steps_health(llama_backend="inprocess")
    assert h is not None
    assert h["active"] is True
    assert h["value"] == 3
    assert h["source"] == "python"


def test_native_decode_step_optional():
    pytest.importorskip("runtime.kv._kv_native")
    from runtime.kv._kv_native import decode_step
    from runtime.kv.native_decode import reset_decode_steps_for_tests

    reset_decode_steps_for_tests()
    base = int(decode_step(0))
    assert int(decode_step(2)) == base + 2
    assert int(decode_step(0)) == base + 2
