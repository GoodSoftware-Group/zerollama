"""Tests for Phase 15 v48 — CPU-only donor-buffer overlay bind.

WHY the native ext is skipped/mocked here: the donor-buffer registry lives in
libllama (built only with LLAMA_KV_EXT_DONOR_BUFFER=1 + a linked llama.cpp),
which is not guaranteed in CI. These tests exercise the Python facade's
argument validation, env gating, and status dict shape via monkeypatching, so
they run everywhere; native-ext-linked behavior is covered by
scripts/phase/phase15_overlay_bind_cpu_smoke.sh.
"""

from __future__ import annotations

import pytest

from runtime.kv import overlay_bind


def test_overlay_bind_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_KV_OVERLAY_BIND", raising=False)
    assert overlay_bind.overlay_bind_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on"])
def test_overlay_bind_enabled_truthy_values(monkeypatch, value):
    monkeypatch.setenv("ZEROLLAMA_KV_OVERLAY_BIND", value)
    assert overlay_bind.overlay_bind_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "", "off"])
def test_overlay_bind_enabled_falsy_values(monkeypatch, value):
    monkeypatch.setenv("ZEROLLAMA_KV_OVERLAY_BIND", value)
    assert overlay_bind.overlay_bind_enabled() is False


def test_register_refused_when_not_enabled(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_KV_OVERLAY_BIND", raising=False)
    with pytest.raises(RuntimeError, match="ZEROLLAMA_KV_OVERLAY_BIND"):
        overlay_bind.register_donor_buffer(0x1000, 4096)


def test_register_refused_without_native_ext(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_OVERLAY_BIND", "1")
    monkeypatch.setattr(overlay_bind, "donor_buffer_available", lambda: False)
    with pytest.raises(RuntimeError, match="native ext"):
        overlay_bind.register_donor_buffer(0x1000, 4096)


def test_register_rejects_non_positive_args(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_OVERLAY_BIND", "1")
    monkeypatch.setattr(overlay_bind, "donor_buffer_available", lambda: True)
    with pytest.raises(ValueError):
        overlay_bind.register_donor_buffer(0, 4096)
    with pytest.raises(ValueError):
        overlay_bind.register_donor_buffer(0x1000, 0)


def test_register_calls_native_ext_when_enabled_and_available(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_OVERLAY_BIND", "1")
    monkeypatch.setattr(overlay_bind, "donor_buffer_available", lambda: True)

    calls = {}

    class _FakeModule:
        @staticmethod
        def register_donor_buffer(ptr, size):
            calls["ptr"] = ptr
            calls["size"] = size
            return 7

    import sys

    monkeypatch.setitem(sys.modules, "runtime.kv._kv_native", _FakeModule)

    donor_id = overlay_bind.register_donor_buffer(0x1000, 4096)
    assert donor_id == 7
    assert calls == {"ptr": 0x1000, "size": 4096}


def test_unregister_noop_without_native_ext(monkeypatch):
    monkeypatch.setattr(overlay_bind, "donor_buffer_available", lambda: False)
    # WHY assert-no-raise: unregister must be a safe no-op when the native ext
    # is not built — teardown paths must never fail because overlay bind was
    # never usable in this build.
    overlay_bind.unregister_donor_buffer(1)


def test_unregister_calls_native_ext_when_available(monkeypatch):
    monkeypatch.setattr(overlay_bind, "donor_buffer_available", lambda: True)

    calls = {}

    class _FakeModule:
        @staticmethod
        def unregister_donor_buffer(donor_id):
            calls["donor_id"] = donor_id

    import sys

    monkeypatch.setitem(sys.modules, "runtime.kv._kv_native", _FakeModule)

    overlay_bind.unregister_donor_buffer(5)
    assert calls == {"donor_id": 5}


def test_donor_buffer_status_none_without_native_ext(monkeypatch):
    monkeypatch.setattr(overlay_bind, "donor_buffer_available", lambda: False)
    assert overlay_bind.donor_buffer_status(1) is None


def test_donor_buffer_status_shape_when_available(monkeypatch):
    monkeypatch.setattr(overlay_bind, "donor_buffer_available", lambda: True)

    class _FakeModule:
        @staticmethod
        def donor_buffer_status(donor_id):
            return {"bound": True, "bytes_used": 1024}

    import sys

    monkeypatch.setitem(sys.modules, "runtime.kv._kv_native", _FakeModule)

    status = overlay_bind.donor_buffer_status(3)
    assert status == {"bound": True, "bytes_used": 1024}


def test_donor_buffer_status_returns_none_on_unknown_id(monkeypatch):
    monkeypatch.setattr(overlay_bind, "donor_buffer_available", lambda: True)

    class _FakeModule:
        @staticmethod
        def donor_buffer_status(donor_id):
            raise KeyError(donor_id)

    import sys

    monkeypatch.setitem(sys.modules, "runtime.kv._kv_native", _FakeModule)

    assert overlay_bind.donor_buffer_status(999) is None


def test_page_bind_health_includes_overlay_bind_fields_disabled(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_KV_OVERLAY_BIND", raising=False)
    from runtime.kv.page_bind import page_bind_health

    h = page_bind_health(native_ext_available=False)
    assert h["overlay_bind_enabled"] is False
    assert h["overlay_bind_auto"] is False
    assert h["overlay_bind_bound"] is False
    assert h["overlay_bind_bytes"] is None
    assert h["overlay_page_catalog"] is None


def test_overlay_bind_auto_follows_overlay_and_kill_switch(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_KV_OVERLAY_BIND", raising=False)
    monkeypatch.delenv("ZEROLLAMA_KV_OVERLAY_AUTO", raising=False)
    assert overlay_bind.overlay_bind_auto_enabled() is False

    monkeypatch.setenv("ZEROLLAMA_KV_OVERLAY_BIND", "1")
    assert overlay_bind.overlay_bind_auto_enabled() is True

    monkeypatch.setenv("ZEROLLAMA_KV_OVERLAY_AUTO", "0")
    assert overlay_bind.overlay_bind_auto_enabled() is False


def test_estimate_overlay_donor_bytes_env_wins(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_OVERLAY_DONOR_BYTES", "12345")
    size, source = overlay_bind.estimate_overlay_donor_bytes(
        None, num_ctx=4096, n_seq_max=2
    )
    assert source == "env"
    assert size == overlay_bind.page_align_bytes(12345)
    assert size % 4096 == 0


def test_estimate_overlay_donor_bytes_pads_estimate(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_KV_OVERLAY_DONOR_BYTES", raising=False)
    monkeypatch.setattr(
        "runtime.gguf_estimate.estimate_kv_cache_bytes",
        lambda *a, **k: 1_000_000,
    )
    monkeypatch.setattr(
        "runtime.gguf_estimate.gguf_arch_hints",
        lambda p: object(),
    )
    from pathlib import Path

    size, source = overlay_bind.estimate_overlay_donor_bytes(
        Path("/tmp/fake.gguf"), num_ctx=4096, n_seq_max=2
    )
    assert source == "estimate"
    # 1e6 * 2 streams * 2 pad + 32MiB
    assert size >= (1_000_000 * 2 * 2) + (32 << 20)
    assert size % 4096 == 0


def test_prepare_auto_donor_skips_when_auto_off(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_OVERLAY_BIND", "1")
    monkeypatch.setenv("ZEROLLAMA_KV_OVERLAY_AUTO", "0")
    assert overlay_bind.prepare_auto_donor(None, num_ctx=512) is None


def test_prepare_auto_donor_registers_aligned_buffer(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_OVERLAY_BIND", "1")
    monkeypatch.delenv("ZEROLLAMA_KV_OVERLAY_AUTO", raising=False)
    monkeypatch.setenv("ZEROLLAMA_KV_OVERLAY_DONOR_BYTES", "8192")
    monkeypatch.setattr(overlay_bind, "donor_buffer_available", lambda: True)

    calls = {}

    class _FakeModule:
        @staticmethod
        def register_donor_buffer(ptr, size):
            calls["ptr"] = ptr
            calls["size"] = size
            return 42

    import sys

    monkeypatch.setitem(sys.modules, "runtime.kv._kv_native", _FakeModule)

    handle = overlay_bind.prepare_auto_donor(None, num_ctx=512, n_seq_max=2)
    assert handle is not None
    assert handle.donor_id == 42
    assert handle.source == "env"
    assert handle.size == 8192
    assert handle.ptr % 4096 == 0
    assert calls["size"] == 8192
    assert calls["ptr"] == handle.ptr


def test_page_bind_health_includes_overlay_bind_auto(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_OVERLAY_BIND", "1")
    monkeypatch.delenv("ZEROLLAMA_KV_OVERLAY_AUTO", raising=False)
    monkeypatch.setattr("runtime.kv.page_bind._native_page_bind_available", lambda: False)
    from runtime.kv.page_bind import page_bind_health

    h = page_bind_health(native_ext_available=False)
    assert h["overlay_bind_enabled"] is True
    assert h["overlay_bind_auto"] is True


def test_page_bind_health_overlay_bind_bound_when_donor_consumed(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_KV_OVERLAY_BIND", "1")
    # WHY patch runtime.kv.overlay_bind (not page_bind): page_bind_health does a
    # local `from runtime.kv.overlay_bind import donor_buffer_status` inside its
    # own body on every call, so the name must be patched at its source module.
    monkeypatch.setattr(
        overlay_bind,
        "donor_buffer_status",
        lambda donor_id: {"bound": True, "bytes_used": 2048},
    )
    monkeypatch.setattr("runtime.kv.page_bind._native_page_bind_available", lambda: False)
    from runtime.kv.page_bind import page_bind_health

    h = page_bind_health(native_ext_available=False, overlay_bind_donor_id=1)
    assert h["overlay_bind_enabled"] is True
    # WHY still False here: native_ext_available=False takes the
    # "not_implemented" branch, which always reports overlay_bind_bound=False
    # regardless of donor status — bound state is only meaningful once the
    # native ext + tensor bind path is actually available.
    assert h["overlay_bind_bound"] is False
