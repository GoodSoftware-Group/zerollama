from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_llama_backend_env(monkeypatch):
    """Operator shells often export Phase 14 / GPU smoke env; keep unit tests hermetic.

    Why: leftover LLAMA_MODEL + RUN_E2E_* turned unit pytest into GPU integration and
    could abort the suite after a wheel heap crash; sched_test.go does the same for Go.
    """
    for key in (
        "ZEROLLAMA_RUNTIME_LLAMA_BACKEND",
        "RUN_E2E_INPROCESS",
        "RUN_E2E_DECODE_LOOP",
        "RUN_E2E_LLAMA_CPP_PYTHON",
        "RUN_E2E_GPU",
        "RUN_E2E_PHASE14",
        "LLAMA_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _reset_go_coordination_mirror():
    """Isolate tests that read Go queue metrics."""
    from runtime.go_coordination import update_go_coordination
    import runtime.gpu.inference_policy as inference_policy

    update_go_coordination({})
    inference_policy._backpressure_latched = False
    yield
    update_go_coordination({})
    inference_policy._backpressure_latched = False


@pytest.fixture(autouse=True)
def _reset_native_page_bind():
    """Clear C-extension page_bind global state before every test.

    WHY: SchedulerLoop.tick() registers page_bind entries in the native C ext's
    global hash table.  Without cleanup, tests that call tick() without completing
    the request leave stale entries that break subsequent tests counting active_binds.
    We bulk-clear slots 0–63 which covers all test workloads.
    """
    try:
        from runtime.kv._kv_native import page_bind_clear, page_bind_stats

        def _clear_all():
            if page_bind_stats().get("active_binds", 0) > 0:
                for slot in range(64):
                    try:
                        page_bind_clear(slot)
                    except Exception:
                        pass

        _clear_all()
        yield
        _clear_all()
    except ImportError:
        yield


_VRAM_YAML_ENV_KEYS = (
    "ZEROLLAMA_RUNTIME_TRAINING_VRAM_RESERVE",
    "ZEROLLAMA_RUNTIME_VRAM_MIN_FREE",
    "ZEROLLAMA_RUNTIME_VRAM_ESTIMATE_FACTOR_AUTOTUNE",
    "ZEROLLAMA_RUNTIME_VRAM_PROBE_CALIBRATE",
    "ZEROLLAMA_RUNTIME_VRAM_CLAMP_NUM_CTX",
)


@pytest.fixture(autouse=True)
def _reset_vram_yaml_defaults():
    """Restore vram_yaml_defaults module state between tests.

    WHY: apply_vram_defaults_from_config sets _APPLIED/_APPLY_RESULT globals and may
    write several ZEROLLAMA_RUNTIME_VRAM_* env vars that leak into tests which
    rely on their default values (e.g. min_free=1GiB, training_reserve=2GiB).
    """
    import os

    saved = {k: os.environ.get(k) for k in _VRAM_YAML_ENV_KEYS}
    yield
    try:
        import runtime.vram_yaml_defaults as mod

        # Only reset if the module was actually applied during this test to
        # avoid disrupting auto-apply that happened before the suite started.
        if mod._APPLIED:
            mod._APPLIED = False
            mod._APPLY_RESULT = None
    except Exception:
        pass
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def cfg_root():
    return Path("/tmp")
