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


@pytest.fixture
def cfg_root():
    return Path("/tmp")
