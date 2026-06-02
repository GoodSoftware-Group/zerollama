import importlib

import pytest

import runtime.gpu.inference_policy as policy
from runtime.gpu.inference_policy import CROSS_QUEUE_PRESSURE_ON
from runtime.go_coordination import update_go_coordination
from runtime.gpu.admission import (
    AdmissionRejected,
    check_inference_first_admission,
)
from runtime.gpu.priority import InferencePriority


def _reset_latch():
    policy._backpressure_latched = False


def test_metrics_drive_combined_pressure(monkeypatch):
    _reset_latch()
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_INFERENCE_POLICY", raising=False)
    update_go_coordination(
        {"defer_waiting": 2, "sched_pending": 2, "sched_active": 1}
    )
    snap = policy.backpressure_snapshot(runtime_waiting=1, runtime_running=1)
    assert snap["cross_queue_pressure"] >= CROSS_QUEUE_PRESSURE_ON
    assert snap["cross_queue_pressure_latched"] is True


def test_high_bypasses_backpressure(monkeypatch):
    _reset_latch()
    update_go_coordination({"ggml_loads_paused": True})
    check_inference_first_admission(
        waiting=0, running=0, priority=InferencePriority.HIGH
    )


def test_low_blocked_by_metrics(monkeypatch):
    _reset_latch()
    update_go_coordination({"ggml_loads_paused": True})
    with pytest.raises(AdmissionRejected, match="ggml loads paused"):
        check_inference_first_admission(
            waiting=0, running=0, priority=InferencePriority.LOW
        )


def test_policy_off(monkeypatch):
    _reset_latch()
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_INFERENCE_POLICY", "off")
    update_go_coordination({"ggml_loads_paused": True})
    check_inference_first_admission(
        waiting=0, running=0, priority=InferencePriority.LOW
    )


def test_backpressure_threshold_env_override(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_GGML_SCHED_BACKLOG_MIN", "9")
    importlib.reload(policy)
    try:
        assert policy.GGML_SCHED_BACKLOG_MIN == 9
        snap = policy.backpressure_snapshot(runtime_waiting=0, runtime_running=0)
        assert snap["thresholds"]["ggml_sched_backlog_min"] == 9
    finally:
        monkeypatch.delenv("ZEROLLAMA_RUNTIME_GGML_SCHED_BACKLOG_MIN", raising=False)
        importlib.reload(policy)
