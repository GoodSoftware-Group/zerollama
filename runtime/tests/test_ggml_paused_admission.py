import pytest

from runtime.go_coordination import cross_queue_depth, update_go_coordination
from runtime.gpu.admission import (
    AdmissionRejected,
    check_ggml_paused_admission,
    check_runtime_backlog_admission,
    ggml_paused_blocks_admission_now,
    runtime_backlog_blocks_admission_now,
)
from runtime.gpu.priority import InferencePriority


def test_ggml_paused_when_mirror_set():
    update_go_coordination({"ggml_loads_paused": False})
    assert not ggml_paused_blocks_admission_now()
    update_go_coordination({"ggml_loads_paused": True})
    assert ggml_paused_blocks_admission_now()


def test_low_rejected_when_ggml_paused():
    update_go_coordination({"ggml_loads_paused": True})
    with pytest.raises(AdmissionRejected, match="ggml loads paused"):
        check_ggml_paused_admission(priority=InferencePriority.LOW)


def test_high_bypasses_ggml_paused():
    update_go_coordination({"ggml_loads_paused": True})
    check_ggml_paused_admission(priority=InferencePriority.HIGH)


def test_runtime_backlog():
    assert not runtime_backlog_blocks_admission_now(waiting=1, running=2)
    assert runtime_backlog_blocks_admission_now(waiting=2, running=2)


def test_runtime_backlog_rejects_low():
    update_go_coordination({})
    with pytest.raises(AdmissionRejected, match="runtime queue backlog"):
        check_runtime_backlog_admission(
            waiting=2, running=2, priority=InferencePriority.LOW
        )


def test_runtime_backlog_allows_normal():
    update_go_coordination({})
    check_runtime_backlog_admission(
        waiting=2, running=2, priority=InferencePriority.NORMAL
    )


def test_cross_queue_depth_snapshot():
    update_go_coordination(
        {
            "defer_waiting": 1,
            "sched_pending": 2,
            "sched_active": 0,
            "ggml_loads_paused": True,
            "runtime_waiting": 3,
            "runtime_running": 1,
        }
    )
    depth = cross_queue_depth(runtime_waiting=5, runtime_running=2)
    assert depth["runtime_backlog"] == 7
    assert depth["go_defer_waiting"] == 1
    assert depth["go_ggml_backlog"] == 2
    assert depth["ggml_loads_paused"] is True
    assert depth["go_runtime_mirror"] == 4
    assert depth["go_mirror_fresh"] is True


def test_cross_queue_depth_stale_mirror_note():
    import runtime.go_coordination as gc

    update_go_coordination({"defer_waiting": 9})
    with gc._lock:
        gc._updated_at = 0.0
    depth = cross_queue_depth(runtime_waiting=0, runtime_running=0)
    assert depth["go_mirror_fresh"] is False
    assert depth["go_defer_waiting"] == 0
    assert "pressure_note" in depth
