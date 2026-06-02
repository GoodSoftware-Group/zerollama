import pytest

from runtime.go_coordination import update_go_coordination
from runtime.gpu.admission import (
    AdmissionRejected,
    check_ggml_backlog_admission,
    ggml_backlog_blocks_admission_now,
)
from runtime.gpu.priority import InferencePriority


def test_ggml_backlog_when_busy():
    update_go_coordination({"sched_pending": 0, "sched_active": 0})
    assert not ggml_backlog_blocks_admission_now()
    update_go_coordination({"sched_pending": 2, "sched_active": 0})
    assert ggml_backlog_blocks_admission_now()


def test_low_rejected_when_ggml_pending():
    update_go_coordination({"sched_pending": 1, "sched_active": 0})
    with pytest.raises(AdmissionRejected, match="ggml scheduler"):
        check_ggml_backlog_admission(priority=InferencePriority.LOW)


def test_high_bypasses_ggml_backlog():
    update_go_coordination({"sched_pending": 3, "sched_active": 1})
    check_ggml_backlog_admission(priority=InferencePriority.HIGH)


def test_loaded_runners_do_not_count_as_ggml_backlog():
    """Product policy: only pending/active sched work blocks batch, not keep-alive loads."""
    update_go_coordination({"sched_pending": 0, "sched_active": 0, "sched_loaded": 2})
    assert not ggml_backlog_blocks_admission_now()
