import time

import pytest

from runtime.go_coordination import go_coordination_is_fresh, update_go_coordination
from runtime.gpu.admission import (
    AdmissionRejected,
    check_training_defer_admission,
    defer_backlog_blocks_admission_now,
    defer_backlog_policy_active,
)
from runtime.gpu.priority import InferencePriority


def test_defer_backlog_when_waiting():
    update_go_coordination({"defer_waiting": 0})
    assert defer_backlog_policy_active()
    assert not defer_backlog_blocks_admission_now()
    update_go_coordination({"defer_waiting": 2})
    assert defer_backlog_blocks_admission_now()


def test_stale_mirror_fails_open(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_GO_COORDINATION_TTL_S", "0.01")
    update_go_coordination({"defer_waiting": 5})
    assert defer_backlog_blocks_admission_now()
    time.sleep(0.02)
    assert not go_coordination_is_fresh()
    assert not defer_backlog_blocks_admission_now()
    check_training_defer_admission(priority=InferencePriority.LOW)


def test_low_rejected_when_defer_waiting():
    update_go_coordination({"defer_waiting": 1})
    with pytest.raises(AdmissionRejected, match="defer backlog"):
        check_training_defer_admission(priority=InferencePriority.LOW)


def test_high_bypasses_defer_backlog():
    update_go_coordination({"defer_waiting": 5})
    check_training_defer_admission(priority=InferencePriority.HIGH)


def test_normal_allowed_while_defer_waiting():
    update_go_coordination({"defer_waiting": 1})
    check_training_defer_admission(priority=InferencePriority.NORMAL)
