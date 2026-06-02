import time

from runtime.go_coordination import (
    go_coordination_health,
    go_coordination_is_fresh,
    go_defer_waiting,
    go_training_gpu_blocked,
    update_go_coordination,
)


def test_go_coordination_roundtrip():
    update_go_coordination({"defer_waiting": 2, "training_gpu_blocked": True})
    snap = go_coordination_health()
    assert snap["defer_waiting"] == 2
    assert snap["training_gpu_blocked"] is True
    assert snap["coordination"]["fresh"] is True
    assert go_defer_waiting() == 2
    assert go_training_gpu_blocked() is True
    update_go_coordination({})
    snap = go_coordination_health()
    assert snap.get("defer_waiting") is None
    assert snap["coordination"]["stale"] is True
    assert go_defer_waiting() == 0
    assert go_training_gpu_blocked() is False


def test_stale_after_ttl(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_GO_COORDINATION_TTL_S", "0.01")
    update_go_coordination({"defer_waiting": 3})
    assert go_coordination_is_fresh()
    time.sleep(0.02)
    assert not go_coordination_is_fresh()
    assert go_defer_waiting() == 0
