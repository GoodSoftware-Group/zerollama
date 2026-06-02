from runtime.gpu import admission as adm
from runtime.gpu.mutex import InferenceGpuCoordinator


def test_go_training_gpu_busy_applies_reserve():
    coord = InferenceGpuCoordinator()
    assert coord.training_reserve_active() is False
    coord.set_go_training_gpu_busy(True)
    assert coord.training_reserve_active() is True
    from runtime.gpu.admission import training_vram_reserve_bytes

    assert (
        training_vram_reserve_bytes(inference_paused=coord.training_reserve_active())
        == adm.TRAINING_VRAM_RESERVE_BYTES
    )
    coord.set_go_training_gpu_busy(False)
    assert training_vram_reserve_bytes(inference_paused=False) == 0


def test_policy_snapshot_gates_active_names():
    coord = InferenceGpuCoordinator()
    snap = coord.policy_snapshot(waiting=0, running=0)
    gates = snap["gates_active"]
    assert "low_would_wait" in gates
    assert "runtime_backlog_pressure" in gates
    assert "defer_would_block_low" in gates
    compat = snap["gates_active_compat"]
    assert compat["batch_backpressure"] == gates["low_would_wait"]
    assert compat["runtime_backlog"] == gates["runtime_backlog_pressure"]


def test_handoff_applies_reserve():
    coord = InferenceGpuCoordinator()
    coord.training_handoff()
    from runtime.gpu.admission import training_vram_reserve_bytes

    assert (
        training_vram_reserve_bytes(inference_paused=coord.training_reserve_active())
        == adm.TRAINING_VRAM_RESERVE_BYTES
    )
