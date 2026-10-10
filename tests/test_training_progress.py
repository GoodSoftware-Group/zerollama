"""Unit tests for T3 training progress metrics helpers."""

from __future__ import annotations

import time

from training_progress import (
    StepMetricsTracker,
    build_progress_metrics,
    estimate_tokens_per_sec,
    format_metrics_message,
    mean_tokens_per_sample,
    training_vram_snapshot,
)


def test_estimate_tokens_per_sec_basic():
    tps = estimate_tokens_per_sec(
        steps_delta=2,
        elapsed_sec=1.0,
        samples_per_step=4.0,
        mean_tokens_per_sample=128.0,
    )
    assert tps == 2 * 4 * 128


def test_estimate_tokens_per_sec_rejects_bad_inputs():
    assert (
        estimate_tokens_per_sec(
            steps_delta=0,
            elapsed_sec=1.0,
            samples_per_step=4.0,
            mean_tokens_per_sample=128.0,
        )
        is None
    )
    assert (
        estimate_tokens_per_sec(
            steps_delta=1,
            elapsed_sec=0.0,
            samples_per_step=4.0,
            mean_tokens_per_sample=128.0,
        )
        is None
    )


def test_build_progress_metrics_and_message():
    m = build_progress_metrics(
        step=10,
        total_steps=100,
        loss=1.23456,
        tokens_per_sec=512.0,
        learning_rate=2e-4,
        vram={"allocated_gib": 3.5},
    )
    assert m["step"] == 10
    assert m["loss"] == 1.23456
    assert m["tokens_per_sec"] == 512.0
    msg = format_metrics_message(m)
    assert "step 10/100" in msg
    assert "loss=1.2346" in msg
    assert "512 tok/s" in msg
    assert "vram=3.50GiB" in msg


def test_mean_tokens_per_sample():
    assert mean_tokens_per_sample({"input_ids": [[1, 2], [1, 2, 3, 4]]}, fallback=64) == 3.0
    assert mean_tokens_per_sample({"input_ids": []}, fallback=64) == 64.0
    assert mean_tokens_per_sample({}, fallback=64) == 64.0


def test_vram_snapshot_without_cuda():
    # No CUDA / no torch in bare CI — empty dict, never raises.
    snap = training_vram_snapshot(torch_mod=None)
    assert isinstance(snap, dict)


def test_step_metrics_tracker_tps():
    tracker = StepMetricsTracker(
        total_steps=20,
        samples_per_step=2.0,
        mean_tokens_per_sample=100.0,
    )
    first = tracker.on_step(step=1, loss=2.0)
    assert first["step"] == 1
    assert first["loss"] == 2.0
    time.sleep(0.05)
    second = tracker.on_step(step=3, loss=1.5)
    assert second["step"] == 3
    assert second["loss"] == 1.5
    assert second.get("tokens_per_sec") is not None
    assert second["tokens_per_sec"] > 0


def test_job_to_dict_includes_metrics():
    from training import Job, JobStatus

    job = Job(id="j1", cmd="train", data={})
    job.status = JobStatus.RUNNING
    job.progress = 42.0
    job.progress_message = "step 3/10 loss=1.5"
    job.progress_metrics = {"step": 3, "total_steps": 10, "loss": 1.5}
    d = job.to_dict()
    assert d["metrics"]["loss"] == 1.5
    assert d["progress"] == 42.0
