"""Bootstrap for embedded CPython (eval'd into __main__ by training_shim.c).

This is not an installable package. It exists to:
  - Replace training.STATE with BridgeState (OOM hooks into Go via ollama_training_native).
  - Start training.job_processor on a daemon thread.
  - Expose ack_vram_headroom and _training_shim_api for the C shim to call.

Why BridgeState keeps _pending_oom_event: Go may ack VRAM relief immediately after fire_oom.
The wait step must use the same threading.Event created *before* notify, or acks are lost
and Python blocks until timeout. See docs/gpu-training.md (OOM synchronization).
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Optional


_oom_acks: Dict[str, threading.Event] = {}
_oom_lock = threading.Lock()
_processor_thread: Optional[threading.Thread] = None


def ack_vram_headroom(job_id: str) -> None:
    with _oom_lock:
        ev = _oom_acks.pop(job_id, None)
    if ev is not None:
        ev.set()


def _register_oom_wait(job_id: str) -> Optional[threading.Event]:
    if not job_id:
        return None
    with _oom_lock:
        ev = _oom_acks.get(job_id)
        if ev is None:
            ev = threading.Event()
            _oom_acks[job_id] = ev
        return ev


def init_ollama_training() -> None:
    global _processor_thread
    import ollama_training_native
    import training

    def emit(job_id: str, _kind: str, payload: dict) -> None:
        # Embedded mode: poll job_status for progress; gRPC-style push to Go is not implemented.
        _ = (job_id, _kind, payload)

    class BridgeState(training.WorkerState):
        def __init__(self) -> None:
            super().__init__()
            self._emit = emit
            self._pending_oom_event: Optional[threading.Event] = None

        def broadcast_to_job_owner(self, job, message: Dict[str, Any]) -> None:
            super().broadcast_to_job_owner(job, message)
            self._emit(job.id, message.get("type", "event"), message)

        def _prepare_vram_relief_wait(self) -> None:
            # Register the event BEFORE notifying Go so Go's ack cannot be lost.
            jid = self.current_job_id
            self._pending_oom_event = _register_oom_wait(jid) if jid else None
            super()._prepare_vram_relief_wait()

        def _notify_cuda_oom(self, exc: BaseException, phase: str = "") -> None:
            jid = self.current_job_id or ""
            msg = f"{phase}: {exc}" if phase else str(exc)
            ollama_training_native.fire_oom(jid, msg)
            super()._notify_cuda_oom(exc, phase)

        def _wait_vram_relief_after_oom(self) -> None:
            # Reuse the event from _prepare — do NOT re-register, which would
            # create a fresh unset event and block for the full timeout even if
            # Go already sent the ack while we were in _notify_cuda_oom.
            ev = self._pending_oom_event
            self._pending_oom_event = None
            if ev is not None:
                ev.wait(timeout=120.0)
            super()._wait_vram_relief_after_oom()

    training.STATE = BridgeState()
    _processor_thread = threading.Thread(target=training.job_processor, daemon=True)
    _processor_thread.start()


def shutdown_ollama_training() -> None:
    import training

    st = training.STATE
    st.running = False
    # If the job thread is blocked in _wait_vram_relief_after_oom, wake it so join can finish.
    pending = getattr(st, "_pending_oom_event", None)
    if pending is not None:
        pending.set()
    if _processor_thread is not None:
        _processor_thread.join(timeout=30.0)


def _job_to_dict(job) -> dict:
    """Convert a Job object or a Job.to_dict() plain dict to the Go-facing wire shape."""
    if isinstance(job, dict):
        g = job.get
        st = g("status") or ""
        cmd = g("cmd") or ""
        jid = g("id") or ""
        progress = float(g("progress") or 0)
        pmsg = g("progress_message") or ""
        result = g("result")
        err = g("error") or ""
        sub = g("submitted_at") or ""
        start = g("started_at") or ""
        done = g("completed_at") or ""
        video_model = g("video_model") or ""
        video_size = g("video_size") or ""
    else:
        st = job.status.value if hasattr(job.status, "value") else str(job.status)
        cmd = job.cmd
        jid = job.id
        progress = float(job.progress)
        pmsg = job.progress_message or ""
        result = job.result
        err = job.error or ""
        sub = job.submitted_at or ""
        start = job.started_at or ""
        done = job.completed_at or ""
        video_model = job.data.get("video_model", "") if cmd == "run_script" else ""
        video_size = job.data.get("video_size", "") if cmd == "run_script" else ""
    out = {
        "jobId": jid,
        "kind": "JOB_KIND_TRAIN" if cmd == "train" else "JOB_KIND_RUN_SCRIPT",
        "status": st,
        "progress": progress,
        "progressMessage": pmsg,
        "resultJson": json.dumps(result) if result is not None else "",
        "error": err,
        "submittedAt": sub,
        "startedAt": start,
        "completedAt": done,
    }
    if video_model:
        out["videoModel"] = video_model
    if video_size:
        out["videoSize"] = video_size
    return out


class _TrainingShimAPI:
    def health(self) -> dict:
        import torch
        import training

        extras = {
            "device": training.STATE.device,
            "cuda_available": torch.cuda.is_available(),
            "model_loaded": training.STATE.current_model_name,
            "training_active": training.STATE.training_active,
            "queue": training.JOB_QUEUE.get_queue_status(),
        }
        return {"status": "ok", "extrasJson": json.dumps(extras)}  # camelCase matches proto JSON

    def submit_job(self, kind: str, payload_json: str) -> dict:
        import training

        cmd = "train" if kind != "run_script" else "run_script"
        try:
            data = json.loads(payload_json) if payload_json else {}
        except json.JSONDecodeError as e:
            return {"error": f"payload_json: {e}"}
        job_id = training.JOB_QUEUE.submit(cmd, data, None)
        return {"jobId": job_id}

    def job_status(self, job_id: str) -> dict:
        import training

        job = training.JOB_QUEUE.get_job(job_id)
        if not job:
            return {"error": "not_found", "job": None}
        return {"job": _job_to_dict(job)}

    def list_jobs(self) -> dict:
        import training

        # list_jobs returns Job.to_dict() dicts already; pass them directly to
        # _job_to_dict to avoid a second per-job lock acquisition.
        raw = training.JOB_QUEUE.list_jobs(50)
        return {"jobs": [_job_to_dict(d) for d in raw]}

    def unload(self) -> dict:
        import training

        training.STATE.unload_model(reason="api_unload")
        return {"status": "ok"}


_training_shim_api = _TrainingShimAPI()


def _training_cancel_job(job_id: str) -> bool:
    import training

    return training.JOB_QUEUE.cancel_job(job_id)
