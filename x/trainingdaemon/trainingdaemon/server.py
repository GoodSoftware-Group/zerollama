"""gRPC TrainingDaemon service (Unix domain socket).

Why UDS (not TCP between Go and Python):
  - No port allocation conflicts; identity is a filesystem path; same-machine IPC only.

Why replace training.STATE with BridgeState at startup:
  - Emit progress/OOM into gRPC without rewriting every call site in training.py; subclass
    hooks keep the legacy module usable as a library.
"""

from __future__ import annotations

import json
import threading
import queue
from concurrent import futures
from typing import Dict, List, Optional, Tuple

import grpc

from . import gpu_session
from . import training_pb2
from . import training_pb2_grpc


def _job_kind_to_cmd(kind: int) -> str:
    if kind == 1:  # JOB_KIND_TRAIN
        return "train"
    if kind == 2:  # JOB_KIND_RUN_SCRIPT
        return "run_script"
    return "train"


def _job_to_pb(job) -> training_pb2.JobInfo:  # noqa: ANN001
    st = job.status.value if hasattr(job.status, "value") else str(job.status)
    res_json = ""
    if job.result is not None:
        res_json = json.dumps(job.result)
    return training_pb2.JobInfo(
        job_id=job.id,
        kind=1 if job.cmd == "train" else 2,
        status=st,
        progress=float(job.progress),
        progress_message=job.progress_message or "",
        result_json=res_json,
        error=job.error or "",
        submitted_at=job.submitted_at or "",
        started_at=job.started_at or "",
        completed_at=job.completed_at or "",
    )


class TrainingServicer(training_pb2_grpc.TrainingDaemonServicer):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._streams: List[Tuple[Optional[str], queue.Queue]] = []
        self._oom_lock = threading.Lock()
        self._oom_acks: Dict[str, threading.Event] = {}
        self._server: Optional[grpc.Server] = None
        self._training = gpu_session.import_training()

        def emit(job_id: str, _kind: str, _payload: dict) -> None:
            self._push_event(job_id, _kind, _payload)

        def register_oom_wait(job_id: str) -> Optional[threading.Event]:
            if not job_id:
                return None
            with self._oom_lock:
                ev = self._oom_acks.get(job_id)
                if ev is None:
                    ev = threading.Event()
                    self._oom_acks[job_id] = ev
                return ev

        self._training.STATE = gpu_session.build_state(
            self._training, emit, register_oom_wait=register_oom_wait
        )
        gpu_session.start_job_processor(self._training)

    def _push_event(self, job_id: str, kind: str, payload: dict) -> None:
        ev = training_pb2.TrainingEvent(job_id=job_id)
        if kind == "progress" or payload.get("type") == "progress":
            ev.progress.CopyFrom(
                training_pb2.ProgressEvent(
                    pct=float(payload.get("progress", 0)),
                    message=str(payload.get("message", "")),
                )
            )
        elif payload.get("type") == "job_completed":
            r = payload.get("result")
            ev.completed.CopyFrom(
                training_pb2.JobCompletedEvent(result_json=json.dumps(r) if r is not None else "")
            )
        elif payload.get("type") == "job_failed":
            ev.failed.CopyFrom(training_pb2.JobFailedEvent(error=str(payload.get("error", ""))))
        elif kind == "oom":
            ev.oom.CopyFrom(training_pb2.OOMEvent(message=str(payload.get("message", ""))))
        else:
            # map generic progress-shaped
            if "progress" in payload:
                ev.progress.CopyFrom(
                    training_pb2.ProgressEvent(
                        pct=float(payload.get("progress", 0)),
                        message=str(payload.get("message", "")),
                    )
                )
            else:
                ev.progress.CopyFrom(training_pb2.ProgressEvent(pct=0, message=json.dumps(payload)))

        with self._lock:
            dead: List[Tuple[Optional[str], queue.Queue]] = []
            for filt, q in self._streams:
                if filt and filt != job_id:
                    continue
                try:
                    q.put_nowait(ev)
                except Exception:
                    dead.append((filt, q))
            for d in dead:
                if d in self._streams:
                    self._streams.remove(d)

    def Health(self, request: training_pb2.HealthRequest, context: grpc.ServicerContext) -> training_pb2.HealthResponse:
        t = self._training
        extras = {
            "device": t.STATE.device,
            "cuda_available": t.torch.cuda.is_available(),
            "model_loaded": t.STATE.current_model_name,
            "queue": t.JOB_QUEUE.get_queue_status(),
        }
        return training_pb2.HealthResponse(status="ok", extras_json=json.dumps(extras))

    def SubmitJob(self, request: training_pb2.SubmitJobRequest, context: grpc.ServicerContext) -> training_pb2.SubmitJobResponse:
        cmd = _job_kind_to_cmd(int(request.kind))
        try:
            data = json.loads(request.payload_json) if request.payload_json else {}
        except json.JSONDecodeError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"payload_json: {e}")
            raise
        job_id = self._training.JOB_QUEUE.submit(cmd, data, None)
        return training_pb2.SubmitJobResponse(job_id=job_id)

    def JobStatus(self, request: training_pb2.JobStatusRequest, context: grpc.ServicerContext) -> training_pb2.JobStatusResponse:
        job = self._training.JOB_QUEUE.get_job(request.job_id)
        if not job:
            context.abort(grpc.StatusCode.NOT_FOUND, "job not found")
            raise
        return training_pb2.JobStatusResponse(job=_job_to_pb(job))

    def CancelJob(self, request: training_pb2.CancelJobRequest, context: grpc.ServicerContext) -> training_pb2.CancelJobResponse:
        ok = self._training.JOB_QUEUE.cancel_job(request.job_id)
        return training_pb2.CancelJobResponse(cancelled=ok)

    def ListJobs(self, request: training_pb2.ListJobsRequest, context: grpc.ServicerContext) -> training_pb2.ListJobsResponse:
        raw = self._training.JOB_QUEUE.list_jobs(50)
        out: List[training_pb2.JobInfo] = []
        for d in raw:
            job = self._training.JOB_QUEUE.get_job(d["id"])
            if job:
                out.append(_job_to_pb(job))
        return training_pb2.ListJobsResponse(jobs=out)

    def StreamEvents(self, request: training_pb2.StreamEventsRequest, context: grpc.ServicerContext):
        filt: Optional[str] = request.job_id or None
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._streams.append((filt, q))
        try:
            while context.is_active():
                try:
                    ev = q.get(timeout=1.0)
                    yield ev
                except queue.Empty:
                    continue
        finally:
            with self._lock:
                try:
                    self._streams.remove((filt, q))
                except ValueError:
                    pass

    def Unload(self, request: training_pb2.UnloadRequest, context: grpc.ServicerContext) -> training_pb2.UnloadResponse:
        self._training.STATE.unload_model(reason="grpc_unload")
        return training_pb2.UnloadResponse(status="ok")

    def AckVRAMHeadroom(self, request: training_pb2.AckVRAMHeadroomRequest, context: grpc.ServicerContext) -> training_pb2.AckVRAMHeadroomResponse:
        with self._oom_lock:
            ev = self._oom_acks.pop(request.job_id, None)
        if ev is not None:
            ev.set()
        return training_pb2.AckVRAMHeadroomResponse()

    def Shutdown(self, request: training_pb2.ShutdownRequest, context: grpc.ServicerContext) -> training_pb2.ShutdownResponse:
        self._training.STATE.running = False
        # stop() ends wait_for_termination so the process can exit after Go closes the connection.
        if self._server is not None:
            # Stop accepting new RPCs; existing streaming calls get a grace period.
            self._server.stop(grace=5)
        return training_pb2.ShutdownResponse()


def serve_uds(uds_path: str) -> None:
    servicer = TrainingServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    training_pb2_grpc.add_TrainingDaemonServicer_to_server(servicer, server)
    listen = f"unix://{uds_path}"
    if not server.add_insecure_port(listen):
        raise RuntimeError(f"cannot bind gRPC {listen}")
    servicer._server = server
    server.start()
    server.wait_for_termination()
