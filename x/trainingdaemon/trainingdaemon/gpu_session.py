"""Load repo-root training.py and expose a patched WorkerState for gRPC event streaming.

Why a BridgeState subclass instead of editing training.py in dozens of places:
  - Keeps training.py usable as a script/library while the daemon adds emit/register_oom_wait hooks.

Why send_progress only calls super (no direct _emit):
  - super().send_progress already triggers broadcast_to_job_owner, which we override to emit once;
    emitting again would duplicate every progress event on the wire.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def repo_root() -> Path:
    # .../ollama/x/trainingdaemon/trainingdaemon/gpu_session.py -> parents[3] == ollama
    return Path(__file__).resolve().parents[3]


def ensure_training_path() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def import_training():
    ensure_training_path()
    import training  # noqa: PLC0415 — runtime path setup

    return training


def build_state(
    training_mod: Any,
    emit: Callable[[str, str, Dict[str, Any]], None],
    register_oom_wait: Optional[Callable[[str], Optional[threading.Event]]] = None,
) -> Any:
    """Return a WorkerState subclass that mirrors legacy behavior and notifies gRPC."""

    reg = register_oom_wait

    class BridgeState(training_mod.WorkerState):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self._emit = emit
            self._register_oom_wait = reg

        def send_progress(self, progress: float, message: str = "") -> None:  # noqa: ANN001
            # Call super (updates JOB_QUEUE, sends legacy socket). broadcast_to_job_owner
            # is overridden below and emits to gRPC, so we must NOT emit here too.
            super().send_progress(progress, message)

        def broadcast_to_job_owner(self, job, message: Dict[str, Any]) -> None:  # noqa: ANN001
            super().broadcast_to_job_owner(job, message)
            self._emit(job.id, message.get("type", "event"), message)

        def _prepare_vram_relief_wait(self) -> None:
            jid = self.current_job_id
            if jid and self._register_oom_wait:
                self._register_oom_wait(jid)
            super()._prepare_vram_relief_wait()

        def _notify_cuda_oom(self, exc: BaseException, phase: str = "") -> None:  # noqa: ANN001
            jid = self.current_job_id
            if jid:
                msg = f"{phase}: {exc}" if phase else str(exc)
                self._emit(jid, "oom", {"message": msg})
            super()._notify_cuda_oom(exc, phase)

        def _wait_vram_relief_after_oom(self) -> None:
            jid = self.current_job_id
            if jid and self._register_oom_wait:
                ev = self._register_oom_wait(jid)
                if ev is not None:
                    ev.wait(timeout=120.0)
            super()._wait_vram_relief_after_oom()

    return BridgeState()


def start_job_processor(training_mod: Any) -> None:
    t = threading.Thread(target=training_mod.job_processor, daemon=True)
    t.start()
