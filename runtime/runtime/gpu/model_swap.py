"""Serialize GGUF loads: finish in-flight and queued work for the active model before swap."""

from __future__ import annotations

import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


def _model_key(gguf: Path | None) -> str:
    if gguf is None:
        return "__default__"
    return str(gguf.resolve())


@dataclass
class _ModelWaiters:
    inflight: int = 0
    waiting: int = 0


@dataclass
class ModelSwapGate:
    """Blocks model swaps until no in-flight or queued acquires remain on the loaded model."""

    _lock: threading.RLock = field(default_factory=threading.RLock)
    _cond: threading.Condition = field(init=False)
    _loaded: str | None = None
    _models: dict[str, _ModelWaiters] = field(
        default_factory=lambda: defaultdict(_ModelWaiters)
    )

    def __post_init__(self) -> None:
        self._cond = threading.Condition(self._lock)

    def reset(self) -> None:
        """Clear loaded state after an external unload (training handoff)."""
        with self._lock:
            self._loaded = None
            self._models.clear()
            self._cond.notify_all()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "loaded_gguf": self._loaded,
                "models": {
                    k: {"inflight": v.inflight, "waiting": v.waiting}
                    for k, v in self._models.items()
                },
            }

    def _can_acquire(self, key: str) -> bool:
        if self._loaded is None or self._loaded == key:
            return True
        active = self._models[self._loaded]
        return active.inflight == 0 and active.waiting == 0

    @contextmanager
    def hold(self, gguf: Path | None) -> Iterator[None]:
        key = _model_key(gguf)
        with self._lock:
            st = self._models[key]
            st.waiting += 1
            while not self._can_acquire(key):
                self._cond.wait()
            st.waiting -= 1
            st.inflight += 1
            self._loaded = key
        try:
            yield
        finally:
            with self._lock:
                st = self._models[key]
                st.inflight -= 1
                if st.inflight == 0 and st.waiting == 0:
                    del self._models[key]
                self._cond.notify_all()
