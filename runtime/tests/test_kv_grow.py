"""In-place KV grow helpers (no llama-server restart)."""

from __future__ import annotations

from runtime.kv.kv_grow import try_grow_worker, try_shrink_worker


class _GrowWorker:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def grow_n_ctx(self, n_ctx: int) -> bool:
        self.calls.append(n_ctx)
        return True


class _FailWorker:
    def grow_n_ctx(self, n_ctx: int) -> bool:
        raise RuntimeError("nope")


def test_try_grow_worker_hook():
    w = _GrowWorker()
    assert try_grow_worker(w, 4096) is True
    assert w.calls == [4096]


def test_try_grow_worker_fail_closed():
    assert try_grow_worker(_FailWorker(), 4096) is False
    assert try_grow_worker(None, 4096) is False
    assert try_grow_worker(object(), 2048) is False


class _ShrinkWorker:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def shrink_n_ctx(self, n_ctx: int) -> bool:
        self.calls.append(n_ctx)
        return True


def test_try_shrink_worker_hook():
    w = _ShrinkWorker()
    assert try_shrink_worker(w, 2048) is True
    assert w.calls == [2048]
    assert try_shrink_worker(None, 2048) is False
