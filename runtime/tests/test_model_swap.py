"""Model swap gate: drain in-flight work before changing loaded GGUF."""

import threading
import time
from pathlib import Path

from runtime.gpu.model_swap import ModelSwapGate


def test_swap_waits_for_inflight_same_model():
    gate = ModelSwapGate()
    path_a = Path("/tmp/model-a.gguf")
    path_b = Path("/tmp/model-b.gguf")
    order: list[str] = []
    started = threading.Event()

    def run_a() -> None:
        with gate.hold(path_a):
            order.append("a_start")
            started.set()
            time.sleep(0.05)
            order.append("a_end")

    def run_b() -> None:
        with gate.hold(path_b):
            order.append("b")

    t1 = threading.Thread(target=run_a)
    t2 = threading.Thread(target=run_b)
    t1.start()
    assert started.wait(timeout=1.0)
    t2.start()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)
    assert order == ["a_start", "a_end", "b"]


def test_reset_allows_immediate_swap():
    gate = ModelSwapGate()
    path_a = Path("/tmp/reset-a.gguf")
    path_b = Path("/tmp/reset-b.gguf")
    with gate.hold(path_a):
        pass
    gate.reset()
    with gate.hold(path_b):
        assert gate._loaded == str(path_b.resolve())
