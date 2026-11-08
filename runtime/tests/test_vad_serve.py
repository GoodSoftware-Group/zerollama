"""Neural VAD ear without onnxruntime or ONNX files."""

from __future__ import annotations

import base64
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import numpy as np

from runtime.asr.vad import Ear
from runtime.asr.vad_serve import _handler


class _Tensor:
    def __init__(self, name: str, shape):
        self.name = name
        self.shape = shape


class _FakeStock:
    """Stock Silero graph: input/state/sr → output/stateN."""

    def get_inputs(self):
        return [
            _Tensor("input", [1, 512]),
            _Tensor("state", [2, 1, 128]),
            _Tensor("sr", []),
        ]

    def get_outputs(self):
        return [_Tensor("output", [1, 1]), _Tensor("stateN", [2, 1, 128])]

    def run(self, output_names, feeds):
        inp = np.asarray(feeds["input"]).reshape(-1)
        # Speech when RMS is high.
        rms = float(np.sqrt(np.mean(inp * inp))) if inp.size else 0.0
        prob = 0.9 if rms > 0.05 else 0.05
        state_n = np.asarray(feeds["state"], dtype=np.float32) + 0.01
        return [np.array([[prob]], dtype=np.float32), state_n]


def _pcm16(samples: list[int]) -> bytes:
    return np.asarray(samples, dtype="<i2").tobytes()


def test_score_finds_last_speech_window():
    ear = Ear("unused.onnx", session=_FakeStock())
    # One quiet window (512 zeros) then one loud window.
    quiet = [0] * 512
    loud = [8000] * 512
    pcm = _pcm16(quiet + loud)
    last = ear.score("s1", pcm, 16000, 0.5)
    assert last == 1023, last
    # Reset clears recurrent state; a quiet-only packet is silence.
    ear.reset("s1")
    last = ear.score("s1", _pcm16(quiet), 16000, 0.5)
    assert last == -1, last


def test_sessions_are_isolated():
    ear = Ear("unused.onnx", session=_FakeStock())
    loud = _pcm16([8000] * 512)
    ear.score("a", loud, 16000, 0.5)
    ear.reset("a")
    # Session b never saw a's state.
    last = ear.score("b", _pcm16([0] * 512), 16000, 0.5)
    assert last == -1


def test_reject_other_rates():
    ear = Ear("unused.onnx", session=_FakeStock())
    try:
        ear.score("s", _pcm16([0] * 512), 24000, 0.5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_leftover_carries_across_packets():
    ear = Ear("unused.onnx", session=_FakeStock())
    # 256 quiet + 256 loud cannot fill a 512 window alone across the split.
    quiet = [0] * 256
    loud = [8000] * 256
    assert ear.score("s", _pcm16(quiet), 16000, 0.5) == -1
    # Leftover quiet + this loud = one full window that is half loud → speech.
    last = ear.score("s", _pcm16(loud), 16000, 0.5)
    assert last == 255, last
    ear.reset("s")
    # After reset, the same loud half-window alone is not scored.
    assert ear.score("s", _pcm16(loud), 16000, 0.5) == -1


def test_serve_score_and_reset():
    ear = Ear("unused.onnx", session=_FakeStock())
    ear.load()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(ear))
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        assert resp.status == 200
        assert json.loads(resp.read().decode()) == {"status": "ok"}

        pcm = base64.b64encode(_pcm16([8000] * 512)).decode()
        body = json.dumps(
            {"session": "live-1", "rate": 16000, "threshold": 0.5, "pcm": pcm}
        )
        conn.request(
            "POST",
            "/score",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        out = json.loads(resp.read().decode())
        assert resp.status == 200
        assert out["last"] == 511

        conn.request(
            "POST",
            "/reset",
            body=json.dumps({"session": "live-1"}),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 200
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
