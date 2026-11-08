"""Kokoro serve without kokoro-onnx or ONNX files."""

from __future__ import annotations

import base64
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import numpy as np

from runtime.tts.kokoro import SAMPLE_RATE, Speaker, lang_for_voice, pcm16_wav, resolve_lang
from runtime.tts.kokoro_serve import _handler


class _FakeKokoro:
    def __init__(self):
        self.voices = {"af_heart": np.zeros(1), "bf_emma": np.zeros(1)}
        self.last_lang = ""
        self.last_speed = 1.0
        self.calls: list[str] = []

    def create(self, text, voice="af_heart", speed=1.0, lang="en-us"):
        if voice not in self.voices:
            raise ValueError(f"voice {voice!r} not found")
        self.last_lang = lang
        self.last_speed = speed
        self.calls.append(text)
        n = max(8, len(text) * 4)
        samples = np.full(n, 0.25, dtype=np.float32)
        return samples, SAMPLE_RATE

    async def create_stream(self, text, voice="af_heart", speed=1.0, lang="en-us"):
        if voice not in self.voices:
            raise ValueError(f"voice {voice!r} not found")
        self.last_lang = lang
        self.last_speed = speed
        self.calls.append(text)
        half = max(4, len(text) * 2)
        yield np.full(half, 0.2, dtype=np.float32), SAMPLE_RATE
        yield np.full(half, -0.2, dtype=np.float32), SAMPLE_RATE


def test_lang_for_voice():
    assert lang_for_voice("af_heart") == "en-us"
    assert lang_for_voice("bf_emma") == "en-gb"
    assert lang_for_voice("bm_lewis") == "en-gb"
    assert resolve_lang("af_heart", "en-GB") == "en-gb"
    assert resolve_lang("bf_emma", "en") == "en-us"
    assert resolve_lang("bf_emma", "") == "en-gb"
    for bad in ("zf_xiaobei", "jf_alpha", "ef_dora"):
        try:
            resolve_lang(bad, "")
            assert False, bad
        except ValueError:
            pass
    try:
        resolve_lang("af_heart", "zh")
        assert False, "zh"
    except ValueError:
        pass


def test_speak_returns_wav():
    sp = Speaker("unused.onnx", "unused.bin", kokoro=_FakeKokoro())
    rate, wav = sp.speak("hello", voice="af_heart", speed=1.0)
    assert rate == SAMPLE_RATE
    assert wav[:4] == b"RIFF"
    assert b"WAVE" in wav[:12]


def test_speak_stream_emits_chunks():
    sp = Speaker("unused.onnx", "unused.bin", kokoro=_FakeKokoro())
    chunks = []
    sp.speak_stream("hello world", voice="af_heart", emit=lambda r, p: chunks.append((r, p)))
    assert len(chunks) == 2
    assert all(r == SAMPLE_RATE and len(p) > 0 for r, p in chunks)


def test_speak_rejects_non_english():
    sp = Speaker("unused.onnx", "unused.bin", kokoro=_FakeKokoro())
    try:
        sp.speak("hi", voice="zf_xiaobei")
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        sp.speak("hi", voice="af_heart", lang="ja")
        assert False, "expected ValueError"
    except ValueError:
        pass
    rate, wav = sp.speak("hi", voice="af_heart", lang="en-GB")
    assert rate == SAMPLE_RATE
    assert wav[:4] == b"RIFF"
    assert sp.kokoro.last_lang == "en-gb"


def test_speak_emotion_and_phoneme_silence():
    import io
    import wave

    from runtime.tts.kokoro import apply_emotion, silence_pcm

    fake = _FakeKokoro()
    sp = Speaker("unused.onnx", "unused.bin", kokoro=fake)
    rate, wav = sp.speak(
        "Hi.",
        voice="af_heart",
        speed=1.0,
        emotion=" Excited ",
        phoneme_silence={".": 0.01},
    )
    assert rate == SAMPLE_RATE
    assert abs(fake.last_speed - 1.2) < 1e-9
    assert fake.calls == ["Hi"]
    with wave.open(io.BytesIO(wav), "rb") as w:
        assert w.getnframes() == 8 + 240
    try:
        sp.speak("hi", emotion="joyful")
        assert False, "unknown emotion"
    except ValueError:
        pass
    try:
        apply_emotion(1.8, "excited")
        assert False, "range"
    except ValueError:
        pass
    chunks = []
    sp.speak_stream(
        "A.B",
        voice="af_heart",
        phoneme_silence=[[".", "0.01"]],
        emit=lambda r, p: chunks.append(p),
    )
    assert len(chunks) == 5
    assert chunks[2] == silence_pcm(SAMPLE_RATE, 0.01)
    assert fake.calls[-2:] == ["A", "B"]


def test_speak_rejects_speed():
    sp = Speaker("unused.onnx", "unused.bin", kokoro=_FakeKokoro())
    try:
        sp.speak("hi", speed=3.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_serve_speak_and_stream():
    sp = Speaker("unused.onnx", "unused.bin", kokoro=_FakeKokoro())
    sp.load()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(sp, "af_heart"))
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        assert resp.status == 200
        assert json.loads(resp.read().decode()) == {"status": "ok", "rate": SAMPLE_RATE}

        body = json.dumps({"text": "hi", "voice": "af_heart", "speed": 1.0})
        conn.request(
            "POST",
            "/speak",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        out = json.loads(resp.read().decode())
        assert resp.status == 200
        raw = base64.b64decode(out["wav"])
        assert raw[:4] == b"RIFF"

        body = json.dumps(
            {"text": "stream me", "voice": "af_heart", "speed": 1.0, "stream": True}
        )
        conn.request(
            "POST",
            "/speak",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 200
        lines = [ln for ln in resp.read().decode().splitlines() if ln.strip()]
        assert json.loads(lines[-1]) == {"done": True}
        assert len(lines) >= 3
        first = json.loads(lines[0])
        assert first["rate"] == SAMPLE_RATE
        assert len(base64.b64decode(first["pcm"])) > 0
        conn.close()

        conn = HTTPConnection(host, port, timeout=5)
        bad = json.dumps({"text": "hi", "voice": "nope", "speed": 1.0, "stream": True})
        conn.request(
            "POST",
            "/speak",
            body=bad,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        err_body = json.loads(resp.read().decode())
        assert resp.status == 400
        assert "nope" in err_body["error"]
        conn.close()

        conn = HTTPConnection(host, port, timeout=5)
        zh = json.dumps({"text": "hi", "voice": "af_heart", "lang": "zh", "speed": 1})
        conn.request("POST", "/speak", body=zh, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 400
        assert "English" in json.loads(resp.read().decode())["error"]
        conn.close()

        conn = HTTPConnection(host, port, timeout=5)
        zf = json.dumps({"text": "hi", "voice": "zf_xiaobei", "speed": 1, "stream": True})
        conn.request("POST", "/speak", body=zf, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 400
        assert "zf_xiaobei" in json.loads(resp.read().decode())["error"]
        conn.close()

        conn = HTTPConnection(host, port, timeout=5)
        zero = json.dumps({"text": "hi", "voice": "af_heart", "speed": 0})
        conn.request(
            "POST",
            "/speak",
            body=zero,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 400
        resp.read()
        conn.close()

        conn = HTTPConnection(host, port, timeout=5)
        mood = json.dumps({"text": "Hi.", "voice": "af_heart", "speed": 1, "emotion": "excited", "phoneme_silence": {".": 0.01}})
        conn.request("POST", "/speak", body=mood, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        out = json.loads(resp.read().decode())
        assert resp.status == 200
        assert abs(sp.kokoro.last_speed - 1.2) < 1e-9
        assert sp.kokoro.calls[-1] == "Hi"
        conn.close()

        conn = HTTPConnection(host, port, timeout=5)
        bad_mood = json.dumps({"text": "hi", "voice": "af_heart", "emotion": "joyful"})
        conn.request("POST", "/speak", body=bad_mood, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 400
        assert "emotion" in json.loads(resp.read().decode())["error"]
        conn.close()
    finally:
        server.shutdown()
        server.server_close()


def test_remote_kokoro_emotion_matches_local():
    import importlib.util
    from pathlib import Path

    from runtime.tts.kokoro import apply_emotion

    path = Path(__file__).resolve().parents[2] / "scripts" / "tts_remote_server.py"
    spec = importlib.util.spec_from_file_location("tts_remote_server_emotion", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for word in ("calm", "sad", "serious", "neutral", "happy", "excited", "angry"):
        got = mod.kokoro_request_speed(1, f" {word.upper()} ")
        want, canon = apply_emotion(1, word)
        assert canon == word
        assert abs(got - want) < 1e-9
    assert mod.kokoro_request_speed(None, "") == 1
    try:
        mod.kokoro_request_speed(1, "joyful")
        assert False, "unknown"
    except ValueError as e:
        assert "emotion" in str(e)
    try:
        mod.kokoro_request_speed(1.8, "excited")
        assert False, "range"
    except ValueError as e:
        assert "0.5" in str(e)


def test_pcm16_wav_roundtrip_header():
    pcm = (np.array([0.5], dtype=np.float32) * 32767).astype("<i2").tobytes()
    wav = pcm16_wav(pcm, SAMPLE_RATE)
    assert wav[:4] == b"RIFF"
