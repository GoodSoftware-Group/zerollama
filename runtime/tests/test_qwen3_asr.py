"""Qwen3-ASR ear without torch or Hub downloads."""

from __future__ import annotations

import json
import os
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

from runtime.asr.qwen3_asr import Ear, model_dir_ready
from runtime.asr.qwen3_serve import _handler


class _FakeResult:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def transcribe(self, audio, language=None, return_time_stamps=False):
        self.calls.append((audio, language))
        assert return_time_stamps is False
        return [_FakeResult("  hello world  ")]


def test_model_dir_ready_requires_directory(tmp_path):
    assert model_dir_ready("") is False
    assert model_dir_ready(str(tmp_path / "missing")) is False
    assert model_dir_ready(str(tmp_path)) is True
    hub = tmp_path / "Qwen"
    hub.write_text("not a dir")
    assert model_dir_ready(str(hub)) is False


def test_ear_transcribe_uses_injected_model(tmp_path):
    fake = _FakeModel()
    ear = Ear(str(tmp_path), device="cpu", model=fake)
    text = ear.transcribe(b"RIFF....wav", language="English")
    assert text == "hello world"
    assert len(fake.calls) == 1
    path, lang = fake.calls[0]
    assert lang == "English"
    assert path.endswith(".wav")
    assert not os.path.exists(path)


def test_ear_rejects_missing_directory_without_model():
    ear = Ear("/no/such/qwen3-asr-dir", device="cpu")
    with pytest.raises(FileNotFoundError):
        ear.load()


def test_parse_multipart_matches_go_writer():
    """Go mime/multipart CreateFormFile uses application/octet-stream."""
    from runtime.asr.qwen3_serve import _parse_multipart

    boundary = b"GoBoundary123"
    wav = b"RIFF....wav-bytes"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="clip.wav"\r\n'
        b"Content-Type: application/octet-stream\r\n"
        b"\r\n" + wav + b"\r\n"
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="language"\r\n'
        b"\r\n"
        b"English\r\n"
        b"--" + boundary + b"--\r\n"
    )
    got_wav, got_lang = _parse_multipart(body, boundary)
    assert got_wav == wav
    assert got_lang == "English"


def test_serve_health_and_transcribe(tmp_path):
    fake = _FakeModel()
    ear = Ear(str(tmp_path), device="cpu", model=fake)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(ear))
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        body = json.loads(resp.read().decode())
        assert resp.status == 200
        assert body == {"status": "ok"}

        boundary = "----zerollama"
        wav = b"RIFF-fake-wav"
        parts = [
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="clip.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n".encode()
            + wav,
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="language"\r\n\r\n'
                "English"
            ).encode(),
            f"--{boundary}--\r\n".encode(),
        ]
        payload = b"\r\n".join(parts)
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(payload)),
        }
        conn.request("POST", "/transcribe", body=payload, headers=headers)
        resp = conn.getresponse()
        out = json.loads(resp.read().decode())
        assert resp.status == 200
        assert out == {"text": "hello world"}
        assert fake.calls[-1][1] == "English"
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
