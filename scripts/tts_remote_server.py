#!/usr/bin/env python3
"""OpenAI-compatible TTS sidecar for zerollama remote-tts backends.

Serves POST /v1/audio/speech and GET /v1/audio/voices.
Select engine with TTS_ENGINE=chatterbox|orpheus|kokoro|echo (default: echo).

  TTS_ENGINE=echo python3 scripts/tts_remote_server.py   # WAV silence smoke
  TTS_PORT=8090 TTS_ENGINE=kokoro python3 scripts/tts_remote_server.py

Upstream zerollama manifests point tts_url at this server (e.g. http://cozmic:8090).
Emotion (JSON field emotion) and X-TTS-Ref-Audio are forwarded to engines that support them.
"""
from __future__ import annotations

import io
import json
import os
import struct
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ENGINE = os.environ.get("TTS_ENGINE", "echo").strip().lower()
HOST = os.environ.get("TTS_HOST", "0.0.0.0")
PORT = int(os.environ.get("TTS_PORT", "8090"))
VOICES_DIR = Path(os.environ.get("TTS_VOICES_DIR", "/mnt/ollama_img/speech/voices"))
REF_AUDIO = os.environ.get("TTS_REF_AUDIO", "")


def _pcm_wav(samples: bytes, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples)
    return buf.getvalue()


def synthesize_echo(text: str, voice: str, emotion: str) -> bytes:
    """Deterministic silent-ish WAV so wiring can be smoke-tested without GPU deps."""
    _ = (voice, emotion)
    n = max(2400, min(48000, len(text) * 200))
    # Quiet tone so players show non-empty audio
    frames = bytearray()
    for i in range(n):
        sample = int(800 * ((i % 40) / 40.0 - 0.5))
        frames += struct.pack("<h", sample)
    return _pcm_wav(bytes(frames), 24000)


def synthesize_kokoro(text: str, voice: str, emotion: str) -> bytes:
    _ = emotion
    try:
        from kokoro_onnx import Kokoro  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "kokoro-onnx not installed; pip install kokoro-onnx misaki[en] "
            "or set TTS_ENGINE=echo for smoke"
        ) from e
    model = os.environ.get("KOKORO_MODEL", "")
    voices = os.environ.get("KOKORO_VOICES", "")
    if not model or not voices:
        raise RuntimeError("set KOKORO_MODEL and KOKORO_VOICES paths")
    k = Kokoro(model, voices)
    samples, rate = k.create(text, voice=voice or "af_heart", speed=1.0)
    # samples may be float32 numpy
    import numpy as np

    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    return _pcm_wav(pcm, int(rate))


def synthesize_chatterbox(text: str, voice: str, emotion: str, ref_audio: str) -> bytes:
    try:
        import torch
        import torchaudio
        from chatterbox.tts import ChatterboxTTS  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "chatterbox-tts not installed; pip install chatterbox-tts "
            "or set TTS_ENGINE=echo for smoke"
        ) from e
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)
    audio_prompt = ref_audio if ref_audio and Path(ref_audio).is_file() else None
    # emotion/exaggeration if supported by installed version
    kwargs: dict[str, Any] = {}
    if audio_prompt:
        kwargs["audio_prompt_path"] = audio_prompt
    if emotion:
        try:
            kwargs["exaggeration"] = float(emotion)
        except ValueError:
            kwargs["exaggeration"] = 0.5
    wav = model.generate(text, **kwargs)
    buf = io.BytesIO()
    torchaudio.save(buf, wav, model.sr, format="wav")
    return buf.getvalue()


def synthesize_orpheus(text: str, voice: str, emotion: str) -> bytes:
    # Prefer tagged text; prepend emotion hint when provided.
    prompt = text
    if emotion and f"<{emotion}>" not in text.lower():
        prompt = f"<{emotion}> {text}"
    try:
        # Optional community Orpheus OpenAI servers often already speak /v1/audio/speech;
        # this path is for in-process adapters when ORPHEUS_SCRIPT is set.
        script = os.environ.get("ORPHEUS_SCRIPT", "")
        if not script:
            raise RuntimeError(
                "Orpheus in-process not configured; point tts_url at an Orpheus "
                "OpenAI server, or set ORPHEUS_SCRIPT / TTS_ENGINE=echo"
            )
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
            out_path = out.name
        try:
            subprocess.run(
                [script, "--text", prompt, "--voice", voice or "tara", "--out", out_path],
                check=True,
            )
            return Path(out_path).read_bytes()
        finally:
            Path(out_path).unlink(missing_ok=True)
    except Exception:
        raise


def synthesize(text: str, voice: str, emotion: str, ref_audio: str) -> bytes:
    if ENGINE in ("echo", "smoke"):
        return synthesize_echo(text, voice, emotion)
    if ENGINE == "kokoro":
        return synthesize_kokoro(text, voice, emotion)
    if ENGINE == "chatterbox":
        return synthesize_chatterbox(text, voice, emotion, ref_audio)
    if ENGINE == "orpheus":
        return synthesize_orpheus(text, voice, emotion)
    raise RuntimeError(f"unknown TTS_ENGINE={ENGINE!r}")


def load_voices() -> list[dict[str, Any]]:
    path = VOICES_DIR / f"{ENGINE}.json"
    if not path.is_file() and ENGINE == "echo":
        return [{"id": "default", "name": "Echo", "backend": "remote-tts"}]
    if not path.is_file():
        return [{"id": "default", "name": ENGINE, "backend": "remote-tts"}]
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return data.get("voices", [])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[tts-remote] {self.address_string()} {fmt % args}")

    def _json(self, code: int, obj: Any) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/health", "/v1/health"):
            self._json(200, {"status": "ok", "engine": ENGINE})
            return
        if path == "/v1/audio/voices":
            qs = parse_qs(urlparse(self.path).query)
            voices = load_voices()
            model = (qs.get("model") or [ENGINE])[0]
            self._json(
                200,
                {
                    "object": "list",
                    "data": [{"model": model, "backend": "remote-tts", "voices": voices}],
                },
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/v1/audio/speech":
            self._json(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return
        text = (body.get("input") or "").strip()
        if not text:
            self._json(400, {"error": "input is required"})
            return
        voice = (body.get("voice") or "default").strip()
        emotion = (body.get("emotion") or "").strip()
        ref = self.headers.get("X-TTS-Ref-Audio") or REF_AUDIO
        try:
            audio = synthesize(text, voice, emotion, ref)
        except Exception as e:  # noqa: BLE001 — surface to client
            self._json(500, {"error": str(e)})
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio)))
        self.send_header("Content-Disposition", 'attachment; filename="speech.wav"')
        self.end_headers()
        self.wfile.write(audio)


def main() -> None:
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[tts-remote] engine={ENGINE} listening on http://{HOST}:{PORT}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
