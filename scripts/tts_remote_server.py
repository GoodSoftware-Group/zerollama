#!/usr/bin/env python3
"""OpenAI-compatible TTS sidecar for zerollama remote-tts backends.

Serves POST /v1/audio/speech and GET /v1/audio/voices.
Select engine with TTS_ENGINE=chatterbox|orpheus|kokoro|echo (default: echo).

  TTS_ENGINE=echo python3 scripts/tts_remote_server.py   # WAV silence smoke
  TTS_PORT=8090 TTS_ENGINE=kokoro python3 scripts/tts_remote_server.py

Upstream zerollama manifests point tts_url at this server (e.g. http://127.0.0.1:8090).
Emotion (JSON field emotion) and X-TTS-Ref-Audio are forwarded to engines that
support them. Kokoro has no emotion head: the same words as local Kokoro scale
speed (calm 0.85, sad 0.80, serious 0.90, neutral 1, happy 1.10, excited 1.20,
angry 1.25). An unknown word, or a product outside 0.5–2, is HTTP 400.

Hand-start defaults (A6): bind 127.0.0.1, 32 MiB body cap, optional TTS_API_KEY,
ref-audio path confined to TTS_REF_AUDIO_DIR (or the TTS_REF_AUDIO file's parent),
generic 500 bodies, and a cached Chatterbox model.
"""
from __future__ import annotations

import io
import json
import logging
import os
import struct
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

log = logging.getLogger("tts-remote")

ENGINE = os.environ.get("TTS_ENGINE", "echo").strip().lower()
# Loopback by default so a hand start is not LAN-open. systemd already sets this.
HOST = os.environ.get("TTS_HOST", "127.0.0.1")
PORT = int(os.environ.get("TTS_PORT", "8090"))
VOICES_DIR = Path(os.environ.get("TTS_VOICES_DIR", "/mnt/ollama_img/speech/voices"))
REF_AUDIO = os.environ.get("TTS_REF_AUDIO", "")
REF_AUDIO_DIR = os.environ.get("TTS_REF_AUDIO_DIR", "").strip()
API_KEY = os.environ.get("TTS_API_KEY", "").strip()
MAX_BODY_BYTES = int(os.environ.get("TTS_MAX_BODY_BYTES", str(32 << 20)))

# Same scales as runtime.tts.kokoro.apply_emotion. Kokoro's graph has no emotion head.
_KOKORO_EMOTION = {
    "calm": 0.85,
    "sad": 0.80,
    "serious": 0.90,
    "neutral": 1.00,
    "happy": 1.10,
    "excited": 1.20,
    "angry": 1.25,
}


def kokoro_request_speed(speed: Any, emotion: str = "") -> float:
    """Requested Kokoro speed after one emotion scale. Empty emotion leaves speed."""
    if speed is None or speed == "":
        speed = 1.0
    try:
        rate = float(speed)
    except (TypeError, ValueError) as e:
        raise ValueError("speed must be a number") from e
    if not 0.5 <= rate <= 2.0:
        raise ValueError("speed must be between 0.5 and 2")
    word = (emotion or "").strip().lower()
    if not word:
        return rate
    scale = _KOKORO_EMOTION.get(word)
    if scale is None:
        names = ", ".join(sorted(_KOKORO_EMOTION))
        raise ValueError(f"kokoro emotion must be one of: {names}")
    out = rate * scale
    if not 0.5 <= out <= 2.0:
        raise ValueError("kokoro emotion pushes speed outside 0.5 to 2")
    return out


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


_kokoro_lock = None
_kokoro_engine = None
_chatterbox_lock = None
_chatterbox_model = None
_chatterbox_device = None


def _kokoro_singleton():
    """Load kokoro-onnx once; duplex speaks many short clauses per turn."""
    global _kokoro_lock, _kokoro_engine
    if _kokoro_lock is None:
        import threading

        _kokoro_lock = threading.Lock()
    with _kokoro_lock:
        if _kokoro_engine is not None:
            return _kokoro_engine
        try:
            from kokoro_onnx import Kokoro  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "kokoro-onnx not installed; pip install kokoro-onnx misaki[en] "
                "or set TTS_ENGINE=echo for smoke"
            ) from e
        model = os.environ.get("KOKORO_MODEL", "").strip()
        voices = os.environ.get("KOKORO_VOICES", "").strip()
        if not model or not voices:
            raise RuntimeError("set KOKORO_MODEL and KOKORO_VOICES paths")
        if not Path(model).is_file():
            raise RuntimeError(f"KOKORO_MODEL not found: {model}")
        if not Path(voices).is_file():
            raise RuntimeError(f"KOKORO_VOICES not found: {voices}")
        _kokoro_engine = Kokoro(model, voices)
        return _kokoro_engine


def synthesize_kokoro(text: str, voice: str, emotion: str, speed: Any = 1.0) -> bytes:
    speed = kokoro_request_speed(speed, emotion)
    k = _kokoro_singleton()
    samples, rate = k.create(text, voice=voice or "af_heart", speed=speed)
    # samples may be float32 numpy
    import numpy as np

    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    return _pcm_wav(pcm, int(rate))


def _chatterbox_singleton():
    """Load Chatterbox once; from_pretrained per request was multi-GB reload."""
    global _chatterbox_lock, _chatterbox_model, _chatterbox_device
    if _chatterbox_lock is None:
        import threading

        _chatterbox_lock = threading.Lock()
    with _chatterbox_lock:
        try:
            import torch
            from chatterbox.tts import ChatterboxTTS  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "chatterbox-tts not installed; pip install chatterbox-tts "
                "or set TTS_ENGINE=echo for smoke"
            ) from e
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if _chatterbox_model is not None and _chatterbox_device == device:
            return _chatterbox_model
        _chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
        _chatterbox_device = device
        return _chatterbox_model


def resolve_ref_audio(raw: str) -> str | None:
    """Allow only ref clips under TTS_REF_AUDIO_DIR (or the default file's parent).

    Rejects absolute paths outside the allowlist and any `..` escape. Empty input
    falls back to TTS_REF_AUDIO when that file exists inside the allowlist.
    """
    candidate = (raw or "").strip() or REF_AUDIO.strip()
    if not candidate:
        return None

    roots: list[Path] = []
    if REF_AUDIO_DIR:
        roots.append(Path(REF_AUDIO_DIR).expanduser().resolve())
    elif REF_AUDIO.strip():
        parent = Path(REF_AUDIO).expanduser().resolve().parent
        roots.append(parent)
    else:
        # No allowlist configured: ignore client-supplied paths (do not open
        # arbitrary server files from X-TTS-Ref-Audio).
        log.warning("ref audio ignored: set TTS_REF_AUDIO_DIR or TTS_REF_AUDIO")
        return None

    try:
        path = Path(candidate).expanduser().resolve()
    except OSError:
        return None
    if not path.is_file():
        return None
    for root in roots:
        try:
            path.relative_to(root)
            return str(path)
        except ValueError:
            continue
    log.warning("ref audio outside allowlist: %s", path)
    return None


def synthesize_chatterbox(text: str, voice: str, emotion: str, ref_audio: str) -> bytes:
    _ = voice
    try:
        import torchaudio
    except ImportError as e:
        raise RuntimeError(
            "chatterbox-tts not installed; pip install chatterbox-tts "
            "or set TTS_ENGINE=echo for smoke"
        ) from e
    model = _chatterbox_singleton()
    audio_prompt = resolve_ref_audio(ref_audio)
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


def synthesize(text: str, voice: str, emotion: str, ref_audio: str, speed: Any = None) -> bytes:
    if ENGINE in ("echo", "smoke"):
        return synthesize_echo(text, voice, emotion)
    if ENGINE == "kokoro":
        return synthesize_kokoro(text, voice, emotion, speed)
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
        log.info("%s %s", self.address_string(), fmt % args)

    def _json(self, code: int, obj: Any) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self) -> bool:
        if not API_KEY:
            return True
        auth = (self.headers.get("Authorization") or "").strip()
        if auth.lower().startswith("bearer ") and auth[7:].strip() == API_KEY:
            return True
        if (self.headers.get("X-API-Key") or "").strip() == API_KEY:
            return True
        self._json(401, {"error": "unauthorized"})
        return False

    def _read_body(self) -> bytes | None:
        raw_len = self.headers.get("Content-Length", "0").strip() or "0"
        try:
            n = int(raw_len)
        except ValueError:
            self._json(400, {"error": "invalid Content-Length"})
            return None
        if n < 0:
            self._json(400, {"error": "invalid Content-Length"})
            return None
        if n > MAX_BODY_BYTES:
            self._json(413, {"error": f"body exceeds {MAX_BODY_BYTES} bytes"})
            return None
        if n == 0:
            return b"{}"
        # Read at most MAX_BODY_BYTES+1 so a lying Content-Length cannot fill RAM.
        data = self.rfile.read(n)
        if len(data) > MAX_BODY_BYTES:
            self._json(413, {"error": f"body exceeds {MAX_BODY_BYTES} bytes"})
            return None
        return data

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            return
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
        if not self._authorized():
            return
        path = urlparse(self.path).path
        if path != "/v1/audio/speech":
            self._json(404, {"error": "not found"})
            return
        raw = self._read_body()
        if raw is None:
            return
        try:
            body = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid json"})
            return
        if not isinstance(body, dict):
            self._json(400, {"error": "invalid json"})
            return
        text = (body.get("input") or "").strip()
        if not text:
            self._json(400, {"error": "input is required"})
            return
        if len(text) > 20000:
            self._json(400, {"error": "input exceeds 20000 characters"})
            return
        voice = (body.get("voice") or "default").strip()
        emotion = (body.get("emotion") or "").strip()
        ref = self.headers.get("X-TTS-Ref-Audio") or REF_AUDIO
        try:
            audio = synthesize(text, voice, emotion, ref, body.get("speed"))
        except ValueError as e:
            self._json(400, {"error": str(e)})
            return
        except Exception:  # noqa: BLE001 — log detail, return generic body
            log.exception("synthesize failed")
            self._json(500, {"error": "tts synthesis failed"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio)))
        self.send_header("Content-Disposition", 'attachment; filename="speech.wav"')
        self.end_headers()
        self.wfile.write(audio)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    if HOST in ("0.0.0.0", "::", "[::]") and not API_KEY:
        log.warning(
            "TTS_HOST=%s with no TTS_API_KEY — LAN-open and unauthenticated; "
            "prefer TTS_HOST=127.0.0.1 or set TTS_API_KEY",
            HOST,
        )
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    log.info("engine=%s listening on http://%s:%s", ENGINE, HOST, PORT)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
