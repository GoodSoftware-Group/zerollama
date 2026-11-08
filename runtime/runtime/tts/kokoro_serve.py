"""Loopback HTTP process for local Kokoro ONNX speech.

Start with::

    python3 -m runtime.tts.kokoro_serve --model /path/to/kokoro-v1.0.onnx \\
        --voices /path/to/voices-v1.0.bin --host 127.0.0.1 --port N

``GET /health`` → ``{"status":"ok","rate":24000}`` after the model loads.
``POST /speak`` JSON: text, voice, speed, lang, emotion, phoneme_silence, stream.
  stream false → ``{"rate":24000,"wav":"<base64>"}``
  stream true → NDJSON lines ``{"rate":24000,"pcm":"..."}`` then ``{"done":true}``
  A voice or synthesis error before the first chunk is HTTP 400, not a 200 body.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from runtime.tts.kokoro import (
    DEFAULT_VOICE,
    SAMPLE_RATE,
    Speaker,
    apply_emotion,
    parse_pauses,
    resolve_lang,
)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse(argv)
    model = (args.model or "").strip()
    voices = (args.voices or "").strip()
    if not model or not os.path.isfile(model):
        print(f"kokoro: model file missing: {model!r}", file=sys.stderr)
        return 2
    if not voices or not os.path.isfile(voices):
        print(f"kokoro: voices file missing: {voices!r}", file=sys.stderr)
        return 2
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    speaker = Speaker(model, voices)
    try:
        speaker.load()
    except Exception as e:
        print(f"kokoro: load failed: {e}", file=sys.stderr)
        return 1
    server = ThreadingHTTPServer((args.host, args.port), _handler(speaker, args.default_voice))
    print(f"kokoro listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _parse(argv: Optional[list[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="runtime.tts.kokoro_serve")
    p.add_argument("--model", required=True, help="Local Kokoro ONNX file")
    p.add_argument("--voices", required=True, help="Local voices-v1.0.bin (numpy archive)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--default-voice", default=DEFAULT_VOICE)
    return p.parse_args(argv)


def _handler(speaker: Speaker, default_voice: str):
    default_voice = (default_voice or DEFAULT_VOICE).strip() or DEFAULT_VOICE

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/health":
                self.send_error(404)
                return
            self._json(200, {"status": "ok", "rate": int(speaker.rate) or SAMPLE_RATE})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/speak":
                self.send_error(404)
                return
            try:
                body = _read_json(self)
            except ValueError as e:
                self._json(400, {"error": str(e)})
                return
            text = str(body.get("text") or "").strip()
            if not text:
                self._json(400, {"error": "text is empty"})
                return
            voice = str(body.get("voice") or "").strip() or default_voice
            try:
                speed = float(body.get("speed") if body.get("speed") is not None else 1.0)
            except (TypeError, ValueError):
                self._json(400, {"error": "speed must be a number"})
                return
            if not 0.5 <= speed <= 2.0:
                self._json(400, {"error": "speed must be between 0.5 and 2"})
                return
            try:
                lang = resolve_lang(voice, str(body.get("lang") or ""))
                apply_emotion(speed, str(body.get("emotion") or ""))
                pauses = body.get("phoneme_silence")
                parse_pauses(pauses)
            except ValueError as e:
                self._json(400, {"error": str(e)})
                return
            emotion = str(body.get("emotion") or "")
            stream = bool(body.get("stream"))
            if stream:
                self._speak_stream(text, voice, speed, lang, pauses, emotion)
                return
            try:
                rate, wav = speaker.speak(
                    text,
                    voice=voice,
                    speed=speed,
                    lang=lang,
                    phoneme_silence=pauses,
                    emotion=emotion,
                )
            except Exception as e:
                self._json(400, {"error": str(e)})
                return
            self._json(
                200,
                {
                    "rate": rate,
                    "wav": base64.b64encode(wav).decode("ascii"),
                },
            )

        def _speak_stream(
            self,
            text: str,
            voice: str,
            speed: float,
            lang: str,
            phoneme_silence: Any,
            emotion: str,
        ) -> None:
            started = False

            def emit(rate: int, pcm: bytes) -> None:
                nonlocal started
                if not started:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    started = True
                line = json.dumps(
                    {
                        "rate": rate,
                        "pcm": base64.b64encode(pcm).decode("ascii"),
                    }
                ).encode("utf-8")
                self.wfile.write(line + b"\n")
                self.wfile.flush()

            try:
                speaker.speak_stream(
                    text,
                    voice=voice,
                    speed=speed,
                    lang=lang,
                    phoneme_silence=phoneme_silence,
                    emotion=emotion,
                    emit=emit,
                )
            except Exception as e:
                if not started:
                    self._json(400, {"error": str(e)})
                    return
                try:
                    self.wfile.write(
                        json.dumps({"error": str(e)}).encode("utf-8") + b"\n"
                    )
                    self.wfile.flush()
                except Exception:
                    pass
                return
            if not started:
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Connection", "close")
                    self.end_headers()
                except Exception:
                    return
            try:
                self.wfile.write(json.dumps({"done": True}).encode("utf-8") + b"\n")
                self.wfile.flush()
            except Exception:
                pass

        def _json(self, status: int, payload: dict) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return Handler


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    try:
        out = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"json body: {e}") from e
    if not isinstance(out, dict):
        raise ValueError("json object required")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
