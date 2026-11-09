"""Loopback HTTP process for resident Qwen3-ASR.

Start with::

    python3 -m runtime.asr.qwen3_serve --model /path/to/weights --host 127.0.0.1 --port N

``GET /health`` returns ``{"status":"ok"}`` after the model is loaded.
``POST /transcribe`` accepts multipart ``file`` and optional ``language``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs

from runtime.asr.qwen3_asr import Ear, model_dir_ready


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse(argv)
    if not model_dir_ready(args.model):
        print(f"qwen3-asr: model directory missing: {args.model!r}", file=sys.stderr)
        return 2

    device = (args.device or os.environ.get("OLLAMA_QWEN3_ASR_DEVICE") or "cpu").strip() or "cpu"
    if device == "cpu":
        # Keep the chat GPU free unless the operator pinned a device.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    ear = Ear(args.model, device=device)
    try:
        ear.load()
    except Exception as e:
        print(f"qwen3-asr: load failed: {e}", file=sys.stderr)
        return 1

    server = ThreadingHTTPServer((args.host, args.port), _handler(ear))
    print(f"qwen3-asr listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _parse(argv: Optional[list[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="runtime.asr.qwen3_serve")
    p.add_argument("--model", required=True, help="Local Qwen3-ASR weight directory")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument(
        "--device",
        default="",
        help="device_map for from_pretrained (default cpu; cuda:0 uses GPU)",
    )
    return p.parse_args(argv)


def _handler(ear: Ear):
    # One GPU call at a time. A second request returns 429 instead of waiting
    # on this lock: duplex cancels the guess as soon as more audio arrives,
    # and a waiting thread would stay up after that socket closed.
    lock = threading.Lock()
    state = threading.Lock()
    busy_since = 0.0

    class Handler(BaseHTTPRequestHandler):
        # Access lines go to stderr. A full stderr pipe blocks this write
        # before the response bytes, so /health would stop answering.
        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/health":
                self.send_error(404)
                return
            payload: dict[str, object] = {"status": "ok"}
            with state:
                if busy_since > 0:
                    payload["busy_ms"] = int((time.monotonic() - busy_since) * 1000)
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            nonlocal busy_since
            path = self.path.split("?", 1)[0]
            if path != "/transcribe":
                self.send_error(404)
                return
            try:
                wav, language = _read_multipart(self)
            except ValueError as e:
                self._json(400, {"error": str(e)})
                return
            if not lock.acquire(blocking=False):
                self._json(429, {"error": "busy"})
                return
            with state:
                busy_since = time.monotonic()
            try:
                text = ear.transcribe(wav, language=language)
            except Exception as e:
                self._json(500, {"error": str(e)})
                return
            finally:
                with state:
                    busy_since = 0.0
                lock.release()
            self._json(200, {"text": text})

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _read_multipart(handler: BaseHTTPRequestHandler) -> tuple[bytes, str]:
    ctype = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length) if length > 0 else b""
    if "multipart/form-data" not in ctype.lower():
        # Also accept raw WAV with ?language=
        language = ""
        if "?" in handler.path:
            qs = parse_qs(handler.path.split("?", 1)[1])
            language = (qs.get("language") or [""])[0]
        if not raw:
            raise ValueError("empty body")
        return raw, language

    boundary = _boundary(ctype)
    if not boundary:
        raise ValueError("multipart boundary missing")
    return _parse_multipart(raw, boundary)


def _boundary(ctype: str) -> bytes:
    for part in ctype.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            val = part.split("=", 1)[1].strip().strip('"')
            return val.encode("utf-8")
    return b""


def _parse_multipart(raw: bytes, boundary: bytes) -> tuple[bytes, str]:
    sep = b"--" + boundary
    wav = b""
    language = ""
    for part in raw.split(sep):
        if not part or part in (b"--", b"--\r\n", b"\r\n"):
            continue
        if part.startswith(b"--"):
            break
        if part.startswith(b"\r\n"):
            part = part[2:]
        header_blob, _, body = part.partition(b"\r\n\r\n")
        if body.endswith(b"\r\n"):
            body = body[:-2]
        headers = header_blob.decode("utf-8", errors="replace").lower()
        if 'name="file"' in headers or "name=file" in headers:
            wav = body
        elif 'name="language"' in headers or "name=language" in headers:
            language = body.decode("utf-8", errors="replace").strip()
    if not wav:
        raise ValueError("multipart file field missing")
    return wav, language


if __name__ == "__main__":
    raise SystemExit(main())
