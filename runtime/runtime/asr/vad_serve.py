"""Loopback HTTP process for neural Silero VAD.

Start with::

    python3 -m runtime.asr.vad_serve --model /path/to/silero_vad.onnx --host 127.0.0.1 --port N

``GET /health`` → ``{"status":"ok"}`` after the graph loads.
``POST /score`` JSON body: session, rate, threshold, pcm (base64 PCM16).
``POST /reset`` JSON body: session.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from runtime.asr.vad import Ear


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse(argv)
    path = (args.model or "").strip()
    if not path or not os.path.isfile(path):
        print(f"neural-vad: model file missing: {path!r}", file=sys.stderr)
        return 2
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    ear = Ear(path)
    try:
        ear.load()
    except Exception as e:
        print(f"neural-vad: load failed: {e}", file=sys.stderr)
        return 1
    server = ThreadingHTTPServer((args.host, args.port), _handler(ear))
    print(f"neural-vad listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _parse(argv: Optional[list[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="runtime.asr.vad_serve")
    p.add_argument("--model", required=True, help="Local Silero ONNX file")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    return p.parse_args(argv)


def _handler(ear: Ear):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/health":
                self.send_error(404)
                return
            self._json(200, {"status": "ok"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                body = _read_json(self)
            except ValueError as e:
                self._json(400, {"error": str(e)})
                return
            if path == "/reset":
                sid = str(body.get("session") or "").strip()
                if not sid:
                    self._json(400, {"error": "session required"})
                    return
                ear.reset(sid)
                self._json(200, {"status": "ok"})
                return
            if path != "/score":
                self.send_error(404)
                return
            sid = str(body.get("session") or "").strip()
            if not sid:
                self._json(400, {"error": "session required"})
                return
            try:
                rate = int(body.get("rate") or 0)
                threshold = float(body.get("threshold") or 0.5)
                pcm_b64 = body.get("pcm") or ""
                pcm = base64.b64decode(pcm_b64)
                last = ear.score(sid, pcm, rate, threshold)
            except Exception as e:
                self._json(400, {"error": str(e)})
                return
            self._json(200, {"last": last})

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
