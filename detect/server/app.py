"""OWLv2 detect sibling for POST /v1/detect.

Point zerollama at this process with ZEROLLAMA_DETECT_URL. A missing weight
directory still serves the route and answers status=wire_only. Lab port is
18097. Do not bind 11434 or 8081.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from detect import Detector, detector_from_env

_detector: Detector | None = None


def detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = detector_from_env()
    return _detector


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/health":
            self._json(404, {"error": "not found"})
            return
        ear = detector()
        ear.load()
        status = "ok" if ear._model is not None else "wire_only"
        self._json(200, {"status": status, "model": "owlv2", "reason": ear.reason})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/v1/detect":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 20 * 1024 * 1024:
            self._json(413, {"error": "image body exceeds 20MB"})
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._json(400, {"error": str(e)})
            return
        if not isinstance(body, dict):
            self._json(400, {"error": "json object required"})
            return
        try:
            self._json(200, detector().detect(body))
        except ValueError as e:
            self._json(400, {"error": str(e)})
        except Exception as e:  # the sibling must answer JSON, not a traceback page
            self._json(500, {"error": str(e)})


def main() -> None:
    parser = argparse.ArgumentParser(description="OWLv2 detect sibling")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18097)
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
