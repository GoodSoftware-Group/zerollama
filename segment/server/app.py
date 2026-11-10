"""SAM2 segment sibling for POST /v1/segment.

This process answers the wire. It does not emit a mask and it does not claim
grasp_ready. Point zerollama at it with ZEROLLAMA_SEGMENT_URL when a lab wants
the sibling process itself; leaving the URL unset makes zerollama return the
same stub. Lab port is 18096. Do not bind 11434 or 8081.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STUB = {
    "status": "wire_only",
    "grasp_ready": False,
    "mask": None,
    "reason": "SAM2 weights are not loaded",
}


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
        self._json(200, {"status": "wire_only", "grasp_ready": False})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/v1/segment":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._json(400, {"error": str(e)})
            return
        image = ""
        if isinstance(body, dict):
            for key in ("image", "image_base64", "image_url", "frame"):
                raw = body.get(key)
                if isinstance(raw, str) and raw.strip():
                    image = raw
                    break
        if not image:
            self._json(400, {"error": "image, image_base64, image_url, or frame is required"})
            return
        self._json(200, STUB)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM2 segment stub")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18096)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
