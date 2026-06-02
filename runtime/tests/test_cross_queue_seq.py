"""HTTP integration for cross-queue FIFO tickets from Go."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from runtime.cross_queue_seq import alloc_cross_queue_seq


class _SeqHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/internal/cross-queue-seq":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        body = json.dumps({"seq": 42}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


def test_alloc_cross_queue_seq_from_go(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _SeqHandler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("ZEROLLAMA_GO_URL", f"http://127.0.0.1:{port}")
        assert alloc_cross_queue_seq() == 42
    finally:
        srv.shutdown()


def test_alloc_cross_queue_seq_fallback_when_go_unreachable(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_GO_URL", "http://127.0.0.1:1")
    a = alloc_cross_queue_seq()
    b = alloc_cross_queue_seq()
    assert a >= (1 << 48)
    assert b > a
