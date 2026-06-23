"""Tests for cancellable llama-server HTTP helpers."""

from __future__ import annotations

import http.server
import json
import socket
import threading
import time

from runtime.kv.native_decode_loop import PrefillAbortedError
from runtime.kv.prefill_cancel import PrefillCancelToken
from runtime.worker.llama_server_http import cancellable_http_post, read_response_body


class _SlowHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        time.sleep(0.3)
        body = json.dumps({"content": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def _start_server() -> tuple[threading.Thread, str, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    sock.close()
    httpd = http.server.HTTPServer((host, port), _SlowHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return thread, host, port


def test_cancellable_http_post_aborts_on_cancel():
    _, host, port = _start_server()
    cancel = PrefillCancelToken()

    def run() -> None:
        time.sleep(0.05)
        cancel.cancel()

    threading.Thread(target=run, daemon=True).start()
    try:
        cancellable_http_post(
            f"http://{host}:{port}",
            "/completion",
            b"{}",
            {"Content-Type": "application/json"},
            prefill_cancel=cancel,
            timeout=5.0,
        )
        raise AssertionError("expected PrefillAbortedError")
    except PrefillAbortedError:
        pass


def test_cancellable_http_post_reads_body():
    _, host, port = _start_server()
    conn, resp = cancellable_http_post(
        f"http://{host}:{port}",
        "/completion",
        b"{}",
        {"Content-Type": "application/json"},
        prefill_cancel=None,
        timeout=5.0,
    )
    raw = read_response_body(conn, resp)
    data = json.loads(raw.decode())
    assert data["content"] == "ok"
