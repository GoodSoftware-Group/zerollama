import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from runtime.supervisor import SupervisorError, wait_for_runtime_health


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_args: object) -> None:
        return


def test_wait_for_runtime_health():
    server = HTTPServer(("127.0.0.1", 0), _HealthHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        wait_for_runtime_health(f"http://127.0.0.1:{port}", timeout_s=5.0)
    finally:
        server.shutdown()


def test_wait_for_runtime_health_timeout():
    with pytest.raises(SupervisorError):
        wait_for_runtime_health("http://127.0.0.1:1", timeout_s=0.5)
