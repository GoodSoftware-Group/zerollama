"""llama-server must wait for /health 200 before /completion."""

from __future__ import annotations

import http.server
import json
import socket
import threading
import time
from pathlib import Path

import pytest

from runtime.kv.native_decode_loop import PrefillAbortedError
from runtime.kv.prefill_cancel import PrefillCancelToken
from runtime.worker.llama_server import LlamaServerError, LlamaServerProcess


class _FakeProc:
    def poll(self) -> None:
        return None


class _LoadingThenReadyHandler(http.server.BaseHTTPRequestHandler):
    """Serves 503 /health until ready_after, then 200. Tracks /completion hits."""

    ready_after = 0.0
    completion_hits = 0
    health_hits = 0
    lock = threading.Lock()

    def do_GET(self) -> None:
        if self.path.rstrip("/") not in ("/health", "/v1/health"):
            self.send_error(404)
            return
        with self.lock:
            type(self).health_hits += 1
            ready = time.monotonic() >= self.ready_after
        if not ready:
            body = b'{"error":{"message":"Loading model","type":"unavailable_error","code":503}}'
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        if self.path.rstrip("/") != "/completion":
            self.send_error(404)
            return
        with self.lock:
            type(self).completion_hits += 1
            # A completion during loading would mean the wait was skipped.
            if time.monotonic() < self.ready_after:
                body = b'{"error":{"message":"Loading model","type":"unavailable_error","code":503}}'
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        body = json.dumps({"content": "ok", "stop": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def _start_loading_server(
    *, load_s: float = 0.4
) -> tuple[http.server.HTTPServer, str, int, type]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    sock.close()
    handler = type(
        "Handler",
        (_LoadingThenReadyHandler,),
        {
            "ready_after": time.monotonic() + load_s,
            "completion_hits": 0,
            "health_hits": 0,
            "lock": threading.Lock(),
        },
    )
    httpd = http.server.HTTPServer((host, port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, host, port, handler


def _alive_worker(host: str, port: int, tmp_path: Path) -> LlamaServerProcess:
    model = tmp_path / "toy.gguf"
    model.write_bytes(b"GGUF")
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    worker = LlamaServerProcess(binary=binary, model=model, host=host, port=port)
    worker._proc = _FakeProc()  # type: ignore[assignment]
    return worker


def test_completion_waits_for_health_before_post(tmp_path: Path):
    httpd, host, port, handler = _start_loading_server(load_s=0.5)
    try:
        worker = _alive_worker(host, port, tmp_path)
        before = time.monotonic()
        out = worker.completion("hi", n_predict=1)
        elapsed = time.monotonic() - before
        assert out.get("content") == "ok"
        assert elapsed >= 0.45
        with handler.lock:
            assert handler.completion_hits == 1
            assert handler.health_hits >= 1
            assert time.monotonic() >= handler.ready_after
    finally:
        httpd.shutdown()


def test_start_on_live_process_waits_for_health(tmp_path: Path):
    httpd, host, port, handler = _start_loading_server(load_s=0.4)
    try:
        worker = _alive_worker(host, port, tmp_path)
        before = time.monotonic()
        worker.start()
        elapsed = time.monotonic() - before
        assert elapsed >= 0.35
        with handler.lock:
            assert handler.completion_hits == 0
            assert handler.health_hits >= 1
    finally:
        httpd.shutdown()


def test_completion_cancel_during_health_wait(tmp_path: Path):
    httpd, host, port, handler = _start_loading_server(load_s=5.0)
    try:
        worker = _alive_worker(host, port, tmp_path)
        cancel = PrefillCancelToken()

        def _cancel() -> None:
            time.sleep(0.1)
            cancel.cancel()

        threading.Thread(target=_cancel, daemon=True).start()
        with pytest.raises(PrefillAbortedError):
            worker.completion("hi", n_predict=1, prefill_cancel=cancel)
        with handler.lock:
            assert handler.completion_hits == 0
    finally:
        httpd.shutdown()


def test_wait_healthy_timeout(tmp_path: Path):
    httpd, host, port, _handler = _start_loading_server(load_s=30.0)
    try:
        worker = _alive_worker(host, port, tmp_path)
        with pytest.raises(LlamaServerError, match="not healthy"):
            worker._wait_healthy(timeout_s=0.6)
    finally:
        httpd.shutdown()


def test_ready_cache_skips_health_on_hot_completion(tmp_path: Path):
    httpd, host, port, handler = _start_loading_server(load_s=0.0)
    try:
        worker = _alive_worker(host, port, tmp_path)
        assert worker.completion("one", n_predict=1).get("content") == "ok"
        with handler.lock:
            first_health = handler.health_hits
            first_completion = handler.completion_hits
        assert first_health >= 1
        assert worker.completion("two", n_predict=1).get("content") == "ok"
        with handler.lock:
            assert handler.completion_hits == first_completion + 1
            assert handler.health_hits == first_health
    finally:
        httpd.shutdown()


def test_second_caller_waits_while_first_loads(tmp_path: Path):
    httpd, host, port, handler = _start_loading_server(load_s=0.6)
    try:
        worker = _alive_worker(host, port, tmp_path)
        results: list[dict] = []
        errors: list[BaseException] = []

        def _call() -> None:
            try:
                results.append(worker.completion("hi", n_predict=1))
            except BaseException as exc:  # noqa: BLE001 — collect for assert
                errors.append(exc)

        t1 = threading.Thread(target=_call)
        t2 = threading.Thread(target=_call)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not errors, errors
        assert len(results) == 2
        assert all(r.get("content") == "ok" for r in results)
        with handler.lock:
            assert handler.completion_hits == 2
            assert time.monotonic() >= handler.ready_after
    finally:
        httpd.shutdown()
