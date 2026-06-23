"""Cancellable HTTP client helpers for llama-server subprocess completions."""

from __future__ import annotations

import http.client
import threading
import time
from typing import Any, Iterator
from urllib.parse import urlparse

from runtime.kv.native_decode_loop import PrefillAbortedError


def _parse_http_base(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def _raise_if_cancelled(prefill_cancel: Any | None) -> None:
    if prefill_cancel is not None and prefill_cancel.is_cancelled():
        raise PrefillAbortedError("KV prefill aborted (client disconnect)")


def cancellable_http_post(
    base_url: str,
    path: str,
    body: bytes,
    headers: dict[str, str],
    *,
    prefill_cancel: Any | None = None,
    timeout: float = 600.0,
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    """POST and return (connection, response). Caller must close the connection.

    WHY http.client: closing the connection aborts an in-flight llama-server
    /completion when the HTTP client disconnects — urllib urlopen cannot be
    interrupted from another thread.
    """
    host, port = _parse_http_base(base_url)
    _raise_if_cancelled(prefill_cancel)

    if prefill_cancel is None:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("POST", path, body, headers)
        return conn, conn.getresponse()

    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    holder: dict[str, Any] = {"resp": None, "err": None, "done": False}

    def worker() -> None:
        try:
            conn.request("POST", path, body, headers)
            holder["resp"] = conn.getresponse()
        except Exception as exc:
            holder["err"] = exc
        finally:
            holder["done"] = True

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while not holder["done"]:
        if prefill_cancel.is_cancelled():
            try:
                conn.close()
            except OSError:
                pass
            raise PrefillAbortedError("KV prefill aborted (client disconnect)")
        time.sleep(0.05)

    if holder["err"] is not None:
        try:
            conn.close()
        except OSError:
            pass
        raise holder["err"]
    assert holder["resp"] is not None
    return conn, holder["resp"]


def read_response_body(
    conn: http.client.HTTPConnection,
    resp: http.client.HTTPResponse,
    *,
    prefill_cancel: Any | None = None,
) -> bytes:
    chunks: list[bytes] = []
    try:
        while True:
            _raise_if_cancelled(prefill_cancel)
            chunk = resp.read(8192)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        try:
            conn.close()
        except OSError:
            pass
    return b"".join(chunks)


def iter_response_chunks(
    conn: http.client.HTTPConnection,
    resp: http.client.HTTPResponse,
    *,
    prefill_cancel: Any | None = None,
) -> Iterator[bytes]:
    try:
        while True:
            _raise_if_cancelled(prefill_cancel)
            chunk = resp.read(4096)
            if not chunk:
                break
            yield chunk
    finally:
        try:
            conn.close()
        except OSError:
            pass
