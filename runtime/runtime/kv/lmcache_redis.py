"""Redis-backed LMCache metadata tier (L3-R4).

WHY Redis first: fleet nodes can share prefix block index without NFS; blob paths
still point at local llama-server slot files — remote tier is metadata for Radix
donor discovery and restart hydration, not cross-node KV VRAM (that needs NIXL/Mooncake).
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from runtime.kv.lmcache_tier import LMCacheBlockRecord, _safe_dir_name


@dataclass(frozen=True)
class RedisConfig:
    host: str
    port: int
    db: int
    password: str | None
    key_prefix: str
    ttl_sec: int | None


def parse_redis_uri(uri: str) -> RedisConfig:
    parsed = urlparse(uri)
    if parsed.scheme not in ("redis", "rediss"):
        raise ValueError(f"not a redis URI: {uri}")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6379
    path = (parsed.path or "/0").lstrip("/")
    db = int(path.split("/")[0] or "0") if path else 0
    password = parsed.password
    if parsed.username and not password:
        password = parsed.username
    return RedisConfig(
        host=host,
        port=port,
        db=db,
        password=password,
        key_prefix="zerollama:lmcache:v1",
        ttl_sec=None,
    )


class _RedisRespClient:
    """Minimal RESP client (GET/SET/PING/DEL/DBSIZE) — no redis-py dependency."""

    def __init__(self, cfg: RedisConfig, *, timeout: float = 2.0) -> None:
        self._cfg = cfg
        self._timeout = timeout

    def _connect(self) -> socket.socket:
        sock = socket.create_connection(
            (self._cfg.host, self._cfg.port), timeout=self._timeout
        )
        if self._cfg.password:
            self._command(sock, "AUTH", self._cfg.password)
        if self._cfg.db:
            self._command(sock, "SELECT", str(self._cfg.db))
        return sock

    @staticmethod
    def _encode_cmd(*args: str) -> bytes:
        parts: list[bytes] = [f"*{len(args)}\r\n".encode()]
        for arg in args:
            b = arg.encode("utf-8")
            parts.append(f"${len(b)}\r\n".encode())
            parts.append(b)
            parts.append(b"\r\n")
        return b"".join(parts)

    def _command(self, sock: socket.socket, *args: str) -> Any:
        sock.sendall(self._encode_cmd(*args))
        return self._read_reply(sock)

    def _read_reply(self, sock: socket.socket) -> Any:
        line = self._read_line(sock)
        if not line:
            raise ConnectionError("redis: empty reply")
        kind = line[0:1]
        payload = line[1:]
        if kind == b"+":
            return payload.decode("utf-8", errors="replace")
        if kind == b"-":
            raise RuntimeError(payload.decode("utf-8", errors="replace"))
        if kind == b":":
            return int(payload)
        if kind == b"$":
            n = int(payload)
            if n < 0:
                return None
            data = self._read_exact(sock, n + 2)
            return data[:-2].decode("utf-8", errors="replace")
        if kind == b"*":
            n = int(payload)
            return [self._read_reply(sock) for _ in range(n)]
        raise RuntimeError(f"redis: unknown reply type {kind!r}")

    @staticmethod
    def _read_line(sock: socket.socket) -> bytes:
        buf = bytearray()
        while True:
            ch = sock.recv(1)
            if not ch:
                break
            buf.extend(ch)
            if buf.endswith(b"\r\n"):
                return bytes(buf[:-2])
        return bytes(buf)

    @staticmethod
    def _read_exact(sock: socket.socket, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            chunk = sock.recv(n - len(out))
            if not chunk:
                break
            out.extend(chunk)
        return bytes(out)

    def get(self, key: str) -> str | None:
        with self._connect() as sock:
            return self._command(sock, "GET", key)

    def set(self, key: str, value: str, *, ttl_sec: int | None = None) -> None:
        with self._connect() as sock:
            if ttl_sec is not None and ttl_sec > 0:
                self._command(sock, "SET", key, value, "EX", str(int(ttl_sec)))
            else:
                self._command(sock, "SET", key, value)

    def delete(self, key: str) -> bool:
        with self._connect() as sock:
            n = self._command(sock, "DEL", key)
            return int(n or 0) > 0

    def ping(self) -> bool:
        with self._connect() as sock:
            return self._command(sock, "PING") == "PONG"

    def dbsize(self) -> int:
        with self._connect() as sock:
            return int(self._command(sock, "DBSIZE") or 0)


class RedisLMCacheTierStore:
    def __init__(self, cfg: RedisConfig, *, uri: str) -> None:
        self._cfg = cfg
        self._uri = uri
        self._client = _RedisRespClient(cfg)

    def _key(self, model_scope: str, block_hash: str) -> str:
        scope = _safe_dir_name(model_scope)
        return f"{self._cfg.key_prefix}:{scope}:{block_hash}"

    def put(self, record: LMCacheBlockRecord) -> None:
        key = self._key(record.model_scope, record.block_hash)
        payload = json.dumps(record.to_dict(), separators=(",", ":"))
        self._client.set(key, payload, ttl_sec=self._cfg.ttl_sec)

    def get(self, *, model_scope: str, block_hash: str) -> LMCacheBlockRecord | None:
        raw = self._client.get(self._key(model_scope, block_hash))
        if not raw:
            return None
        try:
            rec = LMCacheBlockRecord.from_dict(json.loads(raw))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if rec.model_scope != model_scope or rec.block_hash != block_hash:
            return None
        return rec

    def delete(self, *, model_scope: str, block_hash: str) -> bool:
        return self._client.delete(self._key(model_scope, block_hash))

    def health(self) -> dict[str, Any]:
        ok = False
        count = 0
        err: str | None = None
        try:
            ok = self._client.ping()
            count = self._client.dbsize()
        except (OSError, RuntimeError, ConnectionError, ValueError) as e:
            err = str(e)
        return {
            "enabled": True,
            "backend": "redis",
            "uri": self._uri,
            "host": self._cfg.host,
            "port": self._cfg.port,
            "db": self._cfg.db,
            "reachable": ok,
            "record_count": count,
            "ttl_sec": self._cfg.ttl_sec,
            "error": err,
        }
