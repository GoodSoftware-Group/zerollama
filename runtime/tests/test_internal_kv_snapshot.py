"""Phase 15 v7: loopback /internal/kv-snapshot."""

from __future__ import annotations

from fastapi.testclient import TestClient

from runtime.config import RuntimeConfig
from runtime.server.app import create_app


def test_internal_kv_snapshot_loopback(cfg_root):
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=None,
        num_blocks=64,
        block_size=16,
        device_count=1,
    )
    client = TestClient(create_app(cfg))
    r = client.get("/internal/kv-snapshot")
    assert r.status_code == 200
    body = r.json()
    assert "kv_scheduler" in body
    assert "kv_bind" in body
    assert "kv_forward_plans" in body
    assert isinstance(body["kv_forward_plans"], list)
