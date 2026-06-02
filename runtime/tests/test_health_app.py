from fastapi.testclient import TestClient

from runtime.config import RuntimeConfig
from runtime.server.app import create_app


def test_health_without_llama_server(cfg_root):
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=None,
        num_blocks=128,
        block_size=16,
        device_count=2,
    )
    client = TestClient(create_app(cfg))
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert len(body["kv_pools"]) == 2
    assert body["llama_server"] is False
