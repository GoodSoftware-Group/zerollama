from fastapi.testclient import TestClient

from runtime.config import RuntimeConfig
from runtime.server.app import create_app


def test_health_includes_ready_when_running(cfg_root):
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
    assert body["ready"] is True
    assert body["ready_reasons"] == []


def test_ready_503_after_training_handoff(cfg_root):
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
    client.post("/internal/training-handoff")
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False
    assert any("not accepting" in x for x in r.json()["ready_reasons"])


def test_ready_ok_after_resume(cfg_root):
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
    client.post("/internal/training-handoff")
    client.post("/internal/inference/resume")
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_handoff_clears_training_handoff_flag(cfg_root):
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
    client.post("/internal/training-handoff")
    h = client.get("/health").json()
    assert h["inference_state"] == "unloaded"
    assert h["admission"]["training_handoff_active"] is False
