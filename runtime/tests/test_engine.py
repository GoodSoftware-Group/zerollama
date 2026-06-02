from fastapi.testclient import TestClient

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.server.app import create_app


def test_health_engine_without_llama(cfg_root):
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=None,
        num_blocks=64,
        block_size=16,
        device_count=2,
    )
    eng = InferenceEngine(cfg)
    client = TestClient(create_app(cfg, eng))
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["kv_pools"][1]["device_id"] == 1


def test_inference_resume_endpoint(cfg_root):
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
    r = client.post("/internal/inference/resume")
    assert r.status_code == 200
    assert r.json()["inference_state"] == "running"


def test_training_handoff_endpoint(cfg_root):
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
    client = TestClient(create_app(cfg, InferenceEngine(cfg)))
    r = client.post("/internal/training-handoff")
    assert r.status_code == 200
    assert r.json()["inference_state"] == "unloaded"
