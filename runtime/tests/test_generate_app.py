from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from runtime.config import RuntimeConfig
from runtime.engine import GenerateResult
from runtime.server.app import create_app


def test_api_generate_accepts_json_body():
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=__import__("pathlib").Path("/tmp"),
        llama_server_bin=None,
        llama_model=None,
        num_blocks=128,
        block_size=16,
        device_count=1,
    )
    eng = MagicMock()
    eng.health.return_value = {"status": "ok"}
    eng.generate.return_value = GenerateResult(
        content="pong", request_id="t1", llama={}
    )

    client = TestClient(create_app(cfg, engine=eng))
    r = client.post(
        "/api/generate",
        json={
            "model": "smoke",
            "prompt": "Say pong",
            "stream": False,
            "options": {"num_predict": 8},
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["done"] is True
    assert data["response"] == "pong"
    eng.generate.assert_called_once()
