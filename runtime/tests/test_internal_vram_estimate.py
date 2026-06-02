from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.server.app import create_app


@pytest.fixture
def engine(cfg_root, tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=gguf,
        num_blocks=64,
        block_size=16,
        device_count=1,
    )
    return InferenceEngine(cfg)


def test_internal_vram_estimate_loopback(cfg_root, tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
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
    r = client.post(
        "/internal/vram-estimate",
        json={"gguf": str(gguf), "num_ctx": 4096},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["vram_estimate"]["gguf"] == str(gguf.resolve())
    assert body["vram_estimate"]["num_ctx"] == 4096


def test_internal_vram_estimate_rejects_missing(tmp_path: Path):
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=tmp_path,
        llama_server_bin=None,
        llama_model=None,
        num_blocks=8,
        block_size=16,
        device_count=1,
    )
    client = TestClient(create_app(cfg))
    r = client.post(
        "/internal/vram-estimate",
        json={"gguf": str(tmp_path / "nope.gguf")},
    )
    assert r.status_code == 404


def test_vram_estimate_and_budget_host_ram_without_gpu_budget(
    engine, tmp_path: Path,
):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch("runtime.engine.vram_budget_health", return_value=None):
        with patch(
            "runtime.host_memory.host_ram_budget_snapshot",
            return_value={"fits": False, "required_bytes": 9, "load_budget_bytes": 1},
        ):
            with patch("runtime.engine.describe_vram_estimate") as est:
                est.return_value = {"required_per_gpu_bytes": 1}
                _est, budget = engine.vram_estimate_and_budget(gguf)
    assert budget == {"host_ram": {"fits": False, "required_bytes": 9, "load_budget_bytes": 1}}


def test_vram_estimate_and_budget_includes_host_ram(engine, tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    with patch(
        "runtime.host_memory.host_ram_budget_snapshot",
        return_value={"fits": True, "required_bytes": 1, "load_budget_bytes": 2},
    ):
        with patch("runtime.engine.describe_vram_estimate") as est:
            est.return_value = {"required_per_gpu_bytes": 1}
            _est, budget = engine.vram_estimate_and_budget(gguf)
    assert budget is not None
    assert budget["host_ram"]["fits"] is True


def test_vram_estimate_and_budget_uses_llama_kwargs(engine, tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"d")
    engine.config.speculative.draft_model = draft
    engine.config.speculative.method = "draft"

    with patch("runtime.engine.describe_vram_estimate") as est:
        est.return_value = {"required_per_gpu_bytes": 1}
        engine.vram_estimate_and_budget(gguf, options={"num_ctx": 2048})
    kw = est.call_args.kwargs
    assert kw.get("draft_gguf") == draft
    assert kw.get("num_ctx") == 2048
