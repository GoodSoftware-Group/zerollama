from pathlib import Path

import pytest

from runtime.autoconfig import resolve_default_config_path, resolved_config_path


def test_resolved_config_path_prefers_l3_profile_over_darwin_default(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_CONFIG", raising=False)
    monkeypatch.setenv("ZEROLLAMA_L3_PROFILE", "agent")
    monkeypatch.setattr("runtime.autoconfig.sys.platform", "darwin")
    path = resolved_config_path()
    assert path.name == "l3_agent_subprocess.yaml"


def test_resolve_single_gpu_when_one_visible(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_AUTO_CONFIG", "1")
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_CONFIG", raising=False)
    monkeypatch.setattr("runtime.autoconfig.sys.platform", "linux")
    monkeypatch.setattr(
        "runtime.autoconfig.detect_visible_gpu_count", lambda: 1
    )
    path = resolve_default_config_path()
    assert path.name == "single_gpu.yaml"


def test_resolve_dual_when_two_visible(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_AUTO_CONFIG", "1")
    monkeypatch.setattr("runtime.autoconfig.sys.platform", "linux")
    monkeypatch.setattr(
        "runtime.autoconfig.detect_visible_gpu_count", lambda: 2
    )
    path = resolve_default_config_path()
    assert path.name == "dual_4090.yaml"


def test_load_single_gpu_yaml(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_GPU_PROFILE", "0")
    path = Path(__file__).resolve().parents[1] / "configs" / "single_gpu.yaml"
    from runtime.config import RuntimeConfig

    from runtime.worker.factory import llama_backend_source

    cfg = RuntimeConfig.from_file(path)
    assert cfg.device_count == 1
    assert cfg.tensor_parallel == 1
    assert cfg.split_mode == "layer"
    assert cfg.llama_parallel_slots == 1
    assert cfg.llama_backend == "subprocess"
    assert cfg.llama_backend_from_file is False
    assert llama_backend_source(cfg) == "default"


def test_apple_silicon_yaml_inprocess_backend():
    path = Path(__file__).resolve().parents[1] / "configs" / "apple_silicon.yaml"
    from runtime.config import RuntimeConfig
    from runtime.worker.factory import LlamaBackendKind, llama_backend_source, resolve_llama_backend

    cfg = RuntimeConfig.from_file(path)
    assert cfg.llama_backend == "inprocess"
    assert cfg.llama_backend_from_file is True
    assert llama_backend_source(cfg) == "config"
    assert resolve_llama_backend(cfg) == LlamaBackendKind.INPROCESS


def test_load_llama_backend_from_yaml(tmp_path):
    from runtime.config import RuntimeConfig
    from runtime.worker.factory import LlamaBackendKind, resolve_llama_backend

    path = tmp_path / "cfg.yaml"
    path.write_text(
        "device_count: 1\nllama_backend: inprocess\n",
        encoding="utf-8",
    )
    cfg = RuntimeConfig.from_file(path)
    assert cfg.llama_backend == "inprocess"
    assert cfg.llama_backend_from_file is True
    assert resolve_llama_backend(cfg) == LlamaBackendKind.INPROCESS


def test_yaml_llama_backend_env_wins(tmp_path, monkeypatch):
    from runtime.config import RuntimeConfig

    path = tmp_path / "cfg.yaml"
    path.write_text("llama_backend: inprocess\n", encoding="utf-8")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "subprocess")
    cfg = RuntimeConfig.from_file(path)
    assert cfg.llama_backend == "subprocess"


def test_invalid_llama_backend_in_yaml(tmp_path):
    from runtime.config import RuntimeConfig

    path = tmp_path / "cfg.yaml"
    path.write_text("llama_backend: bogus\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown llama backend"):
        RuntimeConfig.from_file(path)


def test_yaml_llama_backend_alias_normalized(tmp_path):
    from runtime.config import RuntimeConfig

    path = tmp_path / "cfg.yaml"
    path.write_text("llama_backend: libllama\n", encoding="utf-8")
    cfg = RuntimeConfig.from_file(path)
    assert cfg.llama_backend == "inprocess"


def test_from_env_loads_yaml_backend(tmp_path, monkeypatch):
    from runtime.config import RuntimeConfig
    from runtime.worker.factory import (
        LlamaBackendKind,
        llama_backend_source,
        resolve_llama_backend,
    )

    path = tmp_path / "cfg.yaml"
    path.write_text("llama_backend: inprocess\n", encoding="utf-8")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_CONFIG", str(path))
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", raising=False)
    cfg = RuntimeConfig.from_env()
    assert cfg.llama_backend == "inprocess"
    assert cfg.llama_backend_from_file is True
    assert llama_backend_source(cfg) == "config"
    assert resolve_llama_backend(cfg) == LlamaBackendKind.INPROCESS


def test_yaml_llama_backend_env_normalizes_case(tmp_path, monkeypatch):
    from runtime.config import RuntimeConfig

    path = tmp_path / "cfg.yaml"
    path.write_text("llama_backend: subprocess\n", encoding="utf-8")
    monkeypatch.setenv("ZEROLLAMA_RUNTIME_LLAMA_BACKEND", "InProcess")
    cfg = RuntimeConfig.from_file(path)
    assert cfg.llama_backend == "inprocess"
