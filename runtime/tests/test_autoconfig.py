from pathlib import Path

from runtime.autoconfig import resolve_default_config_path


def test_resolve_single_gpu_when_one_visible(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_AUTO_CONFIG", "1")
    monkeypatch.delenv("ZEROLLAMA_RUNTIME_CONFIG", raising=False)
    monkeypatch.setattr(
        "runtime.autoconfig.detect_visible_gpu_count", lambda: 1
    )
    path = resolve_default_config_path()
    assert path.name == "single_gpu.yaml"


def test_resolve_dual_when_two_visible(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_AUTO_CONFIG", "1")
    monkeypatch.setattr(
        "runtime.autoconfig.detect_visible_gpu_count", lambda: 2
    )
    path = resolve_default_config_path()
    assert path.name == "dual_4090.yaml"


def test_load_single_gpu_yaml():
    path = Path(__file__).resolve().parents[1] / "configs" / "single_gpu.yaml"
    from runtime.config import RuntimeConfig

    cfg = RuntimeConfig.from_file(path)
    assert cfg.device_count == 1
    assert cfg.tensor_parallel == 1
    assert cfg.split_mode == "layer"
    assert cfg.llama_parallel_slots == 1
