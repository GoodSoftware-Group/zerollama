"""Host memory pre-check before GGUF load."""

from pathlib import Path

import pytest

from runtime.host_memory import check_gguf_host_budget, format_bytes
from runtime.worker.llama_server import LlamaServerError


def test_format_bytes():
    assert format_bytes(0) == "0 B"
    assert "GiB" in format_bytes(5 * 1024**3)


def test_check_gguf_host_budget_rejects_huge_file(monkeypatch, tmp_path: Path):
    gguf = tmp_path / "huge.gguf"
    gguf.write_bytes(b"\0" * 100)
    monkeypatch.setattr(
        "runtime.host_memory.read_linux_host_memory",
        lambda: type(
            "M",
            (),
            {
                "available_bytes": 50,
                "swap_free_bytes": 0,
                "load_budget_bytes": 50,
            },
        )(),
    )
    with pytest.raises(LlamaServerError, match="requires about|weights"):
        check_gguf_host_budget(gguf, margin=1.0)


def test_check_gguf_host_budget_skips_non_linux(monkeypatch, tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\0" * 1024)
    monkeypatch.setattr("runtime.host_memory.read_linux_host_memory", lambda: None)
    check_gguf_host_budget(gguf)
