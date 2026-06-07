"""Darwin host memory and metal-unified VRAM probe tests."""

from __future__ import annotations


import pytest


def test_read_darwin_host_memory_parses_vm_stat(monkeypatch: pytest.MonkeyPatch):
    import runtime.host_memory as hm

    def fake_check_output(cmd, **kwargs):
        if cmd[:2] == ["sysctl", "-n"] and cmd[2] == "hw.pagesize":
            return "4096\n"
        if cmd[0] == "vm_stat":
            return (
                "Pages free:                               100000.\n"
                "Pages inactive:                            50000.\n"
                "Pages speculative:                         10000.\n"
            )
        if cmd == ["sysctl", "vm.swapusage"]:
            # Real macOS format: "vm.swapusage: total = 2048.00M  used = 512.00M  free = 1536.00M"
            return "vm.swapusage: total = 2048.00M  used = 512.00M  free = 1536.00M  (encrypted)\n"
        raise AssertionError(f"unexpected cmd: {cmd}")

    monkeypatch.setattr(hm.sys, "platform", "darwin")
    monkeypatch.setattr(hm.subprocess, "check_output", fake_check_output)
    mem = hm.read_darwin_host_memory()
    assert mem is not None
    # (100000 + 50000 + 10000) * 4096
    assert mem.available_bytes == 160000 * 4096
    # 1536 MiB swap free
    assert mem.swap_free_bytes == 1536 * 1024 * 1024


def test_read_host_memory_delegates_linux(monkeypatch: pytest.MonkeyPatch):
    import runtime.host_memory as hm

    monkeypatch.setattr(hm.sys, "platform", "linux")
    monkeypatch.setattr(
        hm,
        "read_linux_host_memory",
        lambda: hm.HostMemory(available_bytes=123, swap_free_bytes=0),
    )
    mem = hm.read_host_memory()
    assert mem is not None
    assert mem.available_bytes == 123


def test_read_host_memory_delegates_darwin(monkeypatch: pytest.MonkeyPatch):
    import runtime.host_memory as hm

    monkeypatch.setattr(hm.sys, "platform", "darwin")
    monkeypatch.setattr(
        hm,
        "read_darwin_host_memory",
        lambda: hm.HostMemory(available_bytes=456, swap_free_bytes=789),
    )
    mem = hm.read_host_memory()
    assert mem is not None
    assert mem.available_bytes == 456
    assert mem.swap_free_bytes == 789


def test_check_gguf_host_budget_uses_darwin_mem(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import runtime.host_memory as hm
    from runtime.worker.llama_server import LlamaServerError

    gguf = tmp_path / "big.gguf"
    gguf.write_bytes(b"\0" * 100)
    monkeypatch.setattr(hm.sys, "platform", "darwin")
    monkeypatch.setattr(
        hm,
        "read_host_memory",
        lambda: hm.HostMemory(available_bytes=50, swap_free_bytes=0),
    )
    with pytest.raises(LlamaServerError, match="requires about|weights"):
        hm.check_gguf_host_budget(gguf, margin=1.0)


def test_darwin_total_memory_bytes(monkeypatch: pytest.MonkeyPatch):
    import runtime.host_memory as hm

    monkeypatch.setattr(hm.sys, "platform", "darwin")
    monkeypatch.setattr(
        hm.subprocess,
        "check_output",
        lambda *a, **k: "17179869184\n",
    )
    assert hm.darwin_total_memory_bytes() == 17179869184
