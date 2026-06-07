from pathlib import Path

from runtime.host_memory import host_ram_budget_snapshot


def test_host_ram_budget_snapshot(monkeypatch, tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x" * 1000)

    class Mem:
        available_bytes = 10_000_000_000
        swap_free_bytes = 0

        @property
        def load_budget_bytes(self):
            return self.available_bytes + self.swap_free_bytes

    monkeypatch.setattr(
        "runtime.host_memory.read_host_memory", lambda: Mem()
    )
    monkeypatch.setattr(
        "runtime.host_memory.estimate_gguf_ram_bytes", lambda _p: 5000
    )
    snap = host_ram_budget_snapshot(gguf)
    assert snap is not None and snap["fits"] is True

    monkeypatch.setenv("ZEROLLAMA_RUNTIME_RAM_MARGIN", "2.0")
    snap2 = host_ram_budget_snapshot(gguf)
    assert snap2 is not None
    assert snap2["margin"] == 2.0
    assert snap2["required_bytes"] == 10_000
