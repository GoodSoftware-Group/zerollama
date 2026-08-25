import os

from runtime.gpu.vram_budget import apply_vram_budget, parse_vram_budget


def test_parse_vram_budget():
    assert parse_vram_budget("80%") == (0.8, 0)
    assert parse_vram_budget("0.8") == (0.8, 0)
    assert parse_vram_budget("12GiB") == (0.0, 12 << 30)
    assert parse_vram_budget("12GB") == (0.0, 12_000_000_000)
    assert parse_vram_budget("") == (0.0, 0)


def test_parse_vram_budget_rejects_over_100():
    try:
        parse_vram_budget("120%")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_apply_vram_budget(monkeypatch):
    monkeypatch.setenv("ZEROLLAMA_VRAM_BUDGET", "50%")
    total, free = apply_vram_budget(8 << 30, 8 << 30)
    assert total == 4 << 30
    assert free == 4 << 30
    total, free = apply_vram_budget(8 << 30, 1 << 30)
    assert total == 4 << 30
    assert free == 1 << 30


def test_apply_vram_budget_unset(monkeypatch):
    monkeypatch.delenv("ZEROLLAMA_VRAM_BUDGET", raising=False)
    assert apply_vram_budget(8 << 30, 7 << 30) == (8 << 30, 7 << 30)
