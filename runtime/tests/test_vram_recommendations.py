"""Tests for shared VRAM recommendation helpers."""

from __future__ import annotations

from runtime.vram_recommendations import skip_global_vram_factor_export


def test_skip_global_export_when_autotune_off():
    assert not skip_global_vram_factor_export(
        autotune_enabled=False,
        catalog=[{"model": "a.gguf"}],
        factor_source="catalog",
        persisted_factor=1.2,
    )


def test_skip_global_export_when_catalog_present():
    assert skip_global_vram_factor_export(
        autotune_enabled=True,
        catalog=[{"model": "a.gguf"}],
    )


def test_skip_global_export_when_factor_source_catalog():
    assert skip_global_vram_factor_export(
        autotune_enabled=True,
        factor_source="catalog",
    )


def test_skip_global_export_when_persisted_factor():
    assert skip_global_vram_factor_export(
        autotune_enabled=True,
        persisted_factor=1.1,
    )


def test_skip_global_export_false_when_autotune_on_but_no_signals():
    assert not skip_global_vram_factor_export(autotune_enabled=True)
