"""Tests for Phase 15 v38 page copy descriptors."""

from __future__ import annotations

from runtime.kv.page_descriptor import (
    page_copy_descriptor,
    page_copy_descriptors_for_layers,
)


def test_page_copy_descriptor_alias_plan_same_pointer():
    page_map = {
        "page": 0,
        "block_id": 1,
        "kv_layer": 0,
        "n_cells": 16,
        "k_data": 0x1000,
        "v_data": 0x2000,
        "k_span_bytes": 4096,
        "v_span_bytes": 4096,
        "v_transposed": 0,
    }
    alias_plan = {
        "alias_ready": 1,
        "alias_mode": 1,
        "k_same_pointer": 1,
        "v_same_pointer": 1,
    }
    desc = page_copy_descriptor(page_map, alias_plan=alias_plan)
    assert desc["external_buffer_alias_ready"] is True


def test_page_copy_descriptor_no_alias_plan_defaults_false():
    page_map = {
        "page": 0,
        "block_id": 1,
        "kv_layer": 0,
        "n_cells": 16,
        "k_data": 0x1000,
        "v_data": 0x2000,
        "k_span_bytes": 4096,
        "v_span_bytes": 4096,
        "v_transposed": 0,
    }
    desc = page_copy_descriptor(page_map)
    assert desc["external_buffer_alias_ready"] is False


def test_page_copy_descriptor_k_contiguous_v_row_stride():
    page_map = {
        "page": 0,
        "block_id": 3,
        "kv_layer": 1,
        "pos_start": 0,
        "pos_end": 16,
        "n_cells": 16,
        "k_data": 0x1000,
        "v_data": 0x2000,
        "k_span_bytes": 4096,
        "v_span_bytes": 8192,
        "v_transposed": 1,
    }
    desc = page_copy_descriptor(page_map, kv_cache_kv_size=512)
    assert desc["k_copy"]["mode"] == "contiguous"
    assert desc["k_copy"]["byte_length"] == 4096
    assert desc["v_copy"]["mode"] == "row_stride"
    assert desc["v_copy"]["row_stride_elements"] == 512
    assert "scatter/gather" in desc["v_copy"]["warning"]
    assert desc["migration_ready"] is True
    assert desc["external_buffer_alias_ready"] is False


def test_page_copy_descriptor_v_contiguous_fa():
    page_map = {
        "page": 1,
        "block_id": 4,
        "kv_layer": 0,
        "n_cells": 8,
        "k_data": 0x3000,
        "v_data": 0x4000,
        "k_span_bytes": 2048,
        "v_span_bytes": 2048,
        "v_transposed": 0,
    }
    desc = page_copy_descriptor(page_map)
    assert desc["v_copy"]["mode"] == "contiguous"
    assert "Flash-attention" in desc["v_copy"]["note"]


def test_page_copy_descriptor_mla_no_v():
    page_map = {
        "page": 0,
        "block_id": 1,
        "kv_layer": 0,
        "n_cells": 16,
        "k_data": 0x5000,
        "v_data": 0,
        "k_span_bytes": 4096,
        "v_span_bytes": 0,
        "v_transposed": 0,
    }
    desc = page_copy_descriptor(page_map)
    assert desc["v_copy"]["mode"] == "absent"
    assert desc["migration_ready"] is True


def test_page_copy_descriptors_for_layers():
    maps = [
        {"page": 0, "block_id": 1, "kv_layer": 0, "n_cells": 4, "k_data": 1, "v_data": 2,
         "k_span_bytes": 100, "v_span_bytes": 100, "v_transposed": 0},
        {"page": 0, "block_id": 1, "kv_layer": 1, "n_cells": 4, "k_data": 3, "v_data": 4,
         "k_span_bytes": 100, "v_span_bytes": 100, "v_transposed": 0},
    ]
    descs = page_copy_descriptors_for_layers(maps)
    assert len(descs) == 2
    assert descs[0]["kv_layer"] == 0
    assert descs[1]["kv_layer"] == 1


def test_page_bind_health_tensor_layers_bind_complete(monkeypatch):
    from runtime.kv.page_bind import page_bind_health
    from runtime.kv.hybrid_kv_coordinator import HybridKVCacheCoordinator, LayerGroupSpec

    monkeypatch.setattr("runtime.kv.page_bind._native_page_bind_available", lambda: True)

    coord = HybridKVCacheCoordinator(
        kind="hybrid",
        layer_groups=(
            LayerGroupSpec(kind="full", layer_indices=tuple(range(26)), window=None),
            LayerGroupSpec(
                kind="sliding_window",
                layer_indices=tuple(range(26, 36)),
                window=4096,
            ),
        ),
        num_layers=36,
        num_ctx=None,
        swa_effective_window=4096,
    )
    probe = {"kv_n_layers": 26, "tensor_layers_verified": 26}
    h = page_bind_health(
        native_ext_available=True,
        tensor_probe=probe,
        kv_coordinator=coord,
    )
    assert h["tensor_layers_bind_complete"] is True
    assert h["tensor_layers_expected"] == 26

    probe_partial = {"kv_n_layers": 26, "tensor_layers_verified": 20}
    h2 = page_bind_health(
        native_ext_available=True,
        tensor_probe=probe_partial,
        kv_coordinator=coord,
    )
    assert h2["tensor_layers_bind_complete"] is False
