"""Tests for Phase 15 v39 page migration plan export."""

from __future__ import annotations

from unittest.mock import patch

from runtime.kv.page_migration_plan import build_page_migration_plan


def test_build_page_migration_plan_none_without_bind():
    probe = {"tensor_pages_bound": False, "physical_pages_bound": False, "kv_n_layers": 4}
    assert build_page_migration_plan(1, 0, 0, block_size=16, probe=probe) is None


def test_build_page_migration_plan_maps_pages_and_layers():
    probe = {
        "tensor_pages_bound": True,
        "kv_n_layers": 2,
        "llama_token_cells": 20,
        "kv_cache_kv_size": 512,
        "kv_v_transposed": 1,
    }

    def _fake_map(ctx_ptr, seq_id, kv_slot, page_index, *, kv_layer=0):
        return {
            "page": page_index,
            "block_id": page_index + 1,
            "kv_layer": kv_layer,
            "pos_start": page_index * 16,
            "pos_end": (page_index + 1) * 16,
            "n_cells": 16,
            "k_data": 0x1000 + page_index * 0x100 + kv_layer,
            "v_data": 0x2000 + page_index * 0x100 + kv_layer,
            "k_span_bytes": 512,
            "v_span_bytes": 512,
            "v_transposed": 1,
        }

    with patch("runtime.kv.page_migration_plan.map_page", side_effect=_fake_map):
        with patch(
            "runtime.kv.page_migration_plan.export_page_table",
            return_value=[{"page": 0}, {"page": 1}],
        ):
            with patch("runtime.kv.page_migration_plan.tensor_probe_available", return_value=True):
                plan = build_page_migration_plan(99, 0, 3, block_size=16, probe=probe)

    assert plan is not None
    assert plan["pages_live"] == 2
    assert plan["n_layers"] == 2
    assert plan["migration_pages_complete"] is True
    assert len(plan["pages"]) == 2
    assert plan["pages"][0]["layers_mapped"] == 2
    assert plan["pages"][0]["layers"][0]["v_copy"]["mode"] == "row_stride"
    assert plan["external_buffer_alias_ready"] is False


def test_kv_snapshot_includes_page_migration(tmp_path):
    from pathlib import Path

    from runtime.config import RuntimeConfig
    from runtime.engine import InferenceEngine

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=tmp_path,
        llama_server_bin=None,
        llama_model=None,
        num_blocks=64,
        block_size=16,
        device_count=1,
    )
    eng = InferenceEngine(cfg)
    snap = eng.kv_snapshot()
    assert "kv_page_migration" in snap


def test_migration_plan_summary_without_map_page():
    from runtime.kv.page_migration_plan import migration_plan_summary

    probe = {
        "tensor_pages_bound": True,
        "physical_pages_bound": True,
        "kv_n_layers": 4,
        "tensor_layers_verified": 4,
        "llama_token_cells": 32,
        "kv_v_transposed": 1,
        "kv_cache_kv_size": 512,
    }
    with patch(
        "runtime.kv.page_migration_plan.export_page_table",
        return_value=[{"page": 0}, {"page": 1}],
    ):
        summary = migration_plan_summary(probe, block_size=16, kv_slot=0)
    assert summary is not None
    assert summary["pages_live"] == 2
    assert summary["tensor_layers_bind_complete"] is True
    assert summary["full_plan_endpoint"] == "/internal/kv-snapshot"
    assert "src_ptr" not in str(summary)


def test_redact_migration_plan_strips_pointers():
    from runtime.kv.page_migration_plan import redact_migration_plan

    plan = {
        "pages": [
            {
                "layers": [
                    {
                        "k_copy": {"mode": "contiguous", "src_ptr": 4096, "byte_length": 100},
                        "v_copy": {"mode": "row_stride", "src_ptr": 8192, "byte_length": 100},
                    }
                ]
            }
        ]
    }
    redacted = redact_migration_plan(plan)
    assert "src_ptr" not in redacted["pages"][0]["layers"][0]["k_copy"]
    assert redacted["pages"][0]["layers"][0]["k_copy"]["byte_length"] == 100


def test_prepare_migration_plan_respects_env(monkeypatch):
    from runtime.kv.page_migration_plan import prepare_migration_plan_for_export

    plan = {
        "pages": [{"layers": [{"k_copy": {"src_ptr": 1, "byte_length": 2}}]}]
    }
    monkeypatch.delenv("ZEROLLAMA_KV_MIGRATION_INCLUDE_PTRS", raising=False)
    assert "src_ptr" not in prepare_migration_plan_for_export(plan)["pages"][0]["layers"][0]["k_copy"]
    monkeypatch.setenv("ZEROLLAMA_KV_MIGRATION_INCLUDE_PTRS", "1")
    assert prepare_migration_plan_for_export(plan)["pages"][0]["layers"][0]["k_copy"]["src_ptr"] == 1


def test_kv_snapshot_migration_summary_from_last_probe(tmp_path):
    from pathlib import Path
    from unittest.mock import patch

    from runtime.config import RuntimeConfig
    from runtime.engine import InferenceEngine

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=tmp_path,
        llama_server_bin=None,
        llama_model=None,
        num_blocks=64,
        block_size=16,
        device_count=1,
    )
    eng = InferenceEngine(cfg)
    probe = {
        "tensor_pages_bound": True,
        "physical_pages_bound": True,
        "kv_n_layers": 2,
        "tensor_layers_verified": 2,
        "llama_token_cells": 20,
    }
    with patch(
        "runtime.kv.tensor_probe.tensor_probe_available",
        return_value=True,
    ):
        with patch.object(eng, "_inprocess_ctx_for_health", return_value=None):
            with patch(
                "runtime.kv.page_bind.page_bind_last_probe_row_for_health",
                return_value={"kv_slot": 1, "probe": probe},
            ):
                with patch(
                    "runtime.kv.page_bind.page_bind_last_tensor_probe_for_health",
                    return_value=probe,
                ):
                    with patch(
                        "runtime.kv.page_migration_plan.export_page_table",
                        return_value=[{"page": 0}],
                    ):
                        snap = eng.kv_snapshot()
    mig = snap["kv_page_migration"]
    assert mig is not None
    assert mig["source"] == "last_tensor_probe"
    assert "migration_summary" in mig
    assert mig["migration_summary"]["pages_live"] == 1
