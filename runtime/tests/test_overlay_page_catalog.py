"""Phase 15 v51 — overlay donor page-offset catalog (unit tests)."""

from __future__ import annotations

from runtime.kv import overlay_page_catalog as cat


def test_span_in_donor_basic():
    base, size = 1000, 500
    assert cat.span_in_donor(base, size, 1000, 100) is True
    assert cat.span_in_donor(base, size, 1400, 100) is True
    assert cat.span_in_donor(base, size, 1401, 100) is False
    assert cat.span_in_donor(base, size, 999, 10) is False


def test_span_in_donor_absent_ok():
    # MLA null-V / empty span must not fail containment.
    assert cat.span_in_donor(1000, 500, 0, 0) is True
    assert cat.span_in_donor(1000, 500, 1000, 0) is True


def test_page_donor_offsets_in_range():
    base = 0x10000
    size = 0x1000
    pm = {
        "k_data": base + 64,
        "v_data": base + 128,
        "k_span_bytes": 32,
        "v_span_bytes": 32,
        "kv_layer": 0,
        "page": 2,
        "block_id": 7,
    }
    row = cat.page_donor_offsets(base, size, pm)
    assert row["in_donor"] is True
    assert row["k_offset"] == 64
    assert row["v_offset"] == 128
    assert row["block_id"] == 7
    assert row["page"] == 2


def test_page_donor_offsets_out_of_range():
    base = 0x10000
    size = 100
    pm = {
        "k_data": base + 200,
        "v_data": base + 10,
        "k_span_bytes": 16,
        "v_span_bytes": 16,
    }
    row = cat.page_donor_offsets(base, size, pm, block_id=1, page_index=0)
    assert row["in_donor"] is False
    assert row["k_offset"] is None


def test_overlay_page_catalog_summary_strips_pages():
    summary = cat.overlay_page_catalog_summary(
        {
            "donor_base": 1,
            "donor_bytes": 4096,
            "pages_checked": 2,
            "pages_in_donor": 2,
            "all_in_donor": True,
            "pages_live": 2,
            "kv_slot": 0,
            "kv_layer": 0,
            "truncated": False,
            "pages": [{"page": 0}],
        }
    )
    assert summary is not None
    assert "pages" not in summary
    assert summary["all_in_donor"] is True
    assert summary["full_plan_endpoint"] == "/internal/kv-snapshot"


def test_build_catalog_none_without_donor():
    assert (
        cat.build_overlay_page_catalog(
            donor_base=0,
            donor_size=0,
            ctx_ptr=1,
            seq_id=0,
            kv_slot=0,
            block_size=16,
        )
        is None
    )
