"""DI4 — SWA compact re-prefill query budget (YOCO keep-tokens analogue)."""

from __future__ import annotations

from runtime.kv.swa_compact_reprefill import (
    compact_reprefill_keep_tokens,
    compact_reprefill_query_len,
    compact_reprefill_suffix_start,
)


def test_keep_tokens_is_window_plus_backtrack():
    assert compact_reprefill_keep_tokens(sliding_window=128, draft_backtrack=12) == 140
    assert compact_reprefill_keep_tokens(sliding_window=128, draft_backtrack=0) == 128


def test_query_len_truncates_long_prompt():
    assert compact_reprefill_query_len(1000, sliding_window=128, draft_backtrack=12) == 140
    assert compact_reprefill_suffix_start(1000, sliding_window=128, draft_backtrack=12) == 860


def test_query_len_short_prompt_is_identity():
    assert compact_reprefill_query_len(64, sliding_window=128, draft_backtrack=12) == 64
    assert compact_reprefill_suffix_start(64, sliding_window=128, draft_backtrack=12) == 0


def test_no_window_means_full_prompt():
    assert compact_reprefill_query_len(500, sliding_window=0) == 500
    assert compact_reprefill_query_len(500, sliding_window=-1) == 500


def test_negative_inputs_clamped():
    assert compact_reprefill_keep_tokens(sliding_window=-5, draft_backtrack=-3) == 0
    assert compact_reprefill_query_len(-10, sliding_window=128) == 0
