"""Phase 15 v9–v10 — decode prefill + decode step plan export."""

from __future__ import annotations

from runtime.kv.decode_plan import (
    kv_decode_prefill_plan,
    kv_decode_step_plan,
    next_pos_from_llama,
)
from runtime.kv.forward_plan import kv_forward_plan
from runtime.kv.native_decode_batch import native_decode_batch_available
from runtime.scheduler.scheduler import Request


def test_decode_prefill_plan_empty():
    out = kv_decode_prefill_plan([], block_size=16, kv_slot=0)
    assert out["n_prefill_batches"] == 0
    assert out["prefill_chunks"] == []


def test_decode_prefill_plan_single_chunk():
    tokens = list(range(10))
    out = kv_decode_prefill_plan(tokens, block_size=16, kv_slot=1, seq_id=1)
    assert out["n_prefill_batches"] == 1
    chunk = out["prefill_chunks"][0]
    assert chunk["token_count"] == 10
    assert chunk["pos_start"] == 0
    assert chunk["pos_end"] == 9
    assert chunk["page_range"] == [0, 0]
    assert chunk["logits_last"] is True


def test_decode_prefill_plan_page_chunks_when_native():
    tokens = list(range(40))
    out = kv_decode_prefill_plan(tokens, block_size=16, kv_slot=2)
    if native_decode_batch_available():
        assert out["n_prefill_batches"] == 3
        assert out["layout_source"] == "native"
        assert out["prefill_chunks"][0]["batch_layout"]["n_tokens"] == 16
        assert out["prefill_chunks"][-1]["pos_end"] == 39
        for c in out["prefill_chunks"][:-1]:
            assert c["logits_last"] is False, c
        assert out["prefill_chunks"][-1]["logits_last"] is True
    else:
        assert out["n_prefill_batches"] == 1
        assert out["prefill_chunks"][0]["logits_last"] is False
        assert out["layout_source"] == "python"


def test_decode_prefill_plan_logits_last_on_final_chunk_only():
    """Final prefill chunk logits_last=True (v15+ / v23); intermediates False."""
    tokens = list(range(50))  # 4 chunks at block_size=16 when native batch built
    out = kv_decode_prefill_plan(tokens, block_size=16)
    chunks = out["prefill_chunks"]
    if len(chunks) <= 1:
        assert chunks[0]["logits_last"] is True
        return
    for c in chunks[:-1]:
        assert c["logits_last"] is False, c
    assert chunks[-1]["logits_last"] is True


def test_decode_prefill_plan_matches_execute_chunks():
    """Export uses the same helper as libllama_ctypes prefill execution."""
    from runtime.kv.decode_plan import iter_prefill_execute_chunks

    for n_tokens in [1, 10, 16, 17, 32, 33, 48, 50]:
        tokens = list(range(n_tokens))
        plan = kv_decode_prefill_plan(tokens, block_size=16)
        exec_chunks = iter_prefill_execute_chunks(tokens, block_size=16)
        assert plan["n_prefill_batches"] == len(exec_chunks)
        for entry, (et, ep, el) in zip(plan["prefill_chunks"], exec_chunks):
            assert entry["token_count"] == len(et)
            assert entry["pos_start"] == ep
            assert entry["logits_last"] is el


def test_forward_plan_includes_decode_prefill():
    req = Request(
        request_id="dp1",
        prompt_tokens=[0] * 20,
        max_tokens=8,
        num_ctx=64,
        kv_slot=0,
    )
    req.block_table = type(
        "T",
        (),
        {
            "_tables": [type("BT", (), {"block_ids": [1, 2]})()],
            "num_tokens_capacity": 32,
        },
    )()
    plan = kv_forward_plan(req, block_size=16)
    assert "decode_prefill" in plan
    dp = plan["decode_prefill"]
    assert dp["n_prefill_batches"] >= 1
    assert dp["page_bind_slot"] == 0
    assert plan["decode_work"]["phase"] == "admit"


def test_decode_prefill_plan_page_range_exact_boundary():
    """page_range endpoints are correct at exact block boundaries."""
    # 32 tokens at block_size=16 → 2 chunks: [0..15], [16..31]
    out = kv_decode_prefill_plan(list(range(32)), block_size=16)
    chunks = out["prefill_chunks"]
    # Non-native produces one chunk spanning both pages
    if not native_decode_batch_available():
        assert chunks[0]["page_range"] == [0, 1]
        return
    assert len(chunks) == 2
    assert chunks[0]["page_range"] == [0, 0]   # tokens 0-15 → page 0
    assert chunks[1]["page_range"] == [1, 1]   # tokens 16-31 → page 1
    # 33 tokens → 3 chunks: [0..15], [16..31], [32..32]
    out2 = kv_decode_prefill_plan(list(range(33)), block_size=16)
    c2 = out2["prefill_chunks"]
    assert len(c2) == 3
    assert c2[2]["page_range"] == [2, 2]        # token 32 → page 2


def test_forward_plan_omits_decode_prefill_without_block_table():
    req = Request(
        request_id="dp2",
        prompt_tokens=[0] * 5,
        max_tokens=4,
    )
    plan = kv_forward_plan(req, block_size=16)
    assert "decode_prefill" not in plan


def test_next_pos_from_llama():
    assert next_pos_from_llama(-1) == 0
    assert next_pos_from_llama(19) == 20


def test_decode_prefill_plan_remaining_from_pos_start():
    """v10: plan only remaining prompt tokens from current_pos."""
    tokens = list(range(40))
    out = kv_decode_prefill_plan(tokens, block_size=16, pos_start=16)
    assert out.get("prefill_complete") is False
    assert out["pos_start"] == 16
    if native_decode_batch_available():
        assert out["n_prefill_batches"] == 2
        assert out["prefill_chunks"][0]["pos_start"] == 16
        assert out["prefill_chunks"][-1]["pos_end"] == 39
    else:
        assert out["n_prefill_batches"] == 1
        assert out["prefill_chunks"][0]["pos_start"] == 16


def test_decode_prefill_plan_prefill_complete():
    out = kv_decode_prefill_plan(list(range(20)), block_size=16, pos_start=20)
    assert out["prefill_complete"] is True
    assert out["n_prefill_batches"] == 0
    assert out["prefill_chunks"] == []


def test_decode_step_plan_after_prefill():
    out = kv_decode_step_plan(
        current_pos=20,
        n_prompt=20,
        max_tokens=8,
        block_size=16,
        kv_slot=0,
    )
    assert out.get("pending_prefill") is not True
    assert out["tokens_generated"] == 0
    assert out["n_decode_batches_remaining"] == 8
    assert out["step"]["token_count"] == 1
    assert out["step"]["logits_last"] is True
    assert out["pos_range"] == [20, 27]
    assert out["page_range"] == [1, 1]


def test_decode_step_plan_pending_prefill():
    out = kv_decode_step_plan(
        current_pos=10,
        n_prompt=20,
        max_tokens=8,
        block_size=16,
    )
    assert out["pending_prefill"] is True
    assert out["current_pos"] == 10


def test_decode_step_plan_mid_generation():
    out = kv_decode_step_plan(
        current_pos=23,
        n_prompt=20,
        max_tokens=8,
        block_size=16,
    )
    assert out["tokens_generated"] == 3
    assert out["n_decode_batches_remaining"] == 5
    assert out["pos_range"] == [23, 27]


def test_forward_plan_no_decode_steps_without_current_pos():
    """Without current_pos, decode_steps and plan_current_pos are absent."""
    req = Request(
        request_id="dp4",
        prompt_tokens=[0] * 10,
        max_tokens=4,
        kv_slot=0,
    )
    req.block_table = type(
        "T",
        (),
        {
            "_tables": [type("BT", (), {"block_ids": [1]})()],
            "num_tokens_capacity": 16,
        },
    )()
    plan = kv_forward_plan(req, block_size=16)
    assert "decode_steps" not in plan
    assert "plan_current_pos" not in plan
    assert "decode_prefill" in plan
    # decode_work is always present when admitted; without live pos it is admit phase
    assert plan["decode_work"]["phase"] == "admit"


def test_decode_step_plan_pos_range_crosses_page():
    """pos_range and page_range span two pages when decode steps cross a page boundary."""
    # current_pos=15, 4 steps → pos 15..18, page boundary at 16
    out = kv_decode_step_plan(
        current_pos=15,
        n_prompt=15,
        max_tokens=4,
        block_size=16,
    )
    assert out["pos_range"] == [15, 18]
    assert out["page_range"] == [0, 1]


def test_forward_plan_in_progress_with_current_pos():
    req = Request(
        request_id="dp3",
        prompt_tokens=[0] * 20,
        max_tokens=8,
        num_ctx=64,
        kv_slot=1,
    )
    req.block_table = type(
        "T",
        (),
        {
            "_tables": [type("BT", (), {"block_ids": [1, 2]})()],
            "num_tokens_capacity": 32,
        },
    )()
    plan = kv_forward_plan(req, block_size=16, current_pos=20)
    assert plan["plan_current_pos"] == 20
    assert plan["decode_prefill"]["prefill_complete"] is True
    assert "decode_steps" not in plan   # decode_work.decode carries it
    assert plan["decode_work"]["decode"]["n_decode_batches_remaining"] == 8
    assert plan["decode_work"]["phase"] == "decode"
