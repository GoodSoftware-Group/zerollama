"""SWA reachable-tail store filter (vLLM #48911 pattern).

Lookup already clamps to the sliding window; store must not federate / pool
blocks that can never serve a hit under the same alignment rules.
"""

from __future__ import annotations

from typing import Sequence


def is_store_reachable_swa_chunk(
    absolute_chunk_index: int,
    storable_chunk_count: int,
    alignment_chunk_count: int | None,
    sliding_window_chunks: int | None,
    is_eagle_group: bool,
) -> bool:
    """Return whether an SWA chunk can participate in an external-cache hit.

    Port of vLLM ``is_store_reachable_swa_chunk`` — uses **actual** segment
    length for partial final segments (not always ``alignment_chunk_count``).
    """
    if alignment_chunk_count is None:
        return True
    if sliding_window_chunks is None:
        return True
    if alignment_chunk_count <= 0 or storable_chunk_count <= 0:
        return True
    position_in_segment = absolute_chunk_index % alignment_chunk_count
    segment_start = absolute_chunk_index - position_in_segment
    actual_segment_length = min(
        alignment_chunk_count, storable_chunk_count - segment_start
    )
    if actual_segment_length <= 0:
        return False
    reachable_tail = sliding_window_chunks + int(is_eagle_group)
    if reachable_tail >= actual_segment_length:
        return True
    return position_in_segment >= actual_segment_length - reachable_tail


def _keep_junction_and_prompt_end(
    mask: list[bool],
    *,
    num_blocks: int,
    draft_extra: bool,
    shared_prefix_blocks: int,
    prompt_end_blocks: int,
    retention_interval: int | None,
) -> None:
    """Force-store EAGLE junction-1 and optional prompt-end checkpoint."""
    if draft_extra:
        junction = int(shared_prefix_blocks) - 1
        if 0 <= junction < num_blocks:
            mask[junction] = True
    # vLLM #59146: when SWA retention is block-aligned only, still keep the
    # last full prompt block so resume can land on the prompt/decode join.
    if not retention_interval:
        pe = int(prompt_end_blocks) - 1
        if 0 <= pe < num_blocks:
            mask[pe] = True


def swa_reachable_store_mask(
    num_blocks: int,
    *,
    block_size: int,
    sliding_window: int | None,
    retention_interval: int | None = None,
    draft_extra: bool = False,
    alignment_tokens: int | None = None,
    shared_prefix_blocks: int = 0,
    prompt_end_blocks: int = 0,
    extra_retained_tokens: int = 0,
) -> list[bool] | None:
    """Per-block store mask; ``None`` means store every full block (dense).

    Without ``alignment_tokens``: keep the last ``ceil(window/block_size)``
    blocks (+1 if draft/EAGLE), plus optional retention-interval boundary tails.

    With ``alignment_tokens``: per-chunk filter via ``is_store_reachable_swa_chunk``.

    ``shared_prefix_blocks`` (vLLM #62206): with draft/EAGLE, also keep the last
    block of the hash-matched prefix (attention junction-1), which may sit
    outside the current SWA tail after a long decode.

    ``prompt_end_blocks`` (vLLM #59146 analog): when ``retention_interval`` is
    unset/0, also keep the last full prompt block as a resume checkpoint.

    ``extra_retained_tokens`` (vLLM #61455): MTP/EAGLE prefill lookahead kept
    inside the SWA tail in addition to ``sliding_window``.
    """
    if num_blocks <= 0:
        return None
    bs = max(1, int(block_size))
    if sliding_window is None or sliding_window <= 0:
        return None

    window_tokens = int(sliding_window) + max(0, int(extra_retained_tokens))
    window_chunks = max(1, (window_tokens + bs - 1) // bs)
    need = window_chunks + int(draft_extra)

    if alignment_tokens is not None and alignment_tokens > 0:
        align_chunks = max(1, int(alignment_tokens) // bs)
        mask = [
            is_store_reachable_swa_chunk(
                i,
                num_blocks,
                align_chunks,
                window_chunks,
                draft_extra,
            )
            for i in range(num_blocks)
        ]
        if retention_interval is not None and retention_interval > 0:
            per = max(1, int(retention_interval) // bs)
            for i in range(num_blocks):
                if (i + 1) % per == 0:
                    for j in range(max(0, i + 1 - need), i + 1):
                        mask[j] = True
        _keep_junction_and_prompt_end(
            mask,
            num_blocks=num_blocks,
            draft_extra=draft_extra,
            shared_prefix_blocks=shared_prefix_blocks,
            prompt_end_blocks=prompt_end_blocks,
            retention_interval=retention_interval,
        )
        if all(mask):
            return None
        return mask

    if need >= num_blocks:
        return None
    mask = [False] * num_blocks
    for i in range(num_blocks - need, num_blocks):
        mask[i] = True
    if retention_interval is not None and retention_interval > 0:
        per = max(1, int(retention_interval) // bs)
        for i in range(num_blocks):
            if (i + 1) % per == 0:
                for j in range(max(0, i + 1 - need), i + 1):
                    mask[j] = True
    _keep_junction_and_prompt_end(
        mask,
        num_blocks=num_blocks,
        draft_extra=draft_extra,
        shared_prefix_blocks=shared_prefix_blocks,
        prompt_end_blocks=prompt_end_blocks,
        retention_interval=retention_interval,
    )
    if all(mask):
        return None
    return mask


def filter_block_indices(
    mask: list[bool] | None,
    indices: Sequence[int],
) -> list[int]:
    if mask is None:
        return list(indices)
    return [i for i in indices if 0 <= i < len(mask) and mask[i]]
