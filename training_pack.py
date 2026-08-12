"""Sample packing for causal LM SFT (ROADMAP T8 / Unsloth-inspired).

Concatenates short tokenized rows (with EOS separators) into ``max_length``
blocks so each training step sees more real tokens and less pad waste.

Why opt-in (``packing: true``): packed attention sees prior samples in the
same block (standard concat packing, not flash-attn document isolation).
Loss curves differ from the unpacked path — keep default off until operators
opt in. Not bit-identical to Unsloth "uncontaminated" packing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def pack_token_id_lists(
    sequences: Sequence[Sequence[int]],
    max_length: int,
    *,
    eos_token_id: Optional[int] = None,
) -> Dict[str, List[List[int]]]:
    """Greedy pack token-id sequences into blocks of at most ``max_length``.

    Each input sequence is truncated to ``max_length`` first. An EOS id is
    appended between samples when provided and not already present.
    """
    if max_length < 1:
        raise ValueError("max_length must be >= 1")

    packed: List[List[int]] = []
    buf: List[int] = []

    for raw in sequences:
        seq = list(raw)
        if not seq:
            continue
        if eos_token_id is not None and seq[-1] != eos_token_id:
            seq = seq + [eos_token_id]
        if len(seq) > max_length:
            seq = seq[:max_length]

        if len(seq) == max_length:
            if buf:
                packed.append(buf)
                buf = []
            packed.append(seq)
            continue

        if buf and len(buf) + len(seq) > max_length:
            packed.append(buf)
            buf = []
        buf.extend(seq)

    if buf:
        packed.append(buf)

    attention = [[1] * len(x) for x in packed]
    # Causal LM: labels mirror input_ids; collator may still shift.
    labels = [list(x) for x in packed]
    return {
        "input_ids": packed,
        "attention_mask": attention,
        "labels": labels,
    }


def tokenize_and_pack(
    texts: Sequence[str],
    tokenizer: Any,
    max_length: int,
) -> Dict[str, List[List[int]]]:
    """Tokenize texts (no pad) then pack into ``max_length`` blocks."""
    eos_id = getattr(tokenizer, "eos_token_id", None)
    sequences: List[List[int]] = []
    for text in texts:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
        )
        ids = enc["input_ids"]
        if isinstance(ids[0], list):
            ids = ids[0]
        sequences.append(list(ids))
    return pack_token_id_lists(sequences, max_length, eos_token_id=eos_id)


def packing_stats(
    before_n: int,
    packed: Dict[str, List[List[int]]],
) -> Dict[str, Any]:
    after_n = len(packed.get("input_ids") or [])
    lengths = [len(x) for x in packed.get("input_ids") or []]
    total_tok = sum(lengths)
    return {
        "samples_in": before_n,
        "blocks_out": after_n,
        "total_tokens": total_tok,
        "mean_block_len": (total_tok / after_n) if after_n else 0.0,
        "pack_ratio": (before_n / after_n) if after_n else 0.0,
    }
