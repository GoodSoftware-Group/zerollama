"""Sample packing for causal LM SFT (ROADMAP T8 / Unsloth-inspired).

Concatenates short tokenized rows (with EOS separators) into ``max_length``
blocks so each training step sees more real tokens and less pad waste.

Why opt-in (``packing: true``): packed attention sees prior samples in the
same block (standard concat packing, not flash-attn document isolation).
Loss curves differ from the unpacked path — keep default off until operators
opt in. Not bit-identical to Unsloth "uncontaminated" packing.

T9: ``pack_labeled_token_id_lists`` preserves per-sample completion-only
labels across pack boundaries (prompt ``-100`` stays masked inside blocks).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


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
    labeled: List[Tuple[List[int], List[int]]] = []
    for raw in sequences:
        seq = list(raw)
        if not seq:
            continue
        labeled.append((seq, list(seq)))
    # Full-token loss: appended pack EOS stays trainable (mirror input_ids).
    eos_lab = eos_token_id if eos_token_id is not None else -100
    return pack_labeled_token_id_lists(
        labeled,
        max_length,
        eos_token_id=eos_token_id,
        appended_eos_label=eos_lab,
    )


def pack_labeled_token_id_lists(
    sequences: Sequence[Tuple[Sequence[int], Sequence[int]]],
    max_length: int,
    *,
    eos_token_id: Optional[int] = None,
    appended_eos_label: int = -100,
) -> Dict[str, List[List[int]]]:
    """Greedy pack ``(input_ids, labels)`` pairs into ``max_length`` blocks.

    WHY labeled pack: completion-only loss must keep prompt ``-100`` masks when
    short SFT rows are concatenated. Plain ``pack_token_id_lists`` mirrored
    ``input_ids`` into labels and dropped masks (T9 gap).

    Appended pack-boundary EOS defaults to label ``-100`` (completion-only) so
    the separator is not trained as a response token. Full-token packing passes
    ``appended_eos_label=eos_token_id`` to preserve prior T8 semantics.
    """
    if max_length < 1:
        raise ValueError("max_length must be >= 1")

    packed_ids: List[List[int]] = []
    packed_labels: List[List[int]] = []
    buf_ids: List[int] = []
    buf_labels: List[int] = []

    def _flush() -> None:
        nonlocal buf_ids, buf_labels
        if buf_ids:
            packed_ids.append(buf_ids)
            packed_labels.append(buf_labels)
            buf_ids = []
            buf_labels = []

    for raw_ids, raw_labels in sequences:
        ids = list(raw_ids)
        labels = list(raw_labels)
        if not ids:
            continue
        if len(labels) != len(ids):
            # Truncate/pad labels to ids length; missing tail mirrors ids.
            if len(labels) < len(ids):
                labels = labels + ids[len(labels) :]
            else:
                labels = labels[: len(ids)]
        if eos_token_id is not None and ids[-1] != eos_token_id:
            ids = ids + [eos_token_id]
            labels = labels + [int(appended_eos_label)]
        if len(ids) > max_length:
            ids = ids[:max_length]
            labels = labels[:max_length]

        if len(ids) == max_length:
            _flush()
            packed_ids.append(ids)
            packed_labels.append(labels)
            continue

        if buf_ids and len(buf_ids) + len(ids) > max_length:
            _flush()
        buf_ids.extend(ids)
        buf_labels.extend(labels)

    _flush()

    attention = [[1] * len(x) for x in packed_ids]
    return {
        "input_ids": packed_ids,
        "attention_mask": attention,
        "labels": packed_labels,
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
