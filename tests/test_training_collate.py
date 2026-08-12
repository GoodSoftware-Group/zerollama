#!/usr/bin/env python3
"""Unit tests for training_collate (padding-free resolution)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from training_collate import build_sft_collator, resolve_packing, resolve_padding_free


class TestTrainingCollate(unittest.TestCase):
    def test_padding_free_default_on(self):
        self.assertTrue(resolve_padding_free({}))

    def test_padding_free_explicit_off(self):
        self.assertFalse(resolve_padding_free({"padding_free": False}))

    def test_padding_alias_longest(self):
        self.assertFalse(resolve_padding_free({"padding": "longest"}))

    def test_packing_default_off(self):
        self.assertFalse(resolve_packing({}))
        self.assertTrue(resolve_packing({"packing": True}))

    def test_build_flattening_collator(self):
        tok = MagicMock()
        collator, mode = build_sft_collator(tok, padding_free=True)
        self.assertEqual(mode, "flattening")
        # One short + one longer sample → single flattened sequence, no pad.
        batch = collator(
            [
                {"input_ids": [1, 2, 3]},
                {"input_ids": [4, 5]},
            ]
        )
        ids = batch["input_ids"]
        # Shape [1, total] for pt tensors
        if hasattr(ids, "shape"):
            self.assertEqual(tuple(ids.shape), (1, 5))
        else:
            self.assertEqual(len(ids[0]), 5)
        self.assertIn("position_ids", batch)
        self.assertIn("labels", batch)

    def test_build_flash_flattening_collator(self):
        tok = MagicMock()
        collator, mode = build_sft_collator(
            tok, padding_free=True, flash_attn_kwargs=True
        )
        self.assertEqual(mode, "flattening_flash")
        batch = collator(
            [
                {"input_ids": [1, 2, 3]},
                {"input_ids": [4, 5]},
            ]
        )
        keys = set(batch.keys())
        self.assertTrue(
            any(k.startswith("cu_seq_lens") or k.startswith("cu_seqlens") for k in keys),
            keys,
        )

    def test_build_longest_collator(self):
        tok = MagicMock()
        tok.pad_token_id = 0
        tok.padding_side = "right"
        # DataCollatorForLanguageModeling needs a real-ish tokenizer.pad
        def pad(encoded, return_tensors=None, **kwargs):
            import torch

            max_len = max(len(x) for x in encoded["input_ids"])
            input_ids = []
            attention_mask = []
            for row in encoded["input_ids"]:
                pad_n = max_len - len(row)
                input_ids.append(row + [0] * pad_n)
                attention_mask.append([1] * len(row) + [0] * pad_n)
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
            }

        tok.pad = pad
        collator, mode = build_sft_collator(tok, padding_free=False)
        self.assertEqual(mode, "longest")


if __name__ == "__main__":
    unittest.main()
