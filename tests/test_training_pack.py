#!/usr/bin/env python3
"""Unit tests for training_pack (no torch required)."""

from __future__ import annotations

import unittest

from training_pack import pack_token_id_lists, packing_stats


class TestTrainingPack(unittest.TestCase):
    def test_packs_short_sequences(self):
        seqs = [[1, 2], [3, 4], [5]]
        out = pack_token_id_lists(seqs, max_length=5, eos_token_id=9)
        # [1,2,9] + [3,4,9] would be 6 > 5 → first block [1,2,9], then [3,4,9], then [5,9]
        self.assertEqual(out["input_ids"][0], [1, 2, 9])
        self.assertEqual(len(out["input_ids"][0]), len(out["attention_mask"][0]))
        self.assertEqual(out["labels"][0], out["input_ids"][0])

    def test_single_long_sequence(self):
        seqs = [[1, 2, 3, 4, 5, 6, 7]]
        out = pack_token_id_lists(seqs, max_length=4, eos_token_id=None)
        self.assertEqual(out["input_ids"], [[1, 2, 3, 4]])

    def test_fills_block(self):
        seqs = [[1, 2], [3, 4]]
        out = pack_token_id_lists(seqs, max_length=4, eos_token_id=None)
        self.assertEqual(out["input_ids"], [[1, 2, 3, 4]])

    def test_stats(self):
        packed = {"input_ids": [[1, 2, 3], [4, 5]]}
        s = packing_stats(10, packed)
        self.assertEqual(s["samples_in"], 10)
        self.assertEqual(s["blocks_out"], 2)
        self.assertEqual(s["pack_ratio"], 5.0)

    def test_empty_skipped(self):
        out = pack_token_id_lists([[], [1]], max_length=8, eos_token_id=2)
        self.assertEqual(out["input_ids"], [[1, 2]])


if __name__ == "__main__":
    unittest.main()
