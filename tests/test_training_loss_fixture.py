#!/usr/bin/env python3
"""Unit tests for T8 training loss-curve fixture (CPU, tiny GPT-2)."""

from __future__ import annotations

import unittest

from training_loss_fixture import (
    assert_curves_close,
    corpus_texts,
    load_corpus,
    run_loss_curve,
    _FixtureTokenizer,
    _tokenize_rows,
)
from training_pack import packing_stats


class TestTrainingLossFixture(unittest.TestCase):
    def test_corpus_stable(self):
        data = load_corpus()
        texts, max_length = corpus_texts(data)
        self.assertEqual(len(texts), 8)
        self.assertEqual(max_length, 32)
        self.assertTrue(texts[0].startswith("Q: hi"))

    def test_packing_golden_stats(self):
        """Packing block count on the fixed corpus must stay stable."""
        texts, max_length = corpus_texts()
        tok = _FixtureTokenizer()
        rows = _tokenize_rows(texts, tok, max_length, packing=True)
        stats = packing_stats(len(texts), {"input_ids": [r["input_ids"] for r in rows]})
        # 8 short samples → fewer blocks than samples
        self.assertEqual(stats["samples_in"], 8)
        self.assertLess(stats["blocks_out"], stats["samples_in"])
        self.assertGreater(stats["pack_ratio"], 1.0)
        # Exact golden for this tokenizer + corpus + max_length=32
        self.assertEqual(stats["blocks_out"], 4)
        self.assertEqual(stats["pack_ratio"], 2.0)

    def test_baseline_deterministic(self):
        a = run_loss_curve(
            packing=False, padding_free=False, batch_size=1, steps=3, seed=7
        )
        b = run_loss_curve(
            packing=False, padding_free=False, batch_size=1, steps=3, seed=7
        )
        assert_curves_close(a["losses"], b["losses"], label="baseline determinism")

    def test_bs1_padding_free_matches_longest(self):
        """No pads at batch_size=1 → flattening ≡ longest (training hygiene)."""
        longest = run_loss_curve(
            packing=False, padding_free=False, batch_size=1, steps=4, seed=0
        )
        flat = run_loss_curve(
            packing=False, padding_free=True, batch_size=1, steps=4, seed=0
        )
        self.assertEqual(longest["collate"], "longest")
        self.assertEqual(flat["collate"], "flattening")
        assert_curves_close(
            longest["losses"],
            flat["losses"],
            atol=1e-5,
            rtol=1e-4,
            label="bs1 padding_free vs longest",
        )

    def test_packing_deterministic(self):
        a = run_loss_curve(
            packing=True, padding_free=True, batch_size=1, steps=3, seed=3
        )
        b = run_loss_curve(
            packing=True, padding_free=True, batch_size=1, steps=3, seed=3
        )
        assert_curves_close(a["losses"], b["losses"], label="packing determinism")
        self.assertEqual(a["n_rows"], 4)

    def test_packing_may_diverge_from_unpacked(self):
        """Concat packing is not loss-identical to unpacked (documented)."""
        unpacked = run_loss_curve(
            packing=False, padding_free=True, batch_size=1, steps=2, seed=1
        )
        packed = run_loss_curve(
            packing=True, padding_free=True, batch_size=1, steps=2, seed=1
        )
        self.assertEqual(len(unpacked["losses"]), 2)
        self.assertEqual(len(packed["losses"]), 2)
        # Same seed / model init, different data layout → expect divergence.
        # If they ever match by chance, still OK as long as both are finite.
        for x in unpacked["losses"] + packed["losses"]:
            self.assertTrue(x == x and abs(x) < 1e6)  # finite


if __name__ == "__main__":
    unittest.main()
