#!/usr/bin/env python3
"""Unit tests for T9 training_optim + completion-only labels."""

from __future__ import annotations

import unittest

from training_labels import (
    format_sft_prompt_and_full,
    mask_labels_completion_only,
)
from training_optim import (
    build_lora_kwargs,
    resolve_completion_only_loss,
    resolve_gradient_checkpointing,
    resolve_optim,
    resolve_use_rslora,
)


class TestTrainingOptim(unittest.TestCase):
    def test_grad_ckpt_default_cuda(self):
        self.assertTrue(resolve_gradient_checkpointing({}, device="cuda"))
        self.assertFalse(resolve_gradient_checkpointing({}, device="mps"))
        self.assertFalse(
            resolve_gradient_checkpointing({"gradient_checkpointing": False}, device="cuda")
        )

    def test_optim_defaults(self):
        self.assertEqual(resolve_optim({}, use_qlora=False, device="cuda"), "adamw_torch_fused")
        self.assertEqual(resolve_optim({}, use_qlora=True, device="cuda"), "adamw_bnb_8bit")
        self.assertEqual(resolve_optim({}, use_qlora=False, device="mps"), "adamw_torch")

    def test_rslora_default_on(self):
        self.assertTrue(resolve_use_rslora({}))
        self.assertFalse(resolve_use_rslora({"use_rslora": False}))

    def test_completion_only_default_on(self):
        self.assertTrue(resolve_completion_only_loss({}))
        self.assertFalse(resolve_completion_only_loss({"completion_only_loss": False}))

    def test_lora_kwargs_rslora(self):
        kw = build_lora_kwargs({}, lora_rank=8, lora_alpha=16)
        self.assertTrue(kw["use_rslora"])
        self.assertEqual(kw["r"], 8)
        self.assertIn("q_proj", kw["target_modules"])


class TestTrainingLabels(unittest.TestCase):
    def test_chatml_prefix(self):
        prompt, full = format_sft_prompt_and_full(
            {"prompt": "Hi", "response": "Hello"},
            mode="chatml",
        )
        self.assertTrue(full.startswith(prompt))
        self.assertIn("Hello", full[len(prompt) :])
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_alpaca_prefix(self):
        prompt, full = format_sft_prompt_and_full(
            {"prompt": "Q", "response": "A"},
            mode="alpaca",
        )
        self.assertEqual(full, prompt + "A")
        self.assertTrue(prompt.endswith("### Response:\n"))

    def test_mask_labels(self):
        ids = [1, 2, 3, 4, 5]
        labels = mask_labels_completion_only(ids, [1, 2, 3])
        self.assertEqual(labels, [-100, -100, -100, 4, 5])


if __name__ == "__main__":
    unittest.main()
