#!/usr/bin/env python3
"""Unit tests for training_format (no torch required)."""

from __future__ import annotations

import unittest

from training_format import (
    format_alpaca,
    format_chatml,
    format_llama3,
    format_sft_sample,
    resolve_format_mode,
)


class FakeTok:
    def __init__(self, chat_template=None, name_or_path=""):
        self.chat_template = chat_template
        self.name_or_path = name_or_path

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        parts = []
        for m in messages:
            parts.append(f"[{m['role']}]{m['content']}")
        return "|".join(parts)


class TestTrainingFormat(unittest.TestCase):
    def test_alpaca(self):
        s = format_alpaca({"prompt": "Hi", "response": "Hello"})
        self.assertIn("### Instruction:\nHi", s)
        self.assertIn("### Response:\nHello", s)

    def test_chatml(self):
        s = format_chatml({"prompt": "Hi", "response": "Hello", "system": "Be nice"})
        self.assertIn("<|im_start|>system\nBe nice<|im_end|>", s)
        self.assertIn("<|im_start|>user\nHi<|im_end|>", s)
        self.assertIn("<|im_start|>assistant\nHello<|im_end|>", s)

    def test_messages_list(self):
        s = format_chatml(
            {
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        )
        self.assertIn("<|im_start|>user\nu<|im_end|>", s)
        self.assertIn("<|im_start|>assistant\na<|im_end|>", s)

    def test_llama3(self):
        s = format_llama3({"prompt": "Q", "response": "A"})
        self.assertIn("<|begin_of_text|>", s)
        self.assertIn("<|start_header_id|>user<|end_header_id|>", s)
        self.assertIn("Q<|eot_id|>", s)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>", s)

    def test_resolve_auto_prefers_hf_template(self):
        tok = FakeTok(chat_template="x", name_or_path="foo")
        self.assertEqual(resolve_format_mode({}, tok), "hf")

    def test_resolve_qwen_chatml(self):
        tok = FakeTok(name_or_path="Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(resolve_format_mode({"format": "auto"}, tok), "chatml")

    def test_resolve_explicit_alpaca(self):
        self.assertEqual(resolve_format_mode({"format": "alpaca"}, None), "alpaca")

    def test_hf_apply(self):
        tok = FakeTok(chat_template="yes")
        s = format_sft_sample(
            {"prompt": "p", "response": "r"},
            mode="hf",
            tokenizer=tok,
        )
        self.assertEqual(s, "[user]p|[assistant]r")

    def test_default_auto_is_chatml_not_alpaca(self):
        s = format_sft_sample({"prompt": "p", "response": "r"}, mode="auto")
        self.assertIn("<|im_start|>", s)
        self.assertNotIn("### Instruction", s)


if __name__ == "__main__":
    unittest.main()
