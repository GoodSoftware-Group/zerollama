#!/usr/bin/env python3
"""Tests for format=modelfile (Go TEMPLATE via zerollama template render)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from training_format import resolve_format_mode
from training_modelfile import find_zerollama_bin, format_modelfile, resolve_modelfile_template


REPO = Path(__file__).resolve().parents[1]


@unittest.skipUnless(find_zerollama_bin(), "zerollama binary not built (ZEROLLAMA_BIN)")
class TestModelfileFormat(unittest.TestCase):
    def test_resolve_mode(self):
        self.assertEqual(resolve_format_mode({"format": "modelfile"}), "modelfile")
        self.assertEqual(resolve_format_mode({"format": "gotmpl"}), "modelfile")

    def test_stock_chatml_matches_filled_turns(self):
        req = {"format": "modelfile", "template_name": "chatml"}
        tmpl = resolve_modelfile_template(req)
        self.assertIn("im_start", tmpl)
        text = format_modelfile(
            {"prompt": "Hi", "response": "Hello", "system": "Be nice"},
            req,
        )
        self.assertIn("<|im_start|>system\nBe nice<|im_end|>", text)
        self.assertIn("<|im_start|>user\nHi<|im_end|>", text)
        self.assertIn("<|im_start|>assistant\nHello<|im_end|>", text)
        self.assertFalse(text.rstrip().endswith("<|im_start|>assistant"))

    def test_inline_template(self):
        req = {
            "format": "modelfile",
            "template": "{{- range .Messages }}[{{ .Role }}:{{ .Content }}]{{ end }}>>",
        }
        text = format_modelfile(
            {
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ]
            },
            req,
        )
        self.assertEqual(text, "[user:u][assistant:a]>>")


class TestModelfileResolve(unittest.TestCase):
    def test_stock_file(self):
        t = resolve_modelfile_template({"template_name": "chatml"})
        self.assertTrue(t.strip().startswith("{{- range"))

    def test_path(self):
        p = REPO / "template" / "chatml.gotmpl"
        t = resolve_modelfile_template({"template_file": str(p)})
        self.assertIn("im_start", t)


if __name__ == "__main__":
    unittest.main()
