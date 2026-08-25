#!/usr/bin/env python3
"""Unit tests for training_export (T7) — no live serve required."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training_export import (
    collect_path_digests,
    default_register_name,
    file_sha256_digest,
    normalize_quant,
    parse_modelfile_simple,
    quantize_type_arg,
    register_model,
    resolve_export_unload,
    write_adapter_modelfile,
    write_gguf_modelfile,
)


class TestTrainingExport(unittest.TestCase):
    def test_normalize_quant(self):
        self.assertEqual(normalize_quant("Q4_K_M"), "q4_k_m")
        self.assertEqual(normalize_quant("fp16"), "f16")
        self.assertEqual(normalize_quant("q8"), "q8_0")

    def test_quantize_type_arg(self):
        self.assertIsNone(quantize_type_arg("f16"))
        self.assertEqual(quantize_type_arg("q4_k_m"), "Q4_K_M")

    def test_write_adapter_modelfile(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = Path(td) / "lora_adapter"
            adapter.mkdir()
            mf = write_adapter_modelfile(
                from_model="Qwen/Qwen2.5-0.5B-Instruct",
                adapter_path=adapter,
                dest=Path(td) / "Modelfile",
            )
            text = mf.read_text()
            self.assertIn("FROM Qwen/Qwen2.5-0.5B-Instruct", text)
            self.assertIn("ADAPTER ", text)
            self.assertIn(str(adapter.resolve()), text)

    def test_write_gguf_modelfile(self):
        with tempfile.TemporaryDirectory() as td:
            gguf = Path(td) / "m.gguf"
            gguf.write_bytes(b"GGUF")
            mf = write_gguf_modelfile(gguf_path=gguf, dest=Path(td) / "Modelfile")
            self.assertIn(str(gguf.resolve()), mf.read_text())

    def test_default_register_name(self):
        self.assertEqual(
            default_register_name("/tmp/training_output/my-run", "x"),
            "my-run:latest",
        )

    def test_parse_modelfile_adapter(self):
        with tempfile.TemporaryDirectory() as td:
            adapter = Path(td) / "lora"
            adapter.mkdir()
            mf = Path(td) / "Modelfile"
            mf.write_text(
                f'FROM llama3:latest\nADAPTER {adapter}\nSYSTEM """hi"""\n',
                encoding="utf-8",
            )
            parsed = parse_modelfile_simple(mf.read_text(), relative_dir=td)
            self.assertEqual(parsed["from"], "llama3:latest")
            self.assertEqual(parsed["adapter"], str(adapter.resolve()))
            self.assertEqual(parsed["system"], "hi")

    def test_parse_modelfile_gguf_file(self):
        with tempfile.TemporaryDirectory() as td:
            gguf = Path(td) / "m.gguf"
            gguf.write_bytes(b"GGUF")
            parsed = parse_modelfile_simple(
                f"FROM {gguf}\n", relative_dir=td
            )
            self.assertEqual(parsed["from_file"], str(gguf.resolve()))

    def test_file_digest(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.bin"
            p.write_bytes(b"abc")
            d = file_sha256_digest(p)
            self.assertTrue(d.startswith("sha256:"))
            self.assertEqual(len(d), len("sha256:") + 64)
            digests = collect_path_digests(Path(td))
            self.assertEqual(digests[str(p.resolve())], d)

    def test_resolve_export_unload(self):
        self.assertTrue(resolve_export_unload({"export_gguf": True}))
        self.assertFalse(resolve_export_unload({"export_gguf": True, "export_unload": False}))
        self.assertFalse(resolve_export_unload({"export_gguf": False}))

    def test_register_auto_falls_back_to_http(self):
        with tempfile.TemporaryDirectory() as td:
            gguf = Path(td) / "m.gguf"
            gguf.write_bytes(b"GGUFfake")
            mf = write_gguf_modelfile(gguf_path=gguf, dest=Path(td) / "Modelfile")

            def fake_cli(*_a, **_k):
                return {"status": "skipped", "error": "no binary"}

            def fake_http(model_name, modelfile_path, **_k):
                return {
                    "status": "ok",
                    "model": model_name,
                    "modelfile": str(modelfile_path),
                    "via": "http",
                }

            with patch("training_export.register_model_cli", side_effect=fake_cli), patch(
                "training_export.register_model_http", side_effect=fake_http
            ):
                out = register_model("t7-test:latest", mf, via="auto")
            self.assertEqual(out["status"], "ok")
            self.assertEqual(out["via"], "http")
            self.assertIn("cli_fallback", out)


if __name__ == "__main__":
    unittest.main()
