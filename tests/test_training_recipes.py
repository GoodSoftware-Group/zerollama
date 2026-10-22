"""T11 lite recipe ingest (stdlib only; PDF/DOCX optional)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training_recipes import (
    detect_format,
    ingest_csv,
    ingest_file,
    ingest_json,
    ingest_jsonl,
    ingest_txt_pairs,
    write_jsonl,
    write_train_payload,
)


class TestTrainingRecipes(unittest.TestCase):
    def test_detect_format(self):
        self.assertEqual(detect_format(Path("a.csv")), "csv")
        self.assertEqual(detect_format(Path("a.jsonl")), "jsonl")
        self.assertEqual(detect_format(Path("a.pdf")), "pdf")

    def test_ingest_csv_aliases(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.csv"
            p.write_text("question,answer\nhello,world\n", encoding="utf-8")
            rows = ingest_csv(p)
            self.assertEqual(rows, [{"prompt": "hello", "response": "world"}])

    def test_ingest_jsonl_messages(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.jsonl"
            p.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "yo"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = ingest_jsonl(p)
            self.assertEqual(rows, [{"prompt": "hi", "response": "yo"}])

    def test_ingest_json_wrapper(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.json"
            p.write_text(
                json.dumps(
                    {
                        "training_data": [
                            {"prompt": "a", "response": "b"},
                            {"prompt": "", "response": "skip"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = ingest_json(p)
            self.assertEqual(rows, [{"prompt": "a", "response": "b"}])

    def test_ingest_txt_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.txt"
            p.write_text("Q1\nA1 line\n\nQ2\n---\nA2\n", encoding="utf-8")
            rows = ingest_txt_pairs(p)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["prompt"], "Q1")
            self.assertEqual(rows[1]["response"], "A2")

    def test_write_payload(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "payload.json"
            write_train_payload(
                [{"prompt": "p", "response": "r"}],
                out,
                model_name="tiny",
            )
            body = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(body["model_name"], "tiny")
            self.assertEqual(len(body["training_data"]), 1)

    def test_ingest_file_roundtrip_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.csv"
            dst = Path(td) / "out.jsonl"
            src.write_text("prompt,response\nx,y\n", encoding="utf-8")
            rows = ingest_file(src)
            n = write_jsonl(rows, dst)
            self.assertEqual(n, 1)
            again = ingest_file(dst)
            self.assertEqual(again, rows)

    def test_pdf_missing_dep_message(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.pdf"
            p.write_bytes(b"%PDF-1.4 fake")
            try:
                import pypdf  # noqa: F401

                self.skipTest("pypdf installed")
            except ImportError:
                with self.assertRaises(ImportError) as ctx:
                    ingest_file(p)
                self.assertIn("pypdf", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
