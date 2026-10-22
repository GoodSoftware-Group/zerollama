"""T11 lite recipe ingest (stdlib only; PDF/DOCX optional)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from training_recipes import (
    detect_format,
    ingest_csv,
    ingest_file,
    ingest_json,
    ingest_jsonl,
    ingest_text,
    ingest_txt_pairs,
    main_stdin,
    normalize_preset,
    pair_from_obj,
    recipe_result,
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

    def test_sharegpt_preset(self):
        obj = {
            "conversations": [
                {"from": "human", "value": "hi"},
                {"from": "gpt", "value": "hello"},
            ]
        }
        self.assertEqual(pair_from_obj(obj, preset="sharegpt"), ("hi", "hello"))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sg.json"
            p.write_text(json.dumps([obj]), encoding="utf-8")
            rows = ingest_json(p, preset="sharegpt")
            self.assertEqual(rows, [{"prompt": "hi", "response": "hello"}])

    def test_alpaca_preset(self):
        obj = {"instruction": "Add", "input": "2+2", "output": "4"}
        self.assertEqual(pair_from_obj(obj, preset="alpaca"), ("Add\n\n2+2", "4"))

    def test_openai_preset_auto(self):
        obj = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
        self.assertEqual(pair_from_obj(obj, preset="openai"), ("q", "a"))
        self.assertEqual(pair_from_obj(obj, preset="auto"), ("q", "a"))

    def test_ingest_text_jsonl(self):
        text = (
            json.dumps(
                {
                    "conversations": [
                        {"from": "human", "value": "x"},
                        {"from": "gpt", "value": "y"},
                    ]
                }
            )
            + "\n"
        )
        rows = ingest_text(text, fmt="jsonl", preset="sharegpt")
        self.assertEqual(rows, [{"prompt": "x", "response": "y"}])

    def test_recipe_result_envelope(self):
        body = recipe_result(
            [{"prompt": "p", "response": "r"}],
            model_name="m",
            preset="openai",
        )
        self.assertEqual(body["n_rows"], 1)
        self.assertEqual(body["model_name"], "m")
        self.assertEqual(body["preset"], "openai")

    def test_normalize_preset_aliases(self):
        self.assertEqual(normalize_preset("vicuna"), "sharegpt")
        self.assertEqual(normalize_preset("chatgpt"), "openai")
        with self.assertRaises(ValueError):
            normalize_preset("studio")

    def test_main_stdin(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr

        payload = json.dumps(
            {
                "content": json.dumps(
                    [
                        {
                            "instruction": "Say hi",
                            "input": "",
                            "output": "hi",
                        }
                    ]
                ),
                "format": "json",
                "preset": "alpaca",
                "model_name": "demo",
            }
        )
        buf = io.StringIO()
        err = io.StringIO()
        old = sys.stdin
        try:
            sys.stdin = io.StringIO(payload)
            with redirect_stdout(buf), redirect_stderr(err):
                rc = main_stdin()
        finally:
            sys.stdin = old
        self.assertEqual(rc, 0)
        out = json.loads(buf.getvalue())
        self.assertEqual(out["n_rows"], 1)
        self.assertEqual(out["training_data"][0]["response"], "hi")


if __name__ == "__main__":
    unittest.main()
