#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "ltx_mlx_generate", Path(__file__).with_name("ltx_mlx_generate.py")
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader
_SPEC.loader.exec_module(mod)


class TestLtxMlxGenerate(unittest.TestCase):
    def test_parse_size(self):
        self.assertEqual(mod.parse_size("768x480"), (768, 480))

    def test_build_cmd(self):
        cmd = mod.build_cmd(
            {
                "LTX_MLX_MODEL_DIR": "/tmp/LTX",
                "LTX_PROMPT": "a fox",
                "LTX_OUTPUT_PATH": "/tmp/o.mp4",
                "LTX_SIZE": "768x480",
                "LTX_FRAMES": "17",
                "LTX_STEPS": "4",
                "LTX_MLX_PYTHON": "/usr/bin/python3",
            }
        )
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd[cmd.index("--width") + 1], "768")
        self.assertEqual(cmd[cmd.index("--height") + 1], "480")
        self.assertEqual(cmd[cmd.index("--steps") + 1], "4")
    def test_build_cmd_13b_image(self):
        cmd = mod.build_cmd(
            {
                "LTX_MLX_MODEL_DIR": "/tmp/LTX",
                "LTX_PROMPT": "a fox",
                "LTX_OUTPUT_PATH": "/tmp/o.mp4",
                "LTX_SIZE": "1280x720",
                "LTX_FRAMES": "41",
                "LTX_STEPS": "8",
                "LTX_MODEL": "13b",
                "LTX_IMAGE": "/tmp/cel.png",
                "LTX_MLX_PYTHON": "/usr/bin/python3",
            }
        )
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], "13b")
        self.assertEqual(cmd[cmd.index("--width") + 1], "1280")
        self.assertEqual(cmd[cmd.index("--image") + 1], "/tmp/cel.png")

    def test_check_model_dir_13b(self):
        with tempfile.TemporaryDirectory() as raw:
            d = Path(raw)
            (d / "tokenizer").mkdir()
            (d / "text_encoder").mkdir()
            (d / "LTX 13B").mkdir()
            (d / "tokenizer" / "spiece.model").write_bytes(b"x")
            (d / "text_encoder" / "config.json").write_text("{}")
            (d / "LTX 13B" / "ltxv-13b-0.9.8-distilled.safetensors").write_bytes(b"x")
            self.assertEqual(mod.check_model_dir(d, "13b"), [])
        with tempfile.TemporaryDirectory() as raw:
            d = Path(raw)
            (d / "tokenizer").mkdir()
            (d / "text_encoder").mkdir()
            (d / "tokenizer" / "spiece.model").write_bytes(b"x")
            (d / "text_encoder" / "config.json").write_text("{}")
            (d / "ltxv-2b-0.9.8-distilled.safetensors").write_bytes(b"x")
            self.assertEqual(mod.check_model_dir(d), [])


if __name__ == "__main__":
    unittest.main()
