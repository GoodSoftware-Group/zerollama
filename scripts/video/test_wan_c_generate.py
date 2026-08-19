#!/usr/bin/env python3
"""Weightless argv rematch for wan_c_generate.build_cmd."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "wan_c_generate", Path(__file__).with_name("wan_c_generate.py")
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader
_SPEC.loader.exec_module(mod)


class TestWanCGenerate(unittest.TestCase):
    def test_parse_size_star_and_x(self):
        self.assertEqual(mod.parse_size("480*832"), ("480", "832"))
        self.assertEqual(mod.parse_size("768x768"), ("768", "768"))
        self.assertEqual(mod.parse_size(""), ("480", "832"))

    def test_h3_cmd(self):
        cmd = mod.build_cmd(
            {
                "VIDEO_CLI": "/tmp/video-cli",
                "WAN_CKPT_DIR": "/tmp/MiniMax-H3",
                "WAN_PROMPT": "a fox",
                "WAN_OUTPUT_PATH": "/tmp/out.mp4",
                "WAN_SIZE": "32x32",
                "WAN_FRAMES": "5",
                "WAN_STEPS": "2",
                "VIDEO_FAMILY": "h3",
                "VIDEO_H3_LAYERS": "1",
                "WAN_SEED": "7",
            }
        )
        self.assertIsNotNone(cmd)
        self.assertIn("--generate", cmd)
        self.assertEqual(cmd[cmd.index("--family") + 1], "h3")
        self.assertEqual(cmd[cmd.index("--width") + 1], "32")
        self.assertEqual(cmd[cmd.index("--height") + 1], "32")
        self.assertEqual(cmd[cmd.index("--frames") + 1], "5")
        self.assertEqual(cmd[cmd.index("--layers") + 1], "1")
        self.assertEqual(cmd[cmd.index("--seed") + 1], "7")

    def test_h3_768_cmd(self):
        cmd = mod.build_cmd(
            {
                "VIDEO_CLI": "/tmp/video-cli",
                "WAN_CKPT_DIR": "/tmp/MiniMax-H3",
                "WAN_PROMPT": "a fox",
                "WAN_OUTPUT_PATH": "/tmp/out.mp4",
                "WAN_SIZE": "768x768",
                "WAN_FRAMES": "5",
                "WAN_STEPS": "2",
                "VIDEO_FAMILY": "h3",
                "VIDEO_H3_LAYERS": "1",
            }
        )
        self.assertEqual(cmd[cmd.index("--width") + 1], "768")
        self.assertEqual(cmd[cmd.index("--height") + 1], "768")
        self.assertEqual(cmd[cmd.index("--layers") + 1], "1")
        self.assertIn("--generate", cmd)

    def test_h3_default_layers_50(self):
        cmd = mod.build_cmd(
            {
                "VIDEO_CLI": "/tmp/video-cli",
                "WAN_CKPT_DIR": "/tmp/MiniMax-H3",
                "WAN_PROMPT": "a fox",
                "WAN_OUTPUT_PATH": "/tmp/out.mp4",
                "WAN_SIZE": "32x32",
                "VIDEO_FAMILY": "h3",
            }
        )
        self.assertEqual(cmd[cmd.index("--layers") + 1], "50")

    def test_wan_cmd_no_generate(self):
        cmd = mod.build_cmd(
            {
                "VIDEO_CLI": "/tmp/video-cli",
                "WAN_CKPT_DIR": "/tmp/wan",
                "WAN_PROMPT": "a cat",
                "WAN_OUTPUT_PATH": "/tmp/w.mp4",
                "WAN_SIZE": "64*64",
                "VIDEO_FAMILY": "wan",
            }
        )
        self.assertIsNotNone(cmd)
        self.assertNotIn("--generate", cmd)
        self.assertEqual(cmd[cmd.index("--width") + 1], "64")


if __name__ == "__main__":
    unittest.main()
