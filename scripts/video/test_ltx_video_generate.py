#!/usr/bin/env python3
"""Weightless checks for ltx_video_generate.check_weights."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "ltx_video_generate", Path(__file__).with_name("ltx_video_generate.py")
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader
_SPEC.loader.exec_module(mod)


class TestLtxWeights(unittest.TestCase):
    def test_2b_names(self):
        names = mod.weight_names("ltxv_2b_distilled")
        self.assertIn("ltxv-2b-0.9.8-distilled-fp8.safetensors", names)
        self.assertNotIn("ltxv_0.9.8_13B_distilled_quanto_bf16_int8.safetensors", names)

    def test_13b_names(self):
        names = mod.weight_names("ltxv_distilled")
        self.assertIn("ltxv_0.9.8_13B_distilled_quanto_bf16_int8.safetensors", names)

    def test_2b_accepts_official_upscaler_name(self):
        with tempfile.TemporaryDirectory() as raw:
            ckpt = Path(raw)
            for n in (
                "ltxv-2b-0.9.8-distilled-fp8.safetensors",
                "ltxv-spatial-upscaler-0.9.8.safetensors",
                "ltxv_0.9.7_VAE.safetensors",
                "ltxv_scheduler.json",
            ):
                (ckpt / n).write_bytes(b"x")
            t5 = ckpt / "T5_xxl_1.1"
            t5.mkdir()
            (t5 / "T5_xxl_1.1_enc_quanto_bf16_int8.safetensors").write_bytes(b"x")
            self.assertEqual(mod.check_weights(ckpt, "ltxv_2b_distilled"), [])


if __name__ == "__main__":
    unittest.main()
