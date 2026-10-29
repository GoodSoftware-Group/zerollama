#!/usr/bin/env python3
"""Weightless checks for ltx_video_generate.check_weights."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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


class _Pipe:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def init_pipe(self, pipe, kwargs, profile):
        kwargs["budgets"] = {"text_encoder": 100, "transformer": 400, "*": 1000}
        kwargs["pinnedMemory"] = False
        return int(profile)


def _install_fake_torch(free: int, total: int):
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        mem_get_info=lambda: (free, total),
    )
    return patch.dict(sys.modules, {"torch": fake})


class TestT5Budget(unittest.TestCase):
    def test_resident_when_gpu_has_headroom(self):
        w = _Pipe()
        gib = 1024**3
        with patch.dict("os.environ", {"LTX_MMGP_LOCK": "", "LTX_TEXT_ENCODER_STREAM": ""}):
            with _install_fake_torch(14 * gib, 24 * gib):
                mod.pin_text_encoder_budget(w, "ltxv_distilled")
                kwargs: dict = {}
                w.init_pipe({}, kwargs, 5)
        self.assertEqual(kwargs["budgets"]["text_encoder"], 6144)
        self.assertIs(kwargs["pinnedMemory"], False)

    def test_ltx2_keeps_100_mib_window(self):
        w = _Pipe()
        gib = 1024**3
        with _install_fake_torch(20 * gib, 24 * gib):
            mod.pin_text_encoder_budget(w, "ltx2_25_22B_distilled")
            kwargs: dict = {}
            w.init_pipe({}, kwargs, 5)
        self.assertEqual(kwargs["budgets"]["text_encoder"], 100)

    def test_ram_pin_when_gpu_tight(self):
        w = _Pipe()
        with patch.dict("os.environ", {"LTX_MMGP_LOCK": "", "LTX_TEXT_ENCODER_STREAM": ""}):
            with _install_fake_torch(2 * 1024**3, 16 * 1024**3):
                with patch.object(mod, "_mem_available_bytes", return_value=50 * 1024**3):
                    mod.pin_text_encoder_budget(w, "ltxv_distilled")
                    kwargs: dict = {}
                    w.init_pipe({}, kwargs, 5)
        self.assertEqual(kwargs["budgets"]["text_encoder"], 100)
        self.assertEqual(kwargs["pinnedMemory"], ["text_encoder"])

    def test_lock_keeps_window(self):
        w = _Pipe()
        gib = 1024**3
        with patch.dict("os.environ", {"LTX_MMGP_LOCK": "1"}):
            with _install_fake_torch(20 * gib, 24 * gib):
                mod.pin_text_encoder_budget(w, "ltxv_distilled")
                kwargs: dict = {}
                w.init_pipe({}, kwargs, 5)
        self.assertEqual(kwargs["budgets"]["text_encoder"], 100)


if __name__ == "__main__":
    unittest.main()
