#!/usr/bin/env python3
"""Unit smoke for best-fit / migrate policy (no real GPU migrate)."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import cuda_device as cd


class TestCudaDevicePolicy(unittest.TestCase):
    def test_best_fit_picks_smallest_that_fits(self):
        gpus = [
            cd.GpuSnap("0", free_mib=19000, used_mib=5000, movable=[]),
            cd.GpuSnap("1", free_mib=24000, used_mib=0, movable=[]),
        ]
        with mock.patch.object(cd, "snapshot_gpus", return_value=gpus):
            os.environ["ZEROLLAMA_CUDA_POLICY"] = "best_fit"
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            got = cd.pick_cuda_device(need_mib=18000, migrate=False)
        self.assertEqual(got, "0")

    def test_best_fit_skips_undersized(self):
        gpus = [
            cd.GpuSnap("0", free_mib=8000, used_mib=16000, movable=[]),
            cd.GpuSnap("1", free_mib=20000, used_mib=4000, movable=[]),
        ]
        with mock.patch.object(cd, "snapshot_gpus", return_value=gpus):
            os.environ["ZEROLLAMA_CUDA_POLICY"] = "best_fit"
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            got = cd.pick_cuda_device(need_mib=18000, migrate=False)
        self.assertEqual(got, "1")

    def test_migrate_when_only_reclaim_fits(self):
        gpus = [
            cd.GpuSnap(
                "0",
                free_mib=10000,
                used_mib=14000,
                movable=[(111, 9000, "python -m irodori_openai_tts")],
            ),
            cd.GpuSnap("1", free_mib=12000, used_mib=12000, movable=[]),
        ]
        after = [
            cd.GpuSnap("0", free_mib=19000, used_mib=5000, movable=[]),
            cd.GpuSnap("1", free_mib=9000, used_mib=15000, movable=[]),
        ]
        with mock.patch.object(cd, "snapshot_gpus", side_effect=[gpus, after]):
            with mock.patch.object(cd, "_migrate_movables", return_value=True) as mig:
                os.environ["ZEROLLAMA_CUDA_POLICY"] = "best_fit"
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                os.environ.pop("ZEROLLAMA_CUDA_MIGRATE", None)
                got = cd.pick_cuda_device(need_mib=18000, migrate=True)
        self.assertTrue(mig.called)
        self.assertEqual(got, "0")


if __name__ == "__main__":
    unittest.main()
