"""CLI entrypoint for runtime.gpu_health_report."""

import json
import subprocess
import sys
from pathlib import Path

from runtime.gpu_health_report import format_gpu_health_tuning_report

SAMPLE = {
    "status": "ok",
    "vram_autotune": {"enabled": True, "pending_first_calibration": True},
}


def test_format_gpu_health_tuning_report_cli_module():
    env = {**dict(__import__("os").environ), "HEALTH_JSON": json.dumps(SAMPLE)}
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "runtime.gpu_health_report"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "vram_autotune.enabled: True" in proc.stdout
    assert format_gpu_health_tuning_report(SAMPLE) in proc.stdout or proc.stdout.strip()
