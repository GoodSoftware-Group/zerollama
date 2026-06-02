"""Start Python runtime + zerollama serve (Phase 7 single-operator entry)."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


class SupervisorError(Exception):
    pass


def wait_for_runtime_health(base_url: str, timeout_s: float = 120.0) -> None:
    url = base_url.rstrip("/") + "/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.25)
    raise SupervisorError(f"runtime not healthy at {url} within {timeout_s}s")


def _runtime_cmd(config: Path | None, host: str, port: int) -> list[str]:
    cmd = [sys.executable, "-m", "runtime", "serve", "--host", host, "--port", str(port)]
    if config is not None:
        cmd.extend(["--config", str(config)])
    return cmd


def run_stack(
    *,
    zerollama_bin: str | None = None,
    config: Path | None = None,
    runtime_host: str = "127.0.0.1",
    runtime_port: int = 8081,
    health_timeout_s: float = 120.0,
) -> int:
    """Start runtime sidecar, then ``zerollama serve`` in the foreground."""
    zbin = zerollama_bin or shutil.which("zerollama")
    if not zbin:
        raise SupervisorError("zerollama not found on PATH")

    base_url = f"http://{runtime_host}:{runtime_port}"
    env = os.environ.copy()
    env["ZEROLLAMA_RUNTIME_URL"] = base_url
    if config is not None:
        env["ZEROLLAMA_RUNTIME_CONFIG"] = str(config)

    runtime_proc = subprocess.Popen(
        _runtime_cmd(config, runtime_host, runtime_port),
        env=env,
    )

    def _shutdown(*_args: object) -> None:
        if runtime_proc.poll() is None:
            runtime_proc.terminate()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        if runtime_proc.poll() is not None:
            raise SupervisorError("runtime process exited during startup")
        wait_for_runtime_health(base_url, timeout_s=health_timeout_s)
        serve_env = os.environ.copy()
        serve_env["ZEROLLAMA_RUNTIME_URL"] = base_url
        return subprocess.call([zbin, "serve"], env=serve_env)
    finally:
        if runtime_proc.poll() is None:
            runtime_proc.terminate()
            try:
                runtime_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                runtime_proc.kill()
                runtime_proc.wait(timeout=5)
