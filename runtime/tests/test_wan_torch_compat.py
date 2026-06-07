"""Unit tests for Wan LD_LIBRARY_PATH sanitization (no GPU required)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(REPO_SCRIPTS))

from wan_torch_compat import (  # noqa: E402
    _path_may_shadow_cudnn,
    sanitize_ld_library_path_for_pytorch,
    torch_bundled_lib_dir,
)


def test_path_may_shadow_cudnn():
    assert _path_may_shadow_cudnn("/usr/hostlibs") is True
    assert _path_may_shadow_cudnn("/opt/cudnn/lib") is True
    assert _path_may_shadow_cudnn("/usr/local/cuda/lib64") is False


def test_sanitize_prepends_torch_lib(tmp_path):
    venv = tmp_path / "venv"
    torch_lib = venv / "lib" / "python3.10" / "site-packages" / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    py = venv / "bin" / "python3"
    py.parent.mkdir(parents=True)
    py.touch()

    env = {
        "LD_LIBRARY_PATH": "/usr/hostlibs:/usr/local/cuda/lib64:/keep/me",
    }
    sanitize_ld_library_path_for_pytorch(env, python=str(py))
    parts = env["LD_LIBRARY_PATH"].split(":")
    assert parts[0] == str(torch_lib)
    assert "/usr/hostlibs" not in parts
    assert "/keep/me" in parts


def test_torch_bundled_lib_dir_from_fake_venv(tmp_path):
    venv = tmp_path / "venv"
    torch_lib = venv / "lib" / "python3.11" / "site-packages" / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    py = venv / "bin" / "python3"
    py.parent.mkdir(parents=True)
    py.touch()
    assert torch_bundled_lib_dir(python=str(py)) == torch_lib
