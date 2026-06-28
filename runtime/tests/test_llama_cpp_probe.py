"""Sibling llama.cpp probe."""

from __future__ import annotations

from pathlib import Path

from runtime.llama_cpp_probe import (
    _read_cmake_cache_bool,
    default_llama_cpp_root,
    pinned_llama_cpp_version,
    probe_llama_cpp,
)


def test_default_llama_cpp_root_prefers_vendor():
    repo = Path(__file__).resolve().parents[2]
    pin = pinned_llama_cpp_version(repo)
    vendor = repo / "vendor" / f"llama-cpp-{pin}"
    root = default_llama_cpp_root()
    if (vendor / "CMakeLists.txt").is_file():
        assert root == vendor.resolve()
    else:
        assert root.name == "llama.cpp"


def test_pinned_version_from_repo():
    pin = pinned_llama_cpp_version()
    assert pin is not None
    assert pin.startswith("c84b3020")


def test_read_cmake_cache_bool(tmp_path: Path):
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text("GGML_CUDA:BOOL=ON\nGGML_CUDA_GRAPHS:BOOL=OFF\n", encoding="utf-8")
    assert _read_cmake_cache_bool(cache, "GGML_CUDA") is True
    assert _read_cmake_cache_bool(cache, "GGML_CUDA_GRAPHS") is False
    assert _read_cmake_cache_bool(cache, "MISSING") is None


def test_probe_llama_cpp_local_checkout():
    info = probe_llama_cpp()
    assert info["present"] is True
    assert "llama.cpp" in info["root"] or "llama-cpp-" in info["root"]
    assert info["pin_file"] is not None
    assert info["git_describe"] is not None
    assert "epoch_bind_status" in info
