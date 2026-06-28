"""Unified llama.cpp root helpers."""

from __future__ import annotations

from pathlib import Path

from runtime.llama_cpp_unified import (
    is_legacy_checkout,
    normalize_llama_cpp_env,
    path_uses_legacy_checkout,
    resolve_llama_cpp_root,
    unified_health,
    vendor_llama_cpp_root,
)


def test_is_legacy_checkout():
    assert is_legacy_checkout("/tmp/eliza-llama.cpp")
    assert not is_legacy_checkout("/tmp/llama.cpp")


def test_path_uses_legacy_checkout():
    assert path_uses_legacy_checkout(
        "/Users/x/Sites/inference/eliza-llama.cpp/build/bin/llama-server"
    )
    assert not path_uses_legacy_checkout(
        "/Users/x/Sites/inference/llama.cpp/build/bin/llama-server"
    )


def test_unified_health_legacy_warn(monkeypatch, tmp_path: Path):
    legacy = tmp_path / "eliza-llama.cpp"
    legacy.mkdir()
    out = unified_health(llama_cpp_root=legacy, llama_server_bin=None)
    assert out["legacy_checkout"] is True
    assert out["warn"] is True


def test_normalize_llama_cpp_env_legacy(monkeypatch):
    monkeypatch.setenv(
        "LLAMA_SERVER_BIN",
        "/Sites/inference/eliza-llama.cpp/build/bin/llama-server",
    )
    msgs = normalize_llama_cpp_env()
    assert any("legacy" in m.lower() for m in msgs)


def test_resolve_llama_cpp_root_prefers_vendor_over_sibling(
    monkeypatch, tmp_path: Path
):
    repo = tmp_path / "zerollama"
    repo.mkdir()
    pin = repo / "LLAMA_CPP_VERSION"
    pin.write_text("abc123\n", encoding="utf-8")
    vendor = repo / "vendor" / "llama-cpp-abc123"
    vendor.mkdir(parents=True)
    (vendor / "CMakeLists.txt").write_text("# vendor\n", encoding="utf-8")
    sibling = tmp_path / "llama.cpp"
    sibling.mkdir()
    (sibling / "CMakeLists.txt").write_text("# sibling\n", encoding="utf-8")

    monkeypatch.setenv("LLAMA_CPP_ROOT", str(sibling))
    monkeypatch.setattr("runtime.llama_cpp_probe._repo_root", lambda: repo)
    assert resolve_llama_cpp_root() == vendor.resolve()
    assert vendor_llama_cpp_root(repo) == vendor.resolve()
