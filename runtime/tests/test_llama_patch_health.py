"""llama patch doctor / vendor health."""

from __future__ import annotations

from pathlib import Path

from runtime.llama_patch_health import (
    in_tree_patch_markers,
    list_patch_files,
    llama_patch_health,
)


def test_list_patch_files_includes_required():
    names = list_patch_files()
    assert any("0014-ollama-llama-kv-ext" in n for n in names)
    assert any("0017-ollama-kv-seq-copy-endpoint" in n for n in names)


def test_in_tree_seq_copy_markers_present():
    markers = in_tree_patch_markers()
    assert markers["llama/llama.cpp/tools/server/server.cpp"] is True
    assert markers["llama/llama.cpp/include/llama-kv-ext.h"] is True


def test_llama_patch_health_passes_in_repo():
    report = llama_patch_health()
    assert report["required_patches_ok"] is True
    assert report["in_tree_markers"]["llama/llama.cpp/tools/server/server.cpp"] is True
    assert report["status"] == "pass", report.get("issues")


def test_llama_patch_health_external_binary(tmp_path: Path, monkeypatch):
    repo = tmp_path / "zerollama"
    repo.mkdir()
    (repo / "Makefile.sync").write_text(
        "FETCH_HEAD=abc\nFETCH_REF=abc123\nBUILD_NUMBER=1\n", encoding="utf-8"
    )
    ext_bin = Path("/usr/local/lib/ollama/llama-server")
    monkeypatch.setattr(
        "runtime.llama_patch_health.resolve_llama_server_bin",
        lambda _root=None: str(ext_bin),
    )
    monkeypatch.setattr(
        "runtime.llama_patch_health._is_external_llama_install",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "runtime.llama_fork.probe_fork_llama_server",
        lambda _bin: True,
    )
    monkeypatch.setattr(
        "runtime.llama_patch_health.binary_embeds_seq_copy_route",
        lambda _path: False,
    )
    report = llama_patch_health(repo)
    assert report["deployment_mode"] == "external_binary"
    assert report["status"] == "pass"
    assert report["status"] == "pass", report.get("issues")
    assert report["llama_server_binary_seq_copy"] is False


def test_llama_patch_health_fails_missing_in_tree(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("LLAMA_SERVER_BIN", raising=False)
    repo = tmp_path / "zerollama"
    (repo / "llama" / "patches").mkdir(parents=True)
    (repo / "llama" / "patches" / "0014-ollama-llama-kv-ext.patch").write_text("x\n")
    (repo / "llama" / "patches" / "0017-ollama-kv-seq-copy-endpoint.patch").write_text("x\n")
    (repo / "Makefile.sync").write_text(
        "FETCH_HEAD=abc\nFETCH_REF=abc123\nBUILD_NUMBER=1\n", encoding="utf-8"
    )
    report = llama_patch_health(repo)
    assert report["status"] == "fail"
    assert any("in-tree marker missing" in i for i in report["issues"])
