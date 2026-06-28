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


def test_llama_patch_health_fails_missing_in_tree(tmp_path: Path):
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
