"""llama patch doctor / vendor health."""

from __future__ import annotations

from pathlib import Path

from runtime.llama_patch_health import (
    binary_embeds_cuda_graph_invalidate_route,
    binary_embeds_needles,
    binary_embeds_seq_copy_route,
    in_tree_patch_markers,
    list_patch_files,
    llama_patch_health,
    probe_cuda_weight_formats,
    resolve_ggml_cuda_lib,
)


def test_list_patch_files_includes_required():
    names = list_patch_files()
    assert any("ollama-llama-kv-ext" in n for n in names)
    assert any("ollama-kv-seq-copy-endpoint" in n for n in names)
    assert any("cuda_graph_invalidate" in n for n in names)
    assert any("kv-grow-shrink-in-place" in n for n in names)


def test_binary_embeds_seq_copy_checks_server_impl(tmp_path: Path):
    """Thin llama-server wrapper: route string lives in libllama-server-impl."""
    wrapper = tmp_path / "llama-server"
    wrapper.write_bytes(b"thin-wrapper-no-route")
    assert binary_embeds_seq_copy_route(wrapper) is False
    impl = tmp_path / "libllama-server-impl.so"
    impl.write_bytes(b"prefix /kv/seq-copy suffix")
    assert binary_embeds_seq_copy_route(wrapper) is True


def test_binary_embeds_cuda_graph_checks_server_impl(tmp_path: Path):
    wrapper = tmp_path / "llama-server"
    wrapper.write_bytes(b"thin-wrapper-no-route")
    assert binary_embeds_cuda_graph_invalidate_route(wrapper) is False
    impl = tmp_path / "libllama-server-impl.so"
    impl.write_bytes(b"prefix /cuda-graph/invalidate suffix")
    assert binary_embeds_cuda_graph_invalidate_route(wrapper) is True


def test_probe_cuda_weight_formats_from_lib(tmp_path: Path):
    server = tmp_path / "llama-server"
    server.write_bytes(b"thin")
    lib = tmp_path / "libggml-cuda.so"
    lib.write_bytes(b"pad GGML_TYPE_NVFP4 pad GGML_TYPE_MXFP4 pad GGML_TYPE_FP8_E4M3 pad GGML_TYPE_FP8_E5M2 pad")
    assert resolve_ggml_cuda_lib(server) == lib.resolve()
    report = probe_cuda_weight_formats(server)
    assert report["skipped"] is False
    assert report["nvfp4"] is True
    assert report["mxfp4"] is True
    assert report["fp8_e4m3"] is True
    assert report["fp8_e5m2"] is True
    assert binary_embeds_needles(lib, (b"GGML_TYPE_NVFP4",)) is True


def test_probe_cuda_weight_formats_missing_markers(tmp_path: Path):
    server = tmp_path / "llama-server"
    server.write_bytes(b"thin")
    lib = tmp_path / "libggml-cuda.so"
    lib.write_bytes(b"no fp4 here")
    report = probe_cuda_weight_formats(server)
    assert report["nvfp4"] is False
    assert report["mxfp4"] is False
    assert report["fp8_e4m3"] is False
    assert report["fp8_e5m2"] is False


def test_llama_patch_health_warns_missing_nvfp4(tmp_path: Path, monkeypatch):
    repo = tmp_path / "zerollama"
    repo.mkdir()
    (repo / "Makefile.sync").write_text(
        "FETCH_HEAD=abc\nFETCH_REF=abc123\nBUILD_NUMBER=1\n", encoding="utf-8"
    )
    ext_bin = tmp_path / "external" / "llama-server"
    ext_bin.parent.mkdir(parents=True)
    ext_bin.write_bytes(b"\0")
    lib = tmp_path / "external" / "libggml-cuda.so"
    lib.write_bytes(b"cuda without fp4")
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
        lambda _path: True,
    )
    monkeypatch.setattr(
        "runtime.llama_patch_health.binary_embeds_cuda_graph_invalidate_route",
        lambda _path: True,
    )
    report = llama_patch_health(repo)
    assert report["status"] == "pass", report.get("issues")
    assert report["cuda_weight_formats"]["nvfp4"] is False
    assert any("NVFP4" in w for w in report["warnings"])


def test_in_tree_seq_copy_markers_present():
    markers = in_tree_patch_markers()
    assert markers["llama/llama.cpp/tools/server/server.cpp"] is True
    assert markers["llama/llama.cpp/tools/server/server.cpp#cuda-graph"] is True
    assert markers["llama/llama.cpp/tools/server/server.cpp#kv-grow"] is True
    assert markers["llama/llama.cpp/tools/server/server.cpp#kv-shrink"] is True
    assert markers["llama/llama.cpp/include/llama.h"] is True
    assert markers["llama/llama.cpp/include/llama-kv-ext.h"] is True


def test_llama_patch_health_passes_in_repo():
    report = llama_patch_health()
    assert report["required_patches_ok"] is True
    assert report["in_tree_markers"]["llama/llama.cpp/tools/server/server.cpp"] is True
    assert report["in_tree_markers"]["llama/llama.cpp/tools/server/server.cpp#cuda-graph"] is True
    assert report["status"] == "pass", report.get("issues")


def test_llama_patch_health_external_binary(tmp_path: Path, monkeypatch):
    repo = tmp_path / "zerollama"
    repo.mkdir()
    (repo / "Makefile.sync").write_text(
        "FETCH_HEAD=abc\nFETCH_REF=abc123\nBUILD_NUMBER=1\n", encoding="utf-8"
    )
    ext_bin = tmp_path / "external" / "llama-server"
    ext_bin.parent.mkdir(parents=True)
    ext_bin.write_bytes(b"\0")
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
    assert report["status"] == "pass", report.get("issues")
    assert report["llama_server_binary_seq_copy"] is False


def test_llama_patch_health_fails_missing_in_tree(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("LLAMA_SERVER_BIN", raising=False)
    repo = tmp_path / "zerollama"
    (repo / "llama" / "patches").mkdir(parents=True)
    (repo / "llama" / "patches" / "0014-ollama-llama-kv-ext.patch").write_text("x\n")
    (repo / "llama" / "patches" / "0018-ollama-kv-seq-copy-endpoint.patch").write_text("x\n")
    (repo / "Makefile.sync").write_text(
        "FETCH_HEAD=abc\nFETCH_REF=abc123\nBUILD_NUMBER=1\n", encoding="utf-8"
    )
    report = llama_patch_health(repo)
    assert report["status"] == "fail"
    assert any("in-tree marker missing" in i for i in report["issues"])
