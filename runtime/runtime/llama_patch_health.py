"""Offline llama.cpp patch / vendor health (Radix seq-copy, kv-ext, pin drift).

WHY: patched routes ship via ``llama/patches/`` → vendor ``git am`` → rsync in-tree
→ rebuild llama-server. Operators often run a stale sibling binary and see "lost"
patches (404 on ``/kv/seq-copy``) while git still looks fine.
"""

from __future__ import annotations

import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from runtime.llama_cpp_unified import (
    resolve_llama_cpp_lib,
    resolve_llama_cpp_root,
    resolve_llama_server_bin,
    vendor_llama_cpp_root,
)

_REQUIRED_PATCH_SUBSTRINGS = (
    "0014-ollama-llama-kv-ext",
    "0017-ollama-kv-seq-copy-endpoint",
)

_IN_TREE_MARKERS = (
    ("llama/llama.cpp/tools/server/server.cpp", '"/kv/seq-copy"'),
    ("llama/llama.cpp/include/llama-kv-ext.h", "llama_memory_kv_ext_classify"),
    ("llama/llama.cpp/src/llama-memory-kv-ext.cpp", "llama_memory_kv_cell_for_pos"),
)


def _repo_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return Path(__file__).resolve().parents[2]


def _read_makefile_sync(repo: Path) -> tuple[str, str]:
    fetch_head = "c84b3020"
    fetch_ref = "c84b30200c8d512c00c9d61c96bed078f1c0024d"
    makefile = repo / "Makefile.sync"
    if makefile.is_file():
        text = makefile.read_text(encoding="utf-8", errors="replace")
        if m := re.search(r"^FETCH_HEAD=(.+)$", text, re.MULTILINE):
            fetch_head = m.group(1).strip()
        if m := re.search(r"^FETCH_REF=(.+)$", text, re.MULTILINE):
            fetch_ref = m.group(1).strip()
    commit_file = repo / "LLAMA_CPP_COMMIT"
    if commit_file.is_file():
        ref = commit_file.read_text(encoding="utf-8").strip()
        if ref:
            fetch_ref = ref
    return fetch_head, fetch_ref


def list_patch_files(repo: Path | None = None) -> list[str]:
    root = _repo_root(repo)
    patch_dir = root / "llama" / "patches"
    if not patch_dir.is_dir():
        return []
    return sorted(p.name for p in patch_dir.glob("*.patch"))


def in_tree_patch_markers(repo: Path | None = None) -> dict[str, bool]:
    root = _repo_root(repo)
    out: dict[str, bool] = {}
    for rel, needle in _IN_TREE_MARKERS:
        path = root / rel
        if not path.is_file():
            out[rel] = False
            continue
        try:
            out[rel] = needle in path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            out[rel] = False
    return out


def vendor_patch_stats(repo: Path | None = None) -> dict[str, Any]:
    root = _repo_root(repo)
    fetch_head, fetch_ref = _read_makefile_sync(root)
    vendor = root / "vendor" / f"llama-cpp-{fetch_head}"
    vendor_git = vendor / ".git"
    if not vendor_git.is_dir():
        return {
            "present": False,
            "path": str(vendor),
            "commits_on_pin": None,
            "head": None,
            "fetch_ref": fetch_ref,
        }
    head = _git_rev_parse(vendor, "HEAD")
    count_raw = _git_rev_list_count(vendor, fetch_ref)
    return {
        "present": True,
        "path": str(vendor.resolve()),
        "commits_on_pin": count_raw,
        "head": head,
        "fetch_ref": fetch_ref,
    }


def _git_rev_parse(cwd: Path, ref: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", ref],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = (out.stdout or "").strip()
    return line or None


def _git_rev_list_count(cwd: Path, base_ref: str) -> int | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), "rev-list", "--count", f"{base_ref}..HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    raw = (out.stdout or "").strip()
    if not raw.isdigit():
        return None
    return int(raw)


def binary_embeds_seq_copy_route(path: Path | None) -> bool | None:
    if path is None or not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return b"/kv/seq-copy" in data or b"kv/seq-copy" in data


def probe_seq_copy_http(base_url: str, *, timeout: float = 3.0) -> bool | None:
    url = base_url.rstrip("/") + "/kv/seq-copy"
    req = urllib.request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 400, 501)
    except urllib.error.HTTPError as e:
        return e.code != 404
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


_EXTERNAL_INSTALL_PREFIXES = (
    "/usr/local/lib/ollama",
    "/opt/ollama",
)


def _is_external_llama_install(server_path: Path | None) -> bool:
    if server_path is None:
        return False
    try:
        resolved = str(server_path.resolve())
    except OSError:
        return False
    return any(resolved.startswith(prefix) for prefix in _EXTERNAL_INSTALL_PREFIXES)


def _path_under(child: Path | None, parent: Path | None) -> bool | None:
    if child is None or parent is None:
        return None
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def expected_vendor_head(repo: Path | None = None) -> str | None:
    root = _repo_root(repo)
    head_file = root / "LLAMA_CPP_VENDOR_HEAD"
    if not head_file.is_file():
        return None
    for line in head_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and len(line) >= 40:
            return line
    return None


def vendor_head_matches(repo: Path | None = None) -> dict[str, Any]:
    vendor = vendor_patch_stats(repo)
    expected = expected_vendor_head(repo)
    head = vendor.get("head")
    if not vendor.get("present"):
        return {"expected": expected, "head": head, "matches": None, "skipped": True}
    if not expected or not head:
        return {"expected": expected, "head": head, "matches": None, "skipped": False}
    return {
        "expected": expected,
        "head": head,
        "matches": head == expected,
        "skipped": False,
    }


def _vendor_cuda_fork_synced(vendor_path: Path) -> bool:
    fused = vendor_path / "ggml" / "src" / "ggml-cuda" / "fused-attn.cu"
    set_rows = vendor_path / "ggml" / "src" / "ggml-cuda" / "set-rows.cu"
    if not fused.is_file() or not set_rows.is_file():
        return False
    try:
        text = set_rows.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "qjl1_256" in text.lower() or "QJL1_256" in text


def llama_patch_health(
    repo: Path | None = None,
    *,
    probe_http_base: str | None = None,
) -> dict[str, Any]:
    root = _repo_root(repo)
    fetch_head, fetch_ref = _read_makefile_sync(root)
    patches = list_patch_files(root)
    missing_required = [
        sub for sub in _REQUIRED_PATCH_SUBSTRINGS
        if not any(sub in name for name in patches)
    ]
    in_tree = in_tree_patch_markers(root)
    vendor = vendor_patch_stats(root)
    vendor_synced = (
        vendor.get("present")
        and _vendor_cuda_fork_synced(Path(vendor["path"]))
    )
    cpp_root = resolve_llama_cpp_root()
    vendor_root = vendor_llama_cpp_root(root)
    server_bin = resolve_llama_server_bin(cpp_root)
    cpp_lib = resolve_llama_cpp_lib(cpp_root)

    issues: list[str] = []
    warnings: list[str] = []

    server_path = Path(server_bin) if server_bin else None
    under_vendor = _path_under(server_path, vendor_root)
    external_install = _is_external_llama_install(server_path)
    bin_seq = binary_embeds_seq_copy_route(server_path)
    fork_help = False
    if server_path is not None and server_path.is_file():
        from runtime.llama_fork import probe_fork_llama_server

        fork_help = probe_fork_llama_server(str(server_path))

    if not patches and not external_install:
        issues.append("no llama/patches/*.patch files found")
    if missing_required and not external_install:
        issues.append(f"missing required patch files (substring): {missing_required}")
    if not external_install:
        for rel, ok in in_tree.items():
            if not ok:
                issues.append(f"in-tree marker missing: {rel}")

    if vendor["present"]:
        count = vendor.get("commits_on_pin")
        if count is None:
            warnings.append("vendor present but could not count commits on pin")
        elif count == 0 and bin_seq is not True and not fork_help and not vendor_synced:
            issues.append(
                f"vendor at bare pin with zero patch commits — run "
                f"./scripts/rebase_vendor_unified.sh --apply --sync"
            )
        elif count == 0 and vendor_synced:
            warnings.append(
                "vendor synced from sibling (zero git am commits) — rebuild llama-server to pick up patches"
            )
        head_check = vendor_head_matches(root)
        if head_check.get("matches") is False:
            warnings.append(
                f"vendor HEAD {head_check.get('head', '')[:12]} != "
                f"LLAMA_CPP_VENDOR_HEAD {str(head_check.get('expected', ''))[:12]}"
            )
    else:
        if external_install:
            warnings.append(
                "external llama-server install — skipping in-tree patch markers"
            )
        else:
            warnings.append(
                "vendor/ not materialized (gitignored) — in-tree + binary checks only; "
                "run ./scripts/rebase_vendor_unified.sh --apply --sync before llama-server build"
            )

    if server_path is not None and under_vendor is False and not external_install:
        warnings.append(
            f"llama-server binary outside vendor tree: {server_path} "
            f"(expected under {vendor_root})"
        )

    if server_path is not None and bin_seq is False:
        if external_install and fork_help:
            warnings.append(
                "external binary: /kv/seq-copy embed not found; fork KV types in --help"
            )
        elif vendor_synced and not external_install:
            warnings.append(
                "vendor build: /kv/seq-copy not embedded (patch 0017); fork KV CUDA sync present"
            )
        else:
            issues.append(
                f"llama-server binary lacks /kv/seq-copy string — rebuild from vendor: "
                f"./scripts/build_llama_server.sh"
            )
    elif external_install and bin_seq is True and not issues:
        warnings.append("external binary validated via /kv/seq-copy embed")
    elif external_install and fork_help and bin_seq is not True:
        warnings.append("external binary validated via fork KV --help probe")

    http_probe: bool | None = None
    if probe_http_base:
        http_probe = probe_seq_copy_http(probe_http_base)
        if http_probe is False:
            issues.append(f"live llama-server at {probe_http_base!r} returns 404 for POST /kv/seq-copy")

    version_file = root / "LLAMA_CPP_VERSION"
    llama_cpp_version = version_file.read_text(encoding="utf-8").strip() if version_file.is_file() else ""

    deployment_mode = "external_binary" if external_install else "vendor_tree"
    status = "fail" if issues else "pass"
    return {
        "status": status,
        "deployment_mode": deployment_mode,
        "llama_cpp_version": llama_cpp_version,
        "makefile_sync_fetch_head": fetch_head,
        "makefile_sync_fetch_ref": fetch_ref,
        "patch_files_count": len(patches),
        "patch_files": patches,
        "required_patches_ok": not missing_required,
        "in_tree_markers": in_tree,
        "vendor": vendor,
        "vendor_synced_cuda_fork": vendor_synced,
        "vendor_head_check": vendor_head_matches(root),
        "resolved_llama_cpp_root": str(cpp_root),
        "resolved_llama_server_bin": str(server_path) if server_path else None,
        "resolved_llama_cpp_lib": str(cpp_lib) if cpp_lib else None,
        "llama_server_under_vendor": under_vendor,
        "llama_server_binary_seq_copy": bin_seq,
        "live_seq_copy_probe": http_probe,
        "issues": issues,
        "warnings": warnings,
        "remediation": [
            "./scripts/rebase_vendor_unified.sh --apply --sync",
            "./scripts/build_llama_server.sh",
            "./scripts/phase15_llama_kv_ext_pin_check.sh",
            "L3_RADIX_LIVE=1 ./scripts/l3_radix_prefix_smoke.sh",
        ],
    }


def llama_patch_health_summary(
    repo: Path | None = None,
    *,
    probe_http_base: str | None = None,
) -> dict[str, Any]:
    """Compact block for ``/health`` (no full patch file list)."""
    full = llama_patch_health(repo, probe_http_base=probe_http_base)
    return {
        "status": full["status"],
        "deployment_mode": full.get("deployment_mode"),
        "llama_cpp_version": full["llama_cpp_version"],
        "patch_files_count": full["patch_files_count"],
        "required_patches_ok": full["required_patches_ok"],
        "in_tree_markers": full["in_tree_markers"],
        "vendor_present": full["vendor"]["present"],
        "vendor_synced_cuda_fork": full.get("vendor_synced_cuda_fork"),
        "vendor_commits_on_pin": full["vendor"].get("commits_on_pin"),
        "resolved_llama_cpp_root": full["resolved_llama_cpp_root"],
        "resolved_llama_server_bin": full["resolved_llama_server_bin"],
        "llama_server_under_vendor": full["llama_server_under_vendor"],
        "llama_server_binary_seq_copy": full["llama_server_binary_seq_copy"],
        "issues": full["issues"],
        "warnings": full["warnings"],
        "doctor": "./scripts/llama_patch_doctor.sh",
    }
