"""Unified llama.cpp root resolution (parity with llm/llama_cpp_unified.go).

WHY: runtime subprocess + in-process must agree with Go serve on which checkout
is canonical — one ../llama.cpp @ LLAMA_CPP_COMMIT, not legacy eliza-llama siblings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from runtime.llama_cpp_probe import default_llama_cpp_root, pinned_llama_cpp_version

LEGACY_CHECKOUT_NAMES = frozenset(
    {
        "eliza-llama.cpp",
        "eliza_llama.cpp",
        "stock-llama.cpp",
        "ollama-upstream",
    }
)

DEFAULT_PIN = "8f114a9b573b69035299f9b924047f53c1e22c7e"
UNIFIED_REPO = "https://github.com/ggml-org/llama.cpp.git"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sibling_llama_cpp_root() -> Path:
    return (_repo_root().parent / "llama.cpp").resolve()


def _is_bare_sibling_root(path: str | Path) -> bool:
    try:
        return Path(path).resolve() == _sibling_llama_cpp_root()
    except OSError:
        return False


def _server_under_root(bin_path: str | Path, root: str) -> bool:
    try:
        Path(bin_path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def is_legacy_checkout(path: str | Path | None) -> bool:
    if not path:
        return False
    return Path(path).name in LEGACY_CHECKOUT_NAMES


def path_uses_legacy_checkout(path: str | Path | None) -> bool:
    if not path:
        return False
    return any(is_legacy_checkout(part) for part in Path(path).parts)


def unified_llama_server_bin(root: Path | None = None) -> Path | None:
    base = root or default_llama_cpp_root()
    if not base.is_dir():
        return None
    for name in ("llama-server", "llama-server.exe"):
        candidate = base / "build" / "bin" / name
        if candidate.is_file():
            return candidate
    return None


def vendor_llama_cpp_root(repo_root: Path | None = None) -> Path | None:
    """Pinned vendor tree under ``zerollama/vendor/llama-cpp-<pin>``."""
    from runtime.llama_cpp_probe import _vendor_llama_cpp_root

    return _vendor_llama_cpp_root(repo_root)


def resolve_llama_cpp_root() -> Path:
    """Canonical llama.cpp root for runtime + subprocess.

    WHY prefer vendor over bare sibling ``../llama.cpp``: zerollama patches
    (``POST /kv/seq-copy``, cuda-graph invalidate) ship on the vendor pin only;
    stale ``LLAMA_CPP_ROOT`` in the shell must not override the patched tree.
    """
    return default_llama_cpp_root()


def _unique_roots(*roots: Path | None) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        if r is None:
            continue
        try:
            key = str(r.resolve())
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(Path(key))
    return out


def resolve_llama_server_bin(root: Path | None = None) -> Path | None:
    """``llama-server`` path: explicit env → primary root → vendor fallback."""
    override = os.environ.get("LLAMA_SERVER_BIN", "").strip()
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute():
            p = p.resolve()
        return p if p.is_file() else None
    primary = root or resolve_llama_cpp_root()
    for r in _unique_roots(primary, vendor_llama_cpp_root()):
        server = unified_llama_server_bin(r)
        if server is not None:
            return server
    return None


def resolve_llama_cpp_lib(root: Path | None = None) -> Path | None:
    """``libllama`` for in-process backend; follows same root order as server bin."""
    override = os.environ.get("LLAMA_CPP_LIB", "").strip()
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute():
            p = p.resolve()
        return p if p.is_file() else None
    primary = root or resolve_llama_cpp_root()
    names = ("libllama.so", "libllama.dylib", "libllama.dll")
    for r in _unique_roots(primary, vendor_llama_cpp_root()):
        for name in names:
            candidate = r / "build" / "bin" / name
            if candidate.is_file():
                return candidate
    return None


def unified_health(
    *,
    llama_cpp_root: Path | str | None = None,
    llama_server_bin: Path | str | None = None,
) -> dict[str, Any]:
    """Health block for /health.llama_cpp_unified."""
    root = Path(llama_cpp_root) if llama_cpp_root else default_llama_cpp_root()
    root_s = str(root)
    pin = pinned_llama_cpp_version() or DEFAULT_PIN
    server = Path(llama_server_bin) if llama_server_bin else unified_llama_server_bin(root)
    server_s = str(server) if server else None

    legacy_root = is_legacy_checkout(root) or path_uses_legacy_checkout(root_s)
    legacy_server = path_uses_legacy_checkout(server_s)
    warn = legacy_root or legacy_server

    detail_parts = [f"unified {root_s} @ pin {pin[:12]}"]
    if server_s:
        detail_parts.append(f"llama-server {server_s}")
    if legacy_root:
        detail_parts.append("legacy LLAMA_CPP_ROOT name — migrate to ../llama.cpp")
        warn = True
    if legacy_server:
        detail_parts.append("LLAMA_SERVER_BIN outside unified tree or legacy name")
        warn = True
    if server is None:
        detail_parts.append("llama-server not built")
        warn = True

    return {
        "unified_root": root_s,
        "pinned_commit": pin,
        "llama_server_bin": server_s,
        "legacy_checkout": legacy_root or legacy_server,
        "runtime_ready": server is not None and server.is_file(),
        "repo": UNIFIED_REPO,
        "warn": warn,
        "detail": "; ".join(detail_parts),
        "migrate": "./scripts/migrate_llama_cpp_unified.sh",
    }


def normalize_llama_cpp_env() -> list[str]:
    """Return human-readable normalization notes (does not mutate os.environ)."""
    msgs: list[str] = []
    root = resolve_llama_cpp_root()
    env_root = os.environ.get("LLAMA_CPP_ROOT", "").strip()
    env_bin = os.environ.get("LLAMA_SERVER_BIN", "").strip()
    server = unified_llama_server_bin(root)

    if env_root and path_uses_legacy_checkout(env_root):
        msgs.append(f"LLAMA_CPP_ROOT legacy {env_root} → use {root}")
    if env_root and _is_bare_sibling_root(env_root):
        msgs.append(f"LLAMA_CPP_ROOT bare sibling {env_root} → use {root}")
    if env_bin and path_uses_legacy_checkout(env_bin):
        if server:
            msgs.append(f"LLAMA_SERVER_BIN legacy {env_bin} → use {server}")
        else:
            msgs.append(f"LLAMA_SERVER_BIN legacy; build {root}/build/bin/llama-server")
    elif env_bin and server and not _server_under_root(env_bin, str(root)):
        if root.name.startswith("llama-cpp-"):
            msgs.append(f"LLAMA_SERVER_BIN {env_bin} → prefer {server}")
    return msgs
