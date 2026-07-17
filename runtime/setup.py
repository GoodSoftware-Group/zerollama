"""Build native extensions (Phase 15).

Metadata and package discovery live in pyproject.toml; this file configures
``runtime.kv._kv_native`` because optional libllama linking needs compile flags
that pyproject ext-modules cannot express yet.

Optional native decode loop link (v12, auto v25):
  Auto-links libllama when found under LLAMA_CPP_ROOT (or LLAMA_CPP_LIB).
  ZEROLLAMA_KV_DECODE_LOOP=0 — force unlinked ext (default CI without llama).
  ZEROLLAMA_KV_DECODE_LOOP=1 — require libllama (fail if missing).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import Extension, setup


def _should_link_decode_loop(cpp_root: Path) -> bool:
    """Link libllama into _kv_native when available (v25 default).

    WHY auto-link: C prefill/decode is the hot path; operators should not need
    ZEROLLAMA_KV_DECODE_LOOP=1 on every build when libllama is already present.
    Set ZEROLLAMA_KV_DECODE_LOOP=0 to force an unlinked ext (CI without llama).
    Set ZEROLLAMA_KV_DECODE_LOOP=1 to require libllama (fail if missing).
    """
    explicit = os.environ.get("ZEROLLAMA_KV_DECODE_LOOP", "").strip().lower()
    if explicit in ("0", "false", "no", "off"):
        return False
    if explicit in ("1", "true", "yes", "on"):
        # Caller will raise SystemExit when _resolve_libllama_path returns None.
        return True
    return _resolve_libllama_path(cpp_root) is not None


def _resolve_llama_cpp_root() -> Path:
    raw = os.environ.get("LLAMA_CPP_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    repo = Path(__file__).resolve().parent.parent
    for candidate in (
        repo / "llama" / "llama.cpp",
        repo.parent / "llama.cpp",
    ):
        if candidate.is_dir():
            return candidate.resolve()
    return (repo.parent / "llama.cpp").resolve()


def _resolve_libllama_path(cpp_root: Path) -> Path | None:
    env = os.environ.get("LLAMA_CPP_LIB", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.is_file() else None
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    for candidate in (
        cpp_root / "build" / "bin" / f"libllama{suffix}",
        cpp_root / "build" / "lib" / f"libllama{suffix}",
        cpp_root / "build" / "bin" / "libllama.so",
        cpp_root / "build" / "lib" / "libllama.so",
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _kv_native_extension() -> Extension:
    runtime_dir = Path(__file__).resolve().parent
    sources = [
        str(runtime_dir / "native" / "kv_block_pool.c"),
        str(runtime_dir / "native" / "kv_decode_loop.c"),
        str(runtime_dir / "native" / "kv_tensor_probe.c"),
    ]
    define_macros: list[tuple[str, str | None]] = []
    include_dirs: list[str] = [str(runtime_dir / "native")]
    libraries: list[str] = []
    library_dirs: list[str] = []
    extra_link_args: list[str] = []

    cpp_root = _resolve_llama_cpp_root()
    if _should_link_decode_loop(cpp_root):
        libllama = _resolve_libllama_path(cpp_root)
        if libllama is None:
            raise SystemExit(
                "ZEROLLAMA_KV_DECODE_LOOP=1 but libllama not found; "
                "set LLAMA_CPP_LIB or build llama.cpp under LLAMA_CPP_ROOT"
            )
        define_macros.append(("ZEROLLAMA_KV_DECODE_LOOP", "1"))
        define_macros.append(("LLAMA_KV_EXT_WRITABLE_PAGE_MAP", "1"))
        define_macros.append(("LLAMA_KV_EXT_EXTERNAL_ALIAS", "1"))
        define_macros.append(("LLAMA_KV_EXT_DONOR_BUFFER", "1"))
        include_dirs.extend(
            [
                str(cpp_root / "include"),
                str(cpp_root / "ggml" / "include"),
            ]
        )
        libdir = str(libllama.parent)
        library_dirs.append(libdir)
        libraries.append("llama")
        extra_link_args.append(f"-Wl,-rpath,{libdir}")

    return Extension(
        "runtime.kv._kv_native",
        sources=sources,
        define_macros=define_macros,
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_link_args=extra_link_args,
    )


setup(ext_modules=[_kv_native_extension()])
