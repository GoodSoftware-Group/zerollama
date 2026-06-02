"""Select Python vs native KV block pool (Phase 15)."""

from __future__ import annotations

import logging
import os
from typing import Any, Type

from runtime.kv._py_block_pool import BlockPool as PyBlockPool
from runtime.kv._py_block_pool import BlockPoolError as PyBlockPoolError
from runtime.kv._py_block_pool import SequenceBlockTable

_log = logging.getLogger(__name__)

_NATIVE: Any | None = None
_NATIVE_TRIED = False
_BACKEND: str | None = None
_WARNED_NATIVE_UNAVAILABLE = False


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def native_requested() -> bool:
    return _truthy("ZEROLLAMA_RUNTIME_KV_NATIVE")


def _load_native() -> Any | None:
    global _NATIVE, _NATIVE_TRIED
    if _NATIVE_TRIED:
        return _NATIVE
    _NATIVE_TRIED = True
    try:
        from runtime.kv import _kv_native as mod

        _NATIVE = mod
    except ImportError:
        _NATIVE = None
    return _NATIVE


def native_available() -> bool:
    return _load_native() is not None


def _warn_native_unavailable_once() -> None:
    global _WARNED_NATIVE_UNAVAILABLE
    if _WARNED_NATIVE_UNAVAILABLE or not native_requested() or native_available():
        return
    _WARNED_NATIVE_UNAVAILABLE = True
    _log.warning(
        "ZEROLLAMA_RUNTIME_KV_NATIVE is set but runtime.kv._kv_native is not built; "
        "using Python block pool. Build: cd runtime && python3 setup.py build_ext --inplace"
    )


def reset_kv_backend_cache() -> None:
    """Test helper: re-read env and extension availability."""
    global _NATIVE, _NATIVE_TRIED, _BACKEND, _WARNED_NATIVE_UNAVAILABLE
    _NATIVE = None
    _NATIVE_TRIED = False
    _BACKEND = None
    _WARNED_NATIVE_UNAVAILABLE = False


def kv_backend_name() -> str:
    """Effective backend: ``python`` or ``native``."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    _warn_native_unavailable_once()
    if native_requested() and native_available():
        _BACKEND = "native"
    else:
        _BACKEND = "python"
    return _BACKEND


def kv_backend_health() -> dict[str, Any]:
    """Fields for /health: effective backend + whether native was requested/available."""
    _warn_native_unavailable_once()
    requested = native_requested()
    available = native_available()
    backend = kv_backend_name()
    out: dict[str, Any] = {
        "backend": backend,
        "native_requested": requested,
        "native_available": available,
    }
    if requested and not available:
        out["note"] = (
            "ZEROLLAMA_RUNTIME_KV_NATIVE set but extension missing; "
            "run: cd runtime && python3 setup.py build_ext --inplace"
        )
    return out


def block_pool_class() -> Type[Any]:
    _warn_native_unavailable_once()
    mod = _load_native()
    if native_requested() and mod is not None:
        return mod.BlockPool
    return PyBlockPool


def block_pool_error_type() -> type[BaseException]:
    mod = _load_native()
    if native_requested() and mod is not None:
        return mod.BlockPoolError
    return PyBlockPoolError


def create_block_pool(
    *,
    num_blocks: int,
    block_size: int,
    device_id: int = 0,
) -> Any:
    """Construct a block pool using the configured backend (preferred over BlockPool())."""
    cls = block_pool_class()
    return cls(num_blocks=num_blocks, block_size=block_size, device_id=device_id)


__all__ = [
    "BlockPoolError",
    "SequenceBlockTable",
    "create_block_pool",
    "kv_backend_name",
    "kv_backend_health",
    "native_requested",
    "native_available",
    "reset_kv_backend_cache",
    "block_pool_class",
    "block_pool_error_type",
]
