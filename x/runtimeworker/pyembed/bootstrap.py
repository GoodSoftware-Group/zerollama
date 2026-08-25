"""Bootstrap eval'd into __main__ by runtime_shim.c (embedded inference)."""

from __future__ import annotations


def init_ollama_runtime_embed(port: int, runtime_parent: str) -> None:
    """Start runtime HTTP on a daemon thread. runtime_parent is repo/runtime on sys.path."""
    import os
    import sys

    if runtime_parent and runtime_parent not in sys.path:
        sys.path.insert(0, runtime_parent)

    # WHY: Go setenv(ZEROLLAMA_RUNTIME_EMBED_BOOT) after training Py_Initialize does not
    # update Python's os.environ mapping; sync from libc so /health echoes embed_boot.
    try:
        import ctypes

        getenv = ctypes.CDLL(None).getenv
        getenv.argtypes = [ctypes.c_char_p]
        getenv.restype = ctypes.c_char_p
        raw = getenv(b"ZEROLLAMA_RUNTIME_EMBED_BOOT")
        if raw:
            os.environ["ZEROLLAMA_RUNTIME_EMBED_BOOT"] = raw.decode("utf-8", "replace")
    except Exception:
        pass

    from runtime.embed.serve_thread import start_embedded_server

    start_embedded_server(int(port))
