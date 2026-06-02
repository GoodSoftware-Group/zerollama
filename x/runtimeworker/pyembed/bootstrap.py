"""Bootstrap eval'd into __main__ by runtime_shim.c (embedded inference)."""

from __future__ import annotations


def init_ollama_runtime_embed(port: int, runtime_parent: str) -> None:
    """Start runtime HTTP on a daemon thread. runtime_parent is repo/runtime on sys.path."""
    import sys

    if runtime_parent and runtime_parent not in sys.path:
        sys.path.insert(0, runtime_parent)

    from runtime.embed.serve_thread import start_embedded_server

    start_embedded_server(int(port))
