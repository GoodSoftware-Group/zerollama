"""Speculative decoding plugins (llama-server --spec-type flags)."""

from runtime.speculative.plugins import (
    SpeculativeConfig,
    llama_server_args_for,
    resolve_method,
)

__all__ = ["SpeculativeConfig", "llama_server_args_for", "resolve_method"]
