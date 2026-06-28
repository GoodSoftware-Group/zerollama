"""Hash-chained prefix block ids (vLLM BlockPool content addressing, token-level)."""

from __future__ import annotations

import hashlib
from typing import Iterator

_ROOT_HASH = "0" * 64


def model_scope_key(*, model_hash: str, cache_salt: str | None = None) -> str:
    """Scope prefix chains to one model layout + optional tenant salt."""
    salt = (cache_salt or "").strip()
    if salt:
        return f"{model_hash}\0{salt}"
    return model_hash


def prefix_block_hash(
    *,
    scope: str,
    parent_hash: str,
    block_index: int,
    tokens: list[int],
) -> str:
    """Chained block hash: ``SHA256(scope ‖ parent ‖ index ‖ token_ids)``."""
    h = hashlib.sha256()
    h.update(scope.encode())
    h.update(b"\x00")
    h.update(parent_hash.encode())
    h.update(block_index.to_bytes(4, "big", signed=False))
    for tok in tokens:
        h.update(int(tok).to_bytes(4, "big", signed=True))
    return h.hexdigest()


def iter_prefix_blocks(
    tokens: list[int],
    *,
    block_size: int,
    scope: str,
    max_tokens: int | None = None,
) -> Iterator[tuple[int, int, int, str, str]]:
    """Yield ``(block_index, start, end, parent_hash, block_hash)`` for full blocks."""
    if block_size <= 0:
        return
    limit = len(tokens) if max_tokens is None else min(len(tokens), max(0, max_tokens))
    if limit <= 0:
        return
    parent = _ROOT_HASH
    idx = 0
    start = 0
    while start + block_size <= limit:
        end = start + block_size
        chunk = tokens[start:end]
        bh = prefix_block_hash(
            scope=scope,
            parent_hash=parent,
            block_index=idx,
            tokens=chunk,
        )
        yield idx, start, end, parent, bh
        parent = bh
        idx += 1
        start = end


def root_hash() -> str:
    return _ROOT_HASH
