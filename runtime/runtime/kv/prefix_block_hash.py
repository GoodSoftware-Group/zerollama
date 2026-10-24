"""Hash-chained prefix block ids (vLLM BlockPool content addressing, token-level)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Iterator

_ROOT_HASH = "0" * 64


@dataclass(frozen=True)
class MmFeature:
    """One multimodal span in prompt-token space (vLLM ``mm_features`` analog)."""

    identifier: str
    offset: int
    length: int = 1


def parse_mm_features(raw: Any) -> tuple[MmFeature, ...]:
    """Coerce request/options payloads into ``MmFeature`` tuples."""
    if not raw:
        return ()
    out: list[MmFeature] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            feat = _parse_one_mm_feature(item)
            if feat is not None:
                out.append(feat)
    return tuple(out)


def _parse_one_mm_feature(item: Any) -> MmFeature | None:
    if isinstance(item, MmFeature):
        ident = item.identifier.strip()
        return MmFeature(ident, int(item.offset), max(0, int(item.length))) if ident else None
    if isinstance(item, str) and item.strip():
        return MmFeature(item.strip(), 0, 0)
    if not isinstance(item, dict):
        return None
    ident = item.get("identifier") or item.get("hash") or item.get("id")
    if not isinstance(ident, str) or not ident.strip():
        return None
    try:
        offset = int(item.get("offset") or 0)
        length = int(item.get("length") if item.get("length") is not None else 1)
    except (TypeError, ValueError):
        return None
    return MmFeature(ident.strip(), offset, max(0, length))


def mm_extra_keys_for_block(
    features: Sequence[MmFeature] | None,
    start_token_idx: int,
    end_token_idx: int,
) -> tuple[tuple[Any, ...], ...]:
    """Per-block MM extras: ``("mm", identifier, offset_in_block)`` (vLLM #51899)."""
    if not features:
        return ()
    extras: list[tuple[Any, ...]] = []
    for feat in features:
        ident = feat.identifier.strip()
        if not ident:
            continue
        if feat.length <= 0:
            if start_token_idx == 0:
                extras.append(("mm", ident, 0))
            continue
        if end_token_idx <= feat.offset or start_token_idx >= feat.offset + feat.length:
            continue
        extras.append(("mm", ident, int(feat.offset) - int(start_token_idx)))
    return tuple(extras)


def model_scope_key(
    *,
    model_hash: str,
    cache_salt: str | None = None,
    lora_name: str | None = None,
    extra_keys: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Scope prefix chains to one model layout + tagged extra keys.

    vLLM #51899 tags extra hash keys (``cache_salt``) so a salt cannot collide
    with another extra. vLLM #59335 includes LoRA identity in the same chain.
    """
    parts: list[str] = [model_hash]
    extras: list[tuple[str, str]] = []
    salt = (cache_salt or "").strip()
    if salt:
        extras.append(("cache_salt", salt))
    lora = (lora_name or "").strip()
    if lora:
        extras.append(("lora", lora))
    if extra_keys:
        for key, value in extra_keys:
            k, v = str(key).strip(), str(value).strip()
            if k and v:
                extras.append((k, v))
    for key, value in extras:
        parts.append(key)
        parts.append(value)
    return "\0".join(parts)


def prefix_block_hash(
    *,
    scope: str,
    parent_hash: str,
    block_index: int,
    tokens: list[int],
    extra_keys: Sequence[tuple[Any, ...]] | None = None,
) -> str:
    """Chained block hash: ``SHA256(scope ‖ parent ‖ index ‖ token_ids ‖ extras)``."""
    h = hashlib.sha256()
    h.update(scope.encode())
    h.update(b"\x00")
    h.update(parent_hash.encode())
    h.update(block_index.to_bytes(4, "big", signed=False))
    for tok in tokens:
        h.update(int(tok).to_bytes(4, "big", signed=True))
    if extra_keys:
        for item in extra_keys:
            h.update(b"\x00")
            for part in item:
                if isinstance(part, int):
                    h.update(int(part).to_bytes(4, "big", signed=True))
                else:
                    h.update(str(part).encode())
                    h.update(b"\x00")
    return h.hexdigest()


def iter_prefix_blocks(
    tokens: list[int],
    *,
    block_size: int,
    scope: str,
    max_tokens: int | None = None,
    mm_features: Sequence[MmFeature] | None = None,
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
    feats = tuple(mm_features) if mm_features else ()
    while start + block_size <= limit:
        end = start + block_size
        chunk = tokens[start:end]
        bh = prefix_block_hash(
            scope=scope,
            parent_hash=parent,
            block_index=idx,
            tokens=chunk,
            extra_keys=mm_extra_keys_for_block(feats, start, end),
        )
        yield idx, start, end, parent, bh
        parent = bh
        idx += 1
        start = end


def root_hash() -> str:
    return _ROOT_HASH
