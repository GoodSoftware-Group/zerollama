"""Parse llama-server SSE completion stream."""

from __future__ import annotations

import json
from typing import Any, Iterator


def iter_llama_sse_lines(raw: Iterator[bytes]) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from ``data: {...}`` SSE lines."""
    for chunk in raw:
        for line in chunk.decode(errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield data
