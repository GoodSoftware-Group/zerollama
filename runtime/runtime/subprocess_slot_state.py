"""Track llama-server slot sequence positions for L3 SWA cache policy (subprocess).

WHY: in-process reads ``llama_memory_seq_pos_max`` for ``current_pos``; subprocess
has no shared ctx. llama-server exposes per-request context length in completion
``timings`` (``cache_n + prompt_n + predicted_n``). Reuse that on the pinned
``id_slot`` before the next ``cache_prompt`` decision.

After runtime restart, in-memory tracking is empty but llama-server may still hold
KV (``cache_prompt`` + disk slot restore). ``GET /slots`` backfills ``seq_pos``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def seq_pos_from_llama_result(result: dict[str, Any]) -> int | None:
    """Context length after a subprocess completion (llama-server timings)."""
    timings = result.get("timings")
    if not isinstance(timings, dict):
        return None
    if not any(
        k in timings for k in ("cache_n", "prompt_n", "predicted_n", "predict_n")
    ):
        return None
    try:
        cache_n = int(timings.get("cache_n") or 0)
        prompt_n = int(timings.get("prompt_n") or 0)
        predicted_n = int(timings.get("predicted_n") or timings.get("predict_n") or 0)
    except (TypeError, ValueError):
        return None
    return max(0, cache_n + prompt_n + predicted_n)


def seq_pos_from_slot_entry(entry: dict[str, Any]) -> int | None:
    """Best-effort context length from ``GET /slots`` row (idle or cached slot)."""
    if not isinstance(entry, dict):
        return None
    if "n_prompt_tokens" not in entry:
        return None
    try:
        n_prompt = int(entry["n_prompt_tokens"])
        next_t = entry.get("next_token")
        n_decoded = 0
        if isinstance(next_t, dict) and next_t.get("n_decoded") is not None:
            n_decoded = int(next_t["n_decoded"])
    except (TypeError, ValueError):
        return None
    return max(0, n_prompt + n_decoded)


def fetch_llama_server_slots(
    base_url: str,
    *,
    timeout: float = 2.0,
) -> list[dict[str, Any]] | None:
    """``GET /slots`` snapshot; ``None`` when disabled or unreachable."""
    url = f"{base_url.rstrip('/')}/slots"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode(errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    return [row for row in data if isinstance(row, dict)]


class SubprocessSlotState:
    """Last known llama-server context length per ``id_slot``."""

    def __init__(self) -> None:
        self._seq_by_slot: dict[int, int] = {}

    def seq_pos(self, slot_id: int) -> int | None:
        if slot_id < 0:
            return None
        return self._seq_by_slot.get(slot_id)

    def seq_pos_with_fallback(
        self,
        slot_id: int,
        base_url: str | None,
        *,
        timeout: float = 2.0,
    ) -> int | None:
        """Local cache first; on miss, refresh once from ``GET /slots``."""
        local = self.seq_pos(slot_id)
        if local is not None:
            return local
        if slot_id < 0 or not base_url:
            return None
        entries = fetch_llama_server_slots(base_url, timeout=timeout)
        if not entries:
            return None
        self.merge_slots(entries)
        return self.seq_pos(slot_id)

    def merge_slots(self, entries: list[dict[str, Any]]) -> None:
        """Ingest ``GET /slots`` rows (restart / disk-restore backfill)."""
        for entry in entries:
            sid = entry.get("id")
            if sid is None:
                continue
            pos = seq_pos_from_slot_entry(entry)
            if pos is None or pos <= 0:
                continue
            self._seq_by_slot[int(sid)] = pos

    def seed_seq_pos(self, slot_id: int, pos: int) -> None:
        """Pre-completion seq hint after cross-slot Radix KV seed.

        WHY: subprocess ``/slots`` backfill runs after first completion; SWA and
        draft-spec policy need ``seq_pos`` before that to avoid false SWA deny.
        """
        if slot_id >= 0 and pos > 0:
            self._seq_by_slot[slot_id] = int(pos)

    def record_completion(self, slot_id: int, result: dict[str, Any]) -> int | None:
        if slot_id < 0:
            return None
        pos = seq_pos_from_llama_result(result)
        if pos is not None:
            self._seq_by_slot[slot_id] = pos
        return pos

    def clear(self) -> None:
        self._seq_by_slot.clear()

    def snapshot(self) -> dict[str, int]:
        return {str(k): v for k, v in sorted(self._seq_by_slot.items())}
