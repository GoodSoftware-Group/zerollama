"""Prefix cache decision trace + replay (vLLM timed-trace inspired).

Enable with ``ZEROLLAMA_PREFIX_CACHE_TRACE=1``. Records JSONL lines under
``~/.cache/zerollama/prefix-cache-traces/`` for offline replay against
``KVCacheSpec`` — catches SWA/draft-spec regressions without GPU smokes.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from runtime.decode_graph_policy import decode_graph_epoch as slot_decode_graph_epoch
from runtime.kv_cache_spec import KVCacheSpec, PrefixCacheRequest, resolve_kv_cache_spec

_TRACE_LOCK = threading.Lock()
_TRACE_PATH: Path | None = None


def prefix_cache_trace_enabled() -> bool:
    from runtime.env import prefix_cache_trace_enabled as _enabled

    return _enabled()


def _trace_dir() -> Path:
    from runtime.env import prefix_cache_trace_dir

    return prefix_cache_trace_dir()


def trace_path() -> Path | None:
    global _TRACE_PATH
    if not prefix_cache_trace_enabled():
        return None
    trace_dir = _trace_dir()
    if _TRACE_PATH is None or _TRACE_PATH.parent != trace_dir:
        _TRACE_PATH = trace_dir / f"trace-{int(time.time())}.jsonl"
    return _TRACE_PATH


@dataclass(frozen=True)
class PrefixCacheTraceEntry:
    ts_ms: int
    event: str
    kind: str
    prompt_cache_key: str | None
    seq_pos: int | None
    prompt_tokens: int | None
    cache_prompt: bool
    resume_pos: int | None
    spec_method: str
    effective_window: int | None
    speculative_draft: bool
    id_slot: int | None = None
    decode_graph_epoch: int | None = None
    deny_reason: str | None = None
    drop_last_block: bool | None = None
    prefix_block_pool_enabled: bool | None = None
    prefix_block_matched_tokens: int | None = None
    radix_source_slot: int | None = None
    radix_copy_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return {k: v for k, v in out.items() if v is not None}


def record_prefix_cache_decision(
    *,
    spec: KVCacheSpec,
    prompt_cache_key: str | None,
    seq_pos: int | None,
    prompt_tokens: int | None,
    cache_prompt: bool,
    resume_pos: int | None,
    spec_method: str,
    id_slot: int | None = None,
    decode_graph_epoch: int | None = None,
    deny_reason: str | None = None,
    prefix_block_match: Any | None = None,
) -> None:
    if not prefix_cache_trace_enabled():
        return
    path = trace_path()
    if path is None:
        return
    if decode_graph_epoch is None and id_slot is not None:
        decode_graph_epoch = slot_decode_graph_epoch(id_slot)
    drop_last = None
    if (
        seq_pos is not None
        and resume_pos is not None
        and resume_pos < seq_pos
        and cache_prompt
    ):
        drop_last = True
    entry = PrefixCacheTraceEntry(
        ts_ms=int(time.time() * 1000),
        event="cache_decision",
        kind=spec.kind,
        prompt_cache_key=prompt_cache_key,
        seq_pos=seq_pos,
        prompt_tokens=prompt_tokens,
        cache_prompt=cache_prompt,
        resume_pos=resume_pos,
        spec_method=spec_method,
        effective_window=spec.effective_window,
        speculative_draft=spec.speculative_draft,
        id_slot=id_slot,
        decode_graph_epoch=decode_graph_epoch,
        deny_reason=deny_reason,
        drop_last_block=drop_last,
        prefix_block_pool_enabled=(
            prefix_block_match.get("enabled")
            if isinstance(prefix_block_match, dict)
            else None
        ),
        prefix_block_matched_tokens=(
            prefix_block_match.get("matched_tokens")
            if isinstance(prefix_block_match, dict)
            else None
        ),
    )
    line = json.dumps(entry.to_dict(), sort_keys=True)
    with _TRACE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def record_radix_share(
    *,
    prompt_cache_key: str | None,
    id_slot: int | None,
    radix_trace: dict[str, Any],
    spec_method: str = "none",
) -> None:
    """Emit ``radix_seed`` JSONL — WHY separate from ``cache_decision``: copy happens
    after policy admit; operators need donor slot + token count to prove cross-key wins."""
    if not prefix_cache_trace_enabled():
        return
    path = trace_path()
    if path is None:
        return
    entry = PrefixCacheTraceEntry(
        ts_ms=int(time.time() * 1000),
        event="radix_seed",
        kind="radix",
        prompt_cache_key=prompt_cache_key,
        seq_pos=None,
        prompt_tokens=None,
        cache_prompt=True,
        resume_pos=radix_trace.get("copy_tokens"),
        spec_method=spec_method,
        effective_window=None,
        speculative_draft=False,
        id_slot=id_slot,
        decode_graph_epoch=slot_decode_graph_epoch(id_slot) if id_slot is not None else None,
        radix_source_slot=radix_trace.get("source_slot"),
        radix_copy_tokens=radix_trace.get("copy_tokens"),
    )
    line = json.dumps(entry.to_dict(), sort_keys=True)
    with _TRACE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


@dataclass(frozen=True)
class PrefixCacheReplayMismatch:
    line: int
    field: str
    recorded: Any
    expected: Any


def replay_trace_line(
    row: dict[str, Any],
    *,
    spec: KVCacheSpec | None = None,
    cache_enabled: bool = True,
) -> list[PrefixCacheReplayMismatch]:
    """Re-evaluate one recorded decision; return mismatches (empty == pass)."""
    if row.get("event") != "cache_decision":
        return []
    req = PrefixCacheRequest(
        prompt_cache_key=row.get("prompt_cache_key"),
        seq_pos=row.get("seq_pos"),
        prompt_tokens=row.get("prompt_tokens"),
    )
    resolved = spec or resolve_kv_cache_spec(
        spec_method=str(row.get("spec_method") or "none")
    )
    expected_cache = resolved.cache_prompt_allowed(req, cache_enabled=cache_enabled)
    expected_resume = resolved.resume_pos(req, cache_enabled=cache_enabled)

    mismatches: list[PrefixCacheReplayMismatch] = []
    recorded_cache = bool(row.get("cache_prompt"))
    if recorded_cache != expected_cache:
        mismatches.append(
            PrefixCacheReplayMismatch(
                line=0,
                field="cache_prompt",
                recorded=recorded_cache,
                expected=expected_cache,
            )
        )
    recorded_resume = row.get("resume_pos")
    if recorded_resume != expected_resume:
        mismatches.append(
            PrefixCacheReplayMismatch(
                line=0,
                field="resume_pos",
                recorded=recorded_resume,
                expected=expected_resume,
            )
        )
    return mismatches


def iter_trace_file(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                yield data


def replay_trace_file(
    path: Path,
    *,
    spec: KVCacheSpec | None = None,
    cache_enabled: bool = True,
) -> list[PrefixCacheReplayMismatch]:
    """Replay a JSONL trace; return all mismatches."""
    out: list[PrefixCacheReplayMismatch] = []
    for line_no, row in enumerate(iter_trace_file(path), start=1):
        for mm in replay_trace_line(row, spec=spec, cache_enabled=cache_enabled):
            out.append(
                PrefixCacheReplayMismatch(
                    line=line_no,
                    field=mm.field,
                    recorded=mm.recorded,
                    expected=mm.expected,
                )
            )
    return out
