"""Golden prefix cache trace replay (offline regression)."""

from __future__ import annotations

from pathlib import Path

from runtime.kv_cache_spec import KVCacheSpec
from runtime.prefix_cache_trace import replay_trace_file

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "prefix_cache_golden.jsonl"


def test_golden_prefix_cache_trace_replays_clean():
    spec = KVCacheSpec(
        kind="sliding_window",
        effective_window=1024,
        allow_cache_prompt_base=True,
        allow_disk_persist=True,
        disk_ttl_ms=300000,
        speculative_draft=False,
    )
    assert _FIXTURE.is_file()
    mismatches = replay_trace_file(_FIXTURE, spec=spec)
    assert mismatches == [], mismatches
