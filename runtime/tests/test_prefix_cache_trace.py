"""Prefix cache trace record + replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.kv_cache_spec import KVCacheSpec
from runtime.prefix_cache_trace import (
    iter_trace_file,
    prefix_cache_trace_enabled,
    record_prefix_cache_decision,
    replay_trace_file,
    replay_trace_line,
)


def test_prefix_cache_trace_disabled_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ZEROLLAMA_PREFIX_CACHE_TRACE", raising=False)
    assert prefix_cache_trace_enabled() is False


def test_record_and_replay_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_PREFIX_CACHE_TRACE", "1")
    monkeypatch.setenv("ZEROLLAMA_PREFIX_CACHE_TRACE_DIR", str(tmp_path))

    spec = KVCacheSpec(
        kind="sliding_window",
        effective_window=1024,
        allow_cache_prompt_base=True,
        allow_disk_persist=True,
        disk_ttl_ms=300000,
        speculative_draft=False,
    )
    record_prefix_cache_decision(
        spec=spec,
        prompt_cache_key="agent-1",
        seq_pos=0,
        prompt_tokens=512,
        cache_prompt=True,
        resume_pos=0,
        spec_method="none",
        id_slot=2,
        decode_graph_epoch=0,
    )
    record_prefix_cache_decision(
        spec=spec,
        prompt_cache_key="agent-1",
        seq_pos=900,
        prompt_tokens=200,
        cache_prompt=False,
        resume_pos=None,
        spec_method="none",
    )

    files = list(tmp_path.glob("trace-*.jsonl"))
    assert len(files) == 1
    rows = list(iter_trace_file(files[0]))
    assert len(rows) == 2
    assert rows[0]["cache_prompt"] is True
    assert rows[0]["id_slot"] == 2
    assert rows[0]["decode_graph_epoch"] == 0
    assert rows[1]["cache_prompt"] is False

    mismatches = replay_trace_file(files[0], spec=spec)
    assert mismatches == []


def test_replay_detects_mismatch():
    row = {
        "event": "cache_decision",
        "spec_method": "none",
        "prompt_cache_key": "k",
        "seq_pos": 2000,
        "prompt_tokens": 100,
        "cache_prompt": True,
        "resume_pos": 2000,
    }
    spec = KVCacheSpec(
        kind="sliding_window",
        effective_window=1024,
        allow_cache_prompt_base=True,
        allow_disk_persist=True,
        disk_ttl_ms=300000,
        speculative_draft=False,
    )
    mismatches = replay_trace_line(row, spec=spec)
    fields = {m.field for m in mismatches}
    assert "cache_prompt" in fields
    assert "resume_pos" in fields


def test_replay_ignores_non_decision_events(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"event": "other"}) + "\n", encoding="utf-8")
    assert replay_trace_file(path) == []
