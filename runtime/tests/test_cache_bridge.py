"""L3: prompt cache key → llama-server slot bridge."""

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from runtime.cache_bridge import (
    DEFAULT_CACHE_TTLS,
    build_model_hash,
    cache_health,
    default_slot_ttl_ms,
    derive_slot_id,
    evict_expired,
    evict_ttl_ms_for_file,
    llama_server_cache_argv,
    resolve_cache_key_from_options,
    resolve_local_cache_key,
    slot_save_path,
    ttl_ms_for_key,
)
from runtime.config import RuntimeConfig
from runtime.engine import InferenceEngine
from runtime.gpu.priority import InferencePriority
from runtime.kv.block_pool import BlockPool
from runtime.scheduler.loop import SchedulerLoop
from runtime.scheduler.scheduler import Request, Scheduler
from runtime.gpu.mutex import InferenceGpuCoordinator


def test_derive_slot_id_stable_and_bounded():
    key = "agent-thread-abc"
    assert derive_slot_id(key, 4) == derive_slot_id(key, 4)
    slot = derive_slot_id(key, 4)
    assert 0 <= slot < 4
    assert derive_slot_id(key, 1) == 0


def test_derive_slot_id_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_LLAMA_CACHE", "0")
    assert derive_slot_id("x", 8) == -1


def test_resolve_cache_key_precedence():
    opts = {
        "eliza": {
            "conversationId": "c1",
            "prefixHash": "pfx",
            "promptCacheKey": "raw",
        }
    }
    assert resolve_local_cache_key(opts) == "conv:c1"

    opts2 = {
        "eliza": {
            "promptSegments": [
                {"content": "system", "stable": True},
                {"content": "user", "stable": False},
            ],
        }
    }
    seg_key = resolve_local_cache_key(opts2)
    assert seg_key is not None
    assert seg_key.startswith("seg:")

    opts3 = {"prompt_cache_key": "flat-key"}
    assert resolve_cache_key_from_options(opts3) == "flat-key"


def test_resolve_cache_key_from_options_no_redundant_fallback():
    opts = {"eliza": {"promptCacheKey": "from-eliza"}}
    assert resolve_cache_key_from_options(opts) == "from-eliza"
    assert resolve_cache_key_from_options(None) is None


def test_build_model_hash_and_slot_path():
    h = build_model_hash(
        target_model_path="/models/a.gguf",
        cache_type_k="q8_0",
        cache_type_v="q8_0",
    )
    assert len(h) == 16
    assert slot_save_path(h) == slot_save_path(h)


def test_evict_ttl_llama_server_filenames():
    assert evict_ttl_ms_for_file("slot_0_0.bin") == default_slot_ttl_ms()
    assert evict_ttl_ms_for_file("slot_3_12.bin") == default_slot_ttl_ms()
    assert evict_ttl_ms_for_file("custom.short.bin") == ttl_ms_for_key("short")


def test_evict_expired_by_ttl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_LLAMA_CACHE_TTL_MS", str(60 * 1000))
    root = tmp_path / "slots"
    root.mkdir()
    old = root / "slot_0_0.bin"
    old.write_bytes(b"x")
    old_time = time.time() - 120
    os.utime(old, (old_time, old_time))
    fresh = root / "slot_1_0.bin"
    fresh.write_bytes(b"y")
    assert evict_expired(root, now_ms=time.time() * 1000) == 1
    assert not old.exists()
    assert fresh.exists()


def test_evict_expired_class_suffix(tmp_path: Path):
    root = tmp_path / "slots"
    root.mkdir()
    old = root / "slot0.short.bin"
    old.write_bytes(b"x")
    old_time = time.time() - (ttl_ms_for_key("short", DEFAULT_CACHE_TTLS) / 1000 + 10)
    os.utime(old, (old_time, old_time))
    fresh = root / "slot1.long.bin"
    fresh.write_bytes(b"y")
    assert evict_expired(root, now_ms=time.time() * 1000) == 1
    assert not old.exists()
    assert fresh.exists()


def test_llama_server_cache_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_LLAMA_CACHE_ROOT", str(tmp_path))
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    argv = llama_server_cache_argv(model, ["--cache-type-k", "q8_0"])
    assert argv[0] == "--slot-save-path"
    assert Path(argv[1]).is_dir()


def test_cache_health_before_model_download(tmp_path: Path):
    missing = tmp_path / "not-yet-pulled.gguf"
    out = cache_health(missing, ["--cache-type-k", "q8_0"])
    assert out["model_loaded"] is False
    assert out["model_path"] == str(missing)
    assert "model_hash" in out
    assert out["file_count"] == 0


def test_pinned_slot_same_key():
    pool = BlockPool(num_blocks=32, block_size=16, device_id=0)
    sched = Scheduler.for_pools([pool])
    coord = InferenceGpuCoordinator()
    loop = SchedulerLoop(
        scheduler=sched,
        coordinator=coord,
        pools=[pool],
        parallel_slots=4,
        assign_llama_slots=True,
    )
    key = "session-1"
    slot = derive_slot_id(key, 4)
    req1 = Request(
        request_id="a",
        prompt_tokens=[1],
        max_tokens=8,
        prompt_cache_key=key,
        kv_slot=slot,
        slot_pinned=True,
    )
    sched.add_request(req1)
    admitted = loop.tick(max_admit=1)
    assert len(admitted) == 1
    assert admitted[0].kv_slot == slot
    loop.complete(admitted[0])
    assert admitted[0].kv_slot == slot

    req2 = Request(
        request_id="b",
        prompt_tokens=[1],
        max_tokens=8,
        prompt_cache_key=key,
        kv_slot=slot,
        slot_pinned=True,
    )
    sched.add_request(req2)
    admitted2 = loop.tick(max_admit=1)
    assert len(admitted2) == 1
    assert admitted2[0].kv_slot == slot


def test_pinned_slot_busy_requeues():
    pool = BlockPool(num_blocks=64, block_size=16, device_id=0)
    sched = Scheduler.for_pools([pool])
    coord = InferenceGpuCoordinator()
    loop = SchedulerLoop(
        scheduler=sched,
        coordinator=coord,
        pools=[pool],
        parallel_slots=2,
        assign_llama_slots=True,
    )
    slot = derive_slot_id("same", 2)
    req1 = Request(
        request_id="a",
        prompt_tokens=[1],
        max_tokens=8,
        kv_slot=slot,
        slot_pinned=True,
    )
    sched.add_request(req1)
    assert len(loop.tick(max_admit=1)) == 1

    req2 = Request(
        request_id="b",
        prompt_tokens=[1],
        max_tokens=8,
        kv_slot=slot,
        slot_pinned=True,
    )
    sched.add_request(req2)
    assert loop.tick(max_admit=1) == []
    assert len(sched.waiting) == 1


@pytest.fixture
def engine(cfg_root, tmp_path: Path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    cfg = RuntimeConfig(
        host="127.0.0.1",
        port=8081,
        llama_cpp_root=cfg_root,
        llama_server_bin=None,
        llama_model=gguf,
        num_blocks=64,
        block_size=16,
        device_count=1,
    )
    return InferenceEngine(cfg)


def test_admit_one_pins_cache_key_from_options(engine, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine, "_vram_precheck_enqueue", lambda *a, **k: None)
    monkeypatch.setattr(
        engine,
        "_check_admit_policy",
        lambda opts, **k: InferencePriority.NORMAL,
    )
    with patch.object(
        engine, "resolve_num_ctx_for_request", return_value=(512, {})
    ):
        req = engine._admit_one(
            "hi",
            8,
            options={"prompt_cache_key": "sess-1"},
        )
    parallel = engine._effective_llama_parallel_slots()
    assert req.prompt_cache_key == "sess-1"
    assert req.slot_pinned is True
    assert req.kv_slot == derive_slot_id("sess-1", parallel)
    engine.loop.complete(req)

    with patch.object(
        engine, "resolve_num_ctx_for_request", return_value=(512, {})
    ):
        req2 = engine._admit_one(
            "hi again",
            8,
            options={"prompt_cache_key": "sess-1"},
        )
    assert req2.kv_slot == req.kv_slot
