"""L3-R7 — content-addressed LMCache blob federation + cold restore."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from runtime.kv.lmcache_blob import (
    blob_store_health,
    materialize_blob,
    publish_slot_blob,
    reset_lmcache_blob_stats_for_tests,
    resolve_blob_root,
    sha256_file,
)
from runtime.kv.lmcache_tier import (
    reset_lmcache_tier_for_tests,
)
from runtime.kv.prefix_block_pool import (
    build_model_scope,
    get_prefix_block_pool,
    reset_prefix_block_pools_for_tests,
)
from runtime.kv.radix_blob_restore import (
    execute_blob_restore_plan,
    find_blob_restore_plan,
)


@pytest.fixture(autouse=True)
def _reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reset_lmcache_tier_for_tests()
    reset_prefix_block_pools_for_tests()
    reset_lmcache_blob_stats_for_tests()
    root = tmp_path / "lmcache"
    monkeypatch.setenv("ZEROLLAMA_LMCACHE_URI", f"file://{root}")
    monkeypatch.setenv("ZEROLLAMA_LMCACHE_BLOBS", "1")
    monkeypatch.setenv("ZEROLLAMA_PREFIX_BLOCK_POOL", "1")
    monkeypatch.setenv("ZEROLLAMA_RADIX_PREFIX_SHARE", "1")
    monkeypatch.delenv("ZEROLLAMA_LMCACHE_BLOB_ROOT", raising=False)
    yield
    reset_lmcache_tier_for_tests()
    reset_prefix_block_pools_for_tests()
    reset_lmcache_blob_stats_for_tests()


def _tokens(n: int) -> list[int]:
    return [100 + i for i in range(n)]


def test_publish_and_materialize_roundtrip(tmp_path: Path):
    src = tmp_path / "slot.bin"
    src.write_bytes(b"kv-blob-bytes-xyz")
    digest = publish_slot_blob(src)
    assert digest is not None
    assert len(digest) == 64
    root = resolve_blob_root()
    assert root is not None
    assert (root / digest[:2] / f"{digest}.bin").is_file()

    dest = tmp_path / "restored" / "slot_2_0.bin"
    assert materialize_blob(digest, dest) is True
    assert dest.read_bytes() == src.read_bytes()
    assert sha256_file(dest) == digest
    health = blob_store_health()
    assert health["enabled"] is True
    assert health["publish_total"] >= 1
    assert health["materialize_total"] >= 1


def test_register_prefix_publishes_digest(tmp_path: Path):
    scope = build_model_scope(model_hash="mh-blob")
    tokens = _tokens(512)
    slot = tmp_path / "slot_1_0.bin"
    slot.write_bytes(b"donor-slot-payload")
    pool = get_prefix_block_pool(model_scope=scope)
    hashes = pool.register_prefix(
        tokens,
        scope=scope,
        seq_pos=512,
        session_key="a",
        slot_id=1,
        blob_path=str(slot),
        block_size=512,
    )
    assert hashes
    entry = pool._blocks[hashes[0]]
    assert entry.blob_digest is not None
    assert len(entry.blob_digest) == 64

    # Fresh pool + tier hydrate should retain digest.
    reset_prefix_block_pools_for_tests()
    pool2 = get_prefix_block_pool(model_scope=scope)
    found = pool2.find_blob_prefix(tokens, scope=scope, max_tokens=512)
    assert found is not None
    assert found.blob_digest == entry.blob_digest
    assert found.matched_tokens == 512


def test_register_prefix_defers_blob_until_finalize(tmp_path: Path):
    """vLLM #48596 — metadata first; publish when slot file appears."""
    from runtime.kv.prefix_block_pool import pending_blob_finalize_count

    scope = build_model_scope(model_hash="mh-defer")
    tokens = _tokens(512)
    missing = tmp_path / "not-yet.bin"
    pool = get_prefix_block_pool(model_scope=scope)
    reg = pool.register_prefix(
        tokens,
        scope=scope,
        seq_pos=512,
        session_key="a",
        slot_id=7,
        blob_path=str(missing),
        block_size=512,
        finalize_blob=None,
    )
    assert reg.block_hashes
    assert not reg.blob_finalized
    assert pending_blob_finalize_count() >= 1
    entry = pool._blocks[reg.block_hashes[0]]
    assert entry.blob_digest is None

    missing.write_bytes(b"late-slot-bytes")
    digest = pool.finalize_slot_blob(
        scope=scope, slot_id=7, blob_path=str(missing)
    )
    assert digest and len(digest) == 64
    assert pool._blocks[reg.block_hashes[0]].blob_digest == digest
    assert pending_blob_finalize_count() == 0


def test_register_prefix_swa_store_mask_skips_unreachable(tmp_path: Path):
    scope = build_model_scope(model_hash="mh-swa")
    # 4 full blocks of size 256 → 1024 tokens; mask keeps only last two.
    tokens = _tokens(1024)
    pool = get_prefix_block_pool(model_scope=scope)
    mask = [False, False, True, True]
    reg = pool.register_prefix(
        tokens,
        scope=scope,
        seq_pos=1024,
        session_key="s",
        slot_id=1,
        block_size=256,
        store_block_mask=mask,
    )
    assert reg.skipped_swa_blocks == 2
    assert len(reg.block_hashes) == 2


def test_find_blob_restore_plan_and_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_LLAMA_CACHE_ROOT", str(tmp_path / "cache"))
    scope = build_model_scope(model_hash="mh-restore")
    tokens = _tokens(512)
    slot = tmp_path / "slot_0_0.bin"
    slot.write_bytes(b"restore-me-please!!")
    digest = publish_slot_blob(slot)
    assert digest

    pool = get_prefix_block_pool(model_scope=scope)
    pool.register_prefix(
        tokens,
        scope=scope,
        seq_pos=512,
        session_key="donor",
        slot_id=0,
        blob_path=str(slot),
        block_size=512,
    )

    plan = find_blob_restore_plan(
        tokens,
        target_slot=3,
        model_hash="mh-restore",
        seq_pos=0,
    )
    assert plan is not None
    assert plan.blob_digest == digest
    assert plan.restore_tokens == 512

    # Materialize-only path (no lib) — subprocess-style.
    trace = execute_blob_restore_plan(plan, model_hash="mh-restore")
    assert trace["ok"] is True
    assert trace["materialized"] is True
    out_path = Path(trace["path"])
    assert out_path.is_file()
    assert out_path.read_bytes() == slot.read_bytes()


def test_execute_blob_restore_loads_inprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ZEROLLAMA_LLAMA_CACHE_ROOT", str(tmp_path / "cache"))
    scope = build_model_scope(model_hash="mh-load")
    tokens = _tokens(512)
    slot = tmp_path / "slot.bin"
    slot.write_bytes(b"x" * 64)
    pool = get_prefix_block_pool(model_scope=scope)
    pool.register_prefix(
        tokens,
        scope=scope,
        seq_pos=512,
        session_key="d",
        slot_id=0,
        blob_path=str(slot),
        block_size=512,
    )
    plan = find_blob_restore_plan(
        tokens, target_slot=2, model_hash="mh-load", seq_pos=0
    )
    assert plan is not None

    lib = MagicMock()
    lib.llama_state_seq_load_file.return_value = 1
    # n_out written by ctypes — patch load helper instead
    monkeypatch.setattr(
        "runtime.worker.libllama_ctypes.load_slot_cache_disk_file",
        lambda *a, **k: 128,
    )
    trace = execute_blob_restore_plan(
        plan,
        model_hash="mh-load",
        inprocess_lib=lib,
        inprocess_ctx=MagicMock(),
        token_capacity=4096,
    )
    assert trace["ok"] is True
    assert trace["restored_tokens"] == 128


def test_blobs_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("ZEROLLAMA_LMCACHE_BLOBS", "0")
    src = tmp_path / "s.bin"
    src.write_bytes(b"nope")
    assert publish_slot_blob(src) is None


def test_peer_pull_on_materialize_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    from runtime.kv import lmcache_blob_http as blob_http

    blob_http.reset_lmcache_blob_http_stats_for_tests()

    src = tmp_path / "donor.bin"
    payload = b"peer-kv-payload-abc"
    src.write_bytes(payload)
    digest = publish_slot_blob(src)
    assert digest is not None
    root = resolve_blob_root()
    assert root is not None
    blob_path = root / digest[:2] / f"{digest}.bin"
    assert blob_path.is_file()
    # Simulate cold node: remove local blob but keep dest materialize target.
    blob_path.unlink()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == f"/api/kv/blob/{digest}":
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("X-Zerollama-Blob-Digest", digest)
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("ZEROLLAMA_LMCACHE_BLOB_HTTP", "1")
    monkeypatch.setenv("ZEROLLAMA_LMCACHE_BLOB_PEERS", f"http://127.0.0.1:{port}")
    try:
        dest = tmp_path / "cold" / "slot.bin"
        assert materialize_blob(digest, dest) is True
        assert dest.read_bytes() == payload
        assert blob_path.is_file()
        health = blob_store_health()
        assert health["http"]["peer_pull_total"] >= 1
    finally:
        server.shutdown()
        blob_http.reset_lmcache_blob_http_stats_for_tests()


def test_blob_peers_fallback_fleet_peers(monkeypatch: pytest.MonkeyPatch):
    from runtime.kv.lmcache_blob_http import lmcache_blob_peers
    from runtime.go_coordination import update_go_coordination

    monkeypatch.delenv("ZEROLLAMA_LMCACHE_BLOB_PEERS", raising=False)
    monkeypatch.setenv("ZEROLLAMA_FLEET_PEERS", "http://fleet-a:11434, http://fleet-b:11434/")
    update_go_coordination({})
    assert lmcache_blob_peers() == ["http://fleet-a:11434", "http://fleet-b:11434"]

    update_go_coordination({"lmcache_blob_peers": ["http://coord:11434"]})
    assert lmcache_blob_peers() == ["http://coord:11434"]

    monkeypatch.setenv("ZEROLLAMA_LMCACHE_BLOB_PEERS", "http://explicit:11434")
    assert lmcache_blob_peers() == ["http://explicit:11434"]
    update_go_coordination({})
