# L3 — Prompt cache key → llama-server slot bridge

**Audience:** Agent/runtime integrators who want repeat system prompts and conversation threads to skip full prefill.

**Related:** [ROADMAP — borrowings L3](./ROADMAP.md#local-voice--llama-borrowings-eliza-v3), [phase15-native-kv.md](./phase15-native-kv.md), [gpu-profiles-l1.md](./gpu-profiles-l1.md) (`-np` / parallel slots), [gpu-profiles-l2.md](./gpu-profiles-l2.md) (fork KV types affect `model_hash`), eliza-v3 `cache-bridge.ts`.

---

## Why L3 exists

### The problem Phase 15 leaves open

Phase 15 v1 assigns a **dynamic** llama-server `id_slot` per request via `SlotAllocator.acquire()`. When the request finishes, `complete()` releases the slot. llama-server clears or overwrites that slot’s KV — **every turn re-prefills the full prompt**, including multi-kilobyte system prompts and tool schemas.

That is correct for stateless one-shot inference. It is wrong for **agent threads** where:

- The system prompt is identical every turn.
- Only the latest user message grows.
- Effective latency is dominated by prefill, not decode tok/s.

L1 optimizes **peak throughput** (`-b`, `-np`). L2 may shrink **KV bytes** (QJL/TurboQuant). **L3 optimizes repeat-prefix latency** — the third leg of the eliza-v3 inference borrowings track.

### Why hash keys into slots instead of a separate cache store

llama-server already implements parallel slots with in-RAM KV and optional `--slot-save-path` persistence. Reusing that machinery avoids:

- A second KV store in Python (duplicate memory accounting vs Phase 15 PA pools).
- Custom serialization of GGUF-specific KV layouts.
- Divergence from eliza-v3’s proven `deriveSlotId` + `cache_prompt` pattern.

The runtime’s job is **routing**: map a stable client key → slot index → pass `id_slot` + `cache_prompt: true` on `/completion`.

### Why subprocess-first

Disk slot save and `cache_prompt` are llama-server HTTP features. In-process backends get pinned `id_slot` / `seq_id` for RAM reuse within a loaded session, but **on-disk slot files** require the subprocess path today.

---

## How it fits L1 + Phase 15

```text
Request options (conversationId / prompt_cache_key)
        │
        ▼
resolve_local_cache_key()          ← WHY: one precedence chain, eliza-compatible
        │
        ▼
derive_slot_id(key, parallel)      ← parallel from L1 -np / gpu_profile
        │
        ▼
_admit_one() → Request.kv_slot     ← pinned before scheduler tick
        │
        ▼
SchedulerLoop.try_acquire(slot)    ← WHY: block concurrent use of same slot
        │
        ▼
/completion { id_slot, cache_prompt: true }
        │
        ▼
llama-server reuses prefix KV in slot (+ optional disk under --slot-save-path)
```

**Requires `-np > 1`:** with `parallel == 1`, every key maps to slot 0 — pinning works for one session but not concurrent agents. L1 GPU profiles set `-np` per hardware tier.

---

## Cache key precedence

From request `options` (eliza provider shape or flat zerollama fields):

| Priority | Source | Resolved key | Why this order |
|----------|--------|--------------|----------------|
| 1 | `eliza.conversationId` / `conversation_id` | `conv:<id>` | Whole thread shares one slot — simplest agent integration |
| 2 | `eliza.promptSegments` (stable prefix only) | `seg:<hash>` | Skip re-hash when client already split stable vs dynamic parts |
| 3 | `eliza.prefixHash` | `pfx:<hash>` | Client pre-computed stable prefix |
| 4 | `eliza.promptCacheKey` / `prompt_cache_key` | raw string | Explicit fallback |

Slot id: `SHA256(key) mod parallel` (always `0` when `parallel == 1`).

**Why hash mod parallel:** O(1) lookup, no central registry, same algorithm as eliza-v3. Collisions (two keys → same slot) are rare at typical `-np`; concurrent collision re-queues the second request.

---

## Pinned slot lifecycle (WHY release vs keep)

On `complete()` for a pinned request:

1. **`unregister_request_bind(slot)`** then **`SlotAllocator.release(slot)`** — see [Correctness invariants](#correctness-invariants-jun-2026-audit) for ordering.
2. **`kv_slot` kept on the finished Request** — observability only; the live state is in llama-server.
3. **Next turn** — `_admit_one()` re-derives the **same** slot from the cache key hash; llama-server still holds prefix KV in that slot.

**Why not hold the slot in the allocator between turns:** would permanently exhaust `-np` slots for idle sessions. llama-server owns idle KV; Python only serializes concurrent access to the same slot index.

---

## Disk cache

When enabled, runtime appends `--slot-save-path` on llama-server start:

```text
$XDG_CACHE_HOME/zerollama/llama-cache/<modelHash>/
  or ~/.cache/zerollama/llama-cache/<modelHash>/
```

### Why `modelHash` includes more than the GGUF path

`build_model_hash()` mixes target GGUF, draft model (MTP), and `--cache-type-k/v`. **Why:** L2 fork profiles use different KV layouts (tbq3_0 vs q8_0). Reusing the same directory would load incompatible slot blobs after a profile switch.

Paths are **canonicalized** (`resolve()` for existing files) so symlinks and absolute vs relative paths share one hash.

### TTL eviction

llama-server writes `slot_<id>_<seq>.bin` — **no TTL class in the filename**. Eviction uses mtime against **`ZEROLLAMA_LLAMA_CACHE_TTL_MS`** (default **1 hour**).

Optional eliza-style names (`*.short.bin`, `*.long.bin`) still honor class horizons (5m / 1h / 24h) if you write them manually.

Eviction runs synchronously on llama-server startup. **Why then:** small directory (≈ one file per slot); stale files from crashed processes should not fill disk; startup is already a cold path.

On startup, sibling **orphan model-hash directories** under the cache root are TTL-swept and removed when empty (e.g. after switching fork/stock or cache types).

---

## Correctness invariants (Jun 2026 audit)

These rules prevent subtle bugs when L3 shares Phase 15 slot indices with native page bind and batch admission.

### Native bind before slot release (`SchedulerLoop.complete`)

On `complete()`, the runtime must:

1. **`unregister_request_bind(slot)`** — drop Phase 15 v8 native page-table entries for this slot.
2. **`SlotAllocator.release(slot)`** — mark the slot index free for the next admit.

**Why order matters:** if release runs first, another request can `try_acquire` the same slot in the next tick while unregister is still clearing block ids — the new request could decode against stale page mappings. This is a scheduling invariant, not an llama-server detail.

Pinned L3 sessions still follow this order; llama-server keeps prefix KV in RAM/disk independently of Python’s allocator bit.

### Batch keys: no silent flat-key fallback

When `options.prompt_cache_keys` is a list, index `i` uses `keys[i]` only. Out-of-range or empty entries return **no** cache key — they do **not** fall back to `prompt_cache_key`.

**Why:** a batch of three prompts with two explicit keys must not pin rows 2 and 3 to the same slot via a shared flat key. Omit `prompt_cache_keys` entirely to reuse one flat key for every row (single-session batch).

### Canonical GGUF path in `model_hash`

`build_model_hash()` resolves symlinks and existing files before hashing.

**Why:** LM Studio import and operator symlinks often expose the same weights at different paths. Without canonicalization, disk cache fragments across duplicate directories and cold restarts miss saved slots.

### Orphan model-hash directory sweep

On llama-server start, `evict_orphaned_cache_dirs(keep_model_hash=…)` TTL-sweeps **sibling** directories under the cache root and deletes empty ones.

**Why:** switching L2 fork vs stock changes `--cache-type-k/v`, which changes `model_hash`. Old directories linger with stale blobs; sweeping on cold boot reclaims disk without touching the active hash dir.

---

## Environment

| Variable | Default | Effect |
|----------|---------|--------|
| `ZEROLLAMA_LLAMA_CACHE` | `1` | `0` disables key resolution, slot pinning, and `--slot-save-path` |
| `ZEROLLAMA_LLAMA_CACHE_DISK` | `1` | `0` disables in-process `llama_state_seq_*` disk save/load (RAM resume still works) |
| `ZEROLLAMA_LLAMA_CACHE_ROOT` | (XDG / `~/.cache/...`) | Override slot save root |
| `ZEROLLAMA_LLAMA_CACHE_TTL_MS` | `3600000` (1h) | Disk slot file TTL for llama-server `slot_*.bin` names |

---

## `/health`

```json
"llama_cache": {
  "enabled": true,
  "inprocess_disk_cache": true,
  "root": "/Users/you/.cache/zerollama/llama-cache",
  "default_ttl_ms": 3600000,
  "model_path": "/path/to/model.gguf",
  "model_loaded": true,
  "model_hash": "a1b2c3d4e5f67890",
  "slot_save_path": ".../a1b2c3d4e5f67890",
  "file_count": 2,
  "files": [ { "file": "slot_0_0.bin", "size_bytes": 123, "age_ms": 45000 } ]
}
```

**Why `model_loaded`:** operators configure `LLAMA_MODEL` before pull completes; `model_hash` and `slot_save_path` still appear so cache layout is predictable pre-download.

---

## Example request options

Eliza-shaped (via Go proxy `options` passthrough):

```json
{
  "options": {
    "eliza": {
      "conversationId": "thread-42",
      "promptCacheKey": "ignored-when-conversation-set"
    }
  }
}
```

Flat zerollama:

```json
{ "options": { "prompt_cache_key": "my-agent-v1" } }
```

Segment-based (stable system + dynamic user):

```json
{
  "options": {
    "eliza": {
      "promptSegments": [
        { "content": "You are a helpful agent...", "stable": true },
        { "content": "User: hello", "stable": false }
      ]
    }
  }
}
```

Direct `:8081` generate/chat accepts the same `options` shape.

---

## Limitations (Jun 2026)

| Gap | Why deferred |
|-----|----------------|
| Go-side explicit field docs | `options` already flow through runtime proxy |

**Closed (Jun 2026):** batch `generate_batch` + `completions_parallel` per-request `cache_prompt`; `l3_cache_smoke.sh` gate script.

**In-process disk parity (Jun 2026):** `llama_state_seq_save_file` / `load_file` on pinned slots under `~/.cache/zerollama/llama-cache/<modelHash>/slot_<id>_0.bin`. RAM resume remains Phase 15 v17 (`slot_resume_owner_key`). Disable disk only: `ZEROLLAMA_LLAMA_CACHE_DISK=0`. Smokes: `l3_inprocess_smoke.sh`, `l3_agent_bench.sh`. See [In-process disk cache](#in-process-disk-cache-inprocess-backend) below.

---

## Gate scripts

| Script | Role |
|--------|------|
| `scripts/l3_cache_smoke.sh` | Two-turn same `prompt_cache_key`; subprocess path |
| `scripts/l3_inprocess_smoke.sh` | Two-turn in-process + disk file check |
| `scripts/l3_agent_bench.sh` | Multi-turn agent workload (cached vs cold) |
| `scripts/l3_gate_report.sh` | PASS/FAIL verdict from smoke JSON |
| `RUN_E2E_L3=1` in `m3_metal_signoff.sh` | Sign-off hook |

Batch keys: `options.prompt_cache_keys: ["key-a", "key-b"]` aligned with `generate_batch` prompt order. When this list is present, out-of-range indices get **no** cache key (no flat-key fallback) so unrelated batch rows do not share a slot.

---

## In-process disk cache (inprocess backend)

The subprocess backend uses `llama-server --slot-save-path` for disk persistence. The in-process backend (`llama_backend: inprocess`) cannot do that — it owns the `llama_context` directly and has no HTTP slot-save endpoint.

Instead, the runtime calls `llama_state_seq_save_file` / `llama_state_seq_load_file` from `llama.cpp` directly via ctypes after every pinned decode.

### How it works

```text
pinned complete() — non-stream path:
  1. _decode_non_stream(...)               ← fills KV for seq_id
  2. _save_slot_cache_disk(lib, ctx, seq_id, model_hash)
        → sequence_kv_usage(lib, ctx, seq_id) → pos_max
        → llama_state_seq_save_file(ctx, path, seq_id, buf, pos_max+1)
        → writes slot_<id>_0.bin under ~/.cache/zerollama/llama-cache/<modelHash>/

cold pinned slot on next turn (sidecar restarted):
  1. is_resume = False, slot_pinned = True, model_hash set
  2. _try_restore_slot_cache_disk(lib, ctx, seq_id, model_hash, n_ctx_cap)
        → reads slot_<id>_0.bin if present
        → llama_state_seq_load_file(ctx, path, dest_seq_id, tokens_out, cap, &n_out)
        → returns restored token count
  3. decode_pos = _resolve_decode_current_pos(ctx, seq_id, None)
  4. is_resume = True → skip _clear_sequence → partial prefill from decode_pos
```

### Why `pos_max + 1`, not `prompt_tokens`

The `tokens` argument to `llama_state_seq_save_file` is the **sequence's full token history** stored alongside the KV — not just the current request's input. After multi-turn exchanges the KV holds prompt + every generated token from every prior turn.

Saving only the current input prompt's tokens would cause the blob's embedded token vector to under-report the prefix it actually covers. On restore, the `n_past` computed from that metadata would be wrong for turn 3+.

The fix: call `sequence_kv_usage()` to read the live `pos_max` from the KV cache; `n_tokens = pos_max + 1`. This matches exactly what `llama-server` does on `SLOT_SAVE`: `slot->prompt.tokens.get_tokens()` — the accumulated sequence history.

### Why disk restore attempts on any `not is_resume`, not only `decode_pos == 0`

The original guard was:
```python
if not is_resume and decode_pos == 0 and slot_pinned and model_hash:
```

`decode_pos == 0` is true for a cold-started sidecar (the normal restart case). But a still-running sidecar where `_seq_last_owner` was reset (e.g. `is_resume = False` due to mismatched owner, but `decode_pos > 0` from a previous different session) would skip disk restore silently — leaving stale KV in the slot and ignoring the valid on-disk blob.

The fix: remove the `decode_pos == 0` sub-check. When `not is_resume` and the slot is pinned, always attempt disk restore. The C load will overwrite any stale KV regardless of `decode_pos`.

### Why eviction runs once per session, not per save

`prepare_slot_cache_dir` previously called `evict_expired(save_dir)` unconditionally — on every pinned decode, for every agent turn. This scans the cache directory (stat every file) on the hot inference path.

The fix: `prepare_slot_cache_dir(model_hash, evict=False)` by default. Pass `evict=True` only at `LlamaInprocessWorker.start()` — once per sidecar lifetime, on the cold path. Agent turns pay zero eviction cost.

### File layout

```text
~/.cache/zerollama/llama-cache/<modelHash>/
  slot_0_0.bin    ← seq_id=0 save (llama_state_seq_save_file format)
  slot_1_0.bin    ← seq_id=1 save
  ...
```

`modelHash` is `build_model_hash(target_model_path, cache_type_k, cache_type_v)`. Including cache types is **critical**: L2 fork profiles use different KV tensor layouts (`qjl1_256` vs `q8_0`). Loading a blob saved with one layout under a different layout causes silent corruption or a crash.

### Environment

| Variable | Default | Effect |
|----------|---------|--------|
| `ZEROLLAMA_LLAMA_CACHE_DISK` | `1` | `0` disables in-process disk save/load; RAM resume still works |
| `ZEROLLAMA_LLAMA_CACHE` | `1` | `0` disables both RAM resume and disk cache |

### Observability

`infer_trace` events (set `ZEROLLAMA_INFER_TRACE=1`):

| Event | When |
|-------|------|
| `complete.disk_save` | After successful `llama_state_seq_save_file`; includes `nbytes` |
| `complete.disk_restore` | After successful `llama_state_seq_load_file`; includes `restored_tokens`, `decode_pos` |
| `complete.clear` | Slot cleared (not resumed, disk restore also failed or disabled) |

`/health`:
```json
"llama_cache": { "inprocess_disk_cache": true, ... }
```

---

## Code map

| Module | Role | Why here |
|--------|------|----------|
| `runtime/runtime/cache_bridge.py` | Keys, hash, TTL, argv | Session logic ≠ GPU JSON (L1) |
| `runtime/runtime/engine.py` | Admit pin, health, start argv | Owns admission + llama-server lifecycle |
| `runtime/runtime/scheduler/loop.py` | `try_acquire` pinned slots | Serializes concurrent same-slot access |
| `runtime/runtime/kv/slots.py` | `SlotAllocator.try_acquire` | Phase 15 v1 dynamic slots extended for L3 |
| `runtime/runtime/worker/llama_server.py` | `cache_prompt` payload | Subprocess HTTP boundary |
| `runtime/runtime/worker/libllama_ctypes.py` | `_save_slot_cache_disk`, `_try_restore_slot_cache_disk`, ctypes bindings for `llama_state_seq_{save,load}_file` | In-process disk parity; same save dir as subprocess path |
| `runtime/runtime/worker/llama_inprocess.py` | `slot_cache_model_hash`, `prepare_slot_cache_dir(evict=True)` at start | Derives hash from L1 argv cache types; owns session lifecycle |

Tests: `runtime/tests/test_cache_bridge.py`, `runtime/tests/test_l3_inprocess_disk.py`.

---

## Verify

```bash
# Sidecar up with L1 profile (-np > 1) and subprocess or inprocess backend
curl -s :8081/health | jq '{llama_cache, gpu_profile: .gpu_profile.n_parallel}'

# First turn (cold prefill)
curl -s :8081/api/generate -d '{
  "prompt": "System: you are helpful.\nUser: hi",
  "options": { "prompt_cache_key": "smoke-thread-1" }
}'

# Second turn (should hit cached prefix in pinned slot)
curl -s :8081/api/generate -d '{
  "prompt": "System: you are helpful.\nUser: follow up",
  "options": { "prompt_cache_key": "smoke-thread-1" }
}'
```

Compare llama-server timings / `slot_save_path` file mtimes. Disable with `ZEROLLAMA_LLAMA_CACHE=0` to confirm prefill regression.
