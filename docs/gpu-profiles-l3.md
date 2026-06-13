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

1. **`SlotAllocator.release(slot)`** — frees scheduler tracking so another request can use the slot index while this one is not running.
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

### TTL eviction

llama-server writes `slot_<id>_<seq>.bin` — **no TTL class in the filename**. Eviction uses mtime against **`ZEROLLAMA_LLAMA_CACHE_TTL_MS`** (default **1 hour**).

Optional eliza-style names (`*.short.bin`, `*.long.bin`) still honor class horizons (5m / 1h / 24h) if you write them manually.

Eviction runs synchronously on llama-server startup. **Why then:** small directory (≈ one file per slot); stale files from crashed processes should not fill disk; startup is already a cold path.

---

## Environment

| Variable | Default | Effect |
|----------|---------|--------|
| `ZEROLLAMA_LLAMA_CACHE` | `1` | `0` disables key resolution, slot pinning, and `--slot-save-path` |
| `ZEROLLAMA_LLAMA_CACHE_ROOT` | (XDG / `~/.cache/...`) | Override slot save root |
| `ZEROLLAMA_LLAMA_CACHE_TTL_MS` | `3600000` (1h) | Disk slot file TTL for llama-server `slot_*.bin` names |

---

## `/health`

```json
"llama_cache": {
  "enabled": true,
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
| No prefill-skip benchmark gate yet | Need A/B on agent workloads with `-np > 1` |
| Batch `/completions_parallel` | Per-request `cache_prompt` not wired |
| In-process disk save | ctypes path lacks llama-server slot files |
| Go-side explicit field docs | `options` already flow through runtime proxy |

---

## Code map

| Module | Role | Why here |
|--------|------|----------|
| `runtime/runtime/cache_bridge.py` | Keys, hash, TTL, argv | Session logic ≠ GPU JSON (L1) |
| `runtime/runtime/engine.py` | Admit pin, health, start argv | Owns admission + llama-server lifecycle |
| `runtime/runtime/scheduler/loop.py` | `try_acquire` pinned slots | Serializes concurrent same-slot access |
| `runtime/runtime/kv/slots.py` | `SlotAllocator.try_acquire` | Phase 15 v1 dynamic slots extended for L3 |
| `runtime/runtime/worker/llama_server.py` | `cache_prompt` payload | Subprocess HTTP boundary |

Tests: `runtime/tests/test_cache_bridge.py`.

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
