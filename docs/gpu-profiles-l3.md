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
| `ZEROLLAMA_LLAMA_CACHE_DISK` | smart | Unset: off on Darwin, on for Linux subprocess; `0`/`1` overrides |
| `ZEROLLAMA_LLAMA_CACHE_ROOT` | (XDG / `~/.cache/...`) | Override slot save root |
| `ZEROLLAMA_LLAMA_CACHE_TTL_MS` | `3600000` (1h) | Disk slot file TTL for llama-server `slot_*.bin` names |

**YAML profile (preferred for agents):** set `l3:` in runtime config instead of many env vars. Example: `runtime/configs/l3_agent_subprocess.yaml` — `radix_share`, `block_size`, `trace`, `lmcache_uri`. Or one env: **`ZEROLLAMA_L3_PROFILE=agent`** (loads that YAML when `ZEROLLAMA_RUNTIME_CONFIG` unset). Env still overrides any field.

**Debug tier:** **`ZEROLLAMA_DEBUG=l3`** enables prefix-cache JSONL trace + decode-graph trace logging without separate `ZEROLLAMA_PREFIX_CACHE_TRACE` / `ZEROLLAMA_DECODE_GRAPH_TRACE`. Add `infer` for phase-15 infer spans.

**Health:** `curl -s :8081/health | jq .llama_cache.runtime_env` — effective L3 profile, block pool, disk default, `n_parallel` hint.

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

### Prefix cache policy (spec × SWA)

**Why:** vLLM-inspired selective retention — not all GGUF architectures or speculative configs can safely reuse KV slots.

Check `GET /health` → `llama_cache.policy`:

| Field | Meaning |
|-------|---------|
| `kind` | `standard` \| `sliding_window` \| `hybrid` from GGUF attention metadata |
| `allow_cache_prompt` | Whether RAM `cache_prompt` is safe for this model + spec config |
| `allow_disk_persist` | Whether disk slot blobs may be written |
| `effective_window` | SWA token window for prefix matching (pure SWA models) |
| `speculative_draft` | `true` when eagle3/mtp/dflash draft spec is active — disables cache |

**Rules (Jun 2026):**

- **Draft spec** (`ZEROLLAMA_SPEC_METHOD=eagle3|mtp|dflash`, …): RAM `cache_prompt` **enabled**; disk slot blobs **disabled**; last prefix block dropped on resume (vLLM `drop_eagle_block`). **Ngram / none spec:** prefix cache remains enabled when `ZEROLLAMA_LLAMA_CACHE=1`.
- **`cache_salt`:** pass `options.cache_salt` (or `eliza.cacheSalt`, env `ZEROLLAMA_CACHE_SALT`) to isolate tenants sharing the same conversation id.
- **SWA sparse retention:** `ZEROLLAMA_PREFIX_CACHE_RETENTION_INTERVAL` — optional aligned checkpoints for pure SWA models (vLLM analog).
- **Hybrid coordinator (Jun 2026):** Gemma-style full+SWA layer groups; coordinated `cache_prompt` gate via min SWA window (`kv/hybrid_kv_coordinator.py`).
- **Prefix block pool (Jun 2026):** auto-on when L3 + `n_parallel > 1`, Radix share, or LMCache URI; hash-chained blocks verify prefix integrity before reuse. **`ZEROLLAMA_PREFIX_BLOCK_POOL=0`** disables. **`ZEROLLAMA_LMCACHE_URI=file://…`** or **`redis://host:6379/0`** (L3-R4 fleet metadata) persists block index for restart/cold-node hydration.
- **Cross-slot Radix share (Jun 2026):** `ZEROLLAMA_RADIX_PREFIX_SHARE=1` — target slots seed or **catch up** KV from a donor with a longer matching prefix block chain (`llama_memory_seq_cp` in-process; `POST /kv/seq-copy` on llama-server). **Why:** L3 pins one slot per cache key; agents sharing a system prompt but different keys otherwise repeat prefill. **v2 (L3-R2–R5):** warm catch-up on partial targets; ref-count block metadata; optional `redis://` LMCache; Gemma-style hybrid `seq_cp` when prefix ≤ SWA window (`ZEROLLAMA_RADIX_HYBRID_SEQ_COPY`, default on). Requires **vendor** llama-server (patches **0022** + **0071**). Operator guide: [radix-prefix-share.md](./radix-prefix-share.md).

Smoke: `./scripts/phase/l3_spec_cache_smoke.sh` (default `L3_SPEC_METHOD=ngram`). Draft leg: `L3_SPEC_METHOD=eagle3 LLAMA_DRAFT_MODEL=/path/draft.gguf`. Block pool: `./scripts/phase/l3_prefix_block_pool_smoke.sh`. Radix: `./scripts/phase/l3_radix_prefix_smoke.sh` (`L3_RADIX_LIVE=1` for live gate). Implementation: `runtime/runtime/kv_cache_spec.py` + `prefix_cache_policy.py` + `kv/prefix_block_pool.py` + `kv/radix_prefix_share.py`.

**Trace replay (Jun 2026):** set `ZEROLLAMA_PREFIX_CACHE_TRACE=1` to record per-request `(cache_prompt, resume_pos, seq_pos)` JSONL under `~/.cache/zerollama/prefix-cache-traces/`. Trace rows may include `prefix_block_matched_tokens` when block pool is enabled; **`radix_seed`** after successful cross-slot copy (`radix_source_slot`, `radix_copy_tokens`). Replay offline with `prefix_cache_trace.replay_trace_file()` or `./scripts/phase/l3_prefix_cache_trace_replay.sh` (golden fixture, no GPU).

**Health fields (Jun 2026):** `/health.llama_cache.kv_cache_spec`, `.spec_bind`, `.decode_graph.global_epoch`, `.prefix_block_pool`; `/health.kv_resume.prefix_cache_spec` + `.prefix_block_pool`.

**Subprocess (llama-server):** draft drop-last-block is **in-process only**. When eagle3/mtp/dflash would drop the last KV block, subprocess sets `cache_prompt=false` for that turn, bumps decode-graph epoch, and POSTs **`/cuda-graph/invalidate`** on the llama-server child. **Why HTTP:** zerollama Python cannot ctypes into the server process — ggml graph clear must run where `ctx_tgt` lives.

**In-process defense-in-depth (Jun 2026):** even if `cache_prompt=true` reaches libllama with a stale owner match, `_prepare_seq_for_decode` re-checks `KVCacheSpec` and clears the slot when SWA window is exceeded (`spec_bind_swa_block` trace + decode graph epoch bump).

### Decode graph invalidation (CUDA)

**Why:** clearing a slot changes KV contents but ggml-cuda may still hold a captured decode graph keyed by compute topology (not sequence id). Stale graph replay after L3 prefix invalidation is a correctness bug on CUDA; vLLM breaks graphs on KV change — zerollama does the same via epoch bumps + native ggml clear.

On each slot clear (`cache_prompt_disabled`, `spec_bind_swa_block`, `slot_clear`, subprocess SWA/draft deny) or session close:

1. **`decode_graph_policy.bump_decode_graph_epoch`** — increments slot + global epoch (future capture key: `slot:slot_epoch:global_epoch`).
2. **ggml CUDA graph clear** — in-process: `llama_context_cuda_graph_invalidate(ctx)` via native/ctypes; subprocess: `POST {llama-server}/cuda-graph/invalidate` (task queue → same API on `ctx_tgt`).

**Why two transports:** epoch is process-local; ggml's graph map lives in whichever process owns the `llama_context` (runtime in-process vs llama-server subprocess).

Check health:

```bash
curl -s :8081/health | jq '.llama_cache.decode_graph'
```

| Field | Meaning |
|-------|---------|
| `global_epoch` / `slot_epochs` | Invalidation generation counters |
| `capture_ready` | `false` until native graph capture is linked |
| `llama_cpp.graphs_runtime_ready` | Sibling build has CUDA + graphs enabled and env not disabled |

**Subprocess smoke (optional):** with llama-server running, `curl -s -X POST http://127.0.0.1:8082/cuda-graph/invalidate` → `{"ok":true,"backends_cleared":N}`. **Why verify:** older llama-server binaries lack the route — epoch bumps still run but ggml graphs are not cleared until rebuild.

**Operator:** rebuild sibling `../llama.cpp` after pull — `./scripts/build/build_llama_server.sh` (CUDA: `-DGGML_CUDA_GRAPHS=ON`). Kill-switch: `ZEROLLAMA_DECODE_GRAPH_INVALIDATE=0`. **Metal:** invalidate API returns 0 backends; epoch still bumps for trace/future scaffold.

Full guide: [decode-graph-invalidation.md](./decode-graph-invalidation.md).

**Subprocess SWA (Jun 2026):** without in-process `llama_memory_seq_pos_max`, the engine records each pinned-slot completion's `timings.cache_n + prompt_n + predicted_n` and uses that as `seq_pos` on the next turn's `cache_prompt` decision. After runtime restart, a one-shot `GET /slots` backfill restores `seq_pos` from llama-server's retained KV. Check `/health.kv_resume.subprocess_slot_seq_pos` after multi-turn smokes.

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
| `scripts/phase/l3_cache_smoke.sh` | Two-turn same `prompt_cache_key`; subprocess path @ 8k |
| `scripts/phase/l3_inprocess_smoke.sh` | Two-turn in-process + disk file check |
| `scripts/phase/l3_agent_bench.sh` | Multi-turn agent workload (cached vs cold) |
| `scripts/phase/l3_gate_report.sh` | PASS/FAIL from smoke JSON or merged full-gate JSON |
| `scripts/phase/l3_production_gate.sh` | Strict gate on production GGUF @ 27k ctx (`L3_PREFIX_REPEAT=150`) |
| `scripts/phase/l3_cuda_full_gate.sh` | **Production gate** — 8k smoke + 27k production + merged `gate.json` |
| `scripts/phase/l3_spec_cache_smoke.sh` | Spec decode × prefix cache policy (`/health.llama_cache.policy`); `L3_SPEC_METHOD=ngram` default |
| `scripts/phase/l3_full_gate.sh` | Platform dispatcher (CUDA full gate / Darwin smoke) |
| `RUN_E2E_L3=1` in `gpu_5080_session.sh` or `m3_metal_signoff.sh` | Sign-off hooks |

Batch keys: `options.prompt_cache_keys: ["key-a", "key-b"]` aligned with `generate_batch` prompt order. When this list is present, out-of-range indices get **no** cache key (no flat-key fallback) so unrelated batch rows do not share a slot.

### Gate sign-off (Jun 2026)

| Platform | Model | Verdict | Notes |
|----------|-------|---------|-------|
| Metal (M4 Max) | *(see m3 signoff)* | PASS / SOFT PASS | `RUN_E2E_L3=1` on `m3_metal_signoff.sh` |
| CUDA 5080 (CT 1564) | OuteTTS 1B Q8 @ 8k | **SOFT PASS** | Bridge wired: `llama_cache.enabled=true`, `derived_slot=3`, `n_parallel=2`; turn2 wall 1.384s vs turn1 1.379s (ratio 0.996 — no measurable win on tiny prefix). Artifacts: `/tmp/l3-cache-smoke.json`. |
| CUDA 5080 (CT 1564) | eliza-1 9B @ 8k | **STRICT PASS** | `l3_cache_smoke.sh`: cached turn2 **0.66s** vs no-cache **1.13s** (`L3_PREFIX_REPEAT=150`). |
| CUDA 5080 (CT 1564) | eliza-1 9B @ 27k | **PASS** | `l3_production_gate.sh`: cached **0.72s** vs no-cache **1.48s**; `turn2/turn1=1.02` (strict ratio ≤0.75 not met). Artifact: `/tmp/l3-production-gate.json`. |
| Metal (M4 Max) | vendor llama-server | **PASS (Radix live)** | `L3_RADIX_LIVE=1 ./scripts/phase/l3_radix_prefix_smoke.sh` — donor slot 0 → target 2; target **0.58s** vs donor **8.2s**; `radix_seed` 128 tokens; artifact `/tmp/l3-radix-prefix-smoke-live.json` |
| CUDA 5080 (CT 1564) | eliza-1 9B @ 8k | **PASS (Radix live)** | `CUDA_LLAMA_MODEL=… L3_RADIX_LIVE=1 ./scripts/phase/l3_radix_prefix_smoke.sh` — donor slot 1 → target 0; target **0.66s** vs donor **10.6s**; `radix_seed` 128 tokens; `/tmp/l3-radix-prefix-smoke-live.json` |
| CUDA dual-4090 (Jul 2026) | Toppy-M-7B Q5 @ 8k | **PASS** | `l3_cuda_full_gate.sh` + `L3_RUN_RADIX=1` on `/usr/local` pin **`8f114a9b`+0071**; stock fork; cached **0.53s** vs no-cache **0.73s**. Artifact: `/tmp/l3-cuda-full-gate-toppy7b/smoke-8k.json`. |
| CUDA dual-4090 (Jul 2026) | Toppy-M-7B Q5 @ 27k | **PASS** | cached **0.73s** vs no-cache **1.10s**; `turn2/turn1=0.98` (strict ≤0.75 not met). Artifact: `/tmp/l3-cuda-full-gate-toppy7b/production-27k.json`. |
| CUDA dual-4090 (Jul 2026) | Toppy-M-7B Q5 | **PASS (Radix live)** | donor **3.15s** → target **0.21s**; `radix_copy_tokens` **192**; `/tmp/l3-cuda-full-gate-toppy7b/radix-live.json`. Merged: `/tmp/l3-cuda-full-gate-toppy7b/gate.json` → **L3 CUDA full gate PASS**. |

**Why SOFT PASS is OK on 5080:** `l3_gate_report.sh` treats wiring correctness separately from latency improvement. A 1B model with a short smoke prefix is decode-bound, not prefill-bound — cache hit saves little wall time. Production agent threads with multi-kB system prompts are where L3 pays off; run `l3_agent_bench.sh` for agent-scale evidence.

**Radix product gaps (Jun 2026):** L3-R0…R5 shipped — donor seed, warm catch-up, ref-count metadata, Redis block index, hybrid SWA gate. **Still deferred:** llama-level shared KV pages, cross-node KV blob pull, Go scheduler Radix mirror. See [radix-prefix-share.md — Product gaps](./radix-prefix-share.md#product-gaps) and [ROADMAP L3-R](./ROADMAP.md#radix-v2-l3-r--product-gaps).

```bash
# CUDA production gate (5080) — recommended:
export CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf
./scripts/phase/l3_cuda_full_gate.sh
./scripts/phase/l3_gate_report.sh /tmp/l3-cuda-full-gate/gate.json

# Or inside full 5080 session:
RUN_E2E_L3=1 CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf ./scripts/gpu/gpu_5080_session.sh

# Optional spec × cache policy leg (ngram default; eagle3 needs LLAMA_DRAFT_MODEL):
L3_RUN_SPEC_CACHE=1 CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf ./scripts/phase/l3_cuda_full_gate.sh

# Individual legs:
CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf L3_PREFIX_REPEAT=150 L3_COMPARE_NO_CACHE=1 ./scripts/phase/l3_cache_smoke.sh
CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf ./scripts/phase/l3_production_gate.sh
L3_SPEC_METHOD=ngram ./scripts/phase/l3_spec_cache_smoke.sh
```

**Pass criteria (ship bar):**

| Leg | Threshold | Jun 2026 (CT 1564, eliza-1 9B) | Jul 2026 (dual-4090, Toppy-7B) |
|-----|-----------|--------------------------------|--------------------------------|
| 8k smoke | cached turn2 **<** no-cache **or** turn2 **<** turn1 | cached **0.66s** vs no-cache **1.13s** | cached **0.53s** vs no-cache **0.73s** |
| 27k production | cached **<** no-cache **or** strict ratio ≤ 0.75 | cached **0.72s** vs no-cache **1.48s** (ratio **1.02** — strict optional) | cached **0.73s** vs no-cache **1.10s** (ratio **0.98**) |

Optional supernova-class / eliza-1 9B re-validation when that GGUF is on host — not blocking L3 Done on 4090 (Toppy-7B proxy PASS).

## Status (Jun–Jul 2026)

| Platform | Status | Notes |
|----------|--------|-------|
| **Subprocess (5080 CUDA)** | **Done** — `l3_cuda_full_gate.sh` on eliza-1 9B | Optional supernova re-run |
| **Subprocess (dual-4090 CUDA)** | **Done** — Toppy-M-7B Q5 full gate + Radix (`8f114a9b`+0071) | eliza-1 9B re-gate blocked (GGUF missing) |
| **In-process (Metal)** | **Done** — RAM resume + disk parity; `l3_inprocess_smoke.sh` | `RUN_E2E_L3=1` on `m3_metal_signoff.sh` |

**Deferred:** Go-side explicit cache-key field docs (options passthrough works).

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

---

## CUDA 5080 sign-off (Jun 2026, CT 1564)

| Field | 1B Q8 @ 8k | eliza-1 9B @ 8k | eliza-1 9B @ 27k (`l3_production_gate`) |
|-------|------------|-----------------|----------------------------------------|
| Model | `Llama-OuteTTS-1.0-1B-Q8_0.gguf` | `eliza-1-9b-256k.gguf` | same |
| Script | `l3_cache_smoke.sh` | `l3_cache_smoke.sh` | `l3_production_gate.sh` |
| `num_ctx` | 8192 | 8192 | **26624** |
| Prefix | default (~64 repeats) | `L3_PREFIX_REPEAT=150` | `L3_PREFIX_REPEAT=150` |
| Turn 1 / 2 wall | 1.379s / 1.384s | 0.624s / 0.656s | 0.709s / 0.722s |
| vs no-cache turn2 | — | **1.128s** → cached **42% faster** | **1.480s** → cached **51% faster** |
| `turn2/turn1` | 1.004 | 1.05 | **1.02** (need ≤0.75 for strict ratio) |
| Verdict | **SOFT PASS** | **STRICT PASS** (`cached_faster_than_no_cache`) | **PASS** (`cached_faster_than_no_cache`) |

**Why 1B is SOFT only:** decode dominates wall time; prefix too short to beat turn-1 timing noise.

**Why 9B strict needs long prefix + compare leg:** agent-scale system prompt makes prefill the signal; `L3_COMPARE_NO_CACHE=1` proves cache vs cold slot on the same turn-2 prompt.

**Run inside Proxmox CT (CUDA):**

```bash
export CUDA_LLAMA_MODEL=/root/eliza-1-9b-256k.gguf
export L3_PREFIX_REPEAT=150
export L3_COMPARE_NO_CACHE=1
# WHY ZEROLLAMA_GPU_PROFILE_CTX=1 on Linux: l3_cache_smoke.sh sets this — without -c,
# deferred load leaves n_ctx=1024 and long prefix fails.
./scripts/phase/l3_cache_smoke.sh
./scripts/phase/l3_gate_report.sh /tmp/l3-cache-smoke.json
```

Artifact: `/tmp/l3-cache-smoke-9b.json` (or `L3_OUT=…`). Doc: [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md#gate-3--l3-agent-cache-bench).
