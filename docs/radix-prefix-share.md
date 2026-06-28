# Cross-slot Radix prefix share (L3)

**Audience:** Operators running multi-session agent workloads with L3 prompt cache (`-np > 1`).

**Related:** [gpu-profiles-l3.md](./gpu-profiles-l3.md), [vllm-borrowings.md](./vllm-borrowings.md), [decode-graph-invalidation.md](./decode-graph-invalidation.md), [phase17-llama-server.md](./phase17-llama-server.md).

---

## Why this exists

### The problem L3 creates for shared system prompts

L3 maps each `prompt_cache_key` to a **fixed llama-server slot** via `SHA256(key) mod n_parallel`. That design is intentional: one agent thread always lands on the same slot so turn 2+ can skip repeat prefill.

Two different agents with the **same system prompt** but **different cache keys** hash to **different slots**. Without cross-slot sharing, each cold slot pays full prefill even when another slot already holds identical prefix KV.

vLLM’s **RadixAttention** solves this with a content-addressed block pool and ref-counted KV blocks across requests. Zerollama ports a **narrow v1**: hash-chained prefix blocks + one donor slot → one target slot KV seed before decode.

### Why not full RadixAttention yet

| Full Radix (deferred) | Zerollama v1 (shipped) |
|-----------------------|-------------------------|
| Ref-counted block DAG across arbitrary request overlap | Single contiguous donor chain per target |
| Remote LMCache / Mooncake tiers | Local `file://` metadata tier only |
| Partial block copy on all memory types | Full-sequence `seq_cp` on llama-server; hybrid memory skipped in engine |

**Why v1 is enough for agents:** most agent fleets repeat one large system prompt across many conversation IDs. Donor→target seed removes the dominant prefill cost without a full scheduler rewrite.

---

## Architecture

```text
Turn 1 — Agent A (key A → slot 0)
  prefill → complete → register_prefix_block_pool(slot=0)

Turn 2 — Agent B (key B → slot 2), same token prefix, cold slot
  prefix_cache_admission()
        │
        ├─ prefix_block_pool.find_donor_slot_prefix() → donor slot 0
        ├─ execute_radix_share_plan() → llama_memory_seq_cp or POST /kv/seq-copy
        ├─ bump_decode_graph_epoch(target, reason=radix_cross_slot_seed)
        ├─ record_radix_share() → JSONL trace event radix_seed
        └─ resume_pos = copy_tokens → decode only the suffix
```

**WHY block pool first:** Radix seed without hash verification would copy stale KV when the client silently changes the system prompt. The pool denies `cache_prompt` on `prefix_block_hash_mismatch` before any cross-slot copy is considered.

**WHY decode-graph bump after copy:** ggml CUDA graphs key by topology, not sequence id. Seeding slot 2 from slot 0 changes KV without clearing captured graphs → wrong logits on CUDA. See [decode-graph-invalidation.md](./decode-graph-invalidation.md).

---

## Enable

**One-liner (agents):** `ZEROLLAMA_L3_PROFILE=agent` loads `runtime/configs/l3_agent_subprocess.yaml` (`n_parallel=4`, `l3.radix_share=true`). Env overrides any YAML field. Live smoke: `L3_RADIX_LIVE=1 ./scripts/l3_radix_prefix_smoke.sh`.

| Variable | Default | WHY |
|----------|---------|-----|
| `ZEROLLAMA_RADIX_PREFIX_SHARE` | `0` | Master switch; auto-enables prefix block pool |
| `ZEROLLAMA_PREFIX_BLOCK_POOL` | `0` (or implied by radix) | Hash-chained block index + verification |
| `ZEROLLAMA_PREFIX_CACHE_BLOCK_SIZE` | `512` | Block granularity; use `64` in smokes so short prompts register blocks |
| `ZEROLLAMA_PREFIX_CACHE_TRACE` | off | Records `radix_seed` JSONL events |
| `ZEROLLAMA_LMCACHE_TIER` | `0` | Optional `file://` metadata tier (not remote LMCache) |

**Prerequisites:**

1. L3 enabled (`ZEROLLAMA_LLAMA_CACHE=1`).
2. **`n_parallel > 1`** (L1 GPU profile or YAML `llama_parallel_slots`).
3. **Patched vendor llama-server** with `POST /kv/seq-copy` (subprocess path). **WHY vendor, not bare sibling `../llama.cpp`:** seq-copy ships in zerollama patch **0017** on the pinned vendor tree only.

```bash
# Rebuild from vendor (not sibling unless patched)
LLAMA_CPP_ROOT=vendor/llama-cpp-$(grep FETCH_HEAD= Makefile.sync | cut -d= -f2) \
  ./scripts/build_llama_server.sh
```

4. Runtime must use that binary — live smoke forces `LLAMA_CPP_ROOT` to vendor (ignores stale shell `LLAMA_CPP_ROOT=../llama.cpp`).

---

## Subprocess endpoint (`POST /kv/seq-copy`)

**Body:** `{"src_slot": 0, "dst_slot": 2, "pos_end": 128}`

| Field | Role |
|-------|------|
| `src_slot` | Donor llama sequence id (pinned session with prefix KV) |
| `dst_slot` | Cold target sequence id |
| `pos_end` | Tokens copied (metadata; server uses full-sequence copy internally — **WHY:** partial `p1` aborts on several KV backends) |

**Response:** `{"ok": true, "src_slot": 0, "id_slot": 2, "pos_end": N, "n_tokens_copied": M}`

Patch: `llama/patches/0017-ollama-kv-seq-copy-endpoint.patch`

Probe (endpoint exists, no KV mutation):

```bash
curl -s -X POST http://127.0.0.1:8082/kv/seq-copy \
  -H 'Content-Type: application/json' -d '{}'
# 400 = route present; 404 = rebuild llama-server
```

---

## Health and trace

```bash
curl -s :8081/health | jq '.llama_cache.prefix_block_pool, .kv_resume.prefix_block_pool'
```

| Field | Meaning |
|-------|---------|
| `prefix_block_pool.enabled` | Block index active |
| `prefix_block_pool.entry_count` | Registered hash blocks |
| `prefix_block_pool.radix_share.enabled` | Radix env gate |
| `llama_cpp_unified.llama_server_bin` | Must point at **vendor** build for subprocess |

**Trace (`ZEROLLAMA_PREFIX_CACHE_TRACE=1`):**

| Event | When |
|-------|------|
| `cache_decision` | Policy admit (before decode) |
| `radix_seed` | After successful cross-slot copy |

`radix_seed` fields: `radix_source_slot`, `radix_copy_tokens`, `resume_pos`, `id_slot` (target).

---

## Hybrid / recurrent models

**WHY skip:** `llama_memory_seq_cp` on hybrid/recurrent layouts (e.g. some LFM2 paths) can abort or corrupt logits. The engine returns `{"skipped": "hybrid_memory_seq_cp_unsupported"}` when `KVCacheSpec.kind == hybrid`.

Block pool verification still runs; only the KV seed step is skipped. Use a **full-KV transformer** GGUF to gate `radix_seed` in live smokes.

---

## Smoke and CI

| Script | Mode |
|--------|------|
| `./scripts/l3_radix_prefix_smoke.sh` | Offline pytest + plan replay (no GPU) |
| `L3_RADIX_LIVE=1 ./scripts/l3_radix_prefix_smoke.sh` | Two-key live gate (donor + target slots, block pool + `radix_seed`) |
| `./scripts/l3_prefix_block_pool_smoke.sh` | Block pool policy only |
| `./scripts/l3_cache_smoke.sh` | Same-key L3 slot pinning |

Live radix smoke **forces vendor llama-server**, restarts runtime + port 8082, probes `/kv/seq-copy` after donor generate.

---

## Code map

| Path | Role |
|------|------|
| `runtime/runtime/kv/radix_prefix_share.py` | Donor plan from block pool |
| `runtime/runtime/kv/radix_seq_copy.py` | in-process ctypes + subprocess HTTP |
| `runtime/runtime/kv/prefix_block_pool.py` | Hash chain + `find_donor_slot_prefix()` |
| `runtime/runtime/engine.py` | `_prefix_cache_admission()` → `_apply_radix_prefix_share()` |
| `runtime/runtime/prefix_cache_trace.py` | `record_radix_share()` |
| `runtime/runtime/subprocess_slot_state.py` | `seed_seq_pos()` after subprocess copy |
| `llama/patches/0017-ollama-kv-seq-copy-endpoint.patch` | llama-server route |
| `scripts/l3_radix_prefix_smoke.sh` | Operator gate |

---

## Deferred (product gaps)

Full gap matrix with validation status: see **[Product gaps](#product-gaps)** below.

| Item | WHY deferred |
|------|----------------|
| Full RadixAttention ref-count pool | Needs scheduler + block DAG; v1 proves donor seed value |
| Remote LMCache / Mooncake | Local `file://` tier only today |
| Partial-range seq_cp on hybrid memory | Upstream llama.cpp assert on non-full buffers |
| In-process multiseq + LFM2 Radix | Metal crash on partial copy; prefer subprocess + full-KV models |
| Warm-target partial catch-up | v1 only seeds **cold** targets (`seq_pos == 0`); warm slots skip Radix plan |
| 5080 live Radix gate | Same-key L3 signed off on 5080; cross-slot live smoke validated on Mac only (Jun 2026) |
| Fleet / cross-node donor | Donor must live in the **same** llama-server process (`n_parallel` slots) |

---

## Product gaps

**Why document gaps explicitly:** operators comparing zerollama to vLLM RadixAttention or LMCache need to know what v1 **does not** do — not just what shipped. v1 targets agent fleets with one shared system prompt and distinct conversation keys; it is not a drop-in RadixAttention scheduler.

### Scope: what v1 ships

| Capability | v1 behavior | WHY this shape |
|------------|---------------|----------------|
| Donor selection | Longest hash-matched **contiguous** chain from prompt start | Matches dominant agent pattern (shared system prompt prefix); avoids scheduler rewrite |
| Target state | **Cold slot only** (`seq_pos == 0`, no resume) | Copy semantics are “seed empty KV”; warm partial merge needs ref-count DAG (v2) |
| Process boundary | Same llama-server / in-process ctx only | `seq_cp` is in-process memory; no cross-node KV handoff without remote tier |
| Verification | Prefix block pool **before** copy | Without hash chain, silent prompt drift would copy stale logits |
| Post-copy | Decode-graph epoch bump on target | ggml CUDA graphs ignore sequence id — seeded KV without invalidation → wrong logits |
| Default | **Off** (`ZEROLLAMA_RADIX_PREFIX_SHARE=0`) | Surprises operators who did not opt into cross-slot copy; agent profile turns it on |

### Scope: what v1 does not ship

| Gap | User impact | WHY deferred |
|-----|-------------|--------------|
| **Ref-count block DAG** | No arbitrary partial overlap across many concurrent requests; one donor chain per cold target | Full RadixAttention needs block allocator + scheduler integration in Go and Python |
| **Warm-target catch-up** | Target slot with partial KV cannot “sync up” to shared prefix via Radix | Requires merge policy + partial `seq_cp` semantics not defined in v1 |
| **Hybrid / recurrent GGUF** | Block pool verifies; **`seq_cp` skipped** | `llama_memory_seq_cp` aborts or corrupts on hybrid layouts (LFM2, etc.) |
| **Partial-range copy** | API `pos_end` is metadata; server copies **full sequence** | Partial `p1` ranges abort on several llama.cpp KV backends today |
| **Remote LMCache / Mooncake / NIXL** | Only local `file://` block **metadata** tier | Remote blob federation needs connector + fleet routing — not agent-local v1 |
| **Fleet Radix** | Management node does not route by shared-prefix residency | Session-key affinity + L3 slots cover most single-node agent threads; fleet layer is warm-model first |
| **Cross-node donor** | Donor on node A cannot seed target on node B | KV lives in process VRAM; remote tier would need blob pull + load path |
| **Go-side Radix** | All logic in Python engine admission | Go scheduler lacks block pool + live slot KV visibility on decode path |
| **Per-slot CUDA graph capture** | Invalidation after Radix works; capture stub remains | ggml internal capture API not exposed; invalidation is correctness minimum |

### Validation status (Jun 2026)

| Gate | Platform | Status | Notes |
|------|----------|--------|-------|
| Offline pytest + plan replay | CI / any host | **PASS** | Default `./scripts/l3_radix_prefix_smoke.sh` |
| Live two-key smoke | **Mac Metal** (vendor llama-server) | **PASS** | `L3_RADIX_LIVE=1`; donor slot 0 → target 2; `radix_seed` 128 tokens; target ~0.52s vs donor ~8.83s |
| Live two-key smoke | **CUDA 5080** | **Pending** | Same-key L3 signed off; add `L3_RADIX_LIVE=1` to 5080 session when operator runs cross-slot gate |
| Hybrid model live | — | **N/A** | Use full-KV transformer GGUF for Radix gates |

**WHY Mac live first:** agent profile + vendor subprocess path was debugged on Darwin (`cli.py` L3 profile fix, patch doctor). 5080 re-run is operational validation, not new architecture.

### Operator checklist (avoid footguns)

1. **`ZEROLLAMA_L3_PROFILE=agent`** or `ZEROLLAMA_RADIX_PREFIX_SHARE=1`
2. **`n_parallel > 1`** — Radix needs multiple slots in one server
3. **Vendor llama-server** with patch **0017** — `./scripts/llama_patch_doctor.sh`
4. **Full-KV transformer** for live smokes — hybrid models skip copy
5. **Distinct cache keys, same system prompt** — same key uses same-slot L3, not Radix
6. **Cold second thread** — turn 1 on key B after turn 1 on key A completed on donor slot

### Roadmap (next milestones)

See [ROADMAP — Radix v2 (L3-R)](./ROADMAP.md#radix-v2-l3-r--product-gaps). Suggested order:

1. **L3-R1** — 5080 live Radix gate in sign-off table
2. **L3-R2** — Warm-target partial catch-up when `0 < seq_pos < donor_matched`
3. **L3-R3** — Ref-count block pool + scheduler hooks (full RadixAttention parity)
4. **L3-R4** — Remote LMCache connector + fleet prefix metadata
5. **L3-R5** — Hybrid-memory copy path (upstream or model-specific)

