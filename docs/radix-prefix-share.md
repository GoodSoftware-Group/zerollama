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
| Ref-count block DAG | **Partial (L3-R3)** — multi-holder metadata + best donor; physical KV still one `seq_cp` per seed |
| Remote LMCache / Mooncake tiers | **`redis://` metadata (L3-R4)** + content-addressed blobs (**L3-R7**) + **HTTP peer pull (L3-R10)**; NIXL RDMA still deferred |
| Partial block copy on all memory types | Full-sequence `seq_cp`; **Gemma-style hybrid (L3-R5)** when prefix ≤ SWA window; attn+recurrent still conservative |

**Why v1 is enough for agents:** most agent fleets repeat one large system prompt across many conversation IDs. Donor→target seed removes the dominant prefill cost without a full scheduler rewrite.

### Session QoS interaction (`cache_reset` / parent prefer)

**Why document here:** Radix admission historically ran even when same-slot `cache_prompt` was denied (SWA window / shorter shared prefix). That is correct for catch-up — and wrong when the client set `options.zerollama.cache_reset: true`.

| Intent | Radix behavior | Why |
|--------|----------------|-----|
| Normal / SWA deny | May still seed from donor | Shorter matched prefix can fit the window |
| `cache_reset: true` | **Skipped entirely** | Client asked for no KV reuse under the same key this turn |
| `session_parent` / `session_group` | Prefer that donor on **equal-length** ties only | Still hash-verified; relation is tie-break, not override |

Full loop: [agent-qos-and-project-tracking.md](./agent-qos-and-project-tracking.md#session--cache-great-loop-jul-2026).

---

## Architecture

```text
Turn 1 — Agent A (key A → slot 0)
  prefill → complete → register_prefix_block_pool(slot=0)

Turn 2 — Agent B (key B → slot 2), same token prefix
  prefix_cache_admission()
        │
        ├─ prefix_block_pool.find_donor_slot_prefix() → donor slot 0
        ├─ radix_seq_copy_allowed() — hybrid/SWA gate (L3-R5)
        ├─ execute_radix_share_plan() → llama_memory_seq_cp or POST /kv/seq-copy
        ├─ bump_decode_graph_epoch(target, reason=radix_cross_slot_seed)
        ├─ record_radix_share() → JSONL trace event radix_seed
        └─ resume_pos = copy_tokens → decode only the suffix

Warm target (L3-R2): when slot 2 already holds ``[0, seq_pos)`` but donor matched
further, ``verify_target_slot_prefix`` then copy ``[0, donor_matched)`` (full
seq-copy clears target first — redundant tail re-copy is acceptable).
```

**WHY block pool first:** Radix seed without hash verification would copy stale KV when the client silently changes the system prompt. The pool denies `cache_prompt` on `prefix_block_hash_mismatch` before any cross-slot copy is considered.

### Radix v2 track (L3-R2–R5) — Jun 2026

**Why a v2 track:** v1 Radix solved the dominant agent case (cold target, same system prompt, different cache keys) but left four operator-visible gaps vs vLLM RadixAttention. Each milestone closes one gap without rewriting the Go scheduler or llama.cpp KV allocator.

| Milestone | What shipped | WHY it mattered |
|-----------|--------------|-----------------|
| **L3-R2** Warm catch-up | `verify_target_slot_prefix`, donor search with `min_matched=target_pos` | Agent threads often extend a shared prefix on a warm slot while another slot already prefilled further — v1 skipped all `seq_pos > 0` targets |
| **L3-R3** Ref-count metadata | `holder_slots`, `release_slot_holders`, `_best_donor_from_chain` | Two slots registering the same block hash broke donor pick (short vs full overlap); vLLM ref-counts blocks — we ref-count metadata first |
| **L3-R4** Redis LMCache | `lmcache_redis.py` (stdlib RESP), `ZEROLLAMA_LMCACHE_URI=redis://…` | Fleet restarts and cold nodes lost block index on local disk only; Redis federates **metadata** (KV blobs still per-host slot files) |
| **L3-R5** Hybrid Radix | `radix_seq_copy_policy.py`, `ZEROLLAMA_RADIX_HYBRID_SEQ_COPY` | v1 blanket-skipped all `kind=hybrid`; Gemma-style full+SWA layouts are safe when copy ≤ SWA window — attn+recurrent (LFM2) keeps operator kill-switch |

**Still deferred after L3-R5:** llama-level shared KV pages (not metadata-only), NIXL/Mooncake RDMA (HTTP peer pull is L3-R10), `DecodeGraphCache.lookup` capture stub. See [Product gaps](#product-gaps).

---


**One-liner (agents):** `ZEROLLAMA_L3_PROFILE=agent` loads `runtime/configs/l3_agent_subprocess.yaml` (`n_parallel=4`, `l3.radix_share=true`, `l3.kv_unified=true`). Env overrides any YAML field. Size `-c` for **sum** of concurrent live tokens (shared cell pool). Live smoke: `L3_RADIX_LIVE=1 ./scripts/phase/l3_radix_prefix_smoke.sh`.

| Variable | Default | WHY |
|----------|---------|-----|
| `ZEROLLAMA_RADIX_PREFIX_SHARE` | `0` | Master switch; auto-enables prefix block pool |
| `ZEROLLAMA_RADIX_MEDIA_SEQ_COPY` | `1` | Subprocess `/kv/seq-copy` `allow_media` (clone mtmd); `0`/`text_only` = clamp before first media |
| `ZEROLLAMA_RADIX_HYBRID_SEQ_COPY` | `1` | Allow hybrid GGUF `seq_cp` when copy fits SWA window (L3-R5); `0` = conservative skip |
| `ZEROLLAMA_PREFIX_BLOCK_POOL` | `0` (or implied by radix) | Hash-chained block index + verification |
| `ZEROLLAMA_PREFIX_CACHE_BLOCK_SIZE` | `512` | Block granularity; use `64` in smokes so short prompts register blocks |
| `ZEROLLAMA_PREFIX_CACHE_TRACE` | off | Records `radix_seed` JSONL events |
| `ZEROLLAMA_LMCACHE_URI` | — | `file://…` or `redis://host:6379/0` — prefix block metadata tier |
| `ZEROLLAMA_LMCACHE_TTL_SEC` | — | Optional Redis key TTL (seconds) |
| `ZEROLLAMA_LMCACHE_TIER` | *(deprecated)* | Alias for URI-only enable; prefer `ZEROLLAMA_LMCACHE_URI` |

**Prerequisites:**

1. L3 enabled (`ZEROLLAMA_LLAMA_CACHE=1`).
2. **`n_parallel > 1`** (L1 GPU profile or YAML `llama_parallel_slots`).
3. **Patched vendor llama-server** with `POST /kv/seq-copy` (subprocess path). **WHY vendor, not bare sibling `../llama.cpp`:** seq-copy ships in zerollama patch **0017** on the pinned vendor tree only.

```bash
# 5080 / CT 1564 (preferred)
source ./scripts/gpu/5080_env.sh
5080_build_vendor_llama_server

# Manual (any host)
make -f Makefile.sync vendor
LLAMA_CPP_ROOT=vendor/llama-cpp-$(grep '^FETCH_HEAD=' Makefile.sync | cut -d= -f2) \
  ./scripts/build/build_llama_server.sh
./scripts/vendor/llama_patch_doctor.sh
```

4. Runtime must use that binary — live smoke forces `LLAMA_CPP_ROOT` to vendor (ignores stale shell `LLAMA_CPP_ROOT=../llama.cpp`).

---

## Subprocess endpoint (`POST /kv/seq-copy`)

**Body:** `{"src_slot": 0, "dst_slot": 2, "pos_end": 128, "allow_media": true}`

| Field | Role |
|-------|------|
| `src_slot` | Donor llama sequence id (pinned session with prefix KV) |
| `dst_slot` | Cold target sequence id |
| `pos_end` | Requested tokens; server may clamp (media mid-chunk / text-only) and report effective `n_tokens_copied` |
| `allow_media` | Default **true** — clone mtmd chunks via `keep_first`. **false** — pure-text clamp before first media (`ZEROLLAMA_RADIX_MEDIA_SEQ_COPY=0`) |

**Response:** `{"ok": true, "src_slot": 0, "id_slot": 2, "pos_end": N, "n_tokens_copied": M}`

Patches: `0023` (route) + **`0074`** (empty-KV fix) + **`0090`** (Jul 2026 media-aware clone; replaces `check_no_mtmd` on SEQ_COPY).

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
| `prefix_block_pool.block_hashes` | Newest-first capped sample for fleet LA13 (`ZEROLLAMA_RADIX_HEALTH_HASH_CAP`) |
| `prefix_block_pool.radix_share.enabled` | Radix env gate |
| `prefix_block_pool.lmcache_blobs` | Blob store + L3-R10 `http` peer-pull counters |
| `llama_cpp_unified.llama_server_bin` | Must point at **vendor** build for subprocess |

**Trace (`ZEROLLAMA_PREFIX_CACHE_TRACE=1`):**

| Event | When |
|-------|------|
| `cache_decision` | Policy admit (before decode) |
| `radix_seed` | After successful cross-slot copy |

`radix_seed` fields: `radix_source_slot`, `radix_copy_tokens`, `resume_pos`, `id_slot` (target).

---

## Hybrid / SWA models (L3-R5)

**WHY gated:** True attn+recurrent memory (some LFM2 paths) can abort or corrupt `llama_memory_seq_cp`. Gemma-style **hybrid** GGUF layouts (full + SWA layers) are allowed when the donor copy fits the coordinated SWA window.

**WHY not retention on Radix:** `ZEROLLAMA_PREFIX_CACHE_RETENTION_INTERVAL` sparsifies same-slot resume checkpoints. Radix copies live dense donor KV (vLLM #47782 Marconi × retention). Admission also tries Radix when full-prompt `cache_prompt` was denied for window — a shorter matched prefix may still fit.

| Skip reason | Meaning |
|-------------|---------|
| `hybrid_seq_copy_disabled` | `ZEROLLAMA_RADIX_HYBRID_SEQ_COPY=0` |
| `hybrid_missing_effective_window` | Hybrid without a resolved SWA window |
| `hybrid_target_past_swa_window` | Warm target `seq_pos` already ≥ window |
| `hybrid_prefix_exceeds_swa_window` | `copy_tokens` > `effective_window` |

Block pool verification still runs when copy is skipped. Live smokes warn (soft-pass) when a hybrid model produces no `radix_seed`; full-KV models hard-gate `radix_seed`.

---

## Smoke and CI

| Script | Mode |
|--------|------|
| `./scripts/phase/l3_radix_prefix_smoke.sh` | Offline pytest + plan replay (no GPU) |
| `L3_RADIX_LIVE=1 ./scripts/phase/l3_radix_prefix_smoke.sh` | Two-key live gate (donor + target slots, block pool + `radix_seed`) |
| `./scripts/phase/l3_prefix_block_pool_smoke.sh` | Block pool policy only |
| `./scripts/phase/l3_cache_smoke.sh` | Same-key L3 slot pinning |

Live radix smoke **forces vendor llama-server**, restarts the **smoke** runtime + its llama-server port (`ZEROLLAMA_RUNTIME_URL` port and port+1 — never hardcode `:8081`/`:8082` beside production), probes `/kv/seq-copy` after donor generate.

---

## Code map

| Path | Role |
|------|------|
| `runtime/runtime/kv/radix_prefix_share.py` | Donor plan from block pool |
| `runtime/runtime/kv/radix_seq_copy.py` | in-process ctypes + subprocess HTTP |
| `runtime/runtime/kv/radix_seq_copy_policy.py` | Hybrid/SWA `seq_cp` admission (L3-R5) |
| `runtime/runtime/kv/prefix_block_pool.py` | Hash chain + `find_donor_slot_prefix()` |
| `runtime/runtime/engine.py` | `_prefix_cache_admission()` → `_apply_radix_prefix_share()` |
| `runtime/runtime/prefix_cache_trace.py` | `record_radix_share()` |
| `runtime/runtime/subprocess_slot_state.py` | `seed_seq_pos()` after subprocess copy |
| `llama/patches/0017-ollama-kv-seq-copy-endpoint.patch` | llama-server route |
| `scripts/phase/l3_radix_prefix_smoke.sh` | Operator gate |

---

## Deferred (product gaps)

Full gap matrix with validation status: see **[Product gaps](#product-gaps)** below.

| Item | WHY deferred |
|------|----------------|
| Full RadixAttention ref-count pool | Needs scheduler + block DAG; v1 proves donor seed value |
| Remote LMCache / Mooncake | **`redis://` metadata (L3-R4)**; KV blobs remain per-node slot files |
| Partial-range seq_cp on hybrid memory | Upstream llama.cpp assert on non-full buffers |
| In-process multiseq + LFM2 Radix | Metal crash on partial copy; prefer subprocess + full-KV models |
| Warm-target partial catch-up | **Done (L3-R2)** — target with verified partial KV catches up when donor matched > `seq_pos`; full seq-copy (clear+recopy) |
| 5080 live Radix gate | **PASS (Jun 2026)** — eliza-1 9B @ CT 1564: donor **10.6s** → target **0.66s**, `radix_seed` 128 tok |
| Fleet / cross-node donor | Donor must live in the **same** llama-server process (`n_parallel` slots) |

---

## Product gaps

**Why document gaps explicitly:** operators comparing zerollama to vLLM RadixAttention or LMCache need to know what v1 **does not** do — not just what shipped. v1 targets agent fleets with one shared system prompt and distinct conversation keys; it is not a drop-in RadixAttention scheduler.

### Scope: what v1 ships

| Capability | v1 behavior | WHY this shape |
|------------|---------------|----------------|
| Donor selection | Longest hash-matched **contiguous** chain from prompt start | Matches dominant agent pattern (shared system prompt prefix); avoids scheduler rewrite |
| Target state | **Cold** or **warm behind donor** | Cold: seed empty KV. Warm (L3-R2): verify target slot owns prefix blocks, copy when donor matched > `seq_pos` |
| Process boundary | Same llama-server / in-process ctx only | `seq_cp` is in-process memory; no cross-node KV handoff without remote tier |
| Verification | Prefix block pool **before** copy | Without hash chain, silent prompt drift would copy stale logits |
| Post-copy | Decode-graph epoch bump on target | ggml CUDA graphs ignore sequence id — seeded KV without invalidation → wrong logits |
| Default | **Off** (`ZEROLLAMA_RADIX_PREFIX_SHARE=0`) | Surprises operators who did not opt into cross-slot copy; agent profile turns it on |

### Scope: what v1 does not ship

| Gap | User impact | WHY deferred |
|-----|-------------|--------------|
| **Ref-count block DAG** | Multiple slots can reference the same block hash | **Done (L3-R3 metadata)** — `holder_slots`, `release_slot_holders`, best-donor pick; not llama-level shared KV pages |
| **Warm-target catch-up** | Target with partial KV syncs when donor has longer shared prefix | **Done (L3-R2)** — `verify_target_slot_prefix` + skip target-owned blocks in donor search |
| **Hybrid / recurrent GGUF** | **Gemma-style hybrid (L3-R5):** `seq_cp` when copy ≤ SWA window; attn+recurrent: set `ZEROLLAMA_RADIX_HYBRID_SEQ_COPY=0` | True recurrent memory can still abort `seq_cp`; operator kill-switch retained |
| **Partial-range copy** | API `pos_end` is metadata; server copies **full sequence** | Partial `p1` ranges abort on several llama.cpp KV backends today |
| **Remote LMCache / NIXL blobs** | **L3-R4** redis metadata + **L3-R7** content-addressed slot blobs + **L3-R10** HTTP peer pull | NIXL/Mooncake RDMA transport still deferred |
| **Physical shared KV pages (L3-R6)** | **L3-R6a + L3-R6b Done** — cells + tensor fork + used-cell-range pages | Sparse/sub-capacity VRAM alloc still out of scope |
| **L3-R6 readiness** | `/health.kv_resume.l3_r6_metadata` | `complete: true` when metadata path wired |
| **Fleet Radix** | **L3-R8 + L3-R9:** status mirror, soft residency, content-hash longest-prefix assign | NIXL/Mooncake RDMA still deferred |
| **Cross-node donor** | Donor on node A cannot seed target on node B via live `seq_cp` | **L3-R10** HTTP blob pull + load path; live VRAM `seq_cp` remains same-process |
| **Go-side Radix** | **L3-R8 mirror** on status/fleet score; seed/`seq_cp` stay Python | Decode-path Radix remains Python-only |
| **Per-slot CUDA graph capture** | Invalidation after Radix works; capture stub remains | ggml internal capture API not exposed; invalidation is correctness minimum |

### Validation status (Jun 2026)

| Gate | Platform | Status | Notes |
|------|----------|--------|-------|
| Offline pytest + plan replay | CI / any host | **PASS** | Default `./scripts/phase/l3_radix_prefix_smoke.sh` |
| Live two-key smoke | **Mac Metal** (vendor llama-server) | **PASS** | `L3_RADIX_LIVE=1`; donor slot 0 → target 2; `radix_seed` 128 tokens; target ~0.52s vs donor ~8.83s |
| Live two-key smoke | **CUDA 5080** | **Pending** | Same-key L3 signed off; add `L3_RADIX_LIVE=1` to 5080 session when operator runs cross-slot gate |
| Hybrid model live | — | **Soft-pass** | Short prompts within SWA window may `radix_seed`; smoke warns when absent |

**WHY Mac live first:** agent profile + vendor subprocess path was debugged on Darwin (`cli.py` L3 profile fix, patch doctor). 5080 re-run is operational validation, not new architecture.

### Operator checklist (avoid footguns)

1. **`ZEROLLAMA_L3_PROFILE=agent`** or `ZEROLLAMA_RADIX_PREFIX_SHARE=1`
2. **Unified KV (L3-R6):** agent YAML or Radix couple (`ZEROLLAMA_KV_UNIFIED_WITH_RADIX`, default on) — size `n_ctx` for sum of live tokens; kill with `ZEROLLAMA_KV_UNIFIED=0`; check `/health.kv_resume.kv_unified_sizing` + `kv_unified_source`; optional fail-closed: `ZEROLLAMA_KV_UNIFIED_STRICT=1`
3. **Federated blobs (L3-R7/R10/R11):** shared `ZEROLLAMA_LMCACHE_BLOB_ROOT` **or** HTTP pull — set `ZEROLLAMA_LMCACHE_BLOB_PEERS` **or** reuse `ZEROLLAMA_FLEET_PEERS` on nodes (Go coordination mirrors peers to Python); optional `ZEROLLAMA_LMCACHE_BLOB_HTTP_TOKEN`. Agents can build `prefix_block_hashes` with Go `prefixblock.Hashes`.
4. **`n_parallel > 1`** — Radix needs multiple slots in one server
5. **Vendor llama-server** with patch **0017** — `./scripts/vendor/llama_patch_doctor.sh`
6. **Hybrid GGUF** — Radix may seed when prefix ≤ SWA window; full-KV still hard-gates `radix_seed` in smokes
7. **Distinct cache keys, same system prompt** — same key uses same-slot L3, not Radix
8. **Cold second thread** — turn 1 on key B after turn 1 on key A completed on donor slot

### Roadmap (next milestones)

See [ROADMAP — Radix v2 (L3-R)](./ROADMAP.md#radix-v2-l3-r--product-gaps). Suggested order:

1. **L3-R1** — ~~5080 live Radix gate in sign-off table~~ **Done (Jun 2026)**
2. **L3-R2** — ~~Warm-target partial catch-up~~ **Done (Jun 2026)**
3. **L3-R3** — ~~Ref-count block pool + best donor~~ **Done (Jun 2026)** — multi-holder metadata; llama-level shared KV pages still deferred
4. **L3-R4** — ~~Remote LMCache connector~~ **Done (Jun 2026)** — `redis://` metadata
5. **L3-R5** — ~~Hybrid-memory copy path~~ **Done (Jun 2026)** — `radix_seq_copy_policy`; SWA-window gated `seq_cp`

