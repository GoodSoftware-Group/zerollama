# vLLM borrowings (L3 track)

**Why this doc:** Zerollama ports **narrow** vLLM patterns into the GGUF-first Python runtime — not HTTP-to-vLLM, not full RadixAttention ref-count DAG (v1 donor seed shipped), not Model Runner V2.

**Related:** [gpu-profiles-l3.md](./gpu-profiles-l3.md), [decode-graph-invalidation.md](./decode-graph-invalidation.md), [ROADMAP — borrowings L3](./ROADMAP.md#local-voice--llama-borrowings-eliza-v3).

---

## Taken (Jun 2026)

| vLLM concept | Zerollama | WHY |
|--------------|-----------|-----|
| Selective prefix retention (`KVCacheSpec`) | `kv_cache_spec.py` + `prefix_cache_policy.py` | SWA/hybrid/draft guards before `cache_prompt` |
| `drop_eagle_block` | `drop_last_prefix_block` + in-process KV trim | Draft heads need fresh hidden states; rest of prefix still reusable |
| Subprocess draft fallback | `subprocess_drop_last_block_unsupported` → `cache_prompt=false` + `POST /cuda-graph/invalidate` | llama-server has no last-block trim; HTTP clears ggml graphs in child |
| `cache_salt` | `options.cache_salt` → slot hash + owner key | Multi-tenant isolation when thread keys collide |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | `ZEROLLAMA_PREFIX_CACHE_RETENTION_INTERVAL` | SWA sparse checkpoints — memory vs hit rate |
| `HybridKVCacheCoordinator` (HMA) | `kv/hybrid_kv_coordinator.py` + `KVCacheSpec.coordinator` | Per-layer full/SWA groups; coordinated `cache_prompt` gate via min SWA window |
| Breakable CUDA graphs (invalidation) | Epoch + `llama_context_cuda_graph_invalidate` (in-process) + `POST /cuda-graph/invalidate` (subprocess) | ggml graph keys ignore sequence id; child process owns its own ctx |
| Prefix trace replay | `ZEROLLAMA_PREFIX_CACHE_TRACE` + golden JSONL | Offline policy regression without GPU |
| Hash-chained **prefix block pool** | `kv/prefix_block_pool.py` + `prefix_block_hash.py` | Content-addressed prefix verification before `cache_prompt` |
| Optional **LMCache tier** | `kv/lmcache_tier.py` (`ZEROLLAMA_LMCACHE_TIER`) | Filesystem metadata tier for block index across restarts |
| **Cross-slot Radix prefix share (v1)** | `kv/radix_prefix_share.py` + `POST /kv/seq-copy` | Same system prompt, different keys → donor slot KV seed; see [radix-prefix-share.md](./radix-prefix-share.md) |

---

## Deferred (explicit non-goals)

| vLLM feature | WHY not in zerollama L3 |
|--------------|-------------------------|
| Full RadixAttention ref-count block DAG | v1 ships donor→target seed only; gap matrix in [radix-prefix-share.md](./radix-prefix-share.md#product-gaps); milestones [ROADMAP L3-R](./ROADMAP.md#radix-v2-l3-r--product-gaps) |
| LMCache / Mooncake **remote** connectors | Optional local `file://` tier only; Redis/NIXL deferred — **why:** agent-local v1 proves block verification + donor seed before fleet blob federation |
| Warm-target Radix catch-up | v1 seeds **cold** slots only (`seq_pos == 0`) — **why:** partial merge needs ref-count semantics (L3-R2) |
| Fleet / cross-node Radix donor | Donor must be same llama-server process — **why:** KV in VRAM; fleet layer routes warm model, not shared-prefix residency |
| `CUDAGraphDispatcher` + capture handles | ggml internal capture; stub `DecodeGraphCache.lookup` until upstream API — invalidation after Radix seed is wired |
| Scheduler KV preemption loop | LocalAI watchdog + slot allocator; not vLLM-style block preempt yet |

---

## Env reference

| Variable | Default | Role |
|----------|---------|------|
| `ZEROLLAMA_L3_PROFILE` | — | `agent` → `runtime/configs/l3_agent_subprocess.yaml` when `ZEROLLAMA_RUNTIME_CONFIG` unset |
| `ZEROLLAMA_DEBUG` | — | `l3` → prefix trace + decode-graph trace; `infer` → phase-15 infer spans |
| `ZEROLLAMA_LLAMA_CACHE` | `1` | Master L3 enable |
| `ZEROLLAMA_CACHE_SALT` | — | Operator default tenant salt |
| `ZEROLLAMA_PREFIX_CACHE_BLOCK_SIZE` | `512` | EAGLE drop-last-block granularity |
| `ZEROLLAMA_PREFIX_CACHE_RETENTION_INTERVAL` | — | SWA sparse retention (`0`, `N`, unset=dense) |
| `ZEROLLAMA_LMCACHE_URI` | — | Set `file://…` to enable LMCache metadata tier (unset = off) |
| `ZEROLLAMA_LMCACHE_TIER` | *(deprecated)* | Alias for URI-only enable; prefer `ZEROLLAMA_LMCACHE_URI` |
| `ZEROLLAMA_PREFIX_BLOCK_POOL` | auto | Auto-on when Radix, LMCache URI, or `n_parallel > 1`; `=0` to disable |
| `ZEROLLAMA_PREFIX_BLOCK_POOL_MAX` | `8192` | Max in-memory block entries per model scope |
| `ZEROLLAMA_LLAMA_CACHE_DISK` | smart | Unset: off on Darwin, on for Linux subprocess; explicit `0`/`1` overrides |
| `ZEROLLAMA_RADIX_PREFIX_SHARE` | `0` | Cross-slot Radix prefix seed (implies block pool) |
| `ZEROLLAMA_DECODE_GRAPH_INVALIDATE` | `1` | ggml CUDA graph clear on slot invalidation (in-process native/ctypes or subprocess HTTP) |

**Radix operator guide:** [radix-prefix-share.md](./radix-prefix-share.md) — **why** vendor llama-server, live smoke, hybrid skip, trace events.

---

## Subprocess vs in-process (WHY)

| Path | Graph clear mechanism | WHY |
|------|----------------------|-----|
| **In-process** (`ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess`) | `bump_decode_graph_epoch(..., ctx_ptr=…)` → native/ctypes → `llama_context_cuda_graph_invalidate` | Runtime owns `llama_context` in-process |
| **Subprocess** (default) | `bump_decode_graph_epoch(..., base_url=…)` → `POST /cuda-graph/invalidate` | llama-server child owns `ctx_tgt`; ctypes from Python would target the wrong address space |

Rebuild llama-server after pull so the HTTP route exists: `./scripts/build_llama_server.sh`.

---

## Request options

```json
{
  "options": {
    "prompt_cache_key": "agent-thread-42",
    "cache_salt": "tenant-org-9",
    "eliza": { "conversationId": "thread-42", "cacheSalt": "tenant-org-9" }
  }
}
```

Slot id: `SHA256(salt + key) mod parallel` when salt present; owner key: `cache:{salt}:{key}`.
