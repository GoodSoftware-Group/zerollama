# vLLM borrowings (L3 track)

**Why this doc:** Zerollama ports **narrow** vLLM patterns into the GGUF-first Python runtime — not HTTP-to-vLLM, not full RadixAttention ref-count DAG (v1 donor seed shipped), not Model Runner V2.

**Related:** [gpu-profiles-l3.md](./gpu-profiles-l3.md), [decode-graph-invalidation.md](./decode-graph-invalidation.md), [ROADMAP — borrowings L3](./ROADMAP.md#local-voice--llama-borrowings-eliza-v3).

**Local tree:** `../vllm` (Mac: `~/Sites/inference/vllm`). Sibling map + weekly pull ritual: [upstream-siblings.md](./upstream-siblings.md).

**Last checked:** 2026-10-24 — tip `e9acf48614` on `main`. Prior scans: 2026-09-23 (`f9ad9dd6b4`), 2026-08-20 (`f8e0602713`), 2026-07-28 (`118bcde44`). Next weekly: `git pull` in `../vllm`, then triage bring/watch/skip into this doc (and the table in [upstream-siblings.md](./upstream-siblings.md)).

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
| Optional **LMCache tier** | `kv/lmcache_tier.py` + `kv/lmcache_redis.py` | `file://` or `redis://` metadata; block index hydration across restarts / fleet nodes |
| **Cross-slot Radix prefix share (v1)** | `kv/radix_prefix_share.py` + `POST /kv/seq-copy` | Same system prompt, different keys → donor slot KV seed; see [radix-prefix-share.md](./radix-prefix-share.md) |
| **Warm-target Radix catch-up (L3-R2)** | `verify_target_slot_prefix` + donor search past target blocks | Agent thread extended shared prefix while donor already holds longer KV |
| **Ref-count block DAG (L3-R3)** | `holder_slots` + `release_slot_holders` + `_best_donor_from_chain` | Overlapping slot registrations; pick longest donor chain from token 0 |
| **Hybrid Radix gate (L3-R5)** | `radix_seq_copy_policy.py` + `ZEROLLAMA_RADIX_HYBRID_SEQ_COPY` | v1 skipped all hybrid; Gemma full+SWA safe when copy ≤ window; attn+recurrent keeps kill-switch |
| **Marconi × retention preservation** | Radix window-only gate; try Radix even when full-prompt `cache_prompt` denied | vLLM #47782: selective retention must not kill shared-system-prefix hits; retention is same-slot resume, not donor `seq_cp` |
| **Per-request load-tier filter** | `kv/tier_filter.py` + `options.zerollama.kv_load_tiers` | vLLM #48123: skip LMCache/blob secondary lookup when request denies STORAGE |
| **Finish-time / defer blob finalize** | `register_prefix(finalize_blob=…)` + `finalize_slot_blob` / reuse flush | vLLM #48596/#49671: metadata first; publish when slot `.bin` exists; flush before slot reuse |
| **Cache creation vs read tokens** | `cache_creation_tokens` / OpenAI `created_cache_tokens` / Anthropic `cache_creation_input_tokens` | vLLM #48535: `creation = newly_cached − hit_at_admit` |
| **SWA reachable-tail store filter** | `kv/swa_store_filter.py` → `register_prefix(store_block_mask=…)` | vLLM #48911: do not federate blocks outside SWA reachable tail |
| **Partial secondary-tier load** | `_apply_prefix_block_pool` + `PrefixBlockMatch.partial_tier_load` | vLLM #50321: resume from longest LMCache hit when tail blocks absent remotely |
| **Zero-output prefix-cache metrics** | `_stream_done_metrics` + non-stream `OllamaGenerateResponse` metrics | vLLM #48668: keep cache read/creation on done chunks when `eval_count=0` |
| **Retention default → block-aligned** | `prefix_cache_retention_interval()` unset → `0` | vLLM #52216: SWA checkpoints at block boundaries only unless YAML/env sets `N>0` |
| **Skip covered MM payload** | `defer_vision_encode` (ollama-engine + ggml llamarunner) + session prefix strip on llama-server IPC | vLLM #52041: stub GridTHW vision spans, match input-cache by hash, hydrate ViT/mtmd only on the uncached tail; llama-server strips covered `multimodal_data` for Qwen3-VL and other padded-inject families |
| **Tagged prefix extra keys + LoRA** | `prefix_block_hash.model_scope_key` + `options.lora` | vLLM #51899/#59335: `cache_salt` and `lora` are tagged `\0name\0value` extras so salts cannot collide with other keys |
| **EAGLE SWA junction store** | `swa_reachable_store_mask(shared_prefix_blocks=…)` | vLLM #62206: keep the last hash-matched prefix block (junction-1) even when it sits outside the current SWA tail |
| **Prompt-end SWA checkpoint** | `swa_reachable_store_mask(prompt_end_blocks=…)` | vLLM #59146 analog: when retention is `0`, still federate the last full prompt block for resume |
| **Per-block MM extras** | `MmFeature` + `iter_prefix_blocks(mm_features=…)` | vLLM #51899 `_gen_mm_extra_hash_keys`: image/video spans (`identifier`, relative offset) participate in the prefix hash so placeholder tokens cannot false-hit |
| **Explicit EAGLE last-block drop** | `eagle_block_drop_active` + `LayerGroupSpec.is_eagle_group` | vLLM #61455: drop last prefix block only for EAGLE/MTP (not `draft-simple`/`dflash`); tag coordinator groups (fallback: all groups); SWA store keeps `draft_n_max-1` extra tokens |

---

## Watch (Oct 2026 rescan)

| vLLM PR | Topic | Zerollama stance |
|---------|-------|------------------|
| **#50507** / **#54736** | Partial-tail prefix reuse with `hash_block_size` < physical block | **Defer** — needs physical KV pages + COW / OffloadingConnector scheduler; block pool stays 512-token aligned |
| **#50493** | Kimi-K3 DCP partial prefix hit | **Skip** — model-specific distributed checkpoint path |

---

## Deferred (explicit non-goals)

| vLLM feature | WHY not in zerollama L3 |
|--------------|-------------------------|
| Full RadixAttention ref-count block DAG | Metadata multi-holder + best donor shipped (L3-R3); llama-level shared KV pages + Go scheduler mirror still deferred — [radix-prefix-share.md](./radix-prefix-share.md#product-gaps) |
| LMCache / Mooncake **remote blob** connectors | **`redis://` metadata shipped (L3-R4)** — NIXL/Mooncake KV blob pull still deferred |
| Fleet / cross-node Radix donor | Donor must be same llama-server process — **why:** KV in VRAM; fleet layer routes warm model, not shared-prefix residency |
| `CUDAGraphDispatcher` + capture handles | ggml internal capture; stub `DecodeGraphCache.lookup` until upstream API — invalidation after Radix seed is wired |
| Scheduler KV preemption loop | LocalAI watchdog + slot allocator; not vLLM-style block preempt yet |
| KV-cache **admission watermark** (`free_blocks ≥ needed + watermark`) | Skip — zerollama pre-reserves full prompt+`max_tokens`/`num_ctx` and caps concurrency with fixed `n_parallel` slots; Phase 11 `VRAM_MIN_FREE` is the headroom policy. Revisit if PA allocates mid-decode. |
| Partial hybrid prefix hits (`hash_block_size` < physical block) | Needs physical pages + COW; our Radix still full-sequence `seq_cp` — **#50507 watch** |
| Full OffloadingConnector / NIXL workers | Pattern ports only (#48123/#48596/#48535/#48911); not the CUDA offload scheduler |

---

## Env reference

| Variable | Default | Role |
|----------|---------|------|
| `ZEROLLAMA_L3_PROFILE` | — | `agent` → `runtime/configs/l3_agent_subprocess.yaml` when `ZEROLLAMA_RUNTIME_CONFIG` unset |
| `ZEROLLAMA_DEBUG` | — | `l3` → prefix trace + decode-graph trace; `infer` → phase-15 infer spans |
| `ZEROLLAMA_LLAMA_CACHE` | `1` | Master L3 enable |
| `ZEROLLAMA_CACHE_SALT` | — | Operator default tenant salt |
| `ZEROLLAMA_PREFIX_CACHE_BLOCK_SIZE` | `512` | EAGLE drop-last-block granularity |
| `ZEROLLAMA_PREFIX_CACHE_RETENTION_INTERVAL` | `0` | SWA sparse retention (`0` = block-aligned only; `N>0` = every N tokens; unset matches `0`, vLLM #52216) |
| `ZEROLLAMA_LMCACHE_URI` | — | `file://…` or `redis://host:6379/db` — metadata tier (L3-R4) |
| `ZEROLLAMA_LMCACHE_TTL_SEC` | — | Redis key TTL (optional) |
| `ZEROLLAMA_LMCACHE_TIER` | *(deprecated)* | Alias for URI-only enable; prefer `ZEROLLAMA_LMCACHE_URI` |
| `ZEROLLAMA_PREFIX_BLOCK_POOL` | auto | Auto-on when Radix, LMCache URI, or `n_parallel > 1`; `=0` to disable |
| `ZEROLLAMA_PREFIX_BLOCK_POOL_MAX` | `8192` | Max in-memory block entries per model scope |
| `ZEROLLAMA_LLAMA_CACHE_DISK` | smart | Unset: off on Darwin, on for Linux subprocess; explicit `0`/`1` overrides |
| `ZEROLLAMA_RADIX_PREFIX_SHARE` | `0` | Cross-slot Radix prefix seed (implies block pool) |
| `ZEROLLAMA_RADIX_HYBRID_SEQ_COPY` | `1` | Allow hybrid GGUF Radix `seq_cp` when prefix ≤ SWA window (L3-R5); `0` = skip all hybrid copy |
| `ZEROLLAMA_DECODE_GRAPH_INVALIDATE` | `1` | ggml CUDA graph clear on slot invalidation (in-process native/ctypes or subprocess HTTP) |

**Request options:** `options.zerollama.kv_load_tiers` — JSON list of `{medium, locality}` (vLLM #48123). Omit = all secondaries; `[]` = deny LMCache/blob restore.

**Radix operator guide:** [radix-prefix-share.md](./radix-prefix-share.md) — **why** vendor llama-server, live smoke, hybrid SWA gate, trace events.

---

## Subprocess vs in-process (WHY)

| Path | Graph clear mechanism | WHY |
|------|----------------------|-----|
| **In-process** (`ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess`) | `bump_decode_graph_epoch(..., ctx_ptr=…)` → native/ctypes → `llama_context_cuda_graph_invalidate` | Runtime owns `llama_context` in-process |
| **Subprocess** (default) | `bump_decode_graph_epoch(..., base_url=…)` → `POST /cuda-graph/invalidate` | llama-server child owns `ctx_tgt`; ctypes from Python would target the wrong address space |

Rebuild llama-server after pull so the HTTP route exists: `./scripts/build/build_llama_server.sh`.

---

## Request options

```json
{
  "options": {
    "prompt_cache_key": "agent-thread-42",
    "cache_salt": "tenant-org-9",
    "lora": "adapter-a",
    "mm_features": [{"identifier": "img-sha", "offset": 12, "length": 64}],
    "eliza": { "conversationId": "thread-42", "cacheSalt": "tenant-org-9" }
  }
}
```

Slot id: `SHA256(salt + key) mod parallel` when salt present; owner key: `cache:{salt}:{key}`.
