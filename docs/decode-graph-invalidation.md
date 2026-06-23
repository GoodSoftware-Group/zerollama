# Decode graph invalidation (vLLM breakable-graph bind)

**Audience:** Runtime operators and integrators running L3 prefix cache on CUDA (RTX 5080-class) or debugging in-process Phase 15 decode.

**Related:** [L3 prompt cache → slot bridge](./gpu-profiles-l3.md), [phase15-native-kv.md](./phase15-native-kv.md), [ROADMAP — vLLM borrowings](./ROADMAP.md#local-voice--llama-borrowings-eliza-v3), sibling [`../llama.cpp`](../llama.cpp) build (`scripts/build_llama_server.sh`).

---

## Why this exists

### The problem L3 creates for CUDA graphs

L3 reuses KV slots so agent threads skip repeat prefill. When a slot is cleared — SWA window exceeded, owner change, `cache_prompt=false`, draft-spec policy, or session teardown — the **KV tensor contents change** while ggml may still hold a **captured CUDA graph** from the previous decode shape.

ggml-cuda keys internal CUDA graphs by the first node of the compute graph (`cgraph.nodes[0]`), not by llama sequence id. That means:

- Slot 3 and slot 7 can share the same graph key if their decode topology matches.
- Clearing slot 3’s KV does **not** automatically drop ggml’s cached graph.
- Replaying a stale graph after a prefix-cache clear can produce wrong logits or silent corruption.

vLLM solves this with **breakable graphs**: bump an epoch when KV state changes, invalidate captured graphs, recapture on the next decode step. Zerollama ports the **invalidation half** first — capture lookup remains a stub until llama.cpp exposes per-step graph handles.

### Why epoch + native invalidate (two layers)

| Layer | Module | WHY |
|-------|--------|-----|
| Python epoch | `decode_graph_policy.py` | Future `DecodeGraphCache.lookup(slot)` key; trace replay; health without loading libllama |
| ggml invalidate | `llama_context_cuda_graph_invalidate` | Clears **actual** ggml CUDA graph map today — the safety fix on CUDA builds |

Epoch alone is insufficient: ggml does not read zerollama’s epoch counter. Native invalidate alone is insufficient for model swap: slots never individually bumped would keep stale epoch keys in a future capture cache. **`graph_capture_key(slot)` → `slot:slot_epoch:global_epoch`** covers both.

### Why Metal / Apple Silicon is different

ggml-metal does not expose an equivalent CUDA graph cache API. On Mac builds (`GGML_CUDA=OFF`), `llama_context_cuda_graph_invalidate` compiles to a no-op returning `0`. **Epoch bumps and L3 policy still run** — they matter for trace replay and future cross-platform capture scaffolding, not for Metal graph safety today.

---

## Architecture

```text
Prefix cache clear (SWA / owner / cache_prompt=false / session close)
        │
        ▼
bump_decode_graph_epoch(slot, reason=…, ctx_ptr=…)
        │  ├─ slot_epoch++, global_epoch++
        │  └─ invalidate_cuda_graphs(ctx_ptr)   [when ctx wired]
        ▼
runtime/kv/cuda_graph_invalidate.py
        │  ├─ native: runtime.kv._kv_native.invalidate_cuda_graphs(ptr)
        │  └─ fallback: ctypes → llama_context_cuda_graph_invalidate
        ▼
llama.cpp (sibling ../llama.cpp)
        │  iterates ggml_backend_sched CUDA backends
        ▼
ggml_backend_cuda_invalidate_graphs(backend)
        │  cudaStreamSynchronize + cuda_graphs.clear()
        ▼
Next decode step recaptures graph (ggml internal; GGML_CUDA_GRAPHS=ON)
```

**In-process wiring:** `libllama_ctypes._prepare_seq_for_decode` passes `_ctx_ptr(ctx)` on every slot clear path. `LlamaLoadedSession.close()` calls `bump_all_decode_graph_epochs(reason="session_close", ctx_ptr=…)`.

**Subprocess path:** epoch bumps still occur from engine policy; native invalidate requires a live `llama_context` (in-process only today). Subprocess llama-server clears KV via HTTP slot APIs — ggml invalidation happens inside that process on its own context when rebuilt with the new API.

---

## Operator checklist

### 1. Rebuild sibling libllama after pull

Invalidation is a **new public API** in the sibling tree (`../llama.cpp`), not in zerollama’s vendor pin alone:

```bash
# Mac (Metal — API present, CUDA graphs no-op at runtime)
./scripts/build_llama_server.sh

# CUDA 5080 — ensure graphs enabled
GGML_CUDA=ON ./scripts/build_llama_server.sh
```

**Why rebuild:** Python calls `llama_context_cuda_graph_invalidate` via ctypes or the Phase 15 native extension. Without the symbol, invalidation returns `symbol_missing_rebuild_libllama` and epoch bumps still run but ggml graphs are not cleared.

### 2. Health and probe

```bash
curl -s :8081/health | jq '.llama_cache.decode_graph'
```

| Field | Meaning |
|-------|---------|
| `global_epoch` | Monotonic counter; bumps on any slot invalidation or session close |
| `slot_epochs` | Per-slot epoch map (string keys) |
| `capture_key_format` | `"slot_id:slot_epoch:global_epoch"` |
| `capture_ready` | `false` until `DecodeGraphCache.lookup` is wired to native capture |
| `stub` | `true` — lookup always misses today |
| `lookup` | `"epoch_plus_ggml_invalidate"` |
| `llama_cpp` | Sibling tree probe: `GGML_CUDA`, `GGML_CUDA_GRAPHS`, `libllama` path, pin drift |

### 3. Environment variables

| Variable | Default | WHY |
|----------|---------|-----|
| `ZEROLLAMA_DECODE_GRAPH_INVALIDATE` | `1` | Kill-switch for native ggml clear (debug A/B without recompiling) |
| `ZEROLLAMA_DECODE_GRAPH_TRACE` | off | Log epoch bumps: `slot`, `epoch`, `global`, `reason` |
| `GGML_CUDA_DISABLE_GRAPHS` | off | Runtime disable of ggml graph capture (llama.cpp env); probe surfaces in health |
| `LLAMA_CPP_ROOT` | `../llama.cpp` | Override sibling checkout for probe + build scripts |

### 4. Disable invalidation (debug only)

```bash
ZEROLLAMA_DECODE_GRAPH_INVALIDATE=0 zerollama serve
```

**Why you might:** compare decode latency with/without graph recapture overhead on CUDA. **Do not ship disabled** when L3 prefix cache is enabled — stale graphs are a correctness risk.

---

## Invalidation triggers

| Reason | Where | WHY |
|--------|-------|-----|
| `cache_prompt_disabled` | `_prepare_seq_for_decode` | Client or policy forced full prefill — KV cleared |
| `spec_bind_swa_block` | `_prepare_seq_for_decode` | SWA/hybrid spec says resume would exceed window |
| `slot_clear` | `_prepare_seq_for_decode` | Owner mismatch or cold slot — new prefix |
| `session_close` | `LlamaLoadedSession.close()` | Model unload — all slots stale |

Each trigger bumps **both** slot epoch and global epoch (conservative: any slot change may affect shared ggml graph keys).

---

## Tests and smoke

```bash
cd runtime && python3 -m pytest tests/test_decode_graph_policy.py \
  tests/test_decode_graph_cache.py \
  tests/test_cuda_graph_invalidate.py \
  tests/test_llama_cpp_probe.py -v
```

Prefix cache golden trace (includes epoch in JSONL when tracing):

```bash
ZEROLLAMA_PREFIX_CACHE_TRACE=1 ./scripts/l3_prefix_cache_trace_replay.sh
```

---

## Deferred (non-goals today)

| Item | WHY deferred |
|------|----------------|
| `DecodeGraphCache.lookup()` returning a handle | llama.cpp has no per-slot capture export yet; ggml keys graphs internally |
| Per-slot graph capture in zerollama | Depends on upstream graph handle API or epoch-aware ggml keys |
| Metal graph invalidation | No ggml-metal equivalent to `cuda_graphs.clear()` |
| Subprocess epoch → llama-server HTTP invalidate | Separate process owns its own `llama_context`; wire when sidecar exposes hook |

**Taken (Jun 2026):** epoch scaffold, global epoch in capture key, `llama_context_cuda_graph_invalidate` in sibling llama.cpp, Python/native wiring, health probe, env kill-switch.

---

## Code map

| Path | Role |
|------|------|
| `runtime/runtime/decode_graph_policy.py` | Epoch counters + bump API |
| `runtime/runtime/decode_graph_cache.py` | Stub cache + health aggregation |
| `runtime/runtime/kv/cuda_graph_invalidate.py` | Native/ctypes invalidation entry |
| `runtime/runtime/llama_cpp_probe.py` | Sibling build flag probe |
| `runtime/runtime/worker/libllama_ctypes.py` | `_prepare_seq_for_decode` → bump + ctx |
| `runtime/native/kv_decode_loop.c` | C wrapper → `llama_context_cuda_graph_invalidate` |
| `../llama.cpp/include/llama.h` | Public invalidate API |
| `../llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml_backend_cuda_invalidate_graphs` |
