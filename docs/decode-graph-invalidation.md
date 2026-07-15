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
bump_decode_graph_epoch(slot, reason=…, ctx_ptr=…, base_url=…)
        │  ├─ slot_epoch++, global_epoch++
        │  └─ invalidate_cuda_graphs(ctx_ptr | base_url)
        ▼
runtime/kv/cuda_graph_invalidate.py
        │  ├─ in-process native: runtime.kv._kv_native.invalidate_cuda_graphs(ptr)
        │  ├─ in-process fallback: ctypes → llama_context_cuda_graph_invalidate
        │  └─ subprocess: POST {base_url}/cuda-graph/invalidate → llama-server task queue
        ▼
llama.cpp (sibling ../llama.cpp)
        │  iterates ggml_backend_sched CUDA backends
        ▼
ggml_backend_cuda_invalidate_graphs(backend)
        │  cudaStreamSynchronize + cuda_graphs.clear()
        ▼
Next decode step recaptures graph (ggml internal; GGML_CUDA_GRAPHS=ON)
```

**Subprocess wiring:** `engine._prefix_cache_request` bumps epoch and POSTs `/cuda-graph/invalidate` on the llama-server child when prefix-cache policy denies resume or draft drop-last-block falls back to `cache_prompt=false`. The endpoint runs on the server task queue (same thread as slot erase) and calls `llama_context_cuda_graph_invalidate(ctx_tgt)`.

**In-process wiring:** `libllama_ctypes._prepare_seq_for_decode` passes `_ctx_ptr(ctx)` on every slot clear path. `LlamaLoadedSession.close()` calls `bump_all_decode_graph_epochs(reason="session_close", ctx_ptr=…)`.

---

## Operator checklist

### 1. Rebuild sibling libllama after pull

Invalidation is a **public API** on the unified vendor tree (`vendor/llama-cpp-8f114a9b/`, patch **0072**):

```bash
# Mac (Metal — API present, CUDA graphs no-op at runtime)
./scripts/build_llama_server.sh

# CUDA — ensure graphs enabled
GGML_CUDA=ON ./scripts/build_llama_server.sh
# or container (host CUDA skew):
./scripts/build_llama_server_container.sh
```

**Why rebuild:** Python calls `llama_context_cuda_graph_invalidate` via ctypes or the Phase 15 native extension; subprocess POSTs `/cuda-graph/invalidate`. Without the symbol/route, invalidation returns `symbol_missing` / HTTP 404 and epoch bumps still run but ggml graphs are not cleared.

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
| `cache_prompt_disabled` | `_prepare_seq_for_decode` / subprocess policy | Client or policy forced full prefill — KV cleared |
| `spec_bind_swa_block` | `_prepare_seq_for_decode` | SWA/hybrid spec says resume would exceed window |
| `slot_clear` | `_prepare_seq_for_decode` | Owner mismatch or cold slot — new prefix |
| `subprocess_drop_last_block` | `engine._prefix_cache_request` | Draft spec needs last-block trim; llama-server cannot — full prefill + graph break |
| `drop_last_prefix_block` | in-process draft resume | EAGLE-style last block dropped — KV shape changed |
| `session_close` | `LlamaLoadedSession.close()` | Model unload — all slots stale |
| `radix_cross_slot_seed` | `engine._apply_radix_prefix_share` | Cross-slot Radix KV copy seeded target slot — graph key must refresh |

Each trigger bumps **both** slot epoch and global epoch (conservative: any slot change may affect shared ggml graph keys).

---

## Subprocess endpoint (`POST /cuda-graph/invalidate`)

**Why a dedicated route:** the default zerollama backend is subprocess llama-server. Python epoch bumps do not reach ggml in the child process. vLLM breaks graphs inside the worker that owns KV; zerollama mirrors that by POSTing from `engine._prefix_cache_request` when prefix-cache policy invalidates a pinned slot.

**Implementation (sibling `../llama.cpp`):**

- Handler enqueues `SERVER_TASK_TYPE_CUDA_GRAPH_INVALIDATE` on the server task queue (same thread as slot erase — avoids racing active decode).
- Task handler calls `llama_context_cuda_graph_invalidate(ctx_tgt)`.
- Response: `{"ok": true, "backends_cleared": N}` (`N=0` on Metal or when graphs disabled).

**Graceful degradation:** if llama-server predates the endpoint (HTTP 404), `cuda_graph_invalidate.py` logs at debug and returns `ok: false`; epoch bumps still run for trace and future capture keys.

```bash
curl -s -X POST http://127.0.0.1:8082/cuda-graph/invalidate \
  -H 'Content-Type: application/json' \
  -d '{"reason":"smoke"}'
```

---

## Subprocess endpoint (`POST /kv/seq-copy`)

**Why:** cross-slot Radix prefix share seeds a cold target slot by copying KV from a donor slot that already holds a verified hash chain. Python epoch bumps and subprocess SWA policy need the target slot’s context length before the first completion returns timings.

**Body:** `{"src_slot": 2, "dst_slot": 5, "pos_end": 512}` — copies KV positions `[0, pos_end)` from `src_slot` → `dst_slot`.

**Implementation (sibling `../llama.cpp`):**

- Handler enqueues `SERVER_TASK_TYPE_SLOT_SEQ_COPY` on the server task queue.
- Task calls `llama_memory_seq_cp` after clearing the target sequence range.
- Response: `{"ok": true, "pos_end": N}`.

**Enable:** `ZEROLLAMA_RADIX_PREFIX_SHARE=1` (implies prefix block pool). Rebuild llama-server after pull — older binaries return HTTP 404.

```bash
curl -s -X POST http://127.0.0.1:8082/kv/seq-copy \
  -H 'Content-Type: application/json' \
  -d '{"src_slot":0,"dst_slot":1,"pos_end":512}'
```

Offline gate (no GPU): `./scripts/l3_radix_prefix_smoke.sh`.

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

Subprocess policy (no GPU):

```bash
cd runtime && python3 -m pytest tests/test_prefix_cache_subprocess.py \
  tests/test_cuda_graph_invalidate.py -v
```

---

## Deferred (non-goals today)

| Item | WHY deferred |
|------|----------------|
| `DecodeGraphCache.lookup()` returning a handle | llama.cpp has no per-slot capture export yet; ggml keys graphs internally |
| Per-slot graph capture in zerollama | Depends on upstream graph handle API or epoch-aware ggml keys |
| Metal graph invalidation | No ggml-metal equivalent to `cuda_graphs.clear()` |
| Per-slot graph capture in zerollama | Depends on upstream graph handle API or epoch-aware ggml keys |

**Taken (Jun 2026):** epoch scaffold, global epoch in capture key, `llama_context_cuda_graph_invalidate` in sibling llama.cpp, in-process Python/native wiring, **`POST /cuda-graph/invalidate`** for subprocess llama-server, health probe, env kill-switch.

---

## Code map

| Path | Role |
|------|------|
| `runtime/runtime/decode_graph_policy.py` | Epoch counters + bump API |
| `runtime/runtime/decode_graph_cache.py` | Stub cache + health aggregation |
| `runtime/runtime/kv/cuda_graph_invalidate.py` | Native/ctypes/HTTP invalidation entry |
| `runtime/runtime/engine.py` | Subprocess `base_url` → POST on prefix-cache deny |
| `runtime/runtime/llama_cpp_probe.py` | Sibling build flag probe |
| `runtime/runtime/worker/libllama_ctypes.py` | `_prepare_seq_for_decode` → bump + ctx |
| `runtime/native/kv_decode_loop.c` | C wrapper → `llama_context_cuda_graph_invalidate` |
| `../llama.cpp/tools/server/server-context.cpp` | `POST /cuda-graph/invalidate` task handler |
| `../llama.cpp/include/llama.h` | Public invalidate API |
| `../llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml_backend_cuda_invalidate_graphs` |
