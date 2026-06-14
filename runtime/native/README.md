# Native runtime extensions (Phase 15+)

**Why:** PagedAttention block allocation, page-table validation, and decode batch layout run on every admission and decode step. Moving hot bookkeeping — and eventually `llama_decode` — off the Python interpreter reduces GIL contention when training and inference share one embedded CPython (`ZEROLLAMA_RUNTIME_SHARED_PYTHON=1`).

## Shipped (v0–v19)

| Module | Source | Role |
|--------|--------|------|
| `runtime.kv._kv_native` | `native/kv_block_pool.c`, `native/kv_decode_loop.c` | Block pool; tick/decode counters; page bind; batch layout; **v12–v13:** optional libllama link + C decode loop |
| `runtime.kv.decode_plan` | `runtime/kv/decode_plan.py` | **v9–v11:** `decode_prefill`, `decode_work` on forward plans |
| `runtime.kv.native_decode_loop` | `runtime/kv/native_decode_loop.py` | **v12–v14:** `decode_loop_status`, `run_prefill`, `run_step`, `greedy_decode_tokens` |

### v8 — seq-position page bind + decode batch layout

| API | Why |
|-----|-----|
| `page_bind_set/clear/resolve` | Register PA `block_ids` per `kv_slot` at admit; validate token positions before decode — tensor pages still owned by llama until upstream exposes handles |
| `decode_batch_layout` | Build `{token, pos, seq_id, logits}` in C so ctypes path avoids Python list churn per batch |
| `decode_prefill_chunks` | Split long prompts at PA page boundaries — same ranges the C decode loop uses in v13 |

### v11 — decode work plan + link scaffold

| API | Why |
|-----|-----|
| `decode_loop_status()` (C) | Stable probe before libllama links into ext |
| `kv_decode_work_plan()` | Single `{phase, prefill, decode}` object for operators + native loop |
| `current_pos_by_request_from_physical()` | Testable engine wiring from `kv_physical` |

### v12 — libllama link build + probe

| Piece | Why |
|-------|-----|
| `runtime/setup.py` | Auto-links `-lllama` when libllama found (v25); `ZEROLLAMA_KV_DECODE_LOOP=0` for unlinked CI |
| `native/kv_decode_loop.c` | `llama_max_devices()` link probe — no inference |
| `scripts/phase15_kv_decode_loop_build.sh` | Optional operator smoke |

Default CI build: `ZEROLLAMA_KV_DECODE_LOOP=0` (no libllama on GitHub runners). GPU / dev builds auto-link when libllama is under `LLAMA_CPP_ROOT`.

### v13 — C decode loop (`llama_decode` in linked build)

| API | Why |
|-----|-----|
| `decode_loop_prefill(ctx_ptr, tokens, seq_id, block_size)` | Page-aligned prefill + repeated `llama_decode` in C — **why ctx_ptr is int:** same address as ctypes `c_void_p`; one libllama in process |
| `decode_loop_step(ctx_ptr, token, seq_id, current_pos)` | Single-token decode step; sampling stays Python (`llama_sampler_sample` via ctypes) |
| `_decode_stream` fast path | Uses C prefill + C steps when linked; ctypes fallback when not linked or encoder model |

### v14 — GIL release + resume prefill + E2E

| API | Why |
|-----|-----|
| `gil_released` on `decode_loop_status` | Operators confirm C decode releases GIL |
| `decode_loop_prefill(..., pos_start=0)` | Remaining-prefill resume — llama positions = `pos_start + tok_off` |
| `run_prefill` / `run_step` + `kv_slot` | Page-bind validation before C calls (parity with ctypes batch path) |
| `greedy_decode_tokens()` | E2E parity helper for linked-build smoke |

Optional E2E:

```bash
# Auto-links when libllama is built under LLAMA_CPP_ROOT (v25 default)
cd runtime && LLAMA_CPP_ROOT=/path/to/llama.cpp python3 setup.py build_ext --inplace
export RUN_E2E_DECODE_LOOP=1 LLAMA_MODEL=/path/to/small.gguf
../scripts/phase15_kv_decode_loop_build.sh
```

**Still open (v16):** engine passes `current_pos` from `kv_physical` into generate; tensor page bind.

### v15 — sampling in C + resume prefill

| API | Why |
|-----|-----|
| `decode_loop_sample(smpl_ptr, ctx_ptr)` | Post-prefill `llama_sampler_sample` in C |
| `decode_loop_step(..., smpl_ptr=0)` | Decode + sample in one GIL-released call |
| `sampling_in_c` on status | Operator probe |
| `_decode_stream(..., current_pos=0)` | Remaining prefill via `pos_start`; skip when prefill done |

### v18 / v16b / v17 — engine resume + slot ownership + L3 session key + health

| Area | Why |
|------|-----|
| `current_pos_for_seq(lib, ctx, seq_id)` in `kv/physical.py` | Read live llama write position before decode so engine can resume without prefill |
| `InferenceEngine._decode_current_pos_for_request()` | Wires live position into generate / stream / batch completion paths |
| `LlamaLoadedSession.complete(..., current_pos=)` | Skips `_clear_sequence` when resuming the same owner |
| `_seq_last_owner: dict[int, str]` | WHY: `decode_pos > 0` alone is not sufficient — a *different* session can occupy the same slot. Records last-writer owner per slot; cleared on `close()`. |
| `slot_resume_owner_key()` in `cache_bridge.py` | WHY v17: L3 pinned sessions get a new `request_id` every turn; owner must be `cache:{prompt_cache_key}` or multi-turn agent chat always re-prefills |
| `resume_owner_snapshot()` + `/health.kv_resume` | WHY v18: operators need slot→owner map without attaching a debugger |

### v19 — tensor bind scaffold (accounting probe)

| API | Why |
|-----|-----|
| `page_bind_table(kv_slot)` | Export native PA page rows for forward-plan / snapshot parity |
| `page_bind_tensor_probe(ctx, seq_id, kv_slot)` | Linked: `llama_get_memory` + seq positions vs PA page reserve |
| `/health.kv_page_bind.tensor_probe` | Operator visibility; `tensor_bind_ready=false` until upstream page-handle API |

### v20a — forward plan native mirror

| Field | Why |
|-------|-----|
| `kv_forward_plans[].native_page_table` | Same rows as `page_bind_table` without a separate script call |
| `page_table_native_parity` | Quick operator check that C registry matches logical `pages[]` |

### v20 — cell + tensor bind (llama-kv-ext)

| API / field | Why |
|-------------|-----|
| `llama-kv-ext.h` | Staging fork API: cell map + K/V tensor info (standard kv_cache) |
| `page_bind_tensor_probe` | Sets `cell_pages_bound`, `tensor_pages_bound`, `kv_k_data_layer0` when linked |
| `/health.kv_page_bind.status=bound` | Operator signal that PA pages map to live llama KV |

### v32b — writable bind upstream tracker

| API / field | Why |
|-------------|-----|
| `llama_memory_kv_ext_writable_bind_probe` | Static libllama probe — no ctx; flips when `LLAMA_KV_EXT_WRITABLE_PAGE_MAP` ships |
| `page_bind_writable_probe()` | Python binding → `/health.kv_page_bind.writable_bind_available` |
| Pin check upstream watch | CI greps `llama.h` for page-map symbol names |

### v24 — C decode loop bind validation + post-prefill probe

| Piece | Why |
|-------|-----|
| `kv_page_bind_validate_range()` | Defense in depth: C prefill/step validates PA page table before `llama_decode` (not only Python wrapper) |
| Post-prefill `kv_tensor_probe_run` via `kv_decode_loop_post_prefill_probe` | Called after `Py_END_ALLOW_THREADS` so registry writes happen with GIL held |
| Return code `-2` | Distinct from llama_decode failure; surfaced as `LlamaServerError` in Python |

### v25 — auto-link + 131k page-bind cap

| Piece | Why |
|-------|-----|
| `setup.py` auto-link | Links libllama when found; no manual `ZEROLLAMA_KV_DECODE_LOOP=1` on dev/GPU builds |
| `KV_MAX_PAGES_PER_BIND=8192` | 131072 ctx @ block_size=16 (L2 fork-only bench leg) |
| `test_kv_decode_long_ctx.py` | 8192-chunk prefill plan + bind boundary validation without GPU |

### v26 — continuous batch decode (scaffold)

| Piece | Why |
|-------|-----|
| `kv_decode_loop_run_batch_step` | N active sequences → one `llama_decode` (continuous batching hot path) |
| `decode_loop_batch_step` / `run_batch_step` | Python binding + facade with bind error mapping |
| `kv_continuous_batch_step_plan` | `/health` export for operator visibility before engine wiring |

### v27 — engine wiring (shipped)

| Piece | Why |
|-------|-----|
| `complete_parallel` / `_decode_parallel_non_stream` | `generate_batch` prefills per slot, batched decode via `run_batch_step` |
| `native_batch_decode_available()` | Gate on linked ext + `ZEROLLAMA_KV_NATIVE_BATCH` (default on) |
| `_prepare_seq_for_decode` | Shared resume/clear for single- and multi-seq decode |

### v29 — streaming batch decode (shipped)

| Piece | Why |
|-------|-----|
| `_decode_parallel_stream` / `complete_parallel_stream` | Batched autoregressive steps with per-row streaming chunks |
| `completions_parallel_stream` / `stream_generate_batch` | Engine API for interleaved multi-request token streams |
| `_parallel_jobs_and_smpls` | DRY job setup shared by stream and non-stream parallel paths |

### v30 — per-row C batch sampling (shipped)

| Piece | Why |
|-------|-----|
| `smpl_ptrs[]` in `kv_decode_loop_run_batch_step` | Correct logit row + isolated sampler accept state per sequence |
| `run_batch_step(..., smpl_ptrs=)` | Python passes one sampler per active batch row |
| `_decode_parallel_stream` C batch path | Decode + sample in one GIL-released call when native sample enabled |

## Build

```bash
cd runtime
python3 setup.py build_ext --inplace
PYTHONPATH=. python3 -c "from runtime.kv._kv_native import BlockPool; print(BlockPool(8, 16).num_free)"
```

Libllama link (v12–v25, auto when libllama present):

```bash
export LLAMA_CPP_ROOT=/path/to/llama.cpp
export LLAMA_CPP_ROOT=/path/to/llama.cpp
cd runtime && python3 setup.py build_ext --inplace
../scripts/phase15_kv_decode_loop_build.sh
```

Enable at runtime:

```bash
export ZEROLLAMA_RUNTIME_KV_NATIVE=1   # C block pool allocator (optional; page bind needs ext built, not this env)
```

Default block allocator remains Python (`kv.backend: python` on `/health`). Page bind partial status appears when the `.so` is built regardless of `ZEROLLAMA_RUNTIME_KV_NATIVE`. Linked decode loop requires **both** the v12 build flags **and** in-process llama backend with a loaded model.

## Tests

```bash
cd runtime && PYTHONPATH=. python3 -m pytest tests/test_kv_native_parity.py tests/test_kv_page_bind.py tests/test_kv_native_decode_batch.py tests/test_kv_decode_plan.py tests/test_kv_decode_work_plan.py -q
```

CI: `./scripts/phase15_kv_native_ci.sh` — build native + KV pytest bundle.

See [docs/phase15-native-kv.md](../../docs/phase15-native-kv.md).
