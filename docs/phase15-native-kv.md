# Phase 15 — native scheduler + KV

**Status:** Partial (Jul 2026) — **v0–v33 ops** shipped (see slices below). Phase 14 in-process forward **Done** (prerequisite). Default block allocator remains **Python**; C pool is opt-in (`ZEROLLAMA_RUNTIME_KV_NATIVE=1`; sign-off scripts enable it). **GPU sign-off:** `./scripts/phase15_inprocess_signoff.sh` (Linux embed) + `./scripts/phase15_metal_signoff.sh` (Mac uv sidecar) — includes **continuous batch decode** step (v27–v30). **Mac Metal PASS (M4 Max, Jun 2026).** **CUDA 5080 PASS (CT 1564 / cudallama, Jun 2026)** — OuteTTS 1B Q8, `kv_decode_steps=56`, batch decode via `/internal/generate-batch`. **v33 (Jul 2026):** fork writable page-map (`llama_memory_kv_page_map`); Darwin sidecar restarts when `kv_native_build_sha` mismatches build stamp. **Open:** transposed-V layouts, multi-layer fan-out, true external-buffer alias into ggml allocators; scheduler-driven async batching across concurrent HTTP streams.

**Handoff (code map, gaps, next slices):** [handoff-phase15-native-kv.md](./handoff-phase15-native-kv.md)

See also [ROADMAP Phase 15 exit criteria](../ROADMAP.md#phase-15--exit-criteria-partial).

**Why:** Phase 14 moved **forward** in-process; Phase 15 moves **KV bookkeeping** (and eventually decode) off the interpreter so continuous batching does not fight the GIL under `ZEROLLAMA_RUNTIME_SHARED_PYTHON=1`.

---

## Slice index (v0–v19 ops)

| Slice | Summary |
|-------|---------|
| **v0** | Native `BlockPool` (`kv_block_pool.c`), `ZEROLLAMA_RUNTIME_KV_NATIVE`, parity tests, `phase15_kv_native_ci.sh` |
| **v1** | `kv_scheduler` on `/health`, `num_ctx` block reserve, `kv_slot` → subprocess `id_slot` / in-process `seq_id` |
| **v2** | In-process multi-seq shared `llama_context` when `llama_parallel_slots` > 1 (`resolve_parallel_slots`, `-np` wins) |
| **v3** | Logical bind (`kv_bind`, `block_ids`, `assert_kv_capacity`, in-process `kv_token_budget`) |
| **v4** | Post-decode seq-position track (`kv_physical`, native `scheduler_tick()`, strict env) |
| **v5** | `kv_scheduler_tick` `{value, source}`, `kv_physical_recent` (mismatches only), expanded CI |
| **v6** | Native `decode_step(n)`; `/health` + per-response `kv_decode_steps` (in-process only) |
| **v7** | `kv_forward_plans`, `kv_native_stats` (`kv_stats()`), `GET /internal/kv-snapshot`, `phase15_health_smoke.sh` |
| **v8 ops** | Seq-position page bind (`page_bind_*`), C decode batch layout, `/health` `kv_page_bind.status=partial`, Go loopback proxy, opt-in `ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL`, GPU smokes |
| **v9 ops** | `decode_prefill` on `kv_forward_plans` — export page-aligned prefill batch plan from logical page table (no `llama_decode`; bridges to future C decode loop) |
| **v10 ops** | In-progress plans: remaining prefill from `current_pos`, decode info in `decode_work.decode`, wired from live `kv_physical` seq positions |
| **v11 ops** | Unified `decode_work` plan + `kv_decode_loop` link scaffold (`decode_loop_status` in C) |
| **v12 ops** | Optional libllama link at build (`ZEROLLAMA_KV_DECODE_LOOP=1`); `llama_max_devices` probe on `/health.kv_decode_loop` |
| **v13 ops** | `kv_decode_loop_run_prefill` + `kv_decode_loop_run_step` in C; Python bindings; `_decode_stream` C fast path; sampling in C when linked (v15+) |
| **v14 ops** | GIL release around C `llama_decode`; `pos_start` on prefill; page-bind validation in `run_prefill`/`run_step`; `greedy_decode_tokens` + optional E2E smoke |
| **v15 ops** | `llama_sampler_sample` in C; `decode_loop_sample`; `_decode_stream(current_pos=)` resume prefill |
| **v16 ops** | `current_pos_for_seq`; engine→`complete(current_pos=)`; skip seq clear on resume |
| **v16b ops** | `_seq_last_owner` slot-ownership map; skip clear only when same owner last wrote slot; conservative clear otherwise |
| **v17 ops** | `slot_resume_owner_key`: L3 pinned uses `prompt_cache_key` (new `request_id` each turn) |
| **v18 ops** | `/health.kv_resume`; `resume_owner_snapshot()`; Metal L3 two-turn sign-off step |
| **v19 ops** | `page_bind_tensor_probe`; `page_bind_table`; accounting bind vs llama seq cells |
| **v20a ops** | `native_page_table` + `page_table_native_parity` on admitted `kv_forward_plans` |
| **v20 ops** | `llama-kv-ext.h` cell map + K/V tensor verify; `/health.kv_page_bind.status=bound` |
| **v21 ops** | `page_bind_slots` on `/health.kv_page_bind.slots`; post-decode bind warnings |
| **v22 ops** | Fix stale `decode_pos` after `_clear_sequence` on non-resume multiseq path; `infer_trace` debug module |
| **v23 ops** | `iter_prefill_execute_chunks` — shared prefill chunker for `_decode_stream` + `kv_decode_prefill_plan`; sign-off enables C block pool + linked ext build |
| **v24 ops** | C decode loop page-bind validation per chunk; post-prefill tensor probe updates bind flags; sign-off step 4 tensor scaffold |
| **v25 ops** | Auto-link libllama at build when present; `KV_MAX_PAGES_PER_BIND=8192` (131k ctx); 131k chunk/bind validation tests; post-prefill probe GIL fix |
| **v26 ops** | `kv_decode_loop_run_batch_step` — N single-token rows, one `llama_decode`; `decode_batch_layout_multi`; `kv_continuous_batch_step_plan` export |
| **v27 ops** | `complete_parallel` + `_decode_parallel_non_stream`; `generate_batch` → batched decode when `n_seq_max>1`; `ZEROLLAMA_KV_NATIVE_BATCH=0` disables |
| **v28 ops** | `kv_continuous_batch_forward_plan`; `/health.kv_continuous_batch` with `would_batch` for operator sign-off |
| **v29 ops** | `_decode_parallel_stream`; `complete_parallel_stream`; `completions_parallel_stream`; `stream_generate_batch` |
| **v30 ops** | Per-row `smpl_ptrs` in `kv_decode_loop_run_batch_step`; C batch sampling in `_decode_parallel_stream` |

## Continuous batch decode (v26–v30)

**Why this slice exists:** With `llama_parallel_slots>1`, each sequence used to call `llama_decode` separately from Python — N GIL transitions and no shared batch hot path. v26–v30 merge autoregressive rows into one C `llama_decode`, wire the engine batch API, stream interleaved tokens, and sample per-row in C without sampler state bleed.

### Architecture

```
generate_batch / stream_generate_batch (engine)
  → scheduler admits N requests (PA blocks + kv_slot)
  → completions_parallel[_stream] (llama_inprocess)
  → complete_parallel[_stream] (libllama_ctypes)
       1. Sequential prefill per row (different prompts / resume pos)
       2. Sample first token per row immediately (logits still valid)
       3. while active: run_batch_step(N rows) → sample per row (v30: C smpl_ptrs[])
  → POST /internal/generate-batch (loopback GPU smoke only)
```

**Why sequential prefill, batched decode:** Prefill token counts and resume positions differ per request; one shared prefill batch would mis-align page bind and logits. Autoregressive steps are always one token per active row — safe to merge into one `llama_decode`.

**Why one sampler per sequence (v27 audit):** `llama_sampler_sample` calls `llama_sampler_accept`, updating repetition-penalty / grammar / mirostat state. A shared chain bleeds accept state from sequence *i* into *i+1*.

**Why v30 `smpl_ptrs[]`:** After a multi-row batch step the logit matrix has N rows. `run_sample` reads the last row only; ctypes `batch_idx` works but stays in Python. Per-row C samplers keep decode + sample in one GIL-released call with correct indices.

### Engine API

| API | Role |
|-----|------|
| `InferenceEngine.generate_batch(prompts, …)` | Admit ≤ `max_admit`; return full texts (v27) |
| `InferenceEngine.stream_generate_batch(prompts, …)` | Same admission; yield `{request_id, seq_idx, response, done}` chunks (v29) |

Gating: `native_batch_decode_available()` — linked ext + `batch_decode_in_c` + `ZEROLLAMA_KV_NATIVE_BATCH≠0`.

### Loopback HTTP (GPU sign-off / debug)

`POST /internal/generate-batch` — **loopback only** (same middleware as `/internal/kv-snapshot`).

**Why not public `/api/generate` batch yet:** Batch admission policy, streaming NDJSON shape, and OpenAI parity are still evolving; internal endpoint lets GPU smokes exercise the real engine path without committing to an external contract.

```json
{
  "prompts": ["Say: alpha", "Say: beta"],
  "n_predict": 8,
  "max_admit": 2,
  "stream": false,
  "options": {"num_ctx": 4096, "temperature": 0}
}
```

Stream mode (`"stream": true`) returns NDJSON lines with `request_id`, `seq_idx`, `response`, `done`.

### `/health` batch fields

| Field | Why |
|-------|-----|
| `kv_decode_loop.batch_decode_in_c` | Linked ext exposes `kv_decode_loop_run_batch_step` |
| `kv_continuous_batch` | Merged preview of what `run_batch_step` would consume for N running decode-phase rows (v28) |
| `kv_continuous_batch.would_batch` | True when ≥2 candidates + `parallel_slots>1` + native batch available |

### Env knobs (batch path)

| Variable | Default | Why |
|----------|---------|-----|
| `ZEROLLAMA_KV_NATIVE_BATCH` | `1` | Gate engine batch wiring; set `0` to force per-row `completion()` fallback |
| `ZEROLLAMA_KV_NATIVE_DECODE` | `1` | C prefill/step; batch requires native decode loop |
| `ZEROLLAMA_KV_NATIVE_SAMPLE` | `1` | C sampling on prefill + v30 batch rows |
| `llama_parallel_slots` | `1` | Must be >1 for batch merge (`kv_inprocess_n_seq_max` on `/health`) |

Sign-off scripts source `scripts/phase15_runtime_kv_env.sh` — **why:** one place to enable C pool + native decode + build linked ext against sibling `../llama.cpp`.

### GPU sign-off (v27–v30 batch step)

| Script | Platform | Batch step |
|--------|----------|------------|
| `phase15_metal_signoff.sh` | Mac Metal (uv sidecar) | Step **3/5** — `phase15_batch_decode_smoke.sh` |
| `phase15_inprocess_signoff.sh` | Linux embed | End of `phase15_inprocess_multiseq_smoke.sh` |
| `phase15_batch_decode_smoke.sh` | Either (sidecar must be up, `n_seq_max≥2`) | Standalone |

**Jun 2026:** `./scripts/phase15_metal_signoff.sh` **PASS** on M4 Max — `batch_decode_in_c=True`, non-stream + stream batch returned content for both rows, `kv_decode_steps` incremented, **`kv_page_bind.status=bound`** + **`bind_level=tensor`** on linked vendor kv-ext (full gate via `./scripts/metal_signoff.sh` + `eliza-1-2b:latest` qwen35).

---

## What shipped (detail)

### v0 — native block pool

| Piece | Location | Notes |
|-------|----------|--------|
| Native block pool | `runtime/native/kv_block_pool.c` → `runtime.kv._kv_native` | Same API as `runtime.kv._py_block_pool.BlockPool` |
| Backend selector | `runtime/runtime/kv/backend.py` | Env `ZEROLLAMA_RUNTIME_KV_NATIVE=1` |
| Engine wiring | `runtime/runtime/engine.py` | `create_block_pool()`; `/health` → `kv` object |
| Parity tests | `runtime/tests/test_kv_native_parity.py` | Skips if extension not built |
| CI helper | `scripts/phase15_kv_native_ci.sh` | build native + KV pytest bundle |

### v1 — scheduler ↔ llama slots

| Piece | Notes |
|-------|--------|
| `kv_scheduler` on `/health` | `blocks_reserved`, `requests[]` with per-request blocks + `kv_slot` |
| `num_ctx` block reserve | Admission reserves `max(prompt+max_tokens, num_ctx)` blocks |
| `kv_slot` | Subprocess → llama-server `id_slot`; in-process → libllama `seq_id` (same slot index) |
| Slot count | `resolve_parallel_slots(llama_server_args(), default=yaml)` — **`-np` in argv wins** over `llama_parallel_slots` in YAML |

### v1b — L3 prompt cache → pinned slots (Jun 2026)

| Piece | Notes |
|-------|--------|
| **Why** | v1 dynamic slots discard KV on `complete()` — agent threads re-prefill every turn |
| `cache_bridge.py` | Stable keys → `derive_slot_id(key, parallel)`; disk TTL; `--slot-save-path` |
| Admission | `_admit_one()` sets `prompt_cache_key`, pinned `kv_slot`, `slot_pinned` |
| `try_acquire` | Same slot index blocked while another request holds it; re-queue head |
| Subprocess | `cache_prompt: true` + `id_slot` on `/completion` |
| `/health` `llama_cache` | Root, `model_hash`, slot files, `model_loaded` |
| Batch | `prompt_cache_keys[]` per row; strict index (no flat-key fallback when list set) |
| Disk hygiene | Canonical GGUF path in hash; orphan hash-dir TTL sweep on start |
| `complete()` | `unregister_request_bind` before slot release (native bind race) |
| Env | `ZEROLLAMA_LLAMA_CACHE=0` disables; needs L1 `-np > 1` for multi-session |
| Doc | [gpu-profiles-l3.md](./gpu-profiles-l3.md) |

### v2 — in-process multi-seq

| Piece | Notes |
|-------|--------|
| Shared context | When effective `llama_parallel_slots` **> 1**: one shared `llama_context`, `llama_memory_seq_rm` per completion |
| Single-seq default | `llama_parallel_slots == 1` keeps per-request context (Phase 14 default) |
| Wheel backend | Unchanged — per-request; no `kv_slot` map |

### v3 — logical bind (not physical KV tensors)

| Piece | Notes |
|-------|--------|
| `runtime/runtime/kv/bind.py` | Page table: sequence page *i* → pool id `block_ids[i]` (ids are **not** required to equal *i*) |
| `/health` `kv_bind` | `mode`, `physical_bind_level`, `physical_pages_bound` (true when v33 writable page-map mapped all live pages) |
| `/health` `kv_scheduler.requests[].block_ids` | Primary-pool (device 0) page list per request |
| Forward | `assert_kv_capacity` before generate/stream/batch; in-process `kv_token_budget` vs PA reserve |
| Multi-seq load | Shared `n_ctx` capped by `num_blocks * block_size` when pool cap is configured |

### v4 — seq-position physical track

| Piece | Notes |
|-------|--------|
| `runtime/runtime/kv/physical.py` | Post-decode: llama cells used vs PA pages reserved |
| `/health` `kv_physical` | In-process: PA reserve per running request; **live** `llama_pos_*` when `llama_parallel_slots>1` or `ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL=1` (bumps effective `-np` to 2). Single-seq default still runs post-decode checks each completion. |
| `/health` `kv_native_scheduler_tick` | Legacy int: last admission tick when native ext built |
| Native `scheduler_tick()` | `runtime.kv._kv_native.scheduler_tick()` |
| Env | `ZEROLLAMA_RUNTIME_KV_PHYSICAL_STRICT=1` — fail on PA/llama mismatch (in-process) |

### v5 — observability + tick fallback

| Piece | Notes |
|-------|--------|
| `kv_scheduler_tick` | `{value, source: native\|python}` — unified admission tick |
| `kv_physical_recent` | Ring buffer (max 8) of **mismatch-only** post-decode alignment rows |
| `phase15_kv_native_ci.sh` | Full KV pytest bundle + health smoke |

### v6 — native decode-step hook

| Piece | Notes |
|-------|--------|
| Native `decode_step(n)` | Counts in-process `llama_decode` / `llama_encode` (ctypes path) |
| `/health` `kv_decode_steps` | Active for `inprocess` only; else `{active: false, reason}` |
| Per response | `kv_decode_steps` on non-stream `/api/generate`, `/api/chat`, `/v1/completions`, stream final chunk / batch tail (in-process) |
| Python fallback | Same semantics when native ext not built |
| Env | `ZEROLLAMA_RUNTIME_KV_DECODE_HOOK=0` disables hook |

### v7 — forward plan + debug snapshot

| Piece | Notes |
|-------|--------|
| `kv_forward_plans` | `/health` — waiting + running requests: logical page table (see schema below) |
| `kv_native_stats` | Read-only `{scheduler_tick, decode_steps}` via `kv_stats()` (no increment) |
| `GET /internal/kv-snapshot` | Loopback-only KV subset (`engine.kv_snapshot()`) |
| `phase15_health_smoke.sh` | Asserts v7+ `/health` KV keys without GPU or llama-server |

### v8 ops — seq-position page bind + C decode batch (Jun 2026)

**Why this slice exists:** llama.cpp does not expose stable public handles to map PA `block_ids` onto internal KV tensor pages. v8 still moves work off the interpreter for what we *can* bind today: per-`kv_slot` page tables, token-position validation before each batch, and C-built `llama_batch` metadata for the ctypes decode path.

| Piece | Notes |
|-------|--------|
| `page_bind_set/clear/resolve` (C) | Registry keyed by `kv_slot`; `resolve(token_pos)` → `(page, block_id, offset)` |
| `register_request_bind` | Scheduler admit → register from `kv_forward_plan`; clear on `complete()` |
| `validate_token_positions` | Endpoints of each batch checked against registry; raises `LlamaServerError` on overrun |
| `decode_batch_layout` (C) | Returns `{token, pos, seq_id, logits}` lists — Python fills `LlamaBatch` |
| `decode_prefill_chunks` (C) | Splits long prompts at PA page boundaries when `len(prompt) > block_size` |
| `kv_auto_batch` | v32: opt-in coordinator stats when `ZEROLLAMA_KV_AUTO_BATCH=1` — `pending`, `flush_count`, `window_ms` |
| `kv_page_bind` on `/health` | `status` values: `partial` (normal), `misaligned` (llama cells exceed PA reserve), `bound` (tensor verified), `not_implemented` (ext not built). `bind_level` values: `seq_position` → `cell_index` → `tensor` → `physical` (escalating bind quality). `blocker`: probe-reported string when a probe ran, or `llama_kv_ext_not_linked_or_no_decode` when no probe. **v32b:** `writable_bind_available`, `writable_bind_api`, `writable_bind_blocker` — static upstream writable page-map tracker (no live ctx). **v33:** `physical_pages_bound` true when `llama_memory_kv_page_map` resolves writable spans after decode. `slots`: per-slot export from v21 C registry. v19 adds `tensor_probe`, `tensor_bind_ready`, `blocker`, `accounting_aligned`. |
| `kv_live_physical` | Opt-in env bumps in-process effective `-np` to 2 when YAML defaults to 1 |
| Go loopback | `GET :8080/internal/kv-snapshot` proxies Python runtime snapshot |

**Build native ext (required for partial page bind):**

```bash
cd runtime && python3 setup.py build_ext --inplace
export ZEROLLAMA_RUNTIME_KV_NATIVE=1   # optional: C block pool allocator too
```

**Still not shipped:** PA `block_ids` → llama **tensor** KV pages. **v13:** optional C decode loop (`llama_decode` in linked build); default CI build still uses ctypes for decode.

### v9 ops — decode prefill plan export (Jun 2026)

**Why this slice:** exit criterion #6 requires native batch layout **wired to forward plans**. v8 already page-chunks at decode time; v9 exposes the same plan on `/health` and `/internal/kv-snapshot` so operators and a future C decode loop share one source of truth **without running inference**.

**Why export-only (plan does not drive decode yet):** the ctypes path still calls `iter_prefill_decode_chunks` independently at runtime. v9 reuses that function so the exported plan matches real decode; a future native loop will consume the plan directly once libllama links into the extension.

| Piece | Notes |
|-------|--------|
| `runtime/kv/decode_plan.py` | `kv_decode_prefill_plan()` — page-aligned chunks + optional C `batch_layout` summary |
| `kv_forward_plans[].decode_prefill` | Present when request has `prompt_tokens` **and** admitted `block_ids` (waiting requests omit it) |
| `iter_prefill_decode_chunks` | Shared with `libllama_ctypes._prefill_prompt` — **why:** chunk boundaries cannot drift |
| `logits_last: false` | On **every** prefill chunk — **why:** ctypes prefill never requests logits; first sample logit comes from the decode loop’s single-token batch |
| `pos_start=0` | Full prompt at admit time — **why:** continuation positions (`n_pos` after generation) are decode-time; in-progress plans need current `n_pos` (v10+) |
| `seq_id == kv_slot` | Same integer today — **why:** `SlotAllocator` assigns slot N → in-process seq N; subprocess maps to `id_slot` |
| Tests | `tests/test_kv_decode_plan.py` (10 tests); CI via `phase15_kv_native_ci.sh` |

**Example `decode_prefill` object** (native ext built, 40-token prompt, `block_size=16`):

```json
{
  "prefill_chunks": [
    {
      "chunk_index": 0,
      "token_count": 16,
      "pos_start": 0,
      "pos_end": 15,
      "page_range": [0, 0],
      "logits_last": false,
      "batch_layout": {"n_tokens": 16, "first_pos": 0, "last_pos": 15}
    }
  ],
  "n_prefill_batches": 3,
  "layout_source": "native",
  "page_bind_slot": 0
}
```

When the native ext is **not** built, `layout_source` is `"python"` and the whole prompt is one chunk (page-chunking still happens at decode time via the same API once the ext is built).

### v10 ops — in-progress decode plans (Jun 2026)

**Why this slice:** v9 plans assume the full prompt at admit (`pos_start=0`). Operators debugging a **running** request need remaining prefill + decode steps from the **current** llama write position.

| Piece | Notes |
|-------|--------|
| `kv_decode_prefill_plan(..., pos_start=)` | Remaining prompt chunks from `current_pos`; `prefill_complete: true` when done |
| `kv_decode_step_plan()` | Single-token steps, `logits_last: true` — matches `_decode_stream`; `pending_prefill` while prefill incomplete |
| `plan_current_pos` | Next write position on forward plan when live tracking available |
| `decode_steps` | `n_decode_batches_remaining`, `pos_range`, `step` template |
| Engine wiring | `_kv_forward_plans_health()` reads `kv_physical.running[].llama_pos_max` → `next_pos_from_llama()` |

**Why live positions are optional:** single-seq in-process (`llama_parallel_slots==1`) uses per-request ctx — `kv_physical` has no live `llama_pos_*` unless `ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL=1`. Without live data, forward plans keep the v9 admit-time shape.

**Example in-progress forward plan** (20-token prompt, prefill done, 8 decode steps left):

```json
{
  "plan_current_pos": 20,
  "decode_prefill": {
    "prefill_complete": true,
    "n_prefill_batches": 0,
    "prefill_chunks": []
  },
  "decode_steps": {
    "current_pos": 20,
    "n_prompt": 20,
    "tokens_generated": 0,
    "n_decode_batches_remaining": 8,
    "pos_range": [20, 27],
    "step": {"token_count": 1, "logits_last": true}
  }
}
```

### v11 ops — unified decode work plan + libllama link scaffold (Jun 2026)

**Why this slice:** v9–v10 split prefill and decode into separate objects; operators need one `phase` field. The libllama-linked C decode loop is blocked on build wiring — v11 ships the **contract** (`decode_loop_status`) before `llama_decode` moves into the extension.

| Piece | Notes |
|-------|--------|
| `kv_decode_work_plan()` | `{phase, prefill, decode}` — `admit` / `prefill` / `decode` / `done` |
| `kv_forward_plans[].decode_work` | Always on admitted requests; live phase when `current_pos` known |
| `current_pos_by_request_from_physical()` | Engine helper; maps `kv_physical.running` → next write position |
| `/health.kv_decode_loop` | `{available: false, reason, link: "ctypes"}` until native loop linked |
| C `decode_loop_status()` | Build with `ZEROLLAMA_KV_DECODE_LOOP=1` + `LLAMA_CPP_LIB` (future) |

**Build (future native loop — not default):**

```bash
export ZEROLLAMA_KV_DECODE_LOOP=1
export LLAMA_CPP_LIB=/path/to/libllama.so
cd runtime && python3 setup.py build_ext --inplace
```

### v12 ops — libllama link build + probe (Jun 2026)

**Why this slice:** v11 exposed `decode_loop_status` but could not prove libllama was linked. v12 adds build-time wiring so operators verify the extension links before `llama_decode` moves in.

| Piece | Notes |
|-------|--------|
| `runtime/setup.py` | Resolves `LLAMA_CPP_ROOT` / `LLAMA_CPP_LIB`; `-lllama` + rpath when flag set |
| `native/kv_decode_loop.c` | Calls `llama_max_devices()` — no ctx, no inference |
| `/health.kv_decode_loop` | `available: true`, `llama_max_devices` when linked build |
| `scripts/phase15_kv_decode_loop_build.sh` | Optional smoke; skips if libllama missing |

---

### v13 ops — native C decode loop (Jun 2026)

**Why this slice:** v12 proved libllama links. v13 moves `llama_decode` into C, reducing per-token GIL hold time. Sampling (`llama_sampler_sample`) stays in Python because the sampler chain is already wired via ctypes and the sampling call is negligible relative to the forward pass.

| Piece | Notes |
|-------|--------|
| `native/kv_decode_loop.c` | `kv_decode_loop_run_prefill` — page-aligned chunking loop in C; `kv_decode_loop_run_step` — single-token decode |
| `native/kv_decode_loop.h` | Declarations for both functions (gated by `ZEROLLAMA_KV_DECODE_LOOP`) |
| `native/kv_block_pool.c` | `decode_loop_prefill(ctx_ptr, tokens, seq_id, block_size) → steps`; `decode_loop_step(ctx_ptr, token, seq_id, current_pos) → steps`; Python bindings, `#ifdef` gated |
| `runtime/kv/native_decode_loop.py` | `run_prefill(ctx_ptr, tokens, *, seq_id, block_size) → int \| None`; `run_step(ctx_ptr, token, *, seq_id, current_pos) → int \| None`; returns `None` when ctypes build |
| `runtime/worker/libllama_ctypes.py` | `_decode_stream` v13 fast path: C prefill → sample → C step loop; falls back to ctypes when ext not linked or encoder model |
| Tests | `test_run_prefill_returns_none_when_not_linked` / `test_run_step_returns_none_when_not_linked` |
| CI | `scripts/phase15_kv_decode_loop_build.sh` — also verifies `decode_loop_prefill` / `decode_loop_step` symbols exported |

**WHY ctx_ptr is `int`:** ctypes `c_void_p` arrives as a Python `int`; the C binding casts `(void *)(uintptr_t)ctx_ptr` → `struct llama_context *`.  No ABI mismatch because both the ext and `_decode_stream` load the same libllama.

**Still not shipped:** sampling in C (sampler chain lives in Python); GIL release around C `llama_decode`; remaining prefill from `current_pos` in C path; llama tensor KV page management.

---

### v14 ops — harden C decode loop (Jun 2026)

**Why this slice:** v13 proved C decode works but held the GIL (blocking shared embedded Python) and always prefilled from position 0. v14 makes the loop production-safe for operators: release GIL, resume prefill, validate page bind, optional E2E parity vs ctypes.

| Piece | Notes |
|-------|--------|
| `native/kv_block_pool.c` | `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS` around prefill + step; `gil_released: true` on `/health.kv_decode_loop` |
| `native/kv_decode_loop.c` | `pos_start` param — llama write positions are `pos_start + tok_off` |
| `decode_loop_prefill(..., pos_start=0)` | Python binding extended (`\|i` optional arg) |
| `runtime/kv/native_decode_loop.py` | `validate_token_positions` before C calls; `greedy_decode_tokens()` for tests |
| `tests/test_kv_decode_loop_e2e.py` | Gated: `RUN_E2E_DECODE_LOOP=1` + `LLAMA_MODEL` + linked build |
| `scripts/phase15_kv_decode_loop_build.sh` | Asserts `gil_released`; runs E2E when env set |

**WHY GIL release in bindings not in kv_decode_loop.c:** Python owns thread state; only the extension entry points know when it is safe to release before calling into libllama.

**Still open (v17+):** tensor page bind (blocked on llama.cpp API).

---

### v15 ops — sampling in C + resume prefill (Jun 2026)

**Why this slice:** v14 left the last hot-path ctypes call (`llama_sampler_sample`) and hardcoded prefill at position 0. v15 moves sampling into the linked ext and wires `current_pos` for remaining-prefill resume.

| Piece | Notes |
|-------|--------|
| `native/kv_decode_loop.c` | `kv_decode_loop_sample`; optional sampler on `run_step` |
| `decode_loop_sample(smpl_ptr, ctx_ptr)` | Post-prefill first token |
| `decode_loop_step(..., smpl_ptr=0)` | Returns `(steps, token)` when sampling |
| `/health.kv_decode_loop.sampling_in_c` | Operator probe |
| `_decode_stream(..., current_pos=0)` | Remaining prefill via `pos_start`; skip prefill when `current_pos >= n_prompt` |
| `tests/test_kv_decode_stream_resume.py` | Mock wiring tests |

---

### v16 ops — engine resume wiring (Jun 2026)

**Why:** v15 wired `_decode_stream(current_pos=)` but every `complete()` cleared the seq. v16 reads live seq positions before decode and preserves KV when resuming.

| Piece | Notes |
|-------|--------|
| `current_pos_for_seq()` | Live read via `sequence_kv_usage` + `next_pos_from_llama` |
| `InferenceEngine._decode_current_pos_for_request()` | Passed into completion paths |
| `complete(..., current_pos=)` | Skip `_clear_sequence` when `decode_pos > 0` |
| Subprocess backends | Accept and ignore `current_pos` (L3 slot resume is server-side) |

**Hardened by v16b** (slot-ownership guard — see below).

---

### v16b ops — slot-ownership guard (Jun 2026)

**Why:** v16's skip-clear guard (`decode_pos > 0`) was necessary but not sufficient.  A different request can be scheduled onto the same slot after the previous one completes.  Without an ownership check, the new request would inherit stale KV entries from the old request's sequence — logits would be influenced by tokens it never saw, producing incoherent or biased generations.

**Root cause:** `decode_pos` is the live write position for the slot, not a proof that the *current* request wrote it.

**Fix — `_seq_last_owner`** (shipped in v16b as `_seq_last_request_id`, extended in v17):

```
session._seq_last_owner: dict[seq_id, owner_key]
```

Updated at the end of every successful decode (stream and non-stream paths). v16b owner key was `request_id`; v17 uses `slot_resume_owner_key()`. `complete()` only skips `_clear_sequence` when all three hold:

1. `decode_pos > 0` — there is actually something to resume into.
2. Incoming owner is present — caller has a stable identity.
3. `_seq_last_owner[sid] == incoming_owner` — same owner last wrote this slot.

If any condition fails, the slot is cleared (safe fallback = pre-v16 behaviour).

| Piece | Notes |
|-------|--------|
| `LlamaLoadedSession._seq_last_owner` | `dict[int, str]` — per-slot owner map; keyed by normalised `seq_id`; cleared on `close()` |
| `complete()` | `is_resume` flag replaces bare `decode_pos == 0` guard |
| Post-decode write-back | `_seq_last_owner[sid] = incoming_owner` after both stream and non-stream paths |
| `_resolve_decode_current_pos` docstring | WHY the single-seq path is a no-op (always returns 0 — single-seq sessions never resume) |
| `_decode_current_pos_for_request` docstring | WHY the TOCTOU read outside the lock is safe; how `_seq_last_owner` re-validates under the lock |
| Tests | `test_complete_skips_clear_same_request_id`, `test_complete_clears_sequence_different_request_id`, `test_complete_clears_sequence_no_req_id_with_decode_pos` |

**TOCTOU note (low severity, documented):** `_decode_current_pos_for_request` reads the live position outside the `LlamaInprocessWorker` lock.  Worst-case: returns 0 for a slot that gets populated before the lock is acquired.  The `_seq_last_owner` guard under the lock then forces a clear — identical to pre-v16 behaviour.  No corruption risk; only a wasted resume opportunity.

---

### v17 ops — L3 session resume owner (Jun 2026)

**Why:** v16b matched slot ownership on `request_id`. L3 pinned sessions (`slot_pinned=True`, derived from `prompt_cache_key`) allocate a **fresh** `request_id` on every HTTP turn while keeping the same `kv_slot` and llama prefix KV. The owner check always failed → `_clear_sequence` ran every turn → agent chat re-prefilled the full system prompt.

**Fix — `slot_resume_owner_key()`:**

| Condition | Owner key |
|-----------|-----------|
| `slot_pinned` + `prompt_cache_key` | `cache:{prompt_cache_key}` |
| Otherwise | `{request_id}` |

`LlamaLoadedSession._seq_last_owner` (renamed from `_seq_last_request_id`) stores this key per `seq_id`. Same three-condition resume guard as v16b, but turn 2 of an agent session now matches on cache key even when `request_id` differs.

| Piece | Notes |
|-------|--------|
| `runtime/cache_bridge.py` | `slot_resume_owner_key(kv_bind_req)` |
| `LlamaLoadedSession._seq_last_owner` | Renamed map; cleared when sequence cleared or on `close()` |
| Tests | `test_complete_skips_clear_l3_second_turn`, `test_complete_clears_sequence_different_pinned_session`, `test_close_clears_seq_last_owner` |

**Subprocess note:** L3 resume for llama-server remains server-side (`cache_prompt` + stable `id_slot`); v17 fixes the in-process ctypes/C decode path only.

---

### v18 ops — kv_resume health + L3 gate (Jun 2026)

**Why:** Resume state was session-local with no operator visibility. Debugging “did turn 2 reuse prefix KV?” required code inspection of `_seq_last_owner`.

| Field / piece | Meaning |
|---------------|---------|
| `/health.kv_resume.active` | `true` when in-process multi-seq shared ctx is loaded |
| `owners_by_slot` | `{slot_id: owner_key}` from `resume_owner_snapshot()` |
| `owner_key_pinned` / `owner_key_unpinned` | Documents v17 owner scheme |
| `note` | Why inactive (subprocess or single-seq) |
| `scripts/phase15_metal_signoff.sh` step 4 | Two-turn generate with same `prompt_cache_key`; asserts `kv_resume.active` and non-empty owners |

---

### v19 ops — tensor bind scaffold (Jun 2026)

**Why blocked (full bind):** llama.cpp exposes `llama_get_memory` and per-sequence position queries but no stable public API to map external PA `block_ids` onto internal KV tensor page storage. v19 **unblocks the integration path** with what we can verify today.

**Accounting bind (shipped):**

| Piece | Role |
|-------|------|
| `kv_tensor_probe_run` | Linked C: memory non-null, `can_shift`, seq pos min/max, PA pages vs llama cells |
| `page_bind_tensor_probe(ctx, seq_id, kv_slot)` | Python binding → `/health.kv_page_bind.tensor_probe`; **no-op (returns `None`) when ext not linked** — non-linked operators are not affected |
| `page_bind_table(kv_slot)` | Export `{page, block_id, token_start, token_end}` from native registry; returns `[]` when ext not built |
| Post-decode verify | `_tensor_probe_after_decode` warns (or raises under strict env) when cells exceed PA reserve; silent no-op on non-linked builds |
| `/health.kv_page_bind` | `status=partial` (normal) or `status=misaligned` (cells > PA reserve); adds `tensor_bind_ready`, `blocker`, `accounting_aligned`, optional `tensor_probe` (only when a request is running) |

**Operator smoke:**

```bash
cd runtime && python3 setup.py build_ext --inplace
../scripts/phase15_tensor_bind_probe.sh
# Linked probe (GPU host):
export ZEROLLAMA_KV_DECODE_LOOP=1 LLAMA_CPP_LIB=/path/to/libllama.dylib
cd runtime && python3 setup.py build_ext --inplace
../scripts/phase15_tensor_bind_probe.sh
```

**Upstream needed for full tensor bind:** public C API returning writable KV page/cell handles keyed by `(layer, page_index)` or equivalent — then wire `block_ids[i]` → handle in `page_bind_tensor_bind()`.

---

### v20a ops — native page table in forward plans (Jun 2026)

**Why:** v19 exported `page_bind_table` for scripts but operators still had to cross-check `kv_forward_plans.pages[]` against the C registry manually. v20a attaches the native mirror when admitted.

| Piece | Role |
|-------|------|
| `native_page_table` | C registry rows on admitted `kv_forward_plans[]` entries |
| `page_table_native_parity` | `true` when logical `pages[]` matches native registry |
| `page_table_native_parity()` | Shared compare helper in `tensor_probe.py` |
| CI | `phase15_kv_native_ci.sh` includes tensor probe + engine resume tests |

**Still blocked for full tensor bind:** same upstream page-handle API as v19.

---

### v20 ops — cell + tensor bind via llama-kv-ext (Jun 2026)

**Why:** v19 could only verify seq-position accounting. v20 adds a **staging API** in the pinned llama.cpp fork (`include/llama-kv-ext.h`) so zerollama can map PA pages → llama KV **cell indices** and verify **K/V tensor backing** after decode.

| Piece | Role |
|-------|---------|
| `llama-kv-ext.h` | Staging C API: `llama_memory_kv_cell_for_pos`, `llama_memory_kv_cell_map_range`, `llama_memory_kv_tensor_info` |
| `llama-memory-kv-ext.cpp` | Implements ext API for standard `llama_kv_cache` (not hybrid/iSWA/recurrent) |
| `kv_tensor_probe.c` | v20 bind attempt after accounting; sets `cell_pages_bound`, `tensor_pages_bound` |
| `/health.kv_page_bind` | `status=bound` + `bind_level=tensor` when linked probe succeeds; `bind_level=cell_index` when cells only |
| Rebuild | **Must rebuild libllama** from fork before `ZEROLLAMA_KV_DECODE_LOOP=1` native ext link |

**Operator build (GPU host):**

```bash
# 1) Rebuild libllama from zerollama fork (includes llama-kv-ext)
cd llama/llama.cpp && cmake -B build -DBUILD_SHARED_LIBS=ON && cmake --build build -j
# 2) Link native ext
export LLAMA_CPP_ROOT=$PWD ZEROLLAMA_KV_DECODE_LOOP=1
cd ../../runtime && python3 setup.py build_ext --inplace
../scripts/phase15_tensor_bind_probe.sh
```

**Limitations:** pure recurrent-only models → `memory_kind_name=unsupported`. SWA window cache is not the PA bind target (base attn cache is). Writable cross-process PA→tensor migration still needs upstream stable page-handle API; v20/v31 is read-verify bind on live decode.

**Pin tracking:** patch `0014-ollama-llama-kv-ext-Phase-15-tensor-page-bind-b9611.patch` + `./scripts/phase15_llama_kv_ext_pin_check.sh` — see [phase15-llama-kv-ext-upstream.md](./phase15-llama-kv-ext-upstream.md).

### v20b ops — audit fixes (Jun 2026)

| Fix | Role |
|-----|------|
| Per-stream K/V tensor | `kv_tensor_k/v(layer, stream)`; probe exports `kv_stream` |
| Shifted positions | Cell map starts at `seq_pos_min`, not 0 |
| Partial last page | Probe only live token pages; last page may be short |
| Stale bind flags | Clear registry flags before each bind attempt |
| Health state | `status=bound` not overridden when `tensor_pages_bound` |

### v22 ops — stale decode_pos after clear (Jun 2026)

**Root cause:** multiseq shared ctx read `decode_pos` from live KV before the resume guard, cleared the slot on non-resume, but still passed the stale position into `_decode_stream`. Native prefill skipped (`start_pos >= n_prompt`); sampling ran without valid logits → intermittent Metal segfault.

**Fix:** `decode_pos = 0` immediately after `_clear_sequence` when `not is_resume`.

| Piece | Role |
|-------|------|
| `runtime/infer_trace.py` | Opt-in phase logging (`ZEROLLAMA_INFER_TRACE=1`) |
| `scripts/phase15_metal_crash_repro.sh` | Bisect harness (workarounds off) |

### v23 ops — unified prefill chunker + sign-off defaults (Jun 2026)

**Why:** v9–v15 duplicated page-chunk boundaries between `/health` export and `_prefill_prompt`. Sign-off also left C block pool off by default.

| Piece | Role |
|-------|------|
| `iter_prefill_execute_chunks()` | Single source for ctypes prefill + `kv_decode_prefill_plan`; final chunk `logits_last=True` |
| `libllama_ctypes._prefill_prompt` | Calls shared chunker (v23) |
| `scripts/phase15_runtime_kv_env.sh` | `ZEROLLAMA_RUNTIME_KV_NATIVE=1`, native decode/sample defaults; optional ext build |
| Sign-off scripts | `phase15_metal_signoff.sh`, `phase15_inprocess_signoff.sh` source env + build ext when `PHASE15_BUILD_KV_EXT=1` |

---

## Two KV caps (operators)

1. **PA block pool** (`kv_pools`, `kv_scheduler`) — admission and `/health`; sum of reserved blocks × `block_size`.
2. **llama KV** — subprocess or in-process context; sized by `-c` / load `num_ctx` and (in-process) per-request `kv_token_budget`.

On **subprocess**, `kv_token_budget` is not sent to llama-server: PA reserve is bookkeeping until server-side bind exists. On **in-process**, both PA assert and `kv_token_budget` can reject oversize prompt+generation.

---

## Architecture

```text
Today (v0–v12 ops):
  SchedulerLoop → PA block ids + kv_slot → llama seq/slot
  Admit → page_bind_set (native C) from kv_forward_plan page table
  In-process decode → validate token positions vs page table; C batch layout + optional page-chunked prefill
  Post-decode → seq_pos vs PA reserve; scheduler_tick + decode_step hooks
  kv_forward_plans → logical page table + decode_work / decode_prefill export
  kv_decode_loop → ctypes default; libllama linked when built with ZEROLLAMA_KV_DECODE_LOOP=1
  kv_page_bind → partial (seq-position) when native ext built; v20 bound when llama-kv-ext linked + decode ran

Target (hybrid memory + upstream writable handles):
  PA block_ids → llama KV tensor pages (all arch types)
  Native batched decode loop in C/Rust (no ctypes llama_decode)
  Python → config, admission, /health only
```

Phase 14 **inprocess** per-request `llama_context` remains default when `llama_parallel_slots` is 1; multi-seq shared ctx is opt-in via YAML/`-np`.

---

## `/health` — KV fields (complete)

| Field | Meaning |
|-------|---------|
| `kv` | Allocator: `backend` (`python` / `native`), `native_requested`, `native_available`, optional `note` |
| `kv_pools` | Per-device `free`, `utilization` (unchanged) |
| `kv_scheduler` | `blocks_reserved`, `requests[]` (`block_ids`, `kv_slot`, tokens, state) |
| `kv_bind` | `mode`, `physical_bind_level`, `physical_pages_bound` (aggregate from native page-bind stats) |
| `kv_physical` | Running-request PA reserve; live `llama_pos_*` when multi-seq shared ctx |
| `kv_physical_recent` | Last ≤8 **mismatch** alignment rows (`request_id`, `at`, `aligned`, …) |
| `kv_scheduler_tick` | `{value, source}` admission counter |
| `kv_native_scheduler_tick` | Legacy nullable int (prefer `kv_scheduler_tick`) |
| `kv_decode_steps` | Cumulative in-process decode count or `{active: false, reason}` |
| `kv_native_stats` | `{scheduler_tick, decode_steps}` from C when ext built; else `null` |
| `kv_forward_plans` | List of forward-plan objects (waiting + running) |
| `kv_page_bind` | v8 bind status: `partial` + `bind_level=seq_position` when native ext built without linked tensor probe; **`bound` + `bind_level=tensor|physical`** after linked `llama-kv-ext` decode (GPU sign-off success path); `not_implemented` when ext missing. **v33:** `physical_pages_bound` when `llama_memory_kv_page_map` maps all live pages. GPU smokes: `smoke_runtime_assert_kv_snapshot()` accepts partial or bound — **why:** requiring partial-only false-fails linked vendor builds. |
| `kv_live_physical` | Opt-in bump to multi-seq in-process ctx (`ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL`) |

`kv_bind.physical_bind_level` is `seq_position` whenever in-process weights are loaded (not only multi-seq). `kv_physical` may include a `note` when `llama_parallel_slots==1` (no shared ctx for live positions).

---

## `kv_forward_plans` schema (v7)

Each element is one scheduler request (waiting or running):

```json
{
  "request_id": "…",
  "state": "running",
  "kv_slot": 0,
  "block_size": 16,
  "num_ctx": 4096,
  "pages": [
    {"page": 0, "block_id": 3, "token_start": 0, "token_end": 15}
  ],
  "pa_tokens_reserved": 64,
  "tokens_to_reserve": 48
}
```

- **`block_id`** — pool allocator id for page ordinal `page` (not necessarily `page` itself).
- **`token_start` / `token_end`** — inclusive logical token range for that page.
- **`decode_prefill`** (v9, when admitted) — planned prefill batches. Fields:
  - `prefill_chunks[]` — per chunk: `chunk_index`, `token_count`, `pos_start`, `pos_end`, `page_range` `[first_page, last_page]`, `logits_last` (always `false`), optional `batch_layout` (`n_tokens`, `first_pos`, `last_pos`) when native ext built.
  - `n_prefill_batches`, `layout_source` (`native` / `python`), `page_bind_slot` (same as `kv_slot`).
- Empty `pages` when the request has not been admitted to a block table yet — **`decode_prefill` is omitted** in that case (no block table → no meaningful plan).

---

## Internal debug — `GET /internal/kv-snapshot`

Loopback-only (same middleware as `/internal/vram-estimate`). Returns:

`kv`, `kv_bind`, `kv_scheduler`, `kv_physical`, `kv_physical_recent`, `kv_scheduler_tick`, `kv_decode_steps`, `kv_native_stats`, `kv_forward_plans`

```bash
curl -s http://127.0.0.1:8081/internal/kv-snapshot | python3 -m json.tool
# or via Go daemon (embedded/sidecar runtime):
curl -s http://127.0.0.1:8080/internal/kv-snapshot | python3 -m json.tool
```

---

## Operator knobs

| Env | Default | Effect |
|-----|---------|--------|
| `ZEROLLAMA_RUNTIME_KV_NATIVE` | off | Use C `BlockPool` when extension is installed |
| `ZEROLLAMA_RUNTIME_KV_PHYSICAL_STRICT` | off | In-process: error if llama seq cells exceed PA reserve after decode |
| `ZEROLLAMA_RUNTIME_KV_DECODE_HOOK` | on | Count decode steps on in-process ctypes path; set `0` to disable |
| `ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL` | off | In-process: bump effective `-np` to 2 when defaults use 1 for live `kv_physical` (explicit `-np` wins) |
| `llama_parallel_slots` / `-np` | yaml / argv | Slot allocator + in-process `n_seq_max` (**argv wins**) |
| (build) | — | `cd runtime && python3 setup.py build_ext --inplace` |

---

## Quick start

```bash
cd runtime
python3 setup.py build_ext --inplace
pip install -e ".[serve,dev]"   # optional; tests use PYTHONPATH=.
export ZEROLLAMA_RUNTIME_KV_NATIVE=1

# Tests + smokes (no GPU)
cd ..
./scripts/phase15_kv_native_ci.sh      # includes phase15_health_smoke.sh
./scripts/phase15_health_smoke.sh      # /health KV keys only

# With serve (embed or sidecar):
export ZEROLLAMA_RUNTIME_KV_NATIVE=1
./zerollama serve
curl -s http://127.0.0.1:8081/health | python3 -c "import json,sys; h=json.load(sys.stdin); print(h.get('kv'), len(h.get('kv_forward_plans',[])))"
curl -s http://127.0.0.1:8081/internal/kv-snapshot | python3 -m json.tool | head

# GPU in-process (needs LLAMA_CPP_LIB + small GGUF):
export LLAMA_MODEL=/path/to/small.q8_0.gguf
export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
./scripts/phase15_inprocess_signoff.sh
```

---

## CI

| Script | Role |
|--------|------|
| `scripts/phase15_kv_native_ci.sh` | `build_ext --inplace` + KV pytest + `phase15_health_smoke.sh` |
| `scripts/phase15_health_smoke.sh` | Engine `/health` KV key assertions (no GPU) |
| `scripts/phase15_inprocess_signoff.sh` | GPU: KV decode hook + multi-seq (self-contained; needs `LLAMA_CPP_LIB`) |
| `scripts/phase15_batch_decode_smoke.sh` | GPU: continuous batch decode via `POST /internal/generate-batch` (needs multiseq sidecar) |
| `scripts/phase15_inprocess_kv_smoke.sh` | GPU: single-seq decode hook only |
| `scripts/phase15_inprocess_multiseq_smoke.sh` | GPU: `llama_parallel_slots: 2` |
| `.github/workflows/zerollama-regression.yaml` | Runtime pytest; native tests **skip** if `.so` missing |

Optional self-hosted: run `phase15_kv_native_ci.sh` after building the extension on the runner.

---

## Related

- [phase14-inprocess-llama.md](./phase14-inprocess-llama.md) — forward path prerequisite
- [handoff-phase14-inprocess-llama.md](./handoff-phase14-inprocess-llama.md)
- [handoff-phase15-native-kv.md](./handoff-phase15-native-kv.md) — continuation handoff
- [testing-smoke.md](./testing-smoke.md) — script table
- [ROADMAP.md](./ROADMAP.md) — Phase 15 row
- [runtime/native/README.md](../runtime/native/README.md) — build notes
