# Handoff: Phase 15 native scheduler + KV

**Audience:** Engineers continuing runtime KV / scheduler work without this thread.

**Status (Jun 2026):** **Partial (v0–v30 ops)** — Phase 14 prerequisite **Done**. PA block pool (Python + opt-in C), scheduler accounting, `kv_slot`→llama seq/slot, logical + **seq-position page bind (v8 partial)**, C decode batch layout, **v9 decode prefill export**, **v10 in-progress decode plans**, **v11 unified decode work plan**, **v12 libllama link probe**, **v13 native C decode loop**, **v14 GIL release + pos_start + page-bind validation + E2E smoke**, **v15 sampling in C + `_decode_stream` resume**, **v16 engine→`complete(current_pos=)` wiring**, **v16b slot-ownership guard (`_seq_last_owner`)**, **v17 L3 session resume owner (`slot_resume_owner_key`)**, **v18 `/health.kv_resume` + Metal L3 two-turn gate**, **v19–v21 tensor bind probe + per-slot registry**, **v22 stale decode_pos fix**, **v23 unified prefill chunker**, **v24 C decode loop page-bind validation + post-prefill tensor probe**, **v25 auto-link decode loop + 131k page-bind cap + long-ctx validation**, **v26 continuous batch decode in C**, **v27 engine wiring for C batch decode**, **v28 `/health` continuous batch plan**, **v29 streaming batch decode**, **v30 per-row C batch sampling**. **Not done:** hybrid/iSWA bind; upstream-stable writable page handles; scheduler-driven async continuous batching across concurrent HTTP streams.

| Slice | Shipped |
|-------|---------|
| v0 | C `BlockPool`, `ZEROLLAMA_RUNTIME_KV_NATIVE`, `phase15_kv_native_ci.sh` |
| v1 | `kv_scheduler`, `num_ctx` reserve, subprocess `id_slot` |
| v2 | In-process multi-seq shared ctx (`llama_parallel_slots`>1) |
| v3 | `kv_bind`, `block_ids`, `assert_kv_capacity`, `kv_token_budget` |
| v4 | `llama_memory_seq_pos_*` vs PA; native `scheduler_tick()` |
| v5 | Python tick fallback; `kv_scheduler_tick`, `kv_physical_recent` (mismatches only) |
| v6 | Native `decode_step`; `/health` + API `kv_decode_steps`; `ZEROLLAMA_RUNTIME_KV_DECODE_HOOK` |
| v7 | `kv_forward_plans`, `kv_native_stats` (`kv_stats()`), `GET /internal/kv-snapshot`, `phase15_health_smoke.sh` |
| v8 ops | Seq-position `page_bind_*` in C; `decode_batch_layout` + prefill chunks; `/health` `kv_page_bind.status=partial`; Go loopback proxy; `ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL` |
| v9 ops | `decode_prefill` on `kv_forward_plans` — export page-aligned prefill batch plan (bridges to future C decode loop) |
| v10 ops | In-progress plans: `decode_steps`, `plan_current_pos`, remaining prefill from live `kv_physical` seq positions |
| v11 ops | Unified `decode_work` plan; `kv_decode_loop` link scaffold; `decode_loop_status()` in C |
| v12 ops | Optional libllama link at build (`ZEROLLAMA_KV_DECODE_LOOP=1`); `llama_max_devices` probe on `/health.kv_decode_loop` |
| v13 ops | C `kv_decode_loop_run_prefill` + `run_step`; Python bindings; `_decode_stream` C fast path |
| v15 ops | `decode_loop_sample`; C sampling on step; `_decode_stream(current_pos=)` |
| v16 ops | `current_pos_for_seq`; engine→`complete(current_pos=)`; skip seq clear on resume |
| v16b ops | `_seq_last_owner` slot-ownership map; skip clear only when same owner owns slot |
| v17 ops | `slot_resume_owner_key` — L3 pinned uses `prompt_cache_key` |
| v14 ops | GIL release; `pos_start` on prefill; page-bind validation; `greedy_decode_tokens`; E2E smoke |

**Prerequisite:** Phase 14 in-process forward for seq-position checks — [handoff-phase14-inprocess-llama.md](./handoff-phase14-inprocess-llama.md).

**Operator doc:** [phase15-native-kv.md](./phase15-native-kv.md) (knobs, `/health` tables, forward-plan schema).

---

## Documentation index

| Doc | Why |
|-----|-----|
| [phase15-native-kv.md](./phase15-native-kv.md) | Operator knobs, `/health` fields, v7 schema |
| [phase14-inprocess-llama.md](./phase14-inprocess-llama.md) | Forward backends |
| [ROADMAP.md](./ROADMAP.md) | Phase 15 row; Phase 16 edge daemon next |
| [upstream-ollama-diff.md](./upstream-ollama-diff.md) | Upstream delegates KV to llama-server; Phase 15 tensor bind blocked on llama.cpp pin/API |
| [testing-smoke.md](./testing-smoke.md) | `phase15_*` scripts in smoke table |
| [runtime/native/README.md](../runtime/native/README.md) | Build native ext |

---

## Code map

| Area | Path |
|------|------|
| Block pool (facade) | `runtime/runtime/kv/block_pool.py`, `backend.py`, `_py_block_pool.py` |
| Native pool + counters | `runtime/native/kv_block_pool.c` → `runtime.kv._kv_native` (`scheduler_tick`, `decode_step`, `kv_stats`, **v8** `page_bind_*`, `decode_batch_layout`, `decode_prefill_chunks`, **v12–v13** `decode_loop_*` when linked) |
| C decode loop (v13) | `runtime/native/kv_decode_loop.c` — `kv_decode_loop_run_prefill`, `kv_decode_loop_run_step`; build with `ZEROLLAMA_KV_DECODE_LOOP=1` |
| Decode loop facade | `runtime/runtime/kv/native_decode_loop.py` — `decode_loop_status`, `run_prefill`, `run_step` |
| Logical bind | `runtime/runtime/kv/bind.py` |
| Seq-position | `runtime/runtime/kv/physical.py` |
| Tick counter | `runtime/runtime/kv/native_tick.py` |
| Decode hook | `runtime/runtime/kv/native_decode.py` |
| Decode batch layout | `runtime/runtime/kv/native_decode_batch.py` — C-built batch fields; wired from `libllama_ctypes._decode_stream` |
| Forward plan | `runtime/runtime/kv/forward_plan.py` |
| Decode prefill plan (v9) | `runtime/runtime/kv/decode_plan.py` |
| Page bind (v8 partial) | `runtime/runtime/kv/page_bind.py` — admit register, decode validate, `/health` status |
| Live physical opt-in | `runtime/runtime/kv/live_physical.py` |
| Scheduler admit | `runtime/runtime/scheduler/loop.py` |
| Engine / health / snapshot | `runtime/runtime/engine.py` (`kv_snapshot`, `_kv_forward_plans_health`, **`generate_batch` / `stream_generate_batch` v27–v29**) |
| HTTP | `runtime/runtime/server/app.py` (`GET /internal/kv-snapshot`, **`POST /internal/generate-batch` v27–v30 GPU smokes**) |
| Go loopback proxy | `server/runtime_kv_snapshot.go`, `internal/runtimeclient/kv_snapshot.go` (`GET :8080/internal/kv-snapshot`) |
| In-process forward | `runtime/runtime/worker/libllama_ctypes.py` (`_decode_parallel_stream`, `complete_parallel_stream`, `complete_parallel`), `llama_inprocess.py` |
| C batch decode step | `runtime/native/kv_decode_loop.c` — `kv_decode_loop_run_batch_step` (v26 layout, v30 `smpl_ptrs[]`) |
| Batch smoke | `scripts/phase15_batch_decode_smoke.sh`, `scripts/phase15_metal_signoff.sh` step 3/5 |
| Slot count | `runtime/runtime/llama_args.py` (`resolve_parallel_slots`) |

---

## Two KV systems (read this first)

1. **PA block pool** — admission, `/health`, `kv_forward_plans`; sum(reserved blocks) × `block_size`.
2. **llama physical KV** — owned by subprocess or in-process libllama; sized by `num_ctx` / `kv_token_budget`.

**Contract today:** `block_ids[i]` is the pool id for logical page *i*; `kv_slot` selects llama sequence/slot. **No** mapping from pool ids to llama tensor KV pages.

---

## `/health` KV fields (quick reference)

| Field | Meaning |
|-------|---------|
| `kv` | Allocator backend `python` / `native` |
| `kv_scheduler` | Reserved blocks, per-request `block_ids`, `kv_slot` |
| `kv_bind` | `mode`, `physical_bind_level`, `physical_pages_bound: false` |
| `kv_physical` | Live seq positions (multi-seq) or PA-only + `note` (single-seq) |
| `kv_physical_recent` | ≤8 **mismatch-only** post-decode alignment rows |
| `kv_scheduler_tick` | `{value, source: native\|python}` |
| `kv_native_scheduler_tick` | Legacy: last admission tick int (nullable) |
| `kv_decode_steps` | Cumulative in-process decode steps or `{active: false, reason}` |
| `kv_native_stats` | `{scheduler_tick, decode_steps}` from C when ext built; else `null` |
| `kv_forward_plans` | Waiting + running: logical page table per request |
| `kv_page_bind` | v8–v19: `status=partial` (normal) or `misaligned` (llama cells > PA reserve); v19 adds `tensor_probe`, `tensor_bind_ready`, `blocker`, `accounting_aligned` when linked |
| `kv_decode_loop` | v11–v15: `{available, reason, link}`; `llama_max_devices` when linked; **v14:** `gil_released`; **v15:** `sampling_in_c` |
| `kv_resume` | v18: `{active, llama_parallel_slots, owners_by_slot, note}` — in-process L3 resume probe |
| `kv_live_physical` | Opt-in in-process multi-seq for live `kv_physical` (`applied`, `effective`, `reason`) |

---

## v7 — forward plan + snapshot

**Module:** `runtime/runtime/kv/forward_plan.py` — `kv_forward_plan()`, `kv_forward_plans_for_requests()`.

**Included on `/health` and in `engine.kv_snapshot()`** for `scheduler.waiting` + `scheduler.running` only (not completed).

**Per-plan fields:** `request_id`, `state`, `kv_slot`, `block_size`, `num_ctx`, `pages[]`, `pa_tokens_reserved`, `tokens_to_reserve`.

**v10–v11 — `decode_work` (when admitted):** unified `{phase, prefill, decode}`; phase is `admit` / `prefill` / `decode` / `done`. With live seq positions, includes `plan_current_pos`. **Why:** one object for operators and the C decode loop instead of separate prefill/decode exports.

**v9 — `decode_prefill` (when admitted):** page-aligned prefill batch plan. Omitted when `prompt_tokens` is empty or request has no `block_ids` yet (waiting queue). See [phase15-native-kv.md](./phase15-native-kv.md#v9-ops--decode-prefill-plan-export-jun-2026).

**Page entry:** `page` (ordinal), `block_id` (allocator id), `token_start`, `token_end` (inclusive).

**Native stats:** `runtime/runtime/kv/native_stats.py` → `_kv_native.kv_stats()` (read-only; does not call `scheduler_tick` / `decode_step`).

**Loopback endpoint:** `GET /internal/kv-snapshot` in `app.py` — returns:

```text
kv, kv_bind, kv_scheduler, kv_physical, kv_physical_recent,
kv_scheduler_tick, kv_decode_steps, kv_native_stats, kv_forward_plans, kv_page_bind,
kv_decode_loop, kv_resume, kv_live_physical
```

**Smokes:**

```bash
./scripts/phase15_health_smoke.sh    # engine.health() key checks
./scripts/phase15_kv_native_ci.sh    # build_ext + KV pytest + health smoke
./scripts/phase15_inprocess_signoff.sh  # GPU: KV decode hook + multi-seq (needs LLAMA_CPP_LIB)
./scripts/phase15_inprocess_kv_smoke.sh # GPU: self-contained single-seq decode hook
./scripts/phase15_inprocess_multiseq_smoke.sh  # GPU: llama_parallel_slots=2
```

---

## Per-response `kv_decode_steps` (v6, still relevant)

When `llama_backend == inprocess` and decode hook enabled, completions may include:

| Route | Field |
|-------|--------|
| `/api/generate` (non-stream) | `kv_decode_steps` on `OllamaGenerateResponse` |
| `/api/chat` (non-stream) | top-level `kv_decode_steps` |
| `/v1/completions` | `kv_decode_steps` in JSON body |
| Stream / batch | final chunk / tail only |

Subprocess and wheel paths: hook inactive; field omitted or `/health` shows `active: false`.

---

## Operator env

| Env | Effect |
|-----|--------|
| `ZEROLLAMA_RUNTIME_KV_NATIVE=1` | C block pool (needs `build_ext --inplace`) |
| `ZEROLLAMA_RUNTIME_KV_PHYSICAL_STRICT=1` | In-process: fail if llama cells > PA reserve after decode |
| `ZEROLLAMA_RUNTIME_KV_DECODE_HOOK=0` | Disable decode-step counter on ctypes path |
| `ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL=1` | In-process: bump effective `-np` to 2 when defaults use 1 (live `kv_physical`; explicit `-np` wins) |
| `llama_parallel_slots` / `-np` | Slot allocator + in-process `n_seq_max` (argv wins) |

---

## Smokes / CI

```bash
cd runtime && python3 setup.py build_ext --inplace
cd ..
./scripts/phase15_kv_native_ci.sh
./scripts/phase15_health_smoke.sh

cd runtime
PYTHONPATH=. python3 -m pytest \
  tests/test_kv_*.py \
  tests/test_resolve_parallel_slots.py \
  tests/test_internal_kv_snapshot.py \
  -q
```

Regression workflow (`.github/workflows/zerollama-regression.yaml`): runtime pytest; native tests skip if `.so` missing. `check_gpu_scripts.sh` syntax-checks both `phase15_*` scripts.

---

## Known gaps (do not assume shipped)

1. **PA `block_ids` ≠ llama tensor pages** — bookkeeping + seq/slot only.
2. **Subprocess** — PA reserve does not cap llama-server internal KV.
3. **Wheel backend** — no `kv_slot`; accounting-only.
4. **Single-seq in-process** — no live `llama_pos_*` on `/health` unless `ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL=1` (bumps effective `-np` to 2; post-decode checks always run).
5. **Native decode** — C prefill/step/sample when linked ext built (`ZEROLLAMA_KV_DECODE_LOOP=1`); ctypes fallback otherwise.
6. **Tensor/page bind** — **Partial (v8–v20):** seq-position + cell/tensor verify on standard kv_cache; hybrid/iSWA/recurrent unsupported.
7. **Forward plans** — v23: prefill export matches execution via `iter_prefill_execute_chunks`; decode phase export still observability-only.

**v8 audit fixes (Jun 2026):** `SchedulerLoop` uses `pools[0].block_size` (not missing `self.block_size`); bind validation once per batch; `LlamaServerError` on overrun; `n_predict=0` still decodes prompt.

**v9 audit fixes (Jun 2026):** `logits_last=false` on **all** prefill chunks (matches `_prefill_prompt`; first sample logit from decode-loop batch, not final prefill chunk); split `test_forward_plan_omits_decode_prefill_without_block_table` from page-boundary test.

**Done (Jun 2026):** Go `:8080` proxy passthrough for runtime extension fields (`kv_decode_steps`, `vram_num_ctx`, …) on `/api/generate` and `/api/chat` (non-stream raw JSON; stream NDJSON passthrough).

---

## Suggested next work

### v16 — engine resume wiring (shipped)

7. **`current_pos_for_seq`** + **`InferenceEngine._decode_current_pos_for_request`** — live seq read before decode.
8. **`complete(current_pos=)`** — skip `_clear_sequence` when resuming; passed from generate/stream/batch.

### v16b — slot-ownership guard (shipped)

9. **`_seq_last_owner`** — `LlamaLoadedSession` map from normalised `seq_id` to last-writer owner key.  Skip clear only when `decode_pos > 0 AND incoming_owner == _seq_last_owner[sid]`.  Written back after every successful decode (stream + non-stream).  Prevents stale-KV resume by a different session on the same slot.

### v17 — L3 session resume owner (shipped)

10. **`slot_resume_owner_key`** — pinned L3 sessions use `cache:{prompt_cache_key}` instead of per-turn `request_id`.  **Why:** agent multi-turn chat was always clearing prefix KV on turn 2+ for in-process backends.

### v18 — kv_resume health + L3 gate (shipped)

11. **`/health.kv_resume`** + **`resume_owner_snapshot()`** — operator visibility into per-slot resume owners.
12. **`phase15_metal_signoff.sh` step 3** — two-turn L3 generate with same `prompt_cache_key`.

### v19 — tensor bind scaffold (shipped)

13. **`page_bind_tensor_probe`** + **`page_bind_table`** — accounting bind vs llama seq cells; `/health.kv_page_bind.tensor_probe`.
14. **`scripts/phase15_tensor_bind_probe.sh`** — build + table export smoke.

### v20a — forward plan native mirror (shipped)

15. **`native_page_table` + `page_table_native_parity`** on admitted `kv_forward_plans` — C registry vs logical pages without manual cross-check.
16. **CI** — `phase15_kv_native_ci.sh` includes tensor probe + engine resume tests.

### v20 — cell + tensor bind (shipped)

17. **`llama-kv-ext.h`** in pinned fork — cell map + K/V tensor introspection for standard kv_cache.
18. **`kv_tensor_probe.c` v20** — PA pages → llama cells → tensor verify; `/health.kv_page_bind.status=bound`.

### v20b — audit fixes (shipped)

19. **Per-stream tensor + shifted cell map + partial last page** — probe correctness after KV shift and multi-seq.
20. **Stale bind flag clear + health state machine** — `bound` not overridden by misaligned accounting.

### v21 — per-slot bind registry (shipped)

21. **`page_bind_slots()`** — `/health.kv_page_bind.slots` with per-slot `cell_pages_bound` / `tensor_pages_bound`.
22. **Post-decode bind warnings** — incomplete bind (`cell_map_gap`) logged when accounting ok but cells/tensor not bound.

### v21b — probe correctness fixes (shipped)

23. **`kv_tensor_bind_attempt` guard order** — see CHANGELOG v21b.

### v22 — stale decode_pos fix (shipped)

24. **`decode_pos = 0` after `_clear_sequence`** on non-resume multiseq path.
25. **`infer_trace`** — opt-in debug; `scripts/phase15_metal_crash_repro.sh`.

### v23 — unified prefill chunker (shipped)

26. **`iter_prefill_execute_chunks`** — `_decode_stream` + `kv_decode_prefill_plan` share one chunker.
27. **`scripts/phase15_runtime_kv_env.sh`** — sign-off enables C block pool + optional linked ext build.

### v24 — C decode loop page-bind + post-prefill probe (shipped)

27. **`kv_page_bind_validate_range`** — C-side endpoint validation before each native prefill chunk / decode step (defense in depth vs Python-only check).
28. **Post-prefill tensor probe** — `kv_decode_loop_post_prefill_probe` after `Py_END_ALLOW_THREADS` (v25: GIL-held registry write; was inside GIL-released block in v24).
29. **`phase15_metal_signoff.sh` step 4** — tensor bind scaffold smoke.

### v25 — auto-link decode loop + 131k validation (shipped)

30. **`runtime/setup.py` auto-link** — links libllama when present; `ZEROLLAMA_KV_DECODE_LOOP=0` for CI without llama.
31. **`KV_MAX_PAGES_PER_BIND=8192`** — 131072 ctx @ block_size=16 (was 4096 / 65536 tokens).
32. **`kv_decode_loop_post_prefill_probe`** — tensor probe after `Py_END_ALLOW_THREADS` (GIL-held registry write).
33. **`test_kv_decode_long_ctx.py`** — 8192-chunk prefill plan + bind boundary at 131072.

### v26 — continuous batch decode in C (shipped)

34. **`kv_decode_loop_run_batch_step`** — N single-token rows → one `llama_decode`; per-row bind validation + optional C sampling.
35. **`decode_batch_layout_multi` / `kv_continuous_batch_step_plan`** — operator export for batched decode steps.
36. **`run_batch_step`** Python facade; `batch_decode_in_c` on `/health.kv_decode_loop`.

### v27 — engine wiring for C batch decode (shipped)

37. **`complete_parallel` / `_decode_parallel_non_stream`** — sequential prefill per slot, batched autoregressive steps via `run_batch_step`.
38. **`llama_inprocess.completions_parallel`** — uses batch path when `native_batch_decode_available()`; env `ZEROLLAMA_KV_NATIVE_BATCH=0` disables.
39. **`_prepare_seq_for_decode`** — extracted resume/clear helper shared by `complete` and `complete_parallel`.

### v28 — `/health` continuous batch plan export (shipped)

40. **`kv_continuous_batch_forward_plan`** — merged decode-step preview for N running sequences; `would_batch` when `parallel_slots>1` and ≥2 decode-phase rows.
41. **`/health.kv_continuous_batch`** — operator sign-off field before GPU batch decode validation.

### v29 — streaming batch decode (shipped)

42. **`_decode_parallel_stream` / `complete_parallel_stream`** — same prefill + `run_batch_step` path as v27; yields `seq_idx`-tagged chunks.
43. **`completions_parallel_stream` / `stream_generate_batch`** — engine admits N requests and streams interleaved tokens per row.
44. **`_parallel_jobs_and_smpls` / `_finalize_parallel_jobs`** — shared setup/teardown for stream and non-stream parallel decode.

### v30 — per-row C batch sampling (shipped)

45. **`smpl_ptrs[]` in `kv_decode_loop_run_batch_step`** — one sampler per batch row with correct logit index.
46. **`run_batch_step(..., smpl_ptrs=)`** — Python facade; `_decode_parallel_stream` uses C batch sampling when all row samplers are set.

### GPU sign-off — continuous batch decode (Jun 2026)

47. **`POST /internal/generate-batch`** — loopback-only; **why:** batch APIs are engine-internal until public NDJSON contract is settled.
48. **`phase15_batch_decode_smoke.sh`** — asserts `batch_decode_in_c`, non-stream + stream batch for two prompts.
49. **`phase15_metal_signoff.sh` PASS (M4 Max)** — step 3/5 batch decode; `phase15_runtime_kv_env.sh` prefers sibling `../llama.cpp` + venv Python build.
50. **`phase15_inprocess_signoff.sh` PASS (RTX 5080, CT 1564 / cudallama, Jun 2026)** — OuteTTS 1B Q8; `kv_decode_steps=56`, `batch_decode_in_c=True`; multiseq + batch decode PASS. Build notes: host CUDA 12.3 cannot compile **sm_120** — install **cuda-nvcc-12-8** and `-DCMAKE_CUDA_ARCHITECTURES=120-real`; patch **0014** may not apply cleanly to stock sibling b9672 alone — copy kv-ext files from zerollama tree; kill stale `zerollama serve` on `:8080`/`:8081` before embed start. **`ZEROLLAMA_GPU_PROFILE=0`** on multiseq serve (rtx-5080 L1 sets `n_parallel=4` otherwise) — now baked into `phase15_inprocess_multiseq_smoke.sh`.

### v31 — llama-kv-ext pin tracking + hybrid/iSWA resolve (shipped)

50. **`llama/patches/0014-ollama-llama-kv-ext-Phase-15-tensor-page-bind-b9611.patch`** — formal b9672 patch so vendor sync preserves kv-ext.
51. **`llama_memory_kv_ext_classify`** — resolve hybrid/iSWA to attn base cache; `memory_kind_name` on probe.
52. **`scripts/phase15_llama_kv_ext_pin_check.sh`** — pin gate in `phase15_kv_native_ci.sh`.

### v32b — writable bind upstream tracker (shipped)

53. **`llama_memory_kv_ext_writable_bind_probe`** — static probe: is writable page-map API linked (`LLAMA_KV_EXT_WRITABLE_PAGE_MAP`).
54. **`page_bind_writable_probe()`** — Python/C binding; `/health.kv_page_bind.writable_bind_available` + `writable_bind_api` + `writable_bind_blocker`.
55. **Pin check upstream watch** — greps `llama.h` for page-map symbol names.

### v32 — scheduler-driven auto-batch (shipped)

56. **`runtime/kv/auto_batch.py`** — `AutoBatchCoordinator` coalesces concurrent non-stream `generate()` when `ZEROLLAMA_KV_AUTO_BATCH=1`.
57. **`/health.kv_auto_batch`** — `enabled`, `pending`, `window_ms`, `flush_count`, `batched_requests`.
58. **WHY opt-in default off:** batch window adds TTFT latency (``ZEROLLAMA_KV_AUTO_BATCH_MS``, default 5ms); streaming unchanged.

### v20 remaining

23. **Hybrid / iSWA / recurrent** memory types — **v31:** hybrid + iswa attn base resolve; pure recurrent still unsupported.
24. **Upstream-stable writable page handles** — cross-allocator KV migration without fork ext; track [phase15-llama-kv-ext-upstream.md](./phase15-llama-kv-ext-upstream.md).

---

## Verification checklist (last known good)

```bash
./scripts/phase15_kv_native_ci.sh     # KV tests + health smoke (no GPU)
./scripts/phase15_kv_decode_loop_build.sh   # optional: libllama-linked ext (skips if no libllama)
./scripts/check_gpu_scripts.sh
cd runtime && PYTHONPATH=. python3 -m pytest tests/ -q   # full runtime suite (~579 pass, 19 skip)
```

On GPU host with in-process backend:

```bash
# Linux embed:
export LLAMA_MODEL=/path/to/small.q8_0.gguf
export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
./scripts/phase15_inprocess_signoff.sh

# Mac Metal sidecar (includes batch decode step 3/5):
LLAMA_CPP_ROOT=../llama.cpp ./scripts/phase15_metal_signoff.sh
```

After multiseq sidecar is up (`kv_inprocess_n_seq_max≥2`), standalone batch smoke:

```bash
./scripts/phase15_batch_decode_smoke.sh   # POST /internal/generate-batch
```

Confirm `batch_decode_in_c=True` on `/health.kv_decode_loop`, both batch rows return content, and `kv_decode_steps` increments.
