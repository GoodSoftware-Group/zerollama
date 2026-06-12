# Handoff: Phase 15 native scheduler + KV

**Audience:** Engineers continuing runtime KV / scheduler work without this thread.

**Status (Jun 2026):** **Partial (v0–v8 ops)** — Phase 14 prerequisite **Done**. PA block pool (Python + opt-in C), scheduler accounting, `kv_slot`→llama seq/slot, logical + seq-position bind, admission `scheduler_tick`, in-process `decode_step` hook, forward-plan export + `/internal/kv-snapshot`, Go loopback proxy, `kv_page_bind` readiness + opt-in live physical health. **Not done:** tensor KV page map; batched decode in C.

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
| v8 ops | `kv_page_bind` readiness on `/health` + snapshot; Go `:8080` loopback proxy; opt-in `ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL` |

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
| Native pool + counters | `runtime/native/kv_block_pool.c` → `runtime.kv._kv_native` (`scheduler_tick`, `decode_step`, `kv_stats`) |
| Logical bind | `runtime/runtime/kv/bind.py` |
| Seq-position | `runtime/runtime/kv/physical.py` |
| Tick counter | `runtime/runtime/kv/native_tick.py` |
| Decode hook | `runtime/runtime/kv/native_decode.py` |
| Forward plan | `runtime/runtime/kv/forward_plan.py` |
| Page bind readiness | `runtime/runtime/kv/page_bind.py` |
| Live physical opt-in | `runtime/runtime/kv/live_physical.py` |
| Scheduler admit | `runtime/runtime/scheduler/loop.py` |
| Engine / health / snapshot | `runtime/runtime/engine.py` (`kv_snapshot`, `_kv_forward_plans_health`) |
| HTTP | `runtime/runtime/server/app.py` (`GET /internal/kv-snapshot`) |
| Go loopback proxy | `server/runtime_kv_snapshot.go`, `internal/runtimeclient/kv_snapshot.go` (`GET :8080/internal/kv-snapshot`) |
| In-process forward | `runtime/runtime/worker/libllama_ctypes.py`, `llama_inprocess.py` |
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
| `kv_page_bind` | v8 tensor bind status (`not_implemented` until llama API exists) |
| `kv_live_physical` | Opt-in in-process multi-seq for live `kv_physical` (`applied`, `effective`, `reason`) |

---

## v7 — forward plan + snapshot

**Module:** `runtime/runtime/kv/forward_plan.py` — `kv_forward_plan()`, `kv_forward_plans_for_requests()`.

**Included on `/health` and in `engine.kv_snapshot()`** for `scheduler.waiting` + `scheduler.running` only (not completed).

**Per-plan fields:** `request_id`, `state`, `kv_slot`, `block_size`, `num_ctx`, `pages[]`, `pa_tokens_reserved`, `tokens_to_reserve`.

**Page entry:** `page` (ordinal), `block_id` (allocator id), `token_start`, `token_end` (inclusive).

**Native stats:** `runtime/runtime/kv/native_stats.py` → `_kv_native.kv_stats()` (read-only; does not call `scheduler_tick` / `decode_step`).

**Loopback endpoint:** `GET /internal/kv-snapshot` in `app.py` — returns:

```text
kv, kv_bind, kv_scheduler, kv_physical, kv_physical_recent,
kv_scheduler_tick, kv_decode_steps, kv_native_stats, kv_forward_plans, kv_page_bind, kv_live_physical
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
5. **Native decode** — `decode_step` counts `llama_decode` calls; generation still in libllama, not C batching.
6. **Tensor/page bind** — `kv_page_bind.status=not_implemented`; logical `kv_forward_plans` only.

**Done (Jun 2026):** Go `:8080` proxy passthrough for runtime extension fields (`kv_decode_steps`, `vram_num_ctx`, …) on `/api/generate` and `/api/chat` (non-stream raw JSON; stream NDJSON passthrough).

---

## Suggested next work (blocked on llama.cpp)

1. **Tensor/page bind** — implement when llama.cpp exposes stable paged KV handles; flip `kv_page_bind.available`.
2. **Native decode batch** in C/Rust wired to `scheduler_tick` + `kv_forward_plans` page tables.
3. **Subprocess** — extend llama-server slot API if upstream adds KV page export.

---

## Verification checklist (last known good)

```bash
./scripts/phase15_kv_native_ci.sh     # ~38 KV tests + health smoke
./scripts/check_gpu_scripts.sh
cd runtime && PYTHONPATH=. python3 -m pytest tests/ -q   # full runtime suite
```

On GPU host with in-process backend: load a model, admit a request, confirm `kv_forward_plans` non-empty and `kv_decode_steps` increments on completion.
