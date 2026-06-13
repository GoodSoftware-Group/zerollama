# Phase 15 — native scheduler + KV

**Status:** Partial (Jun 2026) — **v0–v8 ops** shipped (see slices below). Phase 14 in-process forward **Done** (prerequisite). Default block allocator remains **Python**; C pool is opt-in. GPU sign-off: `./scripts/phase15_inprocess_signoff.sh` (Linux embed). **Mac Metal:** `./scripts/phase15_metal_signoff.sh` (uv sidecar). **Not done:** PA block ids → llama tensor KV pages; batched decode in C.

**Handoff (code map, gaps, next slices):** [handoff-phase15-native-kv.md](./handoff-phase15-native-kv.md)

See also [ROADMAP Phase 15 exit criteria](../ROADMAP.md#phase-15--exit-criteria-partial).

**Why:** Phase 14 moved **forward** in-process; Phase 15 moves **KV bookkeeping** (and eventually decode) off the interpreter so continuous batching does not fight the GIL under `ZEROLLAMA_RUNTIME_SHARED_PYTHON=1`.

---

## Slice index (v0–v8 ops)

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
| **v8 ops** | `kv_page_bind` readiness; Go loopback `/internal/kv-snapshot`; opt-in `ZEROLLAMA_RUNTIME_KV_LIVE_PHYSICAL`; GPU smokes `phase15_inprocess_signoff.sh` |

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
| `/health` `kv_bind` | `mode`, `physical_pages_bound: false` |
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

### v8 ops — readiness + loopback proxy

| Piece | Notes |
|-------|--------|
| `kv_page_bind` | `/health` + snapshot: `status=not_implemented` until llama tensor API exists |
| `kv_live_physical` | Opt-in env bumps in-process effective `-np` to 2 when YAML defaults to 1 |
| Go loopback | `GET :8080/internal/kv-snapshot` proxies Python runtime snapshot |

**Still not shipped:** PA `block_ids` → llama **tensor** KV pages; batched decode **in** C (hook counts steps only).

---

## Two KV caps (operators)

1. **PA block pool** (`kv_pools`, `kv_scheduler`) — admission and `/health`; sum of reserved blocks × `block_size`.
2. **llama KV** — subprocess or in-process context; sized by `-c` / load `num_ctx` and (in-process) per-request `kv_token_budget`.

On **subprocess**, `kv_token_budget` is not sent to llama-server: PA reserve is bookkeeping until server-side bind exists. On **in-process**, both PA assert and `kv_token_budget` can reject oversize prompt+generation.

---

## Architecture

```text
Today (v0–v8 ops):
  SchedulerLoop → PA block ids + kv_slot → llama seq/slot
  In-process → post-decode seq_pos vs PA reserve; scheduler_tick + decode_step hooks
  kv_forward_plans → logical page table export (not tensor bind)
  kv_page_bind → readiness only; PA block ids are NOT mapped to llama tensor KV pages

Target (v8 implementation — blocked on llama.cpp):
  Native batched decode in C/Rust + block table → llama KV pages (API TBD)
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
| `kv_bind` | `mode`, `physical_bind_level`, `physical_pages_bound: false` |
| `kv_physical` | Running-request PA reserve; live `llama_pos_*` when multi-seq shared ctx |
| `kv_physical_recent` | Last ≤8 **mismatch** alignment rows (`request_id`, `at`, `aligned`, …) |
| `kv_scheduler_tick` | `{value, source}` admission counter |
| `kv_native_scheduler_tick` | Legacy nullable int (prefer `kv_scheduler_tick`) |
| `kv_decode_steps` | Cumulative in-process decode count or `{active: false, reason}` |
| `kv_native_stats` | `{scheduler_tick, decode_steps}` from C when ext built; else `null` |
| `kv_forward_plans` | List of forward-plan objects (waiting + running) |
| `kv_page_bind` | v8 tensor/page bind readiness (`available`, `status`, `reason`) |
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
- Empty `pages` when the request has not been admitted to a block table yet.

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
