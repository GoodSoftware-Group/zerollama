# Phase 15 — `llama-kv-ext.h` upstream tracking

**Why this doc exists:** Tensor page bind (Phase 15 exit criterion #5) needs cell-index, K/V tensor introspection, cross-allocator alias validation, and (v48/v49) zero-copy donor-buffer allocation that upstream llama.cpp does not expose as a stable public API. Zerollama ships a **staging extension** (`llama-kv-ext.h`) as an additive patch series on the vendor pin. This document tracks what is unblocked, what remains blocked, and how to refresh on pin bumps.

---

## Pin alignment

| Source | Pin / path |
|--------|------------|
| `LLAMA_CPP_VERSION` | `86d86ed4` |
| `Makefile.sync` `FETCH_HEAD` | `86d86ed4` |
| In-tree vendored tree | `llama/llama.cpp/` (synced from `vendor/llama-cpp-86d86ed4` + patches) |
| Runtime sibling build | `../llama.cpp` via `LLAMA_CPP_ROOT` |
| Staging patch (base API: classify/cell-map/tensor-info/writable page map) | `llama/patches/0001-chore-stage-ollama-compat-kv-ext-for-vendor.patch` |
| Staging patch (tensor page bind wiring, b9611-era) | `llama/patches/0019-ollama-llama-kv-ext-Phase-15-tensor-page-bind-b9611.patch` |
| Staging patch (external-buffer alias validate, v47) | folded into `0001` (`LLAMA_KV_EXT_EXTERNAL_ALIAS`) |
| Staging patch (CPU-only donor-buffer overlay bind, v48) | `llama/patches/0021-ollama-llama-kv-ext-donor-buffer-v48.patch` |
| Staging patch (Metal device-buft donor consume, v49) | `llama/patches/0022-ollama-llama-kv-ext-donor-buffer-metal-v49.patch` |
| CI pin gate | `./scripts/phase/phase15_llama_kv_ext_pin_check.sh` |

**WHY the patch numbering looks non-sequential:** `0021`/`0022` collide by number with two unrelated patches (`0021-fix-cell_index_for-stage-Ollama-compat-in-CMake.patch`, `0022-ollama-kv-seq-copy-endpoint-Radix-L3-v1.patch`) that were added later in the series for other Phase-15-adjacent work (Radix L3 KV seq-copy). Both numbering collisions are pre-existing in the tree; `git apply`/`make -f Makefile.sync apply-patches` applies patches by filename in the Makefile's explicit list, not by numeric prefix alone, so this does not cause ordering ambiguity in practice — but be aware `ls llama/patches/` will show two files under each of `0021`/`0022`.

**On every pin bump:** run `make -f Makefile.sync clean apply-patches sync`, then `phase15_llama_kv_ext_pin_check.sh`, rebuild libllama, rebuild `_kv_native` with `ZEROLLAMA_KV_DECODE_LOOP=1`.

---

## Staging API (`llama-kv-ext.h`)

| Symbol | Role | Shipped |
|--------|------|---------|
| `llama_memory_kv_cell_for_pos` | `(seq_id, pos)` → cell index + stream | v20 |
| `llama_memory_kv_cell_map_range` | Page-aligned cell map for bind verify | v20 |
| `llama_memory_kv_tensor_info` | K/V tensor backing pointer + size (read-only) | v20 |
| `llama_memory_kv_n_layers` | KV layer count (per-layer scaling / MLA / hybrid offload safety) | v34 |
| `llama_memory_kv_cache_layout` | `kv_size`, `n_stream`, `v_transposed` cache constants | v35 |
| `llama_memory_kv_ext_classify` | Operator visibility: `kv_cache`, `iswa_base`, `hybrid_attn`, … | v20/v31 |
| `llama_memory_kv_ext_writable_bind_probe` | Static probe: is writable page-map API linked (`LLAMA_KV_EXT_WRITABLE_PAGE_MAP`) | v20 |
| `llama_memory_kv_page_map` | Writable K/V spans per (page, kv_layer) | v33/v34 |
| `llama_memory_kv_ext_external_alias_probe` | Static probe: is alias validate API linked (`LLAMA_KV_EXT_EXTERNAL_ALIAS`) | v47 |
| `llama_memory_kv_page_alias_validate` | Compare external K/V ptrs vs `page_map` geometry; no tensor mutation | v47 |
| `llama_kv_ext_register_donor_buffer` | Register an external host buffer as a KV-cache allocation donor | v48 |
| `llama_kv_ext_unregister_donor_buffer` | Unregister a donor (idempotent) | v48 |
| `llama_kv_ext_donor_buffer_status` | Query whether a donor was actually consumed + bytes used | v48 |
| `llama_kv_ext_donor_try_consume` (C++-internal, not C ABI) | CPU-host buft consume via `ggml_backend_cpu_buffer_from_ptr` | v48 |
| `llama_kv_ext_donor_try_consume_dev` (C++-internal, not C ABI) | Device buft consume for `buffer_from_host_ptr`-capable devices (Metal) via `ggml_backend_dev_buffer_from_host_ptr` | v49 |

**WHY separate from `llama.h`:** upstream exposes `llama_get_memory` and seq position queries but not cell/tensor handles keyed for external PA block pools, nor a cross-allocator alias validate/bind pair, nor a zero-copy donor-buffer registration hook. Upstreaming requires a writable page-handle contract, an alias validate/bind pair, **and** a donor-registration contract — not just read probes.

### Writable bind tracker (v32b)

| Field | Meaning |
|-------|---------|
| `/health.kv_page_bind.writable_bind_available` | `true` when staging/upstream writable page-map is linked |
| `writable_bind_api` | Detected symbol name (e.g. `llama_memory_kv_page_map`) or `none` |
| `writable_bind_blocker` | Empty when available; else `staging_writable_page_map_not_implemented` |

### External alias tracker (v47)

**WHY:** `page_map` answers "where are llama's bytes?"; alias validate answers "can **our** pool pointers zero-copy alias those bytes?" — required before v48/v49 overlay bind mutates tensor backing stores.

| Field | Meaning |
|-------|---------|
| `/health.kv_page_bind.external_alias_available` | `true` when `LLAMA_KV_EXT_EXTERNAL_ALIAS` linked |
| `external_alias_api` | `llama_memory_kv_page_alias_validate` or `none` |
| `external_alias_blocker` | Empty when available; else `external_alias_api_not_linked` |

| `alias_mode` (validate plan) | Meaning |
|-----------------------------|---------|
| `SAME_POINTER` (1) | Ext ptrs match `page_map`; `alias_ready=1` |
| `HOST_REBASE` (2) | Host spans match; per-page rebase not implemented (architecturally invalid — see v48 below; superseded by donor-buffer overlay bind) |
| `BLOCKED_DEVICE` (3) | KV on Metal/CUDA buffer — expected pre-v49; **as of v49, Metal KV can be donor-bound**, but `alias_validate`'s `BLOCKED_DEVICE` classification itself is unchanged (it still reflects that per-page rebase is blocked on device buffers, which remains true — v49 does whole-tensor donor consume, not per-page alias) |
| `BLOCKED_V_TRANS` (4) | non-FA V layout — flat alias invalid |
| `BLOCKED_SPAN` (5) | External spans ≠ llama spans |
| `BLOCKED_NO_PAGE` (6) | No live cells / page_map failed |
| `BLOCKED_UNSUPPORTED` (7) | Recurrent-only / unsupported memory |

CI pin check greps `llama.h` for upstream watch symbols (`llama_memory_kv_page_map`, `llama_memory_kv_page_write`, `llama_kv_cache_get_block`) and prints NOTICE when any appear.

```bash
./scripts/phase/phase15_llama_kv_ext_pin_check.sh   # includes writable API + donor API + upstream watch
P15_PIN_JSON=/tmp/phase15-kv-ext-pin-check.json ./scripts/phase/phase15_llama_kv_ext_pin_check.sh
./scripts/phase/phase15_upstream_kv_watch.sh        # scan in-tree + ollama-upstream llama.h
python3 -c "from runtime.kv.tensor_probe import writable_bind_probe; print(writable_bind_probe())"
python3 -c "from runtime.kv.tensor_probe import external_alias_probe; print(external_alias_probe())"
```

---

## Memory type resolution (v31)

`llama-memory-kv-ext.cpp` resolves attn KV through wrappers:

| `llama_memory_t` runtime type | Resolved view | `memory_kind_name` |
|-------------------------------|---------------|-------------------|
| `llama_kv_cache` | direct | `kv_cache` |
| `llama_kv_cache_iswa` | `get_base()` — full-context attn | `iswa_base` |
| `llama_memory_hybrid` | `get_mem_attn()` | `hybrid_attn` |
| `llama_memory_hybrid_iswa` | `get_mem_attn()->get_base()` | `hybrid_iswa_base` |
| `llama_memory_recurrent` only | — | `unsupported` |

**WHY `get_base()` for iSWA:** PA page bind tracks full-context attn KV; the SWA cache is windowed and not the PA reserve target.

**Still unsupported:** pure recurrent-only models (no attn KV cache to bind).

---

## Upstream dependencies (stable at 8f114a9b)

These **must** remain in `include/llama.h` — pin check verifies them:

- `llama_get_memory`
- `llama_memory_can_shift`
- `llama_memory_seq_pos_min` / `llama_memory_seq_pos_max`

If a pin bump removes or renames these, refresh patch **0001**/**0019** and `kv_tensor_probe.c` before merging.

**v48/v49 also depend on these upstream ggml primitives** (not `llama-kv-ext.h`-specific, but load-bearing for the donor-buffer design):

- `ggml_backend_cpu_buffer_from_ptr` (v48) — wraps a host pointer as a CPU `ggml_backend_buffer_t` without taking ownership.
- `ggml_backend_dev_buffer_from_host_ptr` (v49) — device-capability-gated equivalent; implemented for Metal (`newBufferWithBytesNoCopy` + `MTLResourceStorageModeShared`), not implemented for CUDA (`buffer_from_host_ptr = NULL` in the CUDA backend iface). Already used in production by `llama-model.cpp`'s mmap weight-loading path — v49 is the first Phase 15 KV-cache use of it.
- `ggml_backend_alloc_ctx_tensors_from_buft_size` / `ggml_get_max_tensor_size` — size-query helpers the allocation loop uses to decide whether a registered donor is large enough, and to size Metal's internal buffer-window split correctly.

Both primitives are checked by the pin-check script against `ml/backend/ggml/ggml/include/ggml-backend.h` (the Go ml backend's staged ggml headers) — if a pin bump removes either, the donor-buffer hooks (v48 host path, v49 device path) must be re-verified before merging.

---

## What remains blocked (not fixable by fork ext alone)

| Gap | Why blocked | Path forward |
|-----|-------------|--------------|
| **Writable cross-allocator bind** | PA `block_ids` → writable llama tensor spans | **Shipped (v33/v34)** — `llama_memory_kv_page_map`; `physical_pages_bound` on `/health` |
| **External-buffer alias validate** | Can external pool ptrs alias `page_map` spans? | **Shipped (v47)** — `llama_memory_kv_page_alias_validate`; modes + blockers; no mutation |
| **External-buffer alias bind (CPU)** | Overlay external storage into ggml's CPU allocator | **Shipped (v48, patch 0021)** — donor-buffer registry; `ggml_backend_cpu_buffer_from_ptr` at KV-cache construction time; opt-in `ZEROLLAMA_KV_OVERLAY_BIND=1` |
| **External-buffer alias bind (Metal)** | Overlay external storage into ggml's Metal allocator | **Shipped (v49, patch 0022)** — same donor registry; `ggml_backend_dev_buffer_from_host_ptr` for `buffer_from_host_ptr`-capable device bufts; verified with real GPU-offloaded decode (byte-identical output vs. non-donor allocation) |
| **External-buffer alias bind (CUDA)** | Overlay external storage into ggml's CUDA allocator | **Open** — CUDA does not implement `buffer_from_host_ptr` (discrete VRAM, no unified-memory equivalent); would need a fundamentally different mechanism (e.g. wrapping an externally-owned CUDA allocation via `cudaHostRegister`/similar), out of scope for the current donor-buffer design |
| **SWA cache pages** | Windowed SWA tensor is separate from PA full-context reserve | Dual bind registry or upstream unified page API |
| **Recurrent state** | No cell/page model — different memory layout | Out of scope for attn PA bind; separate Phase 15 slice if needed |

Read-verify bind (`status=bound` on standard + hybrid attn models) is **unblocked** with patch **0001**/**0019** + linked libllama. Zero-copy write bind (donor-buffer overlay) is **unblocked for CPU and Metal** with patches **0021**/**0022** + `ZEROLLAMA_KV_OVERLAY_BIND=1`.

---

## Operator workflow

```bash
# 1) Pin + patch present (CI gate)
./scripts/phase/phase15_llama_kv_ext_pin_check.sh

# 2) Rebuild libllama from patched tree (in-tree or sibling)
cd llama/llama.cpp && cmake -B build -DBUILD_SHARED_LIBS=ON && cmake --build build -j
# or: LLAMA_CPP_ROOT=../llama.cpp ./scripts/build/build_llama_server.sh

# 3) Link native ext
source ./scripts/phase/phase15_runtime_kv_env.sh
phase15_runtime_kv_ext_build

# 4) Probe after decode
./scripts/phase/phase15_tensor_bind_probe.sh
curl -s :8081/health | jq '.kv_page_bind'

# 5) Donor-buffer overlay bind sign-off (CPU; add n_gpu_layers>0 model for Metal path)
export ZEROLLAMA_KV_OVERLAY_BIND=1
./scripts/phase/phase15_overlay_bind_cpu_smoke.sh
curl -s :8081/health | jq '.kv_page_bind | {overlay_bind_enabled, overlay_bind_bound, overlay_bind_bytes}'
```

`/health.kv_page_bind.tensor_probe.memory_kind_name` reports resolved memory layout when a request is active.

---

## Upstreaming checklist (for llama.cpp PR)

When proposing upstream merge:

1. [ ] Rename staging symbols to `llama_memory_*` stable names (no `kv_ext` suffix)
2. [ ] Document thread-safety (GIL / decode lock) for cell map reads during `llama_decode`
3. [ ] Add hybrid/iSWA resolution in public API (or document caller must pass attn sub-memory)
4. [ ] Separate read-only probe (v20) from writable page bind (v33/v34)
5. [ ] Add upstream unit test: cell map + tensor info on tiny model fixture
6. [ ] Propose a public donor-buffer / external-buffer registration API for `llama_kv_cache` construction (v48/v49's process-level registry is an additive staging hook, not something upstream would want verbatim — a real proposal would likely thread this through `llama_context_params` instead)
7. [ ] Document the device-capability-gated (`buffer_from_host_ptr`) vs. host-only consume distinction, and that CUDA is out of scope until/unless a discrete-VRAM zero-copy mechanism exists upstream

---

## Related

- [phase15-native-kv.md](./phase15-native-kv.md) — operator knobs, `/health.kv_page_bind`, v48/v49 design detail
- [handoff-phase15-native-kv.md](./handoff-phase15-native-kv.md) — code map
- [ggml-b9509-migration.md](./ggml-b9509-migration.md) — patch series table
- [../runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md) — pin bump checklist
