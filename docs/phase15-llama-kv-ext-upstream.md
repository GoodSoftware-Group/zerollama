# Phase 15 — `llama-kv-ext.h` upstream tracking

**Why this doc exists:** Tensor page bind (Phase 15 exit criterion #5) needs cell-index and K/V tensor introspection that upstream llama.cpp does not expose as a stable public API. Zerollama ships a **staging extension** (`llama-kv-ext.h`) as patches **0014** (page map) and **0019** (alias validate) on the vendor pin. This document tracks what is unblocked, what remains blocked, and how to refresh on pin bumps.

---

## Pin alignment

| Source | Pin / path |
|--------|------------|
| `LLAMA_CPP_VERSION` | `b9781` |
| `Makefile.sync` `FETCH_HEAD` | `b9781` |
| In-tree vendored tree | `llama/llama.cpp/` (synced from `vendor/llama-cpp-b9781` + patches) |
| Runtime sibling build | `../llama.cpp` via `LLAMA_CPP_ROOT` |
| Staging patch (page map) | `llama/patches/0014-ollama-llama-kv-ext-Phase-15-tensor-page-bind-b9611.patch` |
| Staging patch (alias validate) | `llama/patches/0019-ollama-llama-kv-ext-external-buffer-alias-v47.patch` |
| CI pin gate | `./scripts/phase/phase15_llama_kv_ext_pin_check.sh` |

**On every pin bump:** run `make -f Makefile.sync clean apply-patches sync`, then `phase15_llama_kv_ext_pin_check.sh`, rebuild libllama, rebuild `_kv_native` with `ZEROLLAMA_KV_DECODE_LOOP=1`.

---

## Staging API (`llama-kv-ext.h`)

| Symbol | Role |
|--------|------|
| `llama_memory_kv_cell_for_pos` | `(seq_id, pos)` → cell index + stream |
| `llama_memory_kv_cell_map_range` | Page-aligned cell map for bind verify |
| `llama_memory_kv_tensor_info` | K/V tensor backing pointer + size (read-only) |
| `llama_memory_kv_ext_classify` | Operator visibility: `kv_cache`, `iswa_base`, `hybrid_attn`, … |
| `llama_memory_kv_ext_writable_bind_probe` | Static probe: is writable page-map API linked (`LLAMA_KV_EXT_WRITABLE_PAGE_MAP`) |
| `llama_memory_kv_page_map` | Writable K/V spans per (page, kv_layer) — **patch 0014** |
| `llama_memory_kv_ext_external_alias_probe` | Static probe: is alias validate API linked (`LLAMA_KV_EXT_EXTERNAL_ALIAS`) — **patch 0019** |
| `llama_memory_kv_page_alias_validate` | Compare external K/V ptrs vs `page_map` geometry; no tensor mutation — **patch 0019** |

**WHY separate from `llama.h`:** upstream exposes `llama_get_memory` and seq position queries but not cell/tensor handles keyed for external PA block pools. Upstreaming requires a writable page-handle contract **and** a cross-allocator alias validate/bind pair — not just read probes.

### Writable bind tracker (v32b)

| Field | Meaning |
|-------|---------|
| `/health.kv_page_bind.writable_bind_available` | `true` when staging/upstream writable page-map is linked |
| `writable_bind_api` | Detected symbol name (e.g. `llama_memory_kv_page_map`) or `none` |
| `writable_bind_blocker` | Empty when available; else `staging_writable_page_map_not_implemented` |

### External alias tracker (v47)

**WHY:** `page_map` answers “where are llama’s bytes?”; alias validate answers “can **our** pool pointers zero-copy alias those bytes?” — required before v48 overlay bind mutates tensor backing stores.

| Field | Meaning |
|-------|---------|
| `/health.kv_page_bind.external_alias_available` | `true` when patch 0019 + `LLAMA_KV_EXT_EXTERNAL_ALIAS` linked |
| `external_alias_api` | `llama_memory_kv_page_alias_validate` or `none` |
| `external_alias_blocker` | Empty when available; else `external_alias_api_not_linked` |

| `alias_mode` (validate plan) | Meaning |
|-----------------------------|---------|
| `SAME_POINTER` (1) | Ext ptrs match `page_map`; `alias_ready=1` |
| `HOST_REBASE` (2) | Host spans match; overlay bind not implemented yet |
| `BLOCKED_DEVICE` (3) | KV on Metal/CUDA buffer — expected on Mac GPU |
| `BLOCKED_V_TRANS` (4) | non-FA V layout — flat alias invalid |
| `BLOCKED_SPAN` (5) | External spans ≠ llama spans |
| `BLOCKED_NO_PAGE` (6) | No live cells / page_map failed |
| `BLOCKED_UNSUPPORTED` (7) | Recurrent-only / unsupported memory |

CI pin check greps `llama.h` for upstream watch symbols (`llama_memory_kv_page_map`, `llama_memory_kv_page_write`, `llama_kv_cache_get_block`) and prints NOTICE when any appear.

```bash
./scripts/phase/phase15_llama_kv_ext_pin_check.sh   # includes writable API + upstream watch
P15_PIN_JSON=/tmp/phase15-kv-ext-pin-check.json ./scripts/phase/phase15_llama_kv_ext_pin_check.sh
./scripts/phase/phase15_upstream_kv_watch.sh        # scan in-tree + ollama-upstream llama.h
python3 -c "from runtime.kv.tensor_probe import writable_bind_probe; print(writable_bind_probe())"
python3 -c "from runtime.kv.tensor_probe import external_alias_probe; print(external_alias_probe())"
```

---

## Memory type resolution (v31)

`llama-memory-kv-ext.cpp` resolves attn KV through wrappers:

| `llama_memory_t` runtime type | Resolved view | `memory_kind_name` |
|------------------------------|---------------|-------------------|
| `llama_kv_cache` | direct | `kv_cache` |
| `llama_kv_cache_iswa` | `get_base()` — full-context attn | `iswa_base` |
| `llama_memory_hybrid` | `get_mem_attn()` | `hybrid_attn` |
| `llama_memory_hybrid_iswa` | `get_mem_attn()->get_base()` | `hybrid_iswa_base` |
| `llama_memory_recurrent` only | — | `unsupported` |

**WHY `get_base()` for iSWA:** PA page bind tracks full-context attn KV; the SWA cache is windowed and not the PA reserve target.

**Still unsupported:** pure recurrent-only models (no attn KV cache to bind).

---

## Upstream dependencies (stable at b9781)

These **must** remain in `include/llama.h` — pin check verifies them:

- `llama_get_memory`
- `llama_memory_can_shift`
- `llama_memory_seq_pos_min` / `llama_memory_seq_pos_max`

If a pin bump removes or renames these, refresh patch **0014** and `kv_tensor_probe.c` before merging.

---

## What remains blocked (not fixable by fork ext alone)

| Gap | Why blocked | Path forward |
|-----|-------------|--------------|
| **Writable cross-allocator bind** | PA `block_ids` → writable llama tensor spans | **Shipped (v33, patch 0014)** — `llama_memory_kv_page_map`; `physical_pages_bound` on `/health` |
| **External-buffer alias validate** | Can external pool ptrs alias `page_map` spans? | **Shipped (v47, patch 0019)** — `llama_memory_kv_page_alias_validate`; modes + blockers; no mutation |
| **External-buffer alias bind** | Overlay external storage into ggml allocators | **Open (v48+)** — `HOST_REBASE` on CPU; device strategy for Metal |
| **SWA cache pages** | Windowed SWA tensor is separate from PA full-context reserve | Dual bind registry or upstream unified page API |
| **Recurrent state** | No cell/page model — different memory layout | Out of scope for attn PA bind; separate Phase 15 slice if needed |

Read-verify bind (`status=bound` on standard + hybrid attn models) is **unblocked** with patch **0014** + linked libllama.

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
```

`/health.kv_page_bind.tensor_probe.memory_kind_name` reports resolved memory layout when a request is active.

---

## Upstreaming checklist (for llama.cpp PR)

When proposing upstream merge:

1. [ ] Rename staging symbols to `llama_memory_*` stable names (no `kv_ext` suffix)
2. [ ] Document thread-safety (GIL / decode lock) for cell map reads during `llama_decode`
3. [ ] Add hybrid/iSWA resolution in public API (or document caller must pass attn sub-memory)
4. [ ] Separate read-only probe (v20) from writable page bind (future)
5. [ ] Add upstream unit test: cell map + tensor info on tiny model fixture

---

## Related

- [phase15-native-kv.md](./phase15-native-kv.md) — operator knobs, `/health.kv_page_bind`
- [handoff-phase15-native-kv.md](./handoff-phase15-native-kv.md) — code map
- [ggml-b9509-migration.md](./ggml-b9509-migration.md) — patch series table (0014 kv-ext)
- [../runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md) — pin bump checklist
