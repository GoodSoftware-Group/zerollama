# ggml Metal → IOSurface → ANE hook (design)

**Audience:** contributors wiring hybrid Metal+ANE inference. Lab-validated; **not implemented** in ggml/llama-server yet.

**Related:** [ane-hybrid-path.md](./ane-hybrid-path.md), [ane-probe.md](./ane-probe.md), [phase17-llama-server.md](./phase17-llama-server.md).

---

## Goal

After ggml Metal runs a decode/prefill op, **ANE consumes the same activation bytes** without a CPU memcpy. ANE I/O is IOSurface-bound (maderix bridge); Metal already uses **`newBufferWithBytesNoCopy`** on host-visible memory — the hook aligns those two alloc paths.

```text
ggml graph (Metal)                ANE subprocess / bridge
  MUL_MAT / FFN  ──writes──►  IOSurface base  ──eval──►  draft conv / fused MIL
         ▲                           │
         └── MTLBuffer (StorageModeShared, no-copy)
```

---

## What lab already proves

| Lab binary | Pattern |
|------------|---------|
| `ane-metal-handoff-smoke` | Metal fill → `IOSurfaceLookup(id)` → `newBufferWithBytesNoCopy` → ANE eval |
| `ane-prefill-handoff-smoke` | Metal writes **activations** into ANE input surface; weights static on surface |

Bridge API (patched maderix):

```c
uint32_t ane_bridge_input_surface_id(ANEKernelHandle *kernel, int idx);
size_t   ane_bridge_input_surface_bytes(ANEKernelHandle *kernel, int idx);
```

---

## ggml-metal touch points (zerollama tree)

### 1. Shared buffer allocation — primary hook site

`ggml_metal_buffer_init()` in `ml/backend/ggml/ggml/src/ggml-metal/ggml-metal-device.m`:

- Host memory: `ggml_metal_host_malloc(size_aligned)`
- Metal view: `[device newBufferWithBytesNoCopy:res->all_data length:… options:MTLResourceStorageModeShared]`

**Hook option A (ANE-owned surface):** For tensors tagged `ane_eligible`, allocate IOSurface via bridge at compile time, map with `newBufferWithBytesNoCopy` on `IOSurfaceGetBaseAddress` instead of `ggml_metal_host_malloc`. Metal kernels unchanged; ANE reads same bytes after `waitUntilCompleted`.

**Hook option B (map external):** Use existing `ggml_metal_buffer_map()` / `ggml_backend_metal_device_buffer_mapped()` (`ggml-metal.cpp` ~722) to wrap a **pre-allocated IOSurface base** as a mapped shared buffer (`owned = false`). Lab: `tools/ane-metal/ggml_iosurface_map.h` mirrors the page-align + `newBufferWithBytesNoCopy` path.

**Same-process requirement:** ANE bridge IOSurface IDs are **not** `IOSurfaceLookup`-able from a separate process. Production ggml hook and lab daemon `map_fill` run **in-process** with the compiled kernel (subprocess daemon today; in-process CGO optional later).

### 2. Buffer type selection

`ggml_backend_metal_device_get_buffer_type()` picks **shared vs private** based on `use_shared_buffers` (unified memory on Apple Silicon). ANE handoff requires **shared** — already default on M-series.

### 3. Tensor → MTLBuffer at encode time

`ggml_metal_encoder_set_buffer()` uses `ggml_metal_buffer_id { metal, offs }`. ANE hook must preserve **stable base + offset** for the surface-backed allocation for the tensor lifetime of one draft step.

### 4. Synchronization

Before `ane_bridge_eval()`:

1. Metal command buffer **`waitUntilCompleted`**
2. `IOSurfaceLock` / `Unlock` if CPU touches the same surface
3. ANE eval (subprocess or in-process bridge when wired)

Lab order validated in `tools/ane-prefill/prefill_handoff_smoke.m`.

---

## Scheduler integration (llama-server)

**Env:** `ZEROLLAMA_ANE_DRAFT=1` (default off; lab only today).

**Candidate hot paths** (do **not** enable for full 2048² FFN — MPS wins above ~720 IC):

| Subgraph | IC×OC proxy | ANE viable? |
|----------|-------------|-------------|
| Eagle3 / dflash **draft head** | ~256×16 conv | Yes (lab) |
| Full FFN matmul | 2048²+ | No (MPS faster) |
| Vision front-end | model-specific | TBD |

**Draft wiring** (when Eagle3 sidecar exists):

1. `common_speculative_draft()` / server speculative path generates draft tokens on GPU today.
2. With `ZEROLLAMA_ANE_DRAFT=1`, route **draft conv** to ANE MIL compiled from sidecar weights.
3. Base model stays on ggml Metal; IOSurface handoff on draft input only.

References: `llama/llama.cpp/common/speculative.cpp`, `common_speculative_draft()`, server `--spec-draft-model`.

---

## Phased rollout

### Phase 1 — Lab (done)

- Subprocess probes, crossover sweeps, IOSurface handoff smokes
- `ane-prefill-crossover` for width/SEQ decisions

### Phase 2 — Sidecar ANE draft (in progress)

- `ane-draft-surface-smoke --model …` — Metal fill → IOSurface → ANE draft conv; JSON includes `handoff.surface_id` for ggml parent wiring
- Eagle3 drafter GGUF → MIL weight extract: `ane-draft-mil-map` lists tensor slots (`fc.weight`, `blk.0.*`); blocked until sidecar on disk
- Subprocess eval protocol: compile in child, export `surface_id`, parent fills via Metal, child `ane_bridge_eval`
- **`ane-draft-daemon`** — persistent child: compile once, JSON stdin/stdout for repeated eval/bench (`zerollama ane-draft-daemon-smoke`)
- **`ane-draft-router-smoke`** — multi-step scheduler prototype (`ZEROLLAMA_ANE_DRAFT=1`); `discover.ANEDraftRouter` is the serve integration hook

### Phase 3 — ggml buffer backend (API landed)

- **`ggml_metal_buffer_map_iosurface()`** — page-aligned `newBufferWithBytesNoCopy` on IOSurface base (mirrors host-ptr map); retains `IOSurfaceRef` until buffer free
- **`ggml_backend_dev_buffer_from_iosurface()`** — public backend entry in `ggml-metal.h`
- Lab: **`ane-ggml-map-smoke`** + **`ane-draft-router-smoke`**; status: `zerollama ane-ggml-hook-status`
- **Same-process only** — ANE bridge IOSurface IDs are not visible cross-process; llama-server / ggml parent must map in-process
- Guard: only map draft tensors under `DraftANEProxyDims()` (≤512 ch × spatial 16); log when `ZEROLLAMA_ANE_DRAFT=1`
- **Next:** call `ggml_backend_dev_buffer_from_iosurface` from `common_speculative_impl_draft_eagle3` when sidecar weights exist (constructor logs when `ZEROLLAMA_ANE_DRAFT=1`)

### Phase 4 — In-process (optional)

- CGO link `libane_bridge` behind `-tags zerollama_ane` — higher risk on macOS updates

---

## Non-goals

- No IOSurface export for every ggml tensor (memory overhead, ANE compile limits)
- No ANE prefill for eliza 2048 FFN (crossover ~720; see `ane-prefill-crossover`)
- No production serve restart from doctor or lab commands
- MLX models — separate runtime; not this hook

---

## Verification commands

```bash
./scripts/ane_probe_build.sh
./zerollama ane-prefill-crossover --quick
./zerollama ane-prefill-crossover --model tiny-agent --quick
./zerollama ane-prefill-handoff-smoke --model eliza-1-2b-dflash --tokens 128 --quick
./zerollama ane-handoff-smoke --metal --quick
./zerollama ane-ggml-map-smoke --model eliza-1-2b-dflash --quick
ZEROLLAMA_ANE_DRAFT=1 ./zerollama ane-draft-router-smoke --model eliza-1-2b --quick
./zerollama ane-draft-mil-extract --model eliza-1-2b-dflash --out /tmp/ane-draft-weight.bin
# then: ane-draft-daemon --channels 256 --spatial 16 --weight-file /tmp/ane-draft-weight.bin --bench
```

---

## See also

- maderix/ane bridge patch: `tools/ane-patches/0001-bridge-iosurface-export.patch`
- Crossover data: [ane-hybrid-path.md](./ane-hybrid-path.md#ane-vs-mps-crossover-m4-max-jun-2026)
