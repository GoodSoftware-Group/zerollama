# ggml Metal → IOSurface → ANE hook (design)

**Audience:** contributors wiring hybrid Metal+ANE inference. **Implemented (lab)** in llama-common + dflash speculative path when `ZEROLLAMA_ANE_DRAFT=1`.

**Related:** [ane-draft-inprocess.md](./ane-draft-inprocess.md), [ane-hybrid-path.md](./ane-hybrid-path.md), [ane-probe.md](./ane-probe.md).

---

## Goal

After ggml Metal runs a decode step, **ANE consumes the same activation bytes** without a CPU memcpy. **Why:** ANE I/O is IOSurface-bound (maderix bridge); copying activations to CPU and back would erase ANE latency wins and add unified-memory pressure.

```text
llama-server (single PID)
  draft llama_decode (Metal)
       │
       ▼
  common_ane_draft_handoff_after_decode()
       │ llama_get_embeddings_pre_norm_ith
       ▼
  ggml_backend_dev_buffer_from_iosurface(device, surface_id, …)
       │ pack [1, ch, 1, sp] (+ optional gamma on host)
       ▼
  ane_draft_session_eval()  →  libane_bridge
```

---

## Why same-process only

Lab proved:

| Pattern | Result |
|---------|--------|
| Subprocess daemon exports `surface_id` | Parent `IOSurfaceLookup(id)` → **fail** |
| In-process `ane-inprocess-smoke` | Map + eval **ok** |
| In-process llama-server + `ZEROLLAMA_ANE_DRAFT=1` | Handoff + eval **ok** on lab port |

**Implication:** `ane-draft-daemon` remains a scheduling prototype; production wiring is **`ane_draft_session.mm` inside llama-common**, not a Go-spawned child.

---

## Public ggml API (Jun 2026)

```c
// ggml-metal.h — why: stable entry for llama-common without including Metal internals
ggml_backend_buffer_t ggml_backend_dev_buffer_from_iosurface(
    ggml_backend_dev_t device,
    uint32_t surface_id,
    size_t size,
    size_t max_tensor_size);
```

**Why map via backend device:** handoff runs in speculative code with GPU device already selected; IOSurface is owned by ANE bridge in the same process.

Implementation: `ggml_metal_buffer_map_iosurface()` in `ggml-metal-device.m` — page-aligned `newBufferWithBytesNoCopy`, retains `IOSurfaceRef` until buffer free.

---

## Speculative integration

**Files:** `common/ane_draft_hook.cpp`, `common/speculative.cpp`

**When:** `ZEROLLAMA_ANE_DRAFT=1` at dflash / draft-simple init:

1. `common_ane_draft_log_init()` → `ane_draft_session_init(ch, sp, weight, gamma)`
2. `llama_set_embeddings_pre_norm(ctx_dft, true)` — **why:** handoff reads pre-norm hidden, not post-norm logits path
3. After each draft `llama_decode`: `common_ane_draft_handoff_after_decode(ctx_dft, i_batch)`

**Why draft tokens stay Metal:** hook runs ANE conv **telemetry**; token IDs still come from `common_sampler_sample` on Metal until B7 routes logits from ANE subgraph.

---

## Weight / MIL path

| Stage | Source | Why |
|-------|--------|-----|
| Sidecar GGUF | `drafter-*.gguf` on disk | eliza `-dflash` tags ship drafter blob separately from base |
| Extract | `discover/ane_mil_weight_blob.go` | Top-left `ch×ch` slice → maderix BLOBFILE layout |
| Bundle | `ane-draft-mil-bundle` | Manifest + env for server init |
| MIL | `ane_gen_conv_mil` / `ane_gen_conv2_mil` | Lab proxy; phase3 slots in `ane-draft-mil-map` |

**Why host gamma multiply:** MIL `conv × gamma` broadcast failed compile on ANE; scaling activations in handoff pack preserves sidecar norm intent without blocking B2.

---

## Phased rollout

| Phase | Status | Notes |
|-------|--------|-------|
| 1 Lab subprocess probes | Done | crossover, handoff smokes |
| 2 Sidecar extract + daemon | Done | daemon superseded for serve |
| 3 ggml API + hook scaffold | Done | iosurface map, log init |
| **4 In-process B1–B6** | **Partial (lab)** | session, handoff, bundle, A/B — [ane-draft-inprocess.md](./ane-draft-inprocess.md) |
| 5 ANE-driven draft tokens | Not started | B7+ full subgraph |

---

## Build / sync

```bash
./scripts/sync_ane_hook_to_llama_cpp.sh   # unified ../llama.cpp @ c84b3020
./scripts/build_llama_server.sh           # copies libane_bridge.dylib
# Why manual step sometimes needed after build:
install_name_tool -change libane_bridge.dylib @loader_path/libane_bridge.dylib \
  ../llama.cpp/build/bin/libllama-common.0.0.1.dylib
```

---

## Non-goals

- IOSurface on every ggml tensor (memory + compile limits)
- ANE prefill at eliza 2048² FFN
- Cross-process draft daemon in production
- MLX models (separate runtime)

---

## Verification

```bash
./zerollama ane-inprocess-smoke --model eliza-1-2b-dflash --quick
LLAMA_SERVER_BIN=../llama.cpp/build/bin/llama-server \
  ./zerollama ane-draft-ab-smoke --model eliza-1-2b-dflash --quick --e2e
```

See also: maderix patch `tools/ane-patches/0001-bridge-iosurface-export.patch`
