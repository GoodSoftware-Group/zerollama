# ANE hybrid inference path (lab)

**Audience:** operators building toward **Metal base decode + ANE draft/speculative** on Apple Silicon. Not wired into production serve.

**Related:** [ane-probe.md](./ane-probe.md), [scheduling-vram-policy.md](./scheduling-vram-policy.md).

---

## Target architecture

```text
┌─────────────────────────────────────────────────────────────┐
│  Chat request (eliza / qwen35)                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
   ggml Metal (base model)          ANE MIL kernels (draft head)
   prefill + main decode             speculative eagle3 / fused conv
            │                               ▲
            │  activations                  │ IOSurface (shared)
            └──────── newBufferWithBytesNoCopy ─┘
```

**Why IOSurface:** maderix `libane_bridge` binds ANE eval I/O to `IOSurfaceRef`. Metal can write the same memory via `newBufferWithBytesNoCopy` on `IOSurfaceGetBaseAddress` (validated in `ane-metal-handoff-smoke`).

**Why not Core ML:** compile latency and opaque scheduling; research track uses private `_ANEClient` APIs for direct dispatch.

---

## Lab tooling (today)

| Command | What it proves |
|---------|----------------|
| `ane-draft-surface-smoke --model …` | Metal→IOSurface→ANE draft conv + **surface_id** at proxy dims |
| `ane-draft-daemon-smoke --model …` | Persistent compile-once daemon; two bench rounds prove kernel reuse |
| `ane-ggml-map-smoke --model …` | ggml map parent fill on daemon IOSurface → ANE eval (Phase 3 lab POC) |
| `ane-hybrid-smoke --model …` | Draft surface handoff + draft bench at proxy dims |
| `zerollama ane-draft-inspect --model …` | GGUF metadata, sidecar presence, proxy dims (dflash inventory) |
| `zerollama ane-model-resolve` | All local GGUF tags with `embedding_length` for prefill probes |
| `zerollama ane-prefill-sweep --model …` | SEQ grid at that model's hidden size (any pulled GGUF) |
| `zerollama ane-prefill-handoff-smoke --model …` | Metal→IOSurface→ANE prefill handoff at model IC×OC |
| `zerollama ane-handoff-smoke --metal` | Metal→IOSurface→ANE at default 64×16 draft conv |

**Draft/hybrid commands** (`ane-hybrid-smoke`, `ane-draft-*`) use the **dflash / Eagle3 inventory**. **Prefill commands** (`ane-prefill-*` with `--model`) use **any local GGUF** via `ane-model-resolve`.

```bash
./scripts/ane_probe_build.sh
./zerollama ane-hybrid-smoke --model eliza-1-2b-dflash --quick
```

---

## Prefill proxy (single-layer matmul)

Lab benches compare **one** dynamic matmul at prefill geometry `[IC × SEQ] @ [IC × OC]` — not full transformer forward.

```bash
./zerollama ane-prefill-sweep --ic 512 --oc 512 --quick    # width/SEQ crossover probe
./zerollama ane-prefill-sweep --ic 1024 --oc 1024 --quick
./zerollama ane-prefill-bench --compare-metal --ic 704 --oc 704 --seq 512 --quick
```

| Command | What it proves |
|---------|----------------|
| `ane-prefill-bench --compare-metal` | ANE vs naive Metal (+ MPS when built) at one IC×OC×SEQ |
| `ane-prefill-sweep` | SEQ grid; reports `ane_wins`, `metal_wins`, `crossover_seq` |

### M4 Max lab notes (Jun 2026)

**Bench hygiene:** MPS and naive Metal legs run on the **GPU**. If production serve, another Metal workload, or screen compositing is active, MPS times inflate and **crossover shifts toward ANE**. ANE eval uses a separate engine but still competes for memory bandwidth. Reproduce on an **idle GPU** (no concurrent Metal decode, no other GPU-heavy apps):

```bash
# pause other GPU work first; do NOT stop production serve unless you choose to
./scripts/ane_crossover_report.sh
./zerollama ane-prefill-crossover --quick
```

Crossover tables below are from an **idle-GPU run** (2026-06-20). Latencies still vary ±10–20% run-to-run.

Fixed **IC=OC=256**, naive Metal compute shader vs MPS GEMM vs maderix `mil_dynamic` matmul *(idle-GPU runs; re-measure if GPU was shared)*:

| SEQ | ANE (ms) | Metal naive (ms) | MPS (ms) | Faster |
|-----|----------|------------------|----------|--------|
| 128 | 0.09 | 0.42 | 0.20 | ANE ~2.1× vs MPS |
| 512 | 0.12 | 0.56 | 0.23 | ANE ~1.9× vs MPS |
| 2048 | 0.18 | 1.16 | 0.29 | ANE ~1.6× vs MPS |

**IC=OC=2048** (eliza hidden, `--model eliza-1-2b --quick`):

| SEQ | ANE (ms) | Metal naive (ms) | MPS (ms) | Faster |
|-----|----------|------------------|----------|--------|
| 128 | 0.64 | 1.56 | 0.25 | **MPS** ~2.5× vs ANE |
| 512 | 1.08 | 4.31 | 0.59 | **MPS** ~1.8× vs ANE |
| 2048 | 2.32 | 13.78 | 1.57 | **MPS** ~1.5× vs ANE |

At eliza hidden width, **MPS wins** the single-layer matmul proxy over ANE at all tested SEQ. ANE prefill offload only makes sense at smaller IC/OC or after subgraph fusion — not raw 2048² GEMM vs tuned MPS.

### ANE vs MPS crossover (M4 Max, Jun 2026)

Single-layer `mil_dynamic` matmul vs `MPSMatrixMultiplication`. **GPU contention can invert crossover** (MPS slows more than ANE); re-run `./scripts/ane_crossover_report.sh` if the GPU was shared.

**Global grid (SEQ=512, default proxy dims):**

| IC=OC | ANE (ms) | MPS (ms) | Faster |
|-------|----------|----------|--------|
| 512 | 0.15 | 0.26 | ANE |
| 640 | 0.21 | 0.32 | ANE |
| 704 | 0.24 | 0.29 | ANE |
| 736 | 0.27 | 0.21 | MPS |
| 768 | 0.29 | 0.28 | ~tie |
| 896 | 0.32 | 0.24 | MPS |
| 1024 | 0.36 | 0.26 | MPS |
| 2048 | 1.07 | 0.53 | MPS |

**Width crossover ≈ 736** at SEQ=512 on the global grid — below that ANE wins; above that MPS wins. Eliza **2048** is ~3× past crossover.

**Per-model crossover (SEQ=512, `--model` hidden width):**

| Model | crossover IC | Notes |
|-------|----------------|-------|
| qwen3.6 | **704** | full embed 5120: MPS ~4.4× (9.8 vs 2.2 ms) |
| eliza-1-2b | **704** | 2048²: MPS ~2× (1.07 vs 0.55 ms) |
| tiny-agent (896 embed) | **896** | ANE wins through full hidden width |

qwen3.6 @ SEQ=512:

| IC=OC | ANE (ms) | MPS (ms) | Faster |
|-------|----------|----------|--------|
| 512 | 0.17 | 0.20 | ANE |
| 640 | 0.21 | 0.41 | ANE |
| 704 | 0.26 | 0.22 | MPS |
| 2048 | 1.06 | 0.53 | MPS |
| 5120 | 9.77 | 2.24 | MPS |

**By SEQ (IC=OC=512 fixed):**

| SEQ | ANE (ms) | MPS (ms) | Faster |
|-----|----------|----------|--------|
| 128 | 0.11 | 0.22 | ANE |
| 512 | 0.13 | 0.29 | ANE |
| 1024 | 0.21 | 0.25 | ANE |
| 1280 | 0.27 | 0.26 | MPS |
| 2048 | 0.33 | 0.38 | ~tie (variance) |

**SEQ crossover at 512² ≈ 1200 tokens** — long prompts flip to MPS even at moderate width.

**Why MPS wins at large width:** the job becomes a throughput-bound GEMM; MPS uses tuned GPU tiling on unified memory. ANE’s IOSurface + MIL path excels when IC/OC is small and per-op latency dominates.

No crossover for **naive** Metal shader at any tested geometry. Crossover vs ANE appears when IC/OC ≳ **704–736** (at SEQ=512, model-dependent) or SEQ ≳ **1200** (at IC/OC=512).

**Caveats:**

- Metal leg is a **naive** kernel, not ggml GEMM — ggml prefill will differ.
- TFLOPS from `gflop/eval_ms` is inflated at small sizes; use **latency** and sweep crossover.
- Full prefill still needs attention, norms, and 24+ layers on Metal unless subgraph offload lands.

**Production prefill wins today:** L3 `prompt_cache_key` + tail truncate ([gpu-profiles-l3.md](./gpu-profiles-l3.md)) — skip repeat megaprompt eval.

---

## Blockers before hot-path integration

1. **Eagle3 drafter GGUF** — eliza `-dflash` tags set `spec_type: draft-eagle3` but ship **no separate drafter blob**. Check: `zerollama ane-draft-mil-status --model …`. ANE MIL compile from real draft weights blocked until sidecar exists.

2. **ggml Metal hook** — **`ggml_backend_dev_buffer_from_iosurface`** in `ggml-metal.h`; lab: `ane-ggml-map-smoke`, `ane-draft-router-smoke`. Wire into speculative draft when Eagle3 sidecar exists.

3. **Scheduler** — `ZEROLLAMA_ANE_DRAFT=1` enables lab router (`ane-draft-router-smoke`, `discover.ANEDraftRouter`). When wired into serve: route draft steps to ANE subprocess; base stays Metal. Must not restart production serve implicitly.

4. **ggml baseline** — compare prefill proxy against tuned ggml Metal GEMM (not implemented; naive Metal shader is optimistic lower bound for ANE).

5. **Upstream GPU↔ANE samples** — `gpu_ane_share.m` / `gpu_prefill_ane_decode.m` referenced in maderix README but not in public tree yet; IOSurface handoff replicated locally in `ane-metal-handoff-smoke`.

---

## Lab status

```bash
./zerollama ane-lab-status           # binary inventory
./zerollama ane-lab-status --sweep   # + quick 256×256 prefill sweep
```

---

## Env

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANE_REPO` | `~/Sites/inference/ane` | maderix bridge checkout |
| `ZEROLLAMA_ANE_DRAFT` | `0` | Future: enable hybrid draft-on-ANE routing (lab only until ggml hook lands) |

`--model` prefill probes cap IC/OC at **2048** by default; pass **`--full-embed`** to use the model's full `embedding_length` (e.g. 5120 for 27B).

---

## Non-goals

- No automatic serve restart or manifest repair
- No Flash-MoE sidecar path (separate track)
- No ANE training exposure
