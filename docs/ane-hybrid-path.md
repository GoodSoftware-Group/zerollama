# ANE hybrid inference path (lab)

**Audience:** operators building toward **Metal base decode + ANE draft/speculative** on Apple Silicon. In-process dflash hook is **lab-only** until B7; see [ane-draft-inprocess.md](./ane-draft-inprocess.md).

**Related:** [ane-probe.md](./ane-probe.md), [ane-ggml-iosurface-hook.md](./ane-ggml-iosurface-hook.md), [scheduling-vram-policy.md](./scheduling-vram-policy.md).

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
   prefill + main decode             speculative dflash conv proxy
            │                               ▲
            │  activations (same PID)       │ IOSurface (in-process)
            └──────── ggml map + handoff ───┘
```

**Why IOSurface:** maderix `libane_bridge` binds ANE eval I/O to `IOSurfaceRef`. Metal can write the same memory via `newBufferWithBytesNoCopy` on `IOSurfaceGetBaseAddress` (validated in `ane-metal-handoff-smoke`).

**Why same-process (Jun 2026 decision):** subprocess `IOSurfaceLookup(parent_surface_id)` **always fails** — production path compiles ANE inside llama-server (`ane_draft_session.mm`), not a sidecar daemon.

**Why not Core ML for hot path:** compile latency and opaque scheduling; research track uses private `_ANEClient` APIs for direct dispatch and crossover measurement.

---

## Lab tooling (today)

| Command | What it proves |
|---------|----------------|
| `ane-draft-surface-smoke --model …` | Metal→IOSurface→ANE draft conv + **surface_id** at proxy dims |
| `ane-draft-daemon-smoke --model …` | Persistent compile-once daemon; kernel reuse (**subprocess — superseded for serve**) |
| `ane-ggml-map-smoke --model …` | ggml map parent fill on IOSurface → ANE eval |
| `ane-inprocess-smoke --model …` | **B1** same-PID compile + map + eval; export env for server |
| `ane-draft-mil-bundle --model …` | **B3/B6** sidecar extract → cached BLOBFILE + manifest v2 |
| `ane-draft-mil-map --model …` | Sidecar tensor → future MIL slot plan (phase3 subgraph) |
| `ane-draft-ab-smoke --model … --e2e` | **B4** micro ANE vs Metal dflash on lab port **11435** |
| `zerollama ane-draft-inspect --model …` | GGUF metadata, sidecar presence, proxy dims |
| `zerollama ane-prefill-sweep --model …` | SEQ grid at model hidden size (any pulled GGUF) |
| `zerollama ane-prefill-handoff-smoke --model …` | Metal→IOSurface→ANE prefill handoff at model IC×OC |
| `ane-prefill-ffn-slice-smoke [--swiglu]` | In-process expert matmul / fused SwiGLU + map + CPU golden |
| `ane-prefill-ffn-policy-smoke` | Fail-closed `ZEROLLAMA_ANE_FFN_*` policy (no Metal replace) |
| `ane-prefill-ffn-swiglu-force-smoke` | Fused SwiGLU host replace vs CPU golden |
| `ane-prefill-ffn-fuse-unit-smoke` | Topology match up→gate→GLU→down (no ANE) |

**FFN intercept policy (lab):** [ane-prefill-ffn-hook.md](./ane-prefill-ffn-hook.md) — `MUL_MAT` / shexp only; refuses `:11434` / `:8081`.

**Jul 2026 expert-geometry findings** (baseline MIL tax, handoff fill vs map, denormal zeros, fused SwiGLU) — see [Findings](#findings-jul-2026--expert-ffn--moe-geometry) below.

**Draft/hybrid commands** use the **dflash inventory**. **Prefill commands** use **any local GGUF** via `ane-model-resolve`.

```bash
./scripts/ane_probe_build.sh
./scripts/build_llama_server.sh
./zerollama ane-draft-mil-bundle --model eliza-1-2b-dflash
./zerollama ane-draft-ab-smoke --model eliza-1-2b-dflash --quick --e2e
```

Full operator guide: [ane-draft-inprocess.md](./ane-draft-inprocess.md).

---

## Prefill proxy (single-layer matmul)

Lab benches compare **one** dynamic matmul at prefill geometry `[IC × SEQ] @ [IC × OC]` — not full transformer forward.

**Why prefill ANE rarely wins on eliza width:** at IC=OC=2048, MPS beats ANE at all tested SEQ; crossover ~720 at SEQ=512. **Production prefill wins today:** L3 `prompt_cache_key` + tail truncate — skip repeat megaprompt eval.

See crossover tables in prior sections of this doc (M4 Max Jun 2026 idle-GPU runs).

---

## Findings (Jul 2026) — expert FFN / MoE geometry

Lab on M4 Max. Tools: `ane-prefill-bench --variant`, `ane-prefill-handoff-smoke --steady`, `ane-prefill-ffn-slice-smoke [--swiglu]`. Policy: [ane-prefill-ffn-hook.md](./ane-prefill-ffn-hook.md).

### 1. Dense 2048² loses; expert 2048×512 can win

| Geometry | Result |
|----------|--------|
| Square embed (eliza / qwen IC=OC=2048) | MPS wins — do not chase ANE prefill here |
| Expert-up `2048→512` (qwen3.6-mtp FFN) | Optimized ANE beats baseline and (on idle GPU) can beat MPS |
| Square 512 / down-proj `512→2048` | ANE often wins even on older baseline MIL |

`qwen3.6-mtp`: embd **2048**, expert FFN **512**, 8-of-256 routed. That width is the interesting ANE candidate — not full-model prefill.

### 2. Baseline MIL was leaving ~2–3× on the table

Expert-up `2048×512` within-ANE (eval_ms):

| SEQ | baseline (fp32 cast + packed W) | **fp16-blob / fp16-conv** |
|----:|--------------------------------:|--------------------------:|
| 128 | ~0.26 | **~0.09** |
| 512 | ~0.38 | **~0.19** |
| 2048 | ~0.89 | **~0.39** |

What mattered: drop fp32↔fp16 cast tax, put weights in **BLOBFILE** (acts-only upload), prefer channel-first / 1×1 **conv** over a bare reshape sandwich. Ship **`fp16-conv`** or **`fp16-blob`**, not baseline.

### 3. Metal fill handoff eats the win; ggml map does not

| Mode | total @ 2048×512 SEQ=512 |
|------|-------------------------:|
| Per-iter Metal fill + ANE eval | ~0.53 ms (fill ~0.30) |
| **`--steady`** (map once, eval only) | **~0.21 ms** (map ~0.04) |

Implication: zero-copy IOSurface from ggml (draft map pattern) is required for net-positive expert offload. The fill kernel in handoff smoke is **not** the production path.

### 4. “Fast” all-zero outputs = fp16 denormal flush

ANE accumulates in fp16 and flushes denormal mul products. Synth fills like `W=1e-3 × X=1e-2 → 1e-5` look like a working kernel with empty output. **Gate benches on `ane_max_abs > 0`** and keep product scale ≳0.05. Same class of bug as draft P9 blk.1 SwiGLU underflow.

### 5. Fused SwiGLU expert FFN is one ANE eval

`ane_prefill_session_create_swiglu`: gate+up 1×1 conv → `silu(g)*up` → down, three BLOBFILE weights, CPU golden cosine ≥ **0.999999** at `2048→512→2048`:

| SEQ | eval_ms |
|----:|--------:|
| 128 | ~0.13 |
| 512 | ~0.35 |

Better than three separate matmuls + host SiLU for the expert slice. MIL gotcha: `buildInfo` braces need `{{` inside `stringWithFormat`; `appendString` with `{{` emits literal double braces → `InvalidMILProgram`.

### 6. Architecture constraints for any ggml hook

| Do | Don't |
|----|-------|
| Thin `ane_prefill_session` + `ZEROLLAMA_ANE_FFN_*` | Extend `ane_draft_session` (dflash mega-kernel) |
| Intercept shared-expert / dense **`MUL_MAT`** first | Assume routed experts are plain `MUL_MAT` — they are **`MUL_MAT_ID`** |
| Lab ports only (refuse `:11434` / `:8081`) | Flash-MoE + ANE expert offload until up-proj + dyn weight path is proven |

**Bottom line:** ANE can help **narrow MoE/expert FFN** with fp16-blob/conv + zero-copy map + fused SwiGLU. It does **not** help dense 2048² prefill; production prefill remains cache/truncate.

---

## Blockers before hot-path integration

| Blocker | Why it matters | Status (Jun 2026) |
|---------|----------------|-------------------|
| Same-PID IOSurface | Cross-process handoff impossible | **Resolved** — B1 in-process session |
| ggml map API | Pack activations without CPU copy | **Resolved** — `ggml_backend_dev_buffer_from_iosurface` |
| Sidecar weights | MIL needs real BLOBFILE blobs | **Partial** — conv proxy + gamma; not full dflash graph |
| Draft token routing | ANE output ≠ vocab logits | **Open** — B7+ |
| Hook overhead | Per-step map+eval cost | **Measured** — ~10% e2e with stride=2 on 2B |
| conv2 MIL | Chained conv compile fails | **Open** — falls back to conv1 |

**Do not** enable `ZEROLLAMA_ANE_DRAFT=1` on production `:11434` without explicit operator intent.

---

## Lab status

```bash
./zerollama ane-lab-status           # binary inventory
./zerollama ane-lab-status --sweep   # + quick 256×256 prefill sweep
```

---

## Env (dflash in-process)

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANE_REPO` | `~/Sites/inference/ane` | maderix bridge checkout |
| `ZEROLLAMA_ANE_DRAFT` | `0` | Lab hook in llama-server speculative draft path |
| `ZEROLLAMA_ANE_LAB_PORT` | `11435` | A/B e2e avoids production serve port |

See full table in [ane-draft-inprocess.md](./ane-draft-inprocess.md#environment-variables).

---

## Non-goals

- No automatic serve restart or manifest repair
- No Flash-MoE sidecar path (separate track)
- No ANE training exposure
- No ANE for full 2048² base FFN (MPS wins)
- No folding expert FFN into `ane_draft_session` (keep `ZEROLLAMA_ANE_FFN_*`)
