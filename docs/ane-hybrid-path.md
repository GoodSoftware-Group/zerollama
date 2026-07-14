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

**Draft/hybrid commands** use the **dflash inventory**. **Prefill commands** use **any local GGUF** via `ane-model-resolve`.

```bash
./scripts/ane/ane_probe_build.sh
./scripts/build/build_llama_server.sh
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
