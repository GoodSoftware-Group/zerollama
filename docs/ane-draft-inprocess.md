# ANE in-process dflash draft (B1–B6 lab track)

**Audience:** Mac operators and contributors working on **Metal base decode + ANE draft-step research** for eliza `*-dflash` tags. **Not enabled on production `:11434` serve unless you explicitly opt in.**

**Related:** [ane-hybrid-path.md](./ane-hybrid-path.md), [ane-ggml-iosurface-hook.md](./ane-ggml-iosurface-hook.md), [ane-probe.md](./ane-probe.md), [phase17-llama-server.md](./phase17-llama-server.md).

---

## Why this track exists

| Problem | Why ANE (not “just faster Metal”) |
|---------|-----------------------------------|
| **Draft steps are small, latency-bound** | At ~256×16 conv proxy geometry, ANE eval is ~0.08–0.15 ms vs multi-ms Metal draft-graph overhead on some steps — worth measuring before rewriting ggml. |
| **Base model must stay on Metal ggml** | Full 2048² FFN prefill loses to MPS above ~720 IC ([crossover](./ane-hybrid-path.md)); only **draft subgraph** candidates fit ANE today. |
| **IOSurface is the only stable handoff** | maderix `libane_bridge` binds ANE I/O to `IOSurfaceRef`; Metal can share bytes via `newBufferWithBytesNoCopy` without CPU memcpy. |
| **Same-process is non-negotiable** | `IOSurfaceLookup(surface_id)` **fails across PIDs** — subprocess daemons proved scheduling; production path must compile ANE **inside llama-server**. |
| **Draft tokens still Metal (for now)** | ANE today runs a **conv proxy** (sidecar weight slice), not the full dflash graph (`dflash_fc`, attn, lm_head). Hook is telemetry + parity until B7+. |

**Why lab port 11435:** Avoid colliding with daily `./zerollama serve` on `:11434`. Set `ZEROLLAMA_ANE_LAB_PORT` if needed.

---

## Architecture (today)

```text
┌─────────────────────────────────────────────────────────────────┐
│  llama-server (single PID) — lab: ZEROLLAMA_ANE_DRAFT=1         │
├─────────────────────────────────────────────────────────────────┤
│  Base model decode          │  Draft model (Metal ggml)          │
│  ggml Metal                 │  common_speculative_impl_draft_*   │
│                             │       │                            │
│                             │       ▼ llama_decode (draft)       │
│                             │  common_ane_draft_handoff_after_   │
│                             │    decode()                        │
│                             │       │ pack pre-norm hidden       │
│                             │       ▼                            │
│                             │  ggml_backend_dev_buffer_from_     │
│                             │    iosurface() → ANE input surface   │
│                             │       │                            │
│                             │       ▼ ane_draft_session_eval()   │
│                             │  libane_bridge (conv / conv2 MIL)  │
│                             │       │                            │
│                             │       ▼ (telemetry only today)     │
│                             │  draft token sampling still Metal  │
└─────────────────────────────────────────────────────────────────┘
```

**Why tokens stay Metal:** ANE output is a `[channels × spatial]` activation map, not vocabulary logits. Routing draft tokens from ANE requires the full dflash subgraph on ANE (B7+) or a verified fusion point.

---

## Milestones (B1–B6)

| Milestone | What | Why |
|-----------|------|-----|
| **B1** | In-process ANE session (`ane_draft_session.mm`) | Compile-once kernel + IOSurface in same PID as ggml; eliminates cross-process `IOSurfaceLookup` failure. |
| **B2** | ggml IOSurface handoff | Pack draft **pre-norm hidden** into ANE surface via public `ggml_backend_dev_buffer_from_iosurface` — same bytes Metal wrote, no stub fill. |
| **B3** | Sidecar weight bundle | Extract top-left `ffn_gate` slice + norm gamma from real drafter GGUF (Q8_0 dequant for 27B); gamma applied on **host pack** because MIL broadcast mul failed compile. |
| **B4** | A/B bench (`ane-draft-ab-smoke`) | Micro ANE step vs Metal-only dflash e2e on lab port; measures hook overhead without touching production serve. |
| **B5** | Per-step handoff | Handoff after **every** draft `llama_decode`, not once — matches real speculative loop; enables step telemetry. |
| **B6** | Two-conv bundle + golden telemetry | Second weight from `ffn_up`; optional chained conv MIL (falls back to conv1 if compile fails); CPU golden vs ANE for numerical parity; `HANDOFF_STRIDE` to limit overhead. |

**Open (B7):** `ZEROLLAMA_ANE_DRAFT_DRIVE=shadow|force` — ANE conv output → host tied-embed argmax → draft token (`shadow` logs parity, `force` replaces Metal sampler). **Force baseline (2B, Jun 2026):** ~27% draft token acceptance vs Metal, ~35–68% e2e overhead — conv proxy is not dflash; B8 subgraph expansion required.

**Matmul path (P1–P3, Jun 2026):** Prefer `ane-draft-parity-smoke` over the 8-conv chain — lower overhead, real FFN weights.

| Phase | ANE chain | CPU step | Parity metric |
|-------|-----------|----------|---------------|
| **P1** | `h @ ffn_gate` (768→256) | — | `golden_cosine` / `hidden_cos` ≈ 1.0 |
| **P2** | gate → SiLU → `silu(g) @ ffn_up` (256→768) | SiLU on gate | same |
| **P3** | gate + up from `h`, CPU `silu(g)*u`, then `@ ffn_down` (256→768) | SwiGLU multiply | same; **3 ANE evals** per handoff |
| **P4** | P3 + `@ attn_gate` (768→256) | — | `mode=matmul_chain4` golden cos; **4 ANE evals**; B7 drive still uses ffn_down 768-d stash |
| **P5** | P4 + `@ ssm_out` from ffn_down (768→256) | — | `mode=matmul_chain5`; **5 ANE evals**; stride **12** default |

```bash
# P3 parity (default matmul kernel, shadow hidden metrics only)
LLAMA_SERVER_BIN=../llama.cpp/build/bin/llama-server \
  ./zerollama ane-draft-parity-smoke --model eliza-1-2b-dflash --quick --telemetry

# P3 token shadow (tied-embed argmax on ffn_down out + hidden_cos)
LLAMA_SERVER_BIN=../llama.cpp/build/bin/llama-server \
  ./zerollama ane-draft-parity-smoke --model eliza-1-2b-dflash --quick --token-shadow

# P3 force drive (ANE token replaces Metal sampler)
LLAMA_SERVER_BIN=../llama.cpp/build/bin/llama-server \
  ./zerollama ane-draft-parity-smoke --model eliza-1-2b-dflash --quick --force

# P1/P2 baseline (explicit chain; skips auto P3 when FILE3 materialized)
ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN=1 ./zerollama ane-draft-parity-smoke --model eliza-1-2b-dflash --quick --telemetry
ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN=2 ./zerollama ane-draft-parity-smoke --model eliza-1-2b-dflash --quick --telemetry
```

**P3 lab (M4 Max, stride=8 default, async eval):** `shadow_hidden_cos=1.0`, `golden_cosine=1.0`, hook+shadow overhead ~**0–7%** vs Metal-only dflash (quick runs are noisy; use `--telemetry` for golden_cos).

**P3 at stride=4** (override `ZEROLLAMA_ANE_DRAFT_HANDOFF_STRIDE=4`): hook ~**7%** — use when validating per-step handoff parity.

---

## Operator commands

```bash
# Build chain (unified elizaOS pin c84b3020)
./scripts/ane_probe_build.sh
./scripts/build_llama_server.sh          # Darwin: auto-runs sync_ane_hook after pin checkout
# Manual: ./scripts/sync_ane_hook_to_llama_cpp.sh
install_name_tool -change libane_bridge.dylib @loader_path/libane_bridge.dylib \
  ../llama.cpp/build/bin/libllama-common.0.0.1.dylib

# Materialize sidecar weights (manifest v2 = ffn_gate + ffn_up + gamma)
./zerollama ane-draft-mil-bundle --model eliza-1-2b-dflash

# Micro + e2e A/B (lab port 11435)
LLAMA_SERVER_BIN=../llama.cpp/build/bin/llama-server \
  ./zerollama ane-draft-ab-smoke --model eliza-1-2b-dflash --quick --e2e

# B7 shadow (log ANE vs Metal token; hook still uses Metal sampler):
LLAMA_SERVER_BIN=../llama.cpp/build/bin/llama-server \
  ./zerollama ane-draft-ab-smoke --model eliza-1-2b-dflash --quick --e2e --e2e-drive

# B7 force (ANE token replaces Metal sampler — expect lower acceptance until B8 subgraph):
LLAMA_SERVER_BIN=../llama.cpp/build/bin/llama-server \
  ./zerollama ane-draft-ab-smoke --model eliza-1-2b-dflash --quick --e2e --e2e-drive-mode force

# Optional: golden CPU ref telemetry on ANE leg (adds e2e overhead; use for conv2 validation)
# ./zerollama ane-draft-ab-smoke --model eliza-1-2b-dflash --quick --e2e --e2e-telemetry

# Same-process smoke (no full server)
./zerollama ane-inprocess-smoke --model eliza-1-2b-dflash --quick

# Tensor → MIL slot plan
./zerollama ane-draft-mil-map --model eliza-1-2b-dflash
```

---

## Environment variables

| Variable | Default | Why |
|----------|---------|-----|
| `ZEROLLAMA_ANE_DRAFT` | `0` | **Fail closed** — production dflash unchanged unless operator opts in. |
| `ZEROLLAMA_ANE_DRAFT_CHANNELS` | `64` (init) / bundle | Proxy conv width; capped from embed (256 for 2B, 512 for 27B). |
| `ZEROLLAMA_ANE_DRAFT_SPATIAL` | `16` | Lab geometry `[1, ch, 1, sp]` — matches draft conv smoke. |
| `ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE` | — | BLOBFILE conv weights (`ffn_gate` slice). |
| `ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2` | — | B6 second conv (`ffn_up` slice); **dual conv1 kernels** when set (chained conv2 MIL fails compile). |
| `ZEROLLAMA_ANE_DRAFT_GAMMA_FILE` | — | Host-side norm gamma multiply before ANE — **why:** MIL `conv×gamma` broadcast failed compile. |
| `ZEROLLAMA_ANE_DRAFT_WEIGHT_MANIFEST` | — | JSON manifest from `ane-draft-mil-bundle`. |
| `ZEROLLAMA_ANE_DRAFT_DRIVE` | `0` | B7 lab: `shadow` logs ANE vs Metal token; `force` uses ANE tied-embed argmax token. Matmul+shadow skips tied-embed (see `DRIVE_METRICS`). |
| `ZEROLLAMA_ANE_DRAFT_DRIVE_METRICS` | — | `hidden` (default matmul shadow) — `hidden_cos` only, no embed mmap. `tokens` or `both` — load tied-embed for token shadow/force on P3 `ffn_down` output. |
| `ZEROLLAMA_ANE_DRAFT_DRIVE_VOCAB_CAP` | `8192` (when drive on) | Limit host argmax scan for lab speed (full vocab ~248k is slow). |
| `ZEROLLAMA_ANE_DRAFT_TOKEN_EMBD_FILE` | — | B7 mmap tied-embed cache from `ane-draft-mil-bundle` manifest v4. |
| `ZEROLLAMA_ANE_DRAFT_TELEMETRY` | `0` | Log CPU golden vs ANE output per step — validates bridge, not draft quality yet. |
| `ZEROLLAMA_ANE_DRAFT_HANDOFF_STRIDE` | `2` conv / `4` P1–P2 / `8` P3–P4 / `12` P5 matmul | Handoff every N decode steps (Go A/B default per chain depth). |
| `ZEROLLAMA_ANE_DRAFT_CONV_DEPTH` | `0` (unlimited) | Cap compiled conv kernels (`1` = `WEIGHT_FILE` only). **Why:** faster A/B when full v9 chain is materialized but you only want B8/B9 depth. |
| `ZEROLLAMA_ANE_DRAFT_KERNEL` | `conv` | `matmul` — gate (+ optional chain2 up). Static MIL may fail; auto-falls back to **dynamic MIL**. |
| `ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN` | auto | `1`–`5`; P5 auto when `WEIGHT_FILE5` (`ssm_out`) materialized. `3`/`4` pin P3/P4. |
| `ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5` | — | P5 matmul: `blk.0.ssm_out.weight` (768×256 slice). Conv path uses FILE5 for blk.1 gate. |
| `ZEROLLAMA_ANE_DRAFT_MATMUL_OC5` | — | P5 ssm_out output channels. |
| `ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4` | — | P4 matmul: `blk.0.attn_gate.weight` (768×256). Conv path uses FILE4 for `ffn_down` — do not mix. |
| `ZEROLLAMA_ANE_DRAFT_MATMUL_OC4` | — | P4 attn_gate output channels (Go A/B sets from sidecar dims). |
| `ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2` | — | P2/P3: `blk.0.ffn_up.weight` matmul blob. |
| `ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3` | — | P3: `blk.0.ffn_down.weight` matmul blob (256×768). |
| `ZEROLLAMA_ANE_DRAFT_MATMUL_OC2` / `OC3` | — | Up/down output channels (Go A/B sets from sidecar dims). |
| `ZEROLLAMA_ANE_DRAFT_MATMUL_SEQ` | `16` | Matmul spatial; **min 16** at ic=oc=256 (ANE MIL eval floor). |
| `ZEROLLAMA_ANE_DRAFT_EVAL_ASYNC` | matmul on | After IOSurface pack, queue P1–P3 ANE eval on a serial GCD queue (overlaps with next Metal decode). `eval_sync()` before the next pack or force-drive read. Set `0` to block in handoff (~2× hook overhead on M4 Max lab). |
| `ane-draft-parity-smoke` | — | Hidden CLI: matmul + shadow metrics on lab port 11435. JSON: **`hook_overhead_pct`** (handoff+eval only) vs **`shadow_overhead_pct`** (includes B7 tied-embed scan). |
| `ZEROLLAMA_ANE_LAB_PORT` | `11435` | Keeps e2e A/B off production `:11434`. |
| `ANE_REPO` | `~/Sites/inference/ane` | maderix bridge checkout for `libane_bridge.dylib`. |
| `LLAMA_SERVER_BIN` | auto | **Why set explicitly:** vendor tree binary may be wrong arch; sibling build has ANE hook. |

---

## Weight cache layout

Default: `~/.cache/zerollama/ane-draft-weights/`

| File pattern | Source tensor | Why |
|--------------|---------------|-----|
| `drafter-*-{ch}-blk_0_ffn_gate_weight.v3.weight.bin` | `blk.0.ffn_gate.weight` | Lab conv proxy — top-left block transposed `[out,in]` + maderix blob header. |
| `drafter-*-{ch}-blk_0_ffn_up_weight.v3.weight.bin` | `blk.0.ffn_up.weight` | B6 second conv in subgraph expansion. |
| `drafter-*-768-blk_0_ffn_gate_weight.v3.weight.bin.mm768x256.v2.bin` | `blk.0.ffn_gate.weight` | P1–P3 matmul gate (768×256 FP16 blob). |
| `drafter-*-768-blk_0_ffn_up_weight.v3.weight.bin.mm768x256.v2.bin` | `blk.0.ffn_up.weight` | P3 matmul up (768×256). |
| `drafter-*-256-blk_0_ffn_down_weight.v3.weight.bin.mm256x768.v2.bin` | `blk.0.ffn_down.weight` | P3 matmul down (256×768). |
| `drafter-*-{ch}-blk_0_attn_norm_weight.v3.weight.bin` | norm gamma | B3 host-side scale before conv. |
| `drafter-*-{ch}-manifest.json` | — | Bundle version + paths for `ExportEnvForManifest`. |

Manifest **v1 → v2** auto-refresh: old caches without `proxy_conv_w1` are rebuilt on next `ane-draft-mil-bundle`.

---

## Lab results (M4 Max, Jun 2026)

**Micro (2B, 256ch):** steady ~0.08–0.15 ms ANE eval + ~0.15 ms map fill after cold start.

**E2e A/B (`ane-draft-ab-smoke --e2e`, stride=2 on ANE leg):**

| Model | Metal tok/s | ANE hook tok/s | Overhead | Notes |
|-------|-------------|----------------|----------|-------|
| eliza-1-2b-dflash | ~104 | ~103 | **~0.8%** | default (no telemetry); `conv2_chained: true` |
| eliza-1-2b-dflash | ~42 | ~46 | ~-10% | `--e2e-telemetry`; `golden_cosine: 1.0` (bridge validated) |
| eliza-1-27b-256k-dflash | ~6.7 | ~6.0 | ~9% | earlier run |

**Why overhead varies:** B5 per-step handoff vs stride, cached IOSurface reuse, `--e2e-telemetry` (CPU golden ref), server load. JSON reports `handoff_steps` and `conv2_chained` on the ANE leg.

---

## Source files (zerollama tree)

| Path | Role |
|------|------|
| `llama/llama.cpp/common/ane_draft_session.{h,mm}` | Compile-once ANE kernel; IOSurface lifetime |
| `llama/llama.cpp/common/ane_draft_hook.{h,cpp}` | Speculative hook; handoff + telemetry |
| `llama/llama.cpp/common/speculative.cpp` | Calls handoff after draft decodes |
| `discover/ane_draft_weight_bundle.go` | Sidecar extract + manifest |
| `discover/ane_draft_ab.go` | Micro + e2e A/B JSON |
| `discover/ane_draft_mil_map.go` | Tensor → future MIL slot plan |
| `cmd/ane_draft_*.go` | CLI smoke commands |
| `scripts/sync_ane_hook_to_llama_cpp.sh` | Copy hook into unified `../llama.cpp` (also auto-run from `build_llama_server.sh` on Darwin) |
| `tools/ane-patches/` | Canonical ANE sources + `0018` patch scaffolding |
| `zerollama doctor` | **ane draft hook (llama.cpp)** — source + binary B7 marker check |

---

## Non-goals

- No automatic `ZEROLLAMA_ANE_DRAFT=1` on `./zerollama serve :11434`
- No ANE for full base-model prefill at 2048² (MPS wins)
- No subprocess draft daemon in production path (IOSurface PID boundary)
- No commit to Core ML compile latency for hot path

---

## See also

- [bench-cache.md](./bench-cache.md) — client-side model bench (orthogonal)
- [SPECULATIVE.md](../runtime/docs/SPECULATIVE.md) — runtime ngram/MTP (separate from dflash ANE lab)
