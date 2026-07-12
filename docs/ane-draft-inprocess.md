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

**Open (B7):** `ZEROLLAMA_ANE_DRAFT_DRIVE=shadow|force` — ANE matmul output → host tied-embed argmax → draft token (`shadow` logs parity, `force` replaces Metal sampler). **P6 token shadow (M4 Max, Jun 2026):** `shadow_steps=14`, `shadow_match_pct=0%` (expected — blk.0 proxy ≠ full dflash), `shadow_hidden_cos≈1.0` on ffn_down stash vs CPU golden. **Force baseline (P6):** ~27% draft acceptance, large TPS hit from sync eval per handoff. **Open (B8):** attn softmax/KV, lm_head for token parity. **Shipped (Jul 2026):** `common_ane_draft_sync_target_cross` runs after each target decode in speculative `process()` so native Metal dflash reads `cross.v_embd` before draft sync-decode. **Per-layer target export (Jul 2026):** `llama_set_dflash_target_export` + qwen35 graph concat at `dflash.target_layer_ids[]` → `llama_get_dflash_target_features_ith`; ANE hook prefers export over `TARGET_FEAT_TILE` stub when draft arch is native `dflash-draft` (eliza lab sidecar is still qwen35 proxy — export inactive until inventory ships real dflash-draft GGUF).

**Matmul path (P1–P3, Jun 2026):** Prefer `ane-draft-parity-smoke` over the 8-conv chain — lower overhead, real FFN weights.

| Phase | ANE chain | CPU step | Parity metric |
|-------|-----------|----------|---------------|
| **P1** | `h @ ffn_gate` (768→256) | — | `golden_cosine` / `hidden_cos` ≈ 1.0 |
| **P2** | gate → SiLU → `silu(g) @ ffn_up` (256→768) | SiLU on gate | same |
| **P3** | gate + up from `h`, CPU `silu(g)*u`, then `@ ffn_down` (256→768) | SwiGLU multiply | same; **3 ANE evals** per handoff |
| **P4** | P3 + `@ attn_gate` (768→256) | — | `mode=matmul_chain4` golden cos; **4 ANE evals**; B7 drive still uses ffn_down 768-d stash |
| **P5** | P4 + `@ ssm_out` from ffn_down (768→256) | — | `mode=matmul_chain5`; **5 ANE evals**; stride **12** |
| **P6** | P5 + **`attn_qkv` prefix** from `h` (768→256) before FFN | — | `mode=matmul_chain6_qkv`; **6 evals**; stride **16**; `MATMUL_CHAIN=5` pins P5 |
| **P7a** | P6 + **`blk.1 ffn_gate`** from ffn_down (768→256) | — | **7 evals**; stride **20**; **host fp32** on lab (ANE fp16 ~0.58 B6 cos) |
| **P8** | P7a + **`blk.1 ffn_up`** + CPU SwiGLU (768→256) | SiLU× on blk.1 gate/up | **8 evals**; `MATMUL_CHAIN=9`; stride **24**; **host fp32 up** |
| **P9** | P8 + **`blk.1 ffn_down`** (256→768) | — | **host fp32** (ANE fp16 underflow at ~1e-3 SwiGLU); `MATMUL_CHAIN=10`; stride **28**; lab **`shadow_hidden_cos=1.0`** |
| **P7b** | `target_hidden @ dflash_fc` → `n_embd` (native dflash-draft only) | — | **`MATMUL_CHAIN=8`**; `bind_target_ctx` + ctx_tgt pre-norm at **`i_batch=-1`**; **`common_ane_draft_sync_target_cross`** in speculative `process()` → `cross.v_embd`; `TARGET_FEAT_TILE=1` lab stub — **M4 Max Jul 2026: `golden_cosine=1.0`** (eliza qwen35 gate proxy) |
| **P10** | P7b + **host RMS(`dflash_hidden_norm`)** + `@ blk.0.attn_q` | RMS on fc out | **`MATMUL_CHAIN=11`**; `WEIGHT_FILE2=attn_qkv` (qwen35 fallback); stride **20** — **M4 Max Jul 2026: `golden_cosine=1.0`** (gate proxy + `attn_norm` gamma) |

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

# P7b / P10 dflash subgraph (ctx_tgt handoff + cross.v_embd sync)
ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN=8 LLAMA_SERVER_BIN=../llama.cpp/build/bin/llama-server \
  ./zerollama ane-draft-parity-smoke --model eliza-1-2b-dflash --quick --telemetry
ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN=11 LLAMA_SERVER_BIN=../llama.cpp/build/bin/llama-server \
  ./zerollama ane-draft-parity-smoke --model eliza-1-2b-dflash --quick --telemetry
```

**P6 lab (M4 Max, stride=16 default, async eval):** `golden_cosine=1.0`, `matmul_chain=6`, hook overhead ~3–7% vs Metal-only dflash.

**P9 lab (Jul 2026):** blk.1 gate/up/down run on **host fp32** — ANE fp16 loses ~0.58 cos on blk.1 gate and returns zero on blk.1 down (~1e-3 SwiGLU vs ~3 blk.0). With host fp32 P7–P9, `shadow_hidden_cos=1.0` and `golden_cosine=1.0` on eliza `qwen35` sidecar (`ane-draft-parity-smoke --token-shadow --telemetry`). P1–P3 blk.0 FFN remains on ANE.

**P10 lab (Jul 2026):** `MATMUL_CHAIN=11` — dflash_fc (gate proxy) → host RMS norm → `blk.0.attn_qkv` top-left slice on ANE; `mode=matmul_chain11_dflash_attn_q`, **`golden_cosine=1.0`**, stride **20**. qwen35 sidecar lacks `blk.0.attn_q.weight`; Go `ResolveChain11AttnQTensor()` falls back to fused `attn_qkv`.

**Native dflash-draft lab (27b):** `eliza-1-27b-256k-dflash` sidecar is **`dflash-draft`** with real `dflash_fc.weight` `[25600×5120]` (Q8_0), `dflash_hidden_norm.weight`, and full decoder blocks. Inventory tracks **`draft_architecture`** separately from the target base model; chain **8/11** auto-select native weights and skip `TARGET_FEAT_TILE` when export meta is set. **Jul 2026 lab:** chain **8** **`golden_cosine=0.9854`** (512-d export slice → `dflash_fc` on ANE); chain **11** **`golden_cosine=1.0`** (`dflash_fc` → host RMS(`dflash_hidden_norm`) → `blk.0.attn_q` at `5120×512` proxy slice, stride **20**). **B7 token shadow (Jul 2026):** `--token-shadow` on chain **11** loads tied-embed from **base model** (sidecar lacks `token_embd`; Q4_K dequant cache ~2.5GB); **`shadow_steps=18`**, **`shadow_match_pct=0%`** until attn/KV + lm_head complete the draft hidden (ANE argmax uses 512-d `attn_q` padded to `n_embd`). Go wiring: `forceChain==11` enables native `dflash_fc`; `MaterializeANEDraftNormGammaFile` + `MaterializeANEDraftDriveHead` with base-model fallback.

**Staged chain baselines (M4 Max):** `MATMUL_CHAIN=3` → hidden_cos **1.0** (blk.0 down); chain **7/9/10** → **1.0** after P7 host fp32 (chain 7 B7 metric still reports blk.0 down; B6 `chain7_blk1_gate` was ~0.58 on ANE). **Chain 8/11** → **1.0** (dflash_fc subgraph + P10 attn_q).

**B7 token shadow (Jun 2026):** `--token-shadow` stable (~25–30s with `skipMicro`); `shadow_steps=14`, `shadow_match_pct=0%` on eliza `qwen35` drafter (blk.0 proxy ≠ full decode). Parity JSON includes `sidecar_architecture` + `has_dflash_fc_tensor`.

**Lab env hygiene (Jun 2026):** `runDflashServerLeg` strips inherited `ZEROLLAMA_ANE_DRAFT_*` before applying smoke env so stale shell exports (e.g. `DRIVE_METRICS=hidden`) cannot override `--token-shadow`. Operator overrides for `MATMUL_CHAIN` / `HANDOFF_STRIDE` are still read from the shell once, then re-applied.

**llama-server subprocess (Jun 2026):** `runDflashServerLeg` sets `cmd.Dir` to `filepath.Dir(LLAMA_SERVER_BIN)` and prepends that directory to `DYLD_LIBRARY_PATH` so `libane_bridge.dylib` resolves next to the unified build (same rule as `ane-probe`).

**Async eval sync (Jun 2026):** `ane_draft_session_eval_async` pairs `dispatch_group_enter/leave` so `eval_sync()` blocks until ANE eval completes. **B7 drive (shadow/force)** forces **sync eval on handoff** — `try_drive_token` runs in the same `draft()` turn; `eval_sync()` is skipped when drive is active (avoids dispatch_group deadlock).

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
| `ZEROLLAMA_ANE_DRAFT_HANDOFF_STRIDE` | `2` conv / `4` P1–P2 / `8` P3–P4 / `12` P5 / `16` P6 | Handoff every N decode steps (Go A/B default per chain depth). |
| `ZEROLLAMA_ANE_DRAFT_CONV_DEPTH` | `0` (unlimited) | Cap compiled conv kernels (`1` = `WEIGHT_FILE` only). **Why:** faster A/B when full v9 chain is materialized but you only want B8/B9 depth. |
| `ZEROLLAMA_ANE_DRAFT_KERNEL` | `conv` | `matmul` — gate (+ optional chain2 up). Static MIL may fail; auto-falls back to **dynamic MIL**. |
| `ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN` | auto | `1`–`7`; pin with explicit value. P7 auto when `WEIGHT_FILE7` (`blk.1.ffn_gate`) materialized. |
| `ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE4` | — | P4 matmul: `blk.0.attn_gate.weight` (768×256 slice). Conv path: `ffn_down`. |
| `ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE5` | — | P5 matmul: `blk.0.ssm_out.weight` (768×256 slice). Conv path uses FILE5 for blk.1 gate. |
| `ZEROLLAMA_ANE_DRAFT_MATMUL_OC5` | — | P5 ssm_out output channels. |
| `ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE6` | — | P6 matmul: `blk.0.attn_qkv.weight` (768×256 slice). Conv path uses FILE6 for blk.1 `ffn_up`. |
| `ZEROLLAMA_ANE_DRAFT_MATMUL_OC6` | — | P6 qkv output channels. |
| `ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE7` | — | P7 matmul: `blk.1.ffn_gate.weight` (768×256 slice). Conv path uses FILE7 for blk.1 `attn_gate`. |
| `ZEROLLAMA_ANE_DRAFT_MATMUL_OC7` | — | P7 blk.1 gate output channels. |
| `ZEROLLAMA_ANE_DRAFT_MATMUL_OC4` | — | P4 attn_gate output channels (Go A/B sets from sidecar dims). |
| `ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE2` | — | P2/P3: `blk.0.ffn_up.weight` matmul blob. |
| `ZEROLLAMA_ANE_DRAFT_WEIGHT_FILE3` | — | P3: `blk.0.ffn_down.weight` matmul blob (256×768). |
| `ZEROLLAMA_ANE_DRAFT_MATMUL_OC2` / `OC3` | — | Up/down output channels (Go A/B sets from sidecar dims). |
| `ZEROLLAMA_ANE_DRAFT_MATMUL_SEQ` | `16` | Matmul spatial; **min 16** at ic=oc=256 (ANE MIL eval floor). |
| `ZEROLLAMA_ANE_DRAFT_SYNC_CROSS` | `1` | Upsert ctx_tgt pre-norm into draft `cross.v_embd` (speculative `process()` + handoff fallback). |
| `ZEROLLAMA_ANE_DRAFT_TARGET_FEAT_TILE` | `0` | B8 lab: repeat ctx_tgt `n_embd` across `dflash_fc` ic when native sidecar lacks `dflash_fc.weight`. |
| `ZEROLLAMA_ANE_DRAFT_DFLASH_N_TARGET_FEATURES` | auto | Override `dflash.n_target_features` width for cross pack (native dflash-draft). |
| `ZEROLLAMA_ANE_DRAFT_DFLASH_N_TARGET_LAYERS` | `0` | Lab layer-concat stub: tile final pre-norm into N slices until per-layer target export exists. |
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

## Known issues

### Chain 17 B7 token-shadow: intermittent SIGSEGV — RESOLVED Jul 12 2026 (heap buffer overflow in `ane_draft_session_eval_dflash_ffn_gate`)

**Root cause found and fixed.** `ane_draft_session_eval_dflash_ffn_gate()` (`ane_draft_session.mm`)
writes `oc6 * seq` floats into `g_session.outBuf`, but unlike every other step in the dflash
chain that changes the output row-dimension (e.g. `ane_draft_session_eval_dflash_ffn_up_swiglu_down`'s
`ffn_down` write, which checks `if (down_bytes > g_session.outIoBytes) { realloc }`), this one had
**no grow-on-demand check before writing**. `outBuf` was left sized from the previous step
(`attn_wo`, row-dim = n_embd = 5120) and `ffn_gate` then wrote `oc6 * seq = 17408 * 16 = 278,528`
floats into it — for chain 17's real shapes that's a **~3.4x heap buffer overflow**, corrupting
~789KB past the end of a ~320KB allocation into whatever the heap allocator placed next. Fixed by
adding the same grow-then-`calloc` pattern used elsewhere in the file, guarded by comparing
`gate_bytes` against `g_session.outIoBytes` before the write.

This was found via forensic analysis of the fault address (`far` register) across ~10 crash
reports, all of which showed the classic "two adjacent floats overwriting a 64-bit pointer slot"
pattern (see FAR analysis below) — confirming this was real heap corruption, not a live Metal
timing race, before the specific overflow site was located by code audit.

**Verification:** 11/11 back-to-back `ane-draft-parity-smoke --token-shadow` runs on chain 17
completed cleanly (`ok: true`, exit 0) after the fix, vs. the prior ~50% crash rate. All the
Metal-completion-queue/timing "fixes" attempted earlier in this investigation (drain-sync,
residency-heartbeat pause, post-sync sleep) are harmless no-ops now that the actual corruption is
fixed, and are left in the tree as defensive/perf changes, but were never the real fix.

**Follow-up hardening (Jul 12, same day):** audited every `g_session.outBuf` write site across
all dflash chains (11–17), not just the one that crashed, for the same missing-grow-check bug
class. Found one more latent (not yet observed to trigger) instance:
`ane_draft_session_eval_dflash_attn_wo()` wrote `oc5 * seq` floats into `outBuf` with no
size check — currently safe in practice because `oc5` is a fixed session field that happens to
always match the prior step's ending size for chains 14–17, but nothing enforces that invariant,
so a future chain reordering/addition could silently reintroduce the same overflow. Hardened it
with the same grow-check. Also introduced a shared helper,
`static bool ane_session_ensure_out_buf(size_t new_bytes)`, and refactored all three grow-check
call sites (`attn_wo`, `ffn_gate`, `ffn_up_swiglu_down`'s `ffn_down` write) to use it instead of
duplicating the `calloc`/`free` pattern inline — **any new dflash step that writes a
differently-sized row into `outBuf` should call this helper first.** All other `outBuf`
write/read sites in the file either write into a local `std::vector` (safe) or read/write using
`g_session.outIoBytes`/`ioBytes` (the already-current size, self-consistent), so no other overflow
sites were found. Re-verified chains 14, 15, 16, and 17 all pass the token-shadow smoke test
cleanly after this hardening pass.

### (Historical) investigation notes below, kept for the record

**Symptom:** `eliza-1-27b-256k-dflash` chain 17 (`ZEROLLAMA_ANE_DRAFT_MATMUL_CHAIN=17`) with
`--token-shadow` on lab port 11435 crashes `llama-server` with `SIGSEGV`
(`EXC_BAD_ACCESS` / `KERN_INVALID_ADDRESS`) roughly every other run. The fault always lands in
Apple's own AGX/Metal resource-list code (`MTLResourceListAddResource` or
`IOGPUMetalCommandBufferStorageReset`), reached via different call stacks depending on what
else is running at the time:

- Async completion queue: `com.Metal.CompletionQueueDispatch` → `IOGPUMetalCommandBufferStorageReset/Dealloc` (most common / baseline signature)
- Main thread building a new compute encoder (`ggml_metal_encoder_init`)
- Main thread doing an ordinary blit copy inside `llama_decode`/`extract_layer_inputs`

**Root cause: not yet found.** This is a genuine data race / memory-safety bug, most likely a
use-after-free or similar corruption of Metal resource-tracking state tied to `ctx_dft`'s Metal
buffers — but the exact mechanism is still unknown. The bug has a strong **observer effect**:
every diagnostic tool tried either changes the crash signature or suppresses it entirely
(`MTL_DEBUG_LAYER=1` avoided it outright in testing), which makes it very hard to catch live.

**Hypotheses tested and disproven (Jul 2026 lab session):**
1. ~~Host-side latency in the ANE post-eval pipeline (`ane_dflash_post_eval_pipeline`,
   especially `matmul_golden_reference`-based FFN forward in `ane_dflash_host_layer_forward`)
   widens a timing window that exposes a pre-existing race.~~ Multithreading
   `matmul_golden_reference` gave a real ~4.7x speedup (host_layer_tail 31.6s → 6.6s) but the
   crash still reproduced with the same signature — window shrinking alone isn't enough.
2. ~~The `ggml_metal_rsets_init` background residency-set keep-alive heartbeat (5ms period,
   `ggml-metal-device.m`) races a Metal command-encoder/resource-list mutation on `ctx_dft`.~~
   Added `ggml_backend_metal_dev_rsets_pause/resume` (ref-counted; see `ggml-metal.h`,
   `ggml-metal-device.{h,m}`) and paused the heartbeat for the entire ANE host-compute window
   in `common_ane_draft_handoff_after_decode`. Confirmed the pause engaged (debug log fired).
   Crash still reproduced with the exact same signature — the heartbeat is not the culprit.
3. ~~The cached IOSurface buffer in `pack_draft_hidden_into_iosurface` is freed while a Metal
   command buffer still references it (resize-free path).~~ Added a `llama_synchronize(ctx_dft)`
   guard before the resize-free (`ZEROLLAMA_ANE_IOSURFACE_RESIZE_SYNC`, default on). Real
   defensive fix, kept, but the resize-free path was never even exercised in single-request
   repros (no `resize-free` log line), so it cannot explain the crash by itself.
4. Ruled out (not a hypothesis, a check): the GCD async-eval path
   (`ane_draft_session_eval_async` / `ane_handoff_eval_done`) is not implicated — it only runs
   when drive mode is `COMMON_ANE_DRAFT_DRIVE_OFF`, but `--token-shadow` forces
   `COMMON_ANE_DRAFT_DRIVE_SHADOW`, which uses the synchronous eval path exclusively (confirmed
   via drive logs: `"eval ok"`, never `"async ANE eval queued"`). The server's `update_slots()`
   loop is also single-threaded — no cross-thread misuse of `ctx_dft` from our own code found.
   The only genuinely concurrent actor touching Metal state is Apple's own completion-queue
   thread, which lines up with where every crash bottoms out.

**Mitigation for lab work today:** run chain 17 B7 token-shadow smoke tests with
`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` set — this reliably avoided the crash in testing
(one full clean run, vs otherwise ~50% crash rate). This is a **timing-perturbation mitigation,
not a fix** — do not use it as evidence the underlying bug is resolved, and do not extrapolate
"no crash under Metal validation" to production behavior (which will never run with that flag).

**Left in the tree (all lab-only, harmless if unused):**
- `matmul_golden_reference` (`ane_draft_hook.cpp`) — multithreaded, real perf win independent of this bug.
- `ZEROLLAMA_ANE_HANDOFF_DRAIN_SYNC` (default on) — `llama_synchronize(ctx_dft)` before returning from handoff.
- `ZEROLLAMA_ANE_RSETS_PAUSE` (default on) — pauses the residency heartbeat during the ANE host-compute window.
- `ZEROLLAMA_ANE_IOSURFACE_RESIZE_SYNC` (default on) — synchronizes before freeing the cached IOSurface buffer on resize.
- `ggml_backend_metal_dev_rsets_pause/resume` (`ggml-metal.h`) — new reusable API, independent of whether it fixes this bug.

**Tried and disproven (Jul 12 follow-up session):**
5. ~~`llama_synchronize(ctx_dft)`/`ggml_backend_sched_synchronize` (used by our drain-sync and by
   stock `llama_get_logits_ith`) only calls `[cmd_buf waitUntilCompleted]`, which does not
   guarantee Metal's own internal async command-buffer teardown
   (`IOGPUMetalCommandBufferStorageReset/Dealloc`, dispatched via
   `IOGPUNotificationQueueDispatchAvailableCompletionNotifications`) has finished — added a
   tunable post-sync sleep (`ZEROLLAMA_ANE_HANDOFF_DRAIN_SLEEP_US`, default 2000us) as an
   empirical workaround to give that teardown more time.~~ Tested at 3ms: crash still reproduced,
   and the signature **shifted again** — plain `SIGSEGV` in `ggml_mul_impl` called from
   `llm_graph_context::build_norm` during the *next* `llama_decode`'s graph build (CPU-side ggml
   op, not even touching a Metal API directly), one call stack level up from
   `llama_context::process_ubatch`. This is consistent with memory corruption of an `ggml_tensor`
   or its backing buffer from an earlier step (not a live Metal API race at the crash site
   itself) — sleeping longer just changes which subsequent access observes the corruption first.
   Left in the tree, defaults to a small delay since it's cheap and does not appear harmful, but
   it is **not a fix**; do not rely on it.

**New forensic lead (Jul 12 follow-up, FAR analysis):** pulled the fault address (`far` register)
from 10 consecutive crash reports across every signature seen tonight (Metal resource-list code,
`objc_msgSend`/`IOGPUMetalCommandBufferStorageReset`, `ggml_gallocr_alloc_graph`, `ggml_mul_impl`).
Nearly every one has a fault address whose high and low 32-bit halves are identical or nearly
identical (e.g. `0xc0674e94c0674e94`, `0x99fb7c0099fc0`, `0x4c9075404c9080`,
`0xc11259c2c11259ca`). That is the textbook signature of a 64-bit pointer-sized memory slot
(vtable ptr, `next`/`data` field, etc.) having been overwritten by **two adjacent 32-bit floats**
— i.e. real out-of-bounds/stray float write corruption, not a live Metal completion-queue timing
race. This reframes the bug: every crash "moving" under different diagnostics was never a race
changing — it was always the same underlying corruption, and each perturbation just changed
*which* corrupted struct got dereferenced first (explains hypotheses 1–5 above all failing to
fix it while still reproducing the "same" bug family).

Audited (by inspection, not found faulty):
- `pack_draft_hidden_into_iosurface` / `ane_draft_session_pack_matmul_activations` (main IOSurface
  packing path) — verified exact byte math for chain 17's actual runtime shapes
  (ch=512, seq=16, oc=5120, `spIn`=5136, buffer=10,518,528B) — in bounds.
- `drive_head_load` / tied-embed lookup (`W[e*nv+tok]`) — bounds-checked via `tok < nv`, mmap'd
  read-only region sized against `n_embd*n_vocab*2` up front.
- `ane_dflash_stash_inpSA` / `ane_dflash_apply_attn_inpSA_residual` / `ane_dflash_apply_ffn_residual`
  (`ane_draft_session_add_output_row`) — bounds-checked via `min(n, oc)` against
  `ane_session_output_row_dim()`.
- `ane_draft_session_eval_dflash_attn_post_norm` / `..._output_norm` / `..._ffn_up_swiglu_down` —
  each re-derives `outBuf`'s current row-dim (`n_embd`/`oc_gate`/`oc_up`/`oc_down`) from the same
  session field that was used to size `outBuf` (via `calloc`) one step earlier in the chain, so
  the local accounting is internally consistent at each individual call site.

**Not yet audited** (chain 17 touches many more matmul/pack helpers than the above — this list is
not exhaustive): `ane_session_pack_matmul2_activations`, `ane_host_matmul_seq`, the
`matmul{2,4,5,6,7,10}W`/`WCount` weight-load paths, `ane_bridge_read_output`, and every other
`load_gamma_scales` call site (there are ~25). A single off-by-one or stale-length field in any of
these (e.g. a session field updated in one branch but read stale in a sibling branch on a
different code path — chain 17 has several `if/else` forks between "host fp32" and "ANE kernel"
variants of the same step) would produce exactly this kind of intermittent stray-float write.

**Next steps for whoever picks this up:**
- **Start from the FAR-pattern lead, not more synchronization/timing changes.** Grep every
  `calloc`/`resize`/`assign` for `outBuf`, `matmulNW`, `lastDflash*` fields in `ane_draft_session.mm`
  and cross-check each against every read/write site's row-dim math, specifically across the
  host-fp32-vs-ANE-kernel branches within chain 17 — a size field set on one branch and consumed
  on the other after a mode toggle is the most likely single bug class here.
- Alternatively, build with `-fsanitize=address` (ASan) for a single-shot, non-interactive repro
  — ASan poisons redzones around heap allocations, which would catch this class of bug on the
  *write* rather than waiting for a later, unrelated read to crash. (Earlier guidance this session
  was to avoid ASan for iteration speed, but given the FAR evidence now strongly favors a heap
  OOB write over a live race, ASan is likely to catch it in one run instead of ~50% flaky manual
  repros.)
- The crash is probabilistic (confirmed: one clean run followed immediately by a crash under
  identical config), so any "it passed" result needs many repeated runs before it's meaningful
  evidence of anything.
- Since every external observation tool perturbs the timing, consider in-process instrumentation
  instead: e.g. a debug build with `MTLResourceListAddResource` symbolicated watchpoints set via
  a lldb script that only breaks (doesn't slow down other paths), or Instruments' Metal System
  Trace, which is designed to observe without the same degree of interference as `lldb -b`/malloc
  debug flags.
- Consider whether `ctx_dft`'s Metal buffers (KV cache / compute buffers, not just our IOSurface
  cache) get resized/reallocated during a `llama_decode` call that could race an in-flight
  command buffer from a *previous* decode — i.e. look inside stock `llama.cpp`/`ggml-metal`
  buffer lifecycle during ordinary decode, not just our ANE-specific code paths.
- This was all reproduced only on the lab port (11435); never touched production (11434/8081).

## Non-goals

- No automatic `ZEROLLAMA_ANE_DRAFT=1` on `./zerollama serve :11434`
- No ANE for full base-model prefill at 2048² (MPS wins)
- No subprocess draft daemon in production path (IOSurface PID boundary)
- No commit to Core ML compile latency for hot path

---

## See also

- [bench-cache.md](./bench-cache.md) — client-side model bench (orthogonal)
- [SPECULATIVE.md](../runtime/docs/SPECULATIVE.md) — runtime ngram/MTP (separate from dflash ANE lab)
