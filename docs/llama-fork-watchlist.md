# llama.cpp fork watchlist (Jul 2026)

**Audience:** Contributors deciding what to cherry-pick onto the unified ggml-org pin (`86d86ed4` + `llama/patches/`).

**Not a pin swap.** Keep one vendor tree. Treat external forks as **diff mines** and lab binaries.

**Hedge:** [elizaOS/llama.cpp](https://github.com/elizaOS/llama.cpp) (`../eliza-llama.cpp` @ `ad56033`) may stall on upstream rebases. **Zerollama’s ggml-org pin remains source of truth.** Allowlist-import useful Eliza LLM/Metal fixes; never ship `libelizainference` / Kokoro / OmniVoice in our `llama-server`.

Related: [gpu-profiles-l2.md](./gpu-profiles-l2.md), [flash-moe.md](./flash-moe.md), [runtime/docs/SPECULATIVE.md](../runtime/docs/SPECULATIVE.md), [runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md).

---

## Already on our pin

| Capability | Where |
|------------|--------|
| DFlash draft + `--spec-type dflash` | Patches + speculative docs |
| TBQ3/4 + TBQ3_TCQ KV | L2 profiles (`tbq4_0`/`tbq3_0`, `tbq3_tcq`) |
| QJL / Polar KV | L2 `speed` profile |
| Metal + CUDA SET_ROWS / FA for TBQ | Mid-series patches |
| Metal recoverable nil pipeline + bf16 library gate | **0096** (Eliza #11612; renumbered — remote **0093** is llama-bench) |
| Metal Polar + QJL SET_ROWS (embed path) | **0097** (Lab D1 — encode) |
| Metal fused QJL+Polar attn (embed path) | **0098** (Lab D1b — full `speed` unblocked; tok/s still loses) |
| `llama-bench` accepts Eliza L2 KV type names | **0093** |
| Anemll Flash-MoE (separate binary) | [flash-moe.md](./flash-moe.md) / ROADMAP M16 |

Enum IDs are **Eliza-specific** (`TBQ3_0=44` …). Do not assume Bee/TheTom GGUF type IDs match.

---

## Eliza import allowlist (prep, Jul 2026)

Scout: `../eliza-llama.cpp` @ `ad56033` vs vendor `86d86ed4` + patches.

| Item | Verdict |
|------|---------|
| `tools/kokoro`, `tools/omnivoice`, `libelizainference` | **Skip forever** for our binary — build refuses if trees appear (`ZEROLLAMA_ALLOW_ELIZA_VOICE=1` override only) |
| `eliza-shipped/turbo3.metal` / `turbo4.metal` | **Already match** (TBQ revert already in vendor) |
| `tbq_set_rows.metal` | **Ours only** (keep) |
| Metal #11612 nil-pipeline recover + bf16 library gate | **Imported as 0096** (was draft-0093; remote took **0093** for llama-bench) |
| gemma4-assistant | Already on vendor |
| Kokoro Accelerate / OmniVoice diarizer | Voice product — skip |

**Binary gates:** `./scripts/build/build_llama_server.sh` passes `-DLLAMA_BUILD_KOKORO=OFF -DLLAMA_BUILD_OMNIVOICE=OFF` and exits if `tools/kokoro` or `tools/omnivoice` exist unless `ZEROLLAMA_ALLOW_ELIZA_VOICE=1`.

---

## Eliza PR roadmap (draft — do not open yet)

Each future PR into elizaOS/llama.cpp must **sell the merge** in the first screen. Reviewers should not need zerollama context. One theme per PR; flags default off for behavioral features.

### Pitch template

1. **Headline** — one sentence outcome for Eliza product/users
2. **Pain today** — what breaks / costs RAM / blocks agents without this
3. **What you get** — concrete flag/CLI + default stock behavior
4. **Why safe to merge** — default OFF / pure fix; no voice ABI churn
5. **How to try** — 3 commands
6. **Proof** — smoke / Metal build / flags-off == stock
7. **Out of scope** — what this PR deliberately does *not* do

### Water-test first: upstream sync (PR 0)

**Best first PR** is not a feature — it is **bring `elizaOS/llama.cpp` up to a recent ggml-org tip** (e.g. toward `86d86ed4` / current master). That tests whether Eliza still wants / can absorb upstream before we ask them to take Bee/COW/FP8.

| | |
|--|--|
| **Headline** | Eliza’s fork tracks latest llama.cpp again — new models, Metal/CUDA fixes, server APIs without a private freeze. |
| **Pain today** | `main` @ `ad56033` (Jul 3) lags ggml-org; every week of drift makes TBQ/voice/FFI rebases harder and blocks community patches. |
| **What you get** | Merge (or staged rebase) of ggml-org into Eliza `main`; conflict resolution that **keeps** Kokoro/OmniVoice/`elizainference` + TBQ/QJL/Polar. |
| **Why safe** | No new product flags — behavior stays Eliza’s; CI: Metal + CUDA build of `llama-server` + existing eliza workflows. |
| **How to try** | `git fetch ggml-org; git merge <tip>`; build `-DGGML_METAL=ON` / CUDA; smoke `llama-cli` + one fused FFI load if they care. |
| **Proof** | Green `eliza-metal-validation` / `eliza-cuda-validation` (or documented local Metal build); version stamp shows new tip. |
| **Out of scope** | Zerollama-only patches (Bee, COW, media seq-copy); changing voice ABI. |

**Honest risk:** this is the **hardest** engineering PR (conflicts in Metal + fused FFI), but the **easiest sell** (“don’t fall behind upstream”). If they reject or cannot land it, feature PRs are unlikely to land either — stop and keep owning our pin.

### Suggested order after PR 0 lands

| # | Theme | Patch | Sales angle (draft) | Gate / default |
|---|-------|-------|---------------------|----------------|
| 0 | **Upstream sync** | — | Eliza tracks latest ggml-org again — water test. | merge/rebase; keep Eliza deltas |
| 1 | TBQ build fix | 0088 | Unblocks Metal/CPU link when TBQ patches collide — zero behavior change. | none (pure fix) |
| 2 | Metal crash guards | 0096 / #11612 | Stops hard abort on nil Metal pipelines / missing bf16 — Mac reliability. | always-on safety; `GGML_METAL_ABORT_ON_NIL_PIPELINE=1` for debug abort |
| 3 | Bee loop-guard | 0087 | Stops runaway thinking loops that burn tokens and hang agent turns — opt-in. | `--reasoning-loop-guard` **off** |
| 4 | Radix media seq-copy | 0090 | Lets `/kv/seq-copy` seed multimodal slots without corrupt mid-chunk clones. | `allow_media` **stock = reject** until enabled (flip before Eliza drop) |
| 5 | KV COW | 0089 | Safe cross-slot KV fork for Radix without silent alias bugs. | `ZEROLLAMA_KV_COW*` **unset** |
| 6 | Native FP8 | 0076–0079 | Load real FP8 GGUFs without dequant tax — only if Eliza ships FP8 weights. | type support when GGUF has FP8 |
| 7 | (Defer) RotorQuant / OSCAR | external | **Closed no-merge** (5080 + Mac Metal Lab A). Don’t sell. | lab done |

### Marketing rules

- Lead with **Eliza agent / Mac / TBQ** outcomes, not “zerollama ported X”
- Never couple voice FFI with LLM server fixes (except PR 0, which must preserve voice)
- If it can’t be demoed in under 5 minutes with flag off = stock, it’s not ready to open (PR 0 demo = build + version)
- **Gate feature PRs on PR 0** — if upstream sync fails, do not open 0087–0090 onto a stale tip

---

## Lab A — RotorQuant / IsoQuant / PlanarQuant (highest novelty)

**Claim:** block-diagonal rotations beat WHT TurboQuant on PPL + prefill/decode at ~10× KV compression.

| Piece | URL |
|-------|-----|
| Paper / numbers | [scrya-com/rotorquant](https://github.com/scrya-com/rotorquant) |
| Research kernels | [ParaMind2025/isoquant](https://github.com/ParaMind2025/isoquant) |
| **Runnable llama.cpp fork** | [johndpope/llama-cpp-turboquant `feature/planarquant-kv-cache`](https://github.com/johndpope/llama-cpp-turboquant/tree/feature/planarquant-kv-cache) |

**Cache types:** `planar3`, `iso3`, `planar4`, `iso4` (plus Tom `turbo*` in that tree).

**Sibling scout (this machine):** `../llama-cpp-rotorquant` @ `feature/planarquant-kv-cache` (`08e025c`). CUDA build + A/B only on 5080/dual-4090 — Mac has no NVIDIA.

**Type ID collision (must remap on cherry-pick):**

| RotorQuant | ID | Our pin (Eliza) | ID |
|------------|----|-----------------|----|
| `PLANAR3_0` | **44** | `TBQ3_0` | **44** |
| `ISO3_0` | **45** | `TBQ4_0` | **45** |
| `PLANAR4_0` | **46** | `QJL1_256` | **46** |
| `ISO4_0` | **47** | (gap / Polar) | — |

Do **not** merge their enum numbers as-is. Assign free IDs above FP8 (53+) or reclaim unused slots after an A/B win.

### Harness

```bash
# Sibling build (CUDA lab host) — lab ports only
git clone -b feature/planarquant-kv-cache \
  https://github.com/johndpope/llama-cpp-turboquant.git ../llama-cpp-rotorquant
cmake -S ../llama-cpp-rotorquant -B ../llama-cpp-rotorquant/build \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama-cpp-rotorquant/build -j --target llama-server llama-bench

CUDA_LLAMA_MODEL=/path/to.gguf \
ROTORQUANT_LLAMA_SERVER_BIN=../llama-cpp-rotorquant/build/bin/llama-server \
L2_NUM_CTX=8192 \
L2_RQ_ALSO_LLAMA_BENCH=1 \
  ./scripts/phase/l2_rotorquant_ab.sh
# → /tmp/l2-rotorquant-ab.json
```

Default legs: `stock,tbq,qjl,planar3,iso3` (stock/tbq/qjl on **our** binary; planar/iso on RotorQuant binary). Port default **18082**.

### Exit criteria (before cherry-pick)

1. `planar3` or `iso3` **decode ≥ TBQ** at same ctx, or clear **VRAM win** with acceptable PPL.
2. Prefill (`llama-bench -p`) not worse than TBQ by a large margin (their headline claim).
3. FA path stable on 5080 / dual-4090; Metal smoke if we care Mac parity.
4. Type IDs do not collide with `TBQ*` / `QJL` / `POLAR` / FP8 (51/52) on our pin — **RotorQuant currently reuses 44–47**; remap required before vendor merge.

**If it fails:** leave as external lab binary; do not merge.

### 5080 live A/B (Jul 2026, CT 1564) — **no-merge**

Measured with `llama-bench` on RTX 5080 16 GB, FA on. Artifacts: `/var/lib/vz/private/1564/root/bench-5080-alpha/RESULTS.md`.

| Model | Finding |
|-------|---------|
| Llama-3.2-3B Q4_K_M | Stock **f16** wins tg; rotor **turbo3** ~0.82× tg (best compressed); **planar3/iso3** prefill collapses (~0.11–0.16× pp2048) |
| Llama-3.1-8B Q4_K_M | Stock **q8_0** beats f16 (**157 vs 113** tg); matches L1 `rtx-5080.json` |
| 8B depth tg @ 8k/16k | **q8_0** still best; adaptive **turbo2** near f16 at 16k; **turbo3** falls off hard; planar/iso not competitive |
| 8B depth tg @ 32k/65k | **q8_0 ≈ f16** (**85/59** vs **88/58** tg); turbo2/3 collapse (~51/32, ~25/14) — VRAM labs only |

**Verdict:** Do **not** cherry-pick planar/iso. Keep TBQ as VRAM opt-in only. Production stock stays **q8_0** (see `serve_gpu_example.sh` / `rtx-5080.json`). Optional external lab: [craftogrammer/llama.cpp-adaptive-turboquant](https://github.com/craftogrammer/llama.cpp-adaptive-turboquant) `turbo2` (CUDA **12.9** + `GGML_CUDA_NO_MXFP4` rebuild available under `bench-5080-alpha`).

**Harness fix:** patch **0093** — `llama-bench` `-ctk/-ctv` now accepts Eliza TBQ/QJL/Polar names (was rejecting while `llama-server`/`common` already worked).

### Mac Metal A/B (Jul 2026, M4 Max) — **no-merge** (closes Apple gap)

Measured with `llama-bench` Metal FA on Llama-3.2-3B Q4_K_M. **Trust v2** (`tmp/metal-ab/v2/`, quiet GPU); discard `*.noisy.log`. RotorQuant binary: `../llama-cpp-rotorquant` @ `08e025c`.

| Leg | pp512 | pp2048 | tg128 | Finding |
|-----|------:|-------:|------:|---------|
| stock **f16** | **2115** | **1997** | **151** | Best on this Mac |
| stock q8_0 | 1850 | 1849 | 137 | ~0.91× tg — close, not a win |
| TBQ tbq4/tbq3 | 1521 | 1355 | 32 | FAIL merge tok/s (VRAM opt-in) |
| turbo3 (rotor) | 1189 | 942 | 43 | Same story as TBQ |
| planar3/f16 | 1087 | 865 | 64 | Best rotor compressed; still loses |
| planar3 | 380 | 125 | 43 | Prefill collapse (~0.06× pp2048) |
| iso3 | 353 | 100 | 29 | Worse than planar3 |
| QJL/Polar (pre-0097) | — | — | — | **Abort** — Metal `SET_ROWS` on `cache_v` |
| f16/q4_polar (D1) | 1709 | 1455 | 36 | **PASS** SET_ROWS; ~0.25× f16 tg |
| qjl1_256/q4_polar (D1b) | 934 | **350** | **37** | **PASS** after **0098** (~0.26× f16 tg; pp2048 ~0.18×) |

**Verdict:** Do **not** cherry-pick planar/iso for Metal. Stock KV on Mac 3B stays **f16** (5080’s q8_0 win does not transfer). TBQ/turbo3 stay VRAM opt-in. QJL `speed` runs after **0097+0098** but remains a tok/s FAIL merge.

---

## Lab B — BeeLlama server controls (product UX)

Source: [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp) ([features](https://github.com/Anbeeld/beellama.cpp/blob/main/docs/beellama-features.md)).

Local scout checkout (optional): `/tmp/llama-fork-scout/bee/beellama.cpp` (sparse `common` + `tools/server` + `docs`).

We already have DFlash + TBQ/TCQ-class KV. Bee adds **server-facing** controls we lack.

### Sized cherry-pick order (Jul 2026 scout)

| Priority | Feature | Surface | Size / risk | Verdict |
|----------|---------|---------|-------------|---------|
| **B0** | Reasoning-loop guard | `server-loop-guard.{h,cpp}` + CLI/schema + `process_token` wiring | **Landed as patch 0087** (default **off**; force-close/stop opt-in). Uses `reasoning_budget_tracking` + `process_token` (no Bee accept-callbacks on our pin). | **Done** — Mac lab `:18082` smoke PASS; optional CUDA sanity on 5080 |
| **B1** | Adaptive draft-max | `server-adaptive-dm.h` (**~1680 LOC**, mostly header) + `common.h` dm_* fields + **~58** hooks in Bee `server-context.cpp` | Bee `server-context.cpp` is **~8.7k** LOC vs our **~5.6k** — not a clean format-patch | **Dedicated port**, not drive-by; needs DFlash accept/reject telemetry we may lack |
| **B2** | DDTree / sampled draft | `--spec-branch-budget`, `--spec-draft-temp` | Tied to Bee DFlash tree path | After flat DFlash acceptance is solid |
| **B3** | CopySpec / suffix / recycle | `--spec-type copyspec` etc. | Separate speculative backends | Low–med |
| **B4** | Multi-GPU DFlash tape | Bee CUDA accept path | Dual-4090 only | Measure vs our speculative path first |

**Do not** wholesale merge Bee’s Turbo enum layout — remap or keep Eliza IDs.

### B0 — reasoning-loop guard (**0087**, landed)

**Usage (lab `llama-server` only — not production `:11434`):**

```bash
# force-close hidden reasoning when a loop is detected (needs think tags / budget sampler)
./build/bin/llama-server ... --reasoning-loop-guard force-close
# or stop the whole completion:
./build/bin/llama-server ... --reasoning-loop-guard stop
```

Per-request JSON: `reasoning_loop_guard`, `reasoning_loop_min_tokens`, `reasoning_loop_window`, …

**Adaptation vs Bee:** Bee hooks sampler accept-callbacks; we check in `process_token` after each emitted token while the reasoning-budget sampler is in `COUNTING`. Requires chat path to populate `reasoning_budget_start`/`end` tags (same as think budget). Default mode is **off**. Native `/completion` JSON includes `stop_detail` + `loop_guard.{triggered,action,reason}` when triggered.

**Also in 0087:** replace stale `check_no_mtmd` in `SLOT_SEQ_COPY` with `check_slot_no_media` (compile fix; matches save/restore/erase).

**Exit:** Qwen3 think model that previously looped force-closes or stops within window; no regression on non-reasoning chat when left at default `off`.

### Lab A sibling checkout (Mac scout / CUDA build host)

```bash
# Already cloned (shallow) at ../llama-cpp-rotorquant @ feature/planarquant-kv-cache
# CUDA A/B must run on the 5080/dual-4090 host — this Mac has no NVIDIA.
ROTORQUANT_LLAMA_SERVER_BIN=../llama-cpp-rotorquant/build/bin/llama-server \
CUDA_LLAMA_MODEL=/path/to.gguf \
  ./scripts/phase/l2_rotorquant_ab.sh
```

### B1 — adaptive DM (why it’s heavy)

- Controller logic is concentrated in `tools/server/server-adaptive-dm.h` (`profit` / `fringe`, EWMA, probe/off-dwell).
- Runtime effect is **not** a drop-in: Bee overrides `get_n_draft_max`-equivalent paths using DFlash cycle stats (`adaptive_n_max`, profit keys, continuation preserve).
- Our pin already has static `--spec-draft-n-max` + slot `get_n_draft_max()` (ctx/remaining clamp only) in `tools/server/server-context.cpp` — the integration seam is there, but Bee’s hooks assume richer DFlash metrics.

**Minimum viable port (if we proceed):**

1. Add `dm_*` fields to `common_params_speculative` + CLI (mirror Bee names).
2. Copy `server-adaptive-dm.h` as-is (header-only helpers).
3. Inherit/compose state on `server_slot`; call update on draft accept/reject with **acceptance rate + cycle time** we already log.
4. Gate behind `--spec-dm-adaptive` default **off** until A/B on 5080 with `eliza-*-dflash`.

**Defer** B1 until a clear DFlash acceptance win (Lab A RotorQuant closed no-merge on both 5080 + Mac — don’t stack B1 on top).

Lineage Bee cites: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant), [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp).

---

## Lab D — Mac Metal Polar SET_ROWS + asymmetric K/V (Apple)

**Root cause (Metal Lab A):** `FORK_PROFILE=speed` is `qjl1_256` K + `q4_polar` V. Pre-0097: Metal `SET_ROWS` allowed TBQ + QJL K but **not** `Q4_POLAR` V. Also: `GGML_METAL_EMBED_LIBRARY=ON` only embeds `ggml-metal.metal` text — eliza-shipped QJL SET_ROWS `.air` never loaded.

| Next | Why |
|------|-----|
| **D0** Asymmetric bench: `-ctk f16\|q8_0 -ctv tbq3_0` (TheTom: V≈free **quality**) | **Done (quiet):** f16/tbq3 tg **45** (~0.30× f16); f16/tbq4 tg **43**; q8/tbq3 tg **34**. V-only TBQ still hammers Metal decode — “free” is PPL, not tok/s |
| **D1** Metal Polar + QJL SET_ROWS (**0097**) | **Done:** `f16/q4_polar` smoke PASS (tg ~33); QJL encode in embed |
| **D1b** Wire Metal `fused_attn_qjl_polar` (**0098**) | **Done (quiet v3):** `qjl1_256/q4_polar` pp512 **934** / pp2048 **350** / tg **37** (~0.26× f16). Tok/s FAIL — do not advertise as speed win |
| **D2** Default Mac fork advice | Prefer `FORK_PROFILE=vram` (TBQ) for RAM only; `speed` is experimental / tok/s FAIL; don’t expect asymmetric TBQ-V to save tg |

---

## Lab C — Blackwell TQ3 weight path (5080-only)

[turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3) — TQ3_4S GGUF with runtime **FP4 tile** prefill on `sm_120` / `sm_121`.

Only worth it if we adopt TQ3 weights. Orthogonal to KV RotorQuant. Env knobs: `GGML_CUDA_TQ3_4S_FP4*`.

---

## Explicit non-goals

| Fork / project | Why skip as pin |
|----------------|-----------------|
| ik_llama.cpp | CUDA MoE/TP ideas only; Metal second-class; upstream tensor split covers multi-GPU first |
| Kobold / Unsloth network forks | Product packaging |
| BigMoeOnEdge | Interesting **API-only** MoE streaming; watch for Anemll alternatives later |
| llama-swap | Control-plane proxy — fleet/LA ideas, not kernels |
| PrismML Q2_0 **g128** / `PQ2_0` | Fork-only / future type id — do not merge onto pin |
| PrismML ternary **g64** | On pin (`QK2_0=64` + CUDA **0082**). Use `*-Q2_g64.gguf`. Doc: [prism-ternary.md](./prism-ternary.md) |
| Dual llama-server (TTS + vendor) | Retire via **0099–0101** — [llama-server-unify.md](./llama-server-unify.md) |

---

## Lab Q — Dual Chunk Attention / Qwen long-ctx (Jul 2026)

**Claim:** Qwen 1M / official long-ctx needs DCA (3-way FA + DualChunk RoPE), not YaRN alone. Upstream ggml-org has no DCA; vLLM V0 backend removed; SGLang still has `dual_chunk_flash_attn` (**oracle only**).

| Piece | Status |
|-------|--------|
| Native DualChunk RoPE + 3× FA + LSE merge (Qwen2/2.5) | **In vendor** — hparams `dca.*`, `llama-dca.h`, `qwen2.cpp` `build_attn_dca`, FA `ggml_flash_attn_ext_set_lse`; patches **0095+** |
| GGUF `*.attention.dca.*` on convert | **0094** keys + convert stamp |
| Oracle: SGLang dense vs native logits | `scripts/dca_oracle_logits.py` (n=0 ≈ stock FA; n≥1 ≈ SGLang) |
| Serve path | **Stock llama-server / zerollama-runtime** with stamped GGUF |
| SGLang sidecar `inference=sglang` | **Lab / legacy only** — not product long-ctx |

**Product model:** native ggml DCA. Fail ship on oracle drift. Sparse deferred.

---

## Suggested operator sequence

1. **Mac (done):** pin `86d86ed4` through **0088**; B0 smoke; Metal L2 stock vs TBQ (FAIL merge — expected). Eliza #11612 Metal guards imported as **0096**.
2. **5080 (done Jul 2026):** RotorQuant/planar/iso + vendor TBQ A/B — **no-merge** (CUDA table above). Stock path stays **q8_0**. Patch **0093** unblocks fork names in `llama-bench`.
3. **Mac Metal Lab A (done Jul 2026):** Llama-3.2-3B planar/iso/TBQ/QJL A/B — **no-merge** planar/iso; stock **f16**.
4. Optional: adaptive-turboquant **turbo2** long-ctx lab (external binary; CUDA **12.9** / NVFP4 is 5080-only).
5. **Mac Lab D:** D1 SET_ROWS **0097** + D1b fused **0098** done (`speed` runs, tok/s FAIL); asymmetric TBQ-V already measured — no tg win.
6. **Qwen 1M / DCA:** native GGUF + patched llama-server; gate with `scripts/dca_oracle_logits.py` (SGLang = oracle recipe only).
7. Defer **B1 adaptive DM** until a clear DFlash acceptance win.
8. Revisit turbo-tan only with TQ3 models in the fleet.
9. **Later (optional):** Eliza PRs per [roadmap](#eliza-pr-roadmap-draft--do-not-open-yet) — **start with PR 0 upstream sync**, then feature PRs only if that lands.
