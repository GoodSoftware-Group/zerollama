# llama.cpp fork watchlist (Jul 2026)

**Audience:** Contributors deciding what to cherry-pick onto the unified ggml-org pin (`86d86ed4` + `llama/patches/`).

**Not a pin swap.** Keep one vendor tree. Treat external forks as **diff mines** and lab binaries.

Related: [gpu-profiles-l2.md](./gpu-profiles-l2.md), [flash-moe.md](./flash-moe.md), [runtime/docs/SPECULATIVE.md](../runtime/docs/SPECULATIVE.md).

---

## Already on our pin

| Capability | Where |
|------------|--------|
| DFlash draft + `--spec-type dflash` | Patches + speculative docs |
| TBQ3/4 + TBQ3_TCQ KV | L2 profiles (`tbq4_0`/`tbq3_0`, `tbq3_tcq`) |
| QJL / Polar KV | L2 `speed` profile |
| Metal + CUDA SET_ROWS / FA for TBQ | Mid-series patches |
| Anemll Flash-MoE (separate binary) | [flash-moe.md](./flash-moe.md) / ROADMAP M16 |

Enum IDs are **Eliza-specific** (`TBQ3_0=44` …). Do not assume Bee/TheTom GGUF type IDs match.

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

**Verdict:** Do **not** cherry-pick planar/iso. Keep TBQ as VRAM opt-in only. Optional external lab: [craftogrammer/llama.cpp-adaptive-turboquant](https://github.com/craftogrammer/llama.cpp-adaptive-turboquant) `turbo2` for long-ctx (built here on CUDA 12.8 + `GGML_CUDA_NO_MXFP4`).

**Harness fix:** patch **0093** — `llama-bench` `-ctk/-ctv` now accepts Eliza TBQ/QJL/Polar names (was rejecting while `llama-server`/`common` already worked).

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

**Defer** until RotorQuant Lab A results are in (don’t stack two large vendor diffs).

Lineage Bee cites: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant), [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp).

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
| PrismML Q2_0 g128 | Upstream Q2_0 g64 already moving |

---

## Suggested operator sequence

1. **Mac (done):** pin `86d86ed4` through **0088**; B0 smoke; Metal L2 stock vs TBQ (FAIL merge — expected).
2. **5080 (done Jul 2026):** RotorQuant/planar/iso + vendor TBQ A/B — **no-merge** (see table above). Stock path stays **q8_0**. Patch **0093** unblocks fork names in `llama-bench`.
3. Optional: adaptive-turboquant **turbo2** long-ctx lab (external binary); CUDA **12.9** if chasing NVFP4 / their Windows-tuned path.
4. Defer **B1 adaptive DM** until a clear DFlash acceptance win.
5. Revisit turbo-tan only with TQ3 models in the fleet.
