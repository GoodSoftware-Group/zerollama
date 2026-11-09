# m4-prefill borrowings — findings

Audit trail for staging + wire of [m4-prefill-engine](https://github.com/mohamedhossammohamed/m4-prefill-engine).

## 2026-09-01 — stage only

- Cloned sibling to `../m4-prefill-engine` (Apache-2.0).
- Extracted production kernels from `unified_kernels.metal` into
  `ml/backend/ggml/ggml/src/ggml-metal/m4-prefill-shipped/` (+ upstream `LICENSE`).
- Skipped probes, naive attn, and `llamacpp_style_mul_mm_q4_0` (baseline only).

## 2026-09-01 — P0 wire (0122)

- Added `m4-prefill-shipped/m4_fused_swiglu.metal` (`fused_gate_up_swiglu_q4_0_f32`):
  F32 activations/dst, ggml `block_q4_0` layout, dual half4 A loads (no `half8`).
- Registered `X(M4_FUSED_SWIGLU, m4_fused_swiglu)` in `GGML_METAL_LIBS`.
- Embed script copies `m4-prefill-shipped/`; CMake links AIR for non-embed builds.
- `m4_prefill_metal_hook.cpp` intercepts `ggml_metal_op_mul_mat` **before** ANE:
  `ane_ffn_swiglu_fuse_match` → encode fused kernel → skip through GLU (down stock).
- Opt-in: `ZEROLLAMA_M4_PREFILL_SWIGLU=1` (+ optional `ZEROLLAMA_M4_PREFILL_TELEMETRY=1`).
- Mac build regenerated 33 embeds; binary contains kernel name + env strings.
- **Follow-ups still open:** F16 act, scales, holey MoE. Lab A/B done below.

## 2026-09-01 — P1 partial (0123/0124)

- Stock Metal already has Q8_0 `SET_ROWS` + `kernel_flash_attn_ext_q8_0_*`.
- Prefill `nq≥32` was forcing Q8→F16 scratch then F16 FA (llama.cpp #27390 tradeoff).
- Shipped `ZEROLLAMA_M4_PREFILL_Q8_KV=1` in `ggml_metal_op_flash_attn_ext_use_kv_f16`:
  for `GGML_TYPE_Q8_0` return false → keep native Q8 FA.
- **Did not** wire staged `quantize_kv_to_q8_0.metal` / `flash_attn_q8_0_causal.metal`
  (D=64 / layout / mask blockers). Kept as reference under `m4-prefill-shipped/`.
- Operator pair: `OLLAMA_KV_CACHE_TYPE=q8_0` + FA on + lab port only.

## 2026-09-01 — Lab A/B (M4 Max, `:11435`)

- **Model:** `m4-lab-qwen05-q4_0` (Qwen2.5-0.5B-Instruct **Q4_0**, bartowski HF GGUF). Local eliza/Qwen3.5 labs are **Q4_K** → P0 gate never fires.
- **Host:** Apple M4 Max; Metal OK after unsandboxed serve. Need `ZEROLLAMA_UMA_SCHED=off` or loads hang (`HOLD_GPU`).
- **Method:** `go run ./cmd/bench` — 4 epochs, warmup 1, `prompt-tokens=512`, `max-tokens=32`, `num-ctx=2048`, raw. CSVs under `.cache/m4-prefill-lab/bench/`. Bench `ttft` column stayed 0; use prefill tok/s / NS_PER_COUNT.
- **First P0+P1 run was invalid for P1:** `FlashAttentionSupported` only matched `Library=="Metal"`, but discovery reports **`MTL`**. Log: `flash attention enabled but not supported by gpu` → FA forced off → quantized KV stripped → F16 KV, no `native Q8_0 FA` telemetry.

| Config | Median prefill tok/s | vs stock | Notes |
|--------|---------------------:|---------:|-------|
| stock | **4649** | — | high epoch variance |
| P0 only (`SWIGLU=1`) | 3884 | **−16%** | fused path fires |
| P0+P1 (pre-fix, FA never stuck) | 3727 | **−20%** | not a real P1 measurement |

## 2026-09-01 — MTL FA gate fix + re-A/B

- **Fix:** `ml.MetalLikeLibrary` treats `Metal`/`MTL`; `FlashAttentionSupported` + `MinimumMemory` use it. Test: `ml/device_flash_attn_test.go`.
- After rebuild, lab load shows `flash_attn=enabled`, `K/V (q8_0)`, telemetry `m4_prefill: native Q8_0 FA (skip kv_f16)`.

| Config | Median prefill tok/s | vs stock |
|--------|---------------------:|---------:|
| P1 only (FA + `KV=q8_0` + `Q8_KV=1`, no SWIGLU) | **5469** | **+18%** |
| P0+P1 after MTL fix (SWIGLU + FA + q8) | 3532 | **−24%** |

**Verdict:** **P1 (native Q8 FA) is a real win** on this 0.5B Q4_0 once MTL is recognized. **P0 fused SwiGLU still regresses** and drags P0+P1 below stock. Keep both opt-in; prefer documenting P1 as the useful lab knob. Do **not** default P0. P2 still deferred until P0 is fixed or a larger-model win appears.

## 2026-09-01 — 1.5B Q4_0 rematch (disk ~16 Gi free)

- **Model:** `m4-lab-qwen15-q4_0` (Qwen2.5-1.5B-Instruct **Q4_0**, bartowski; GGUF in `.cache/m4-prefill-lab/`).
- **Method:** same as 0.5B — 4 epochs, warmup 1, prompt≈512, max=32, num_ctx=2048, lab `:11435`, `ZEROLLAMA_UMA_SCHED=off`. CSVs: `stock-1p5b.csv`, `m4-p1-1p5b.csv`, `m4-p0-1p5b.csv`.
- **P1 telemetry OK:** `flash_attn=enabled`, `K/V (q8_0)`, `m4_prefill: native Q8_0 FA (skip kv_f16) nq=511 head=128`.

| Config | Prefill samples (tok/s) | Median | vs stock median |
|--------|-------------------------|-------:|----------------:|
| stock | 1116, **2070**, **2062**, 604† | **1589** | — |
| P1 only | 1582, 1682, 1710, 1594 | **1638** | **+3%** |
| P0 only | 1158, 1247, 1291, 1284 | **1265** | **−20%** |

† Stock epoch 4 (604) looks like interference; drop-min stock median ≈ **2062**. Against that clean peak, P1 (~1640) is about **−21%**. P1 runs were much more stable than stock.

**Verdict (1.5B):** Protocol median shows a tiny P1 edge, but **hot stock F16-FA epochs beat native Q8 FA**. Matches llama.cpp’s rationale for Q8→F16 scratch on larger prefill (compute-bound). **Do not default P1**; keep opt-in / size-sensitive. P0 still regresses. P2 still deferred.

## 2026-09-01 — P1 auto head gate

- **Change:** `ZEROLLAMA_M4_PREFILL_Q8_KV=1|auto` only skips kv_f16 when `head_dim ≤ ZEROLLAMA_M4_PREFILL_Q8_KV_MAX_HEAD` (default **64**). `always` / `force` restores unconditional native Q8 FA.
- **Telemetry:** native line when gate hits; one-shot `Q8_KV auto → stock kv_f16 (head=… exceeds max)` when it does not.
- **Lab script:** `scripts/phase/m4_prefill_lab_serve.sh` defaults to `Q8_KV=auto`.

| Config | Median prefill tok/s | Notes |
|--------|---------------------:|-------|
| stock 0.5B (no FA/q8) | **4860** | |
| P1-auto 0.5B (native, head=64) | **3545** | gate fires native; this run **slower** than no-FA stock (earlier always-on P1 was +18% — noisy / baseline-dependent) |
| stock 1.5B | **2006** | |
| P1-auto 1.5B | **1934** | gate correctly **falls back** to kv_f16 (head=128); ~parity with stock |

**Verdict:** Auto gate behaves as designed (native on head≤64, stock path on head=128). Keep opt-in; do not default. Revisit 0.5B win with paired FA-on baselines before promoting auto as a speedup.

## 2026-09-01 — Paired FA+q8 baseline (0.5B) — decisive

Same model `m4-lab-qwen05-q4_0`, both arms: `OLLAMA_FLASH_ATTENTION=1` + `OLLAMA_KV_CACHE_TYPE=q8_0` + UMA off. Only difference is native skip vs stock kv_f16. 6 epochs, warmup 2, prompt≈512.

| Arm | Prefill samples (tok/s) | Mean | Median |
|-----|-------------------------|-----:|-------:|
| A stock kv_f16 (`Q8_KV=0`) | 5648, 5759, 5768, 5709, 5664, 5751 | **5716** | **5730** |
| B native (`Q8_KV=always`) | 4626, 5475, 5536, 5429, 5421, 4566 | **5176** | **5425** |

**Delta native vs kv_f16:** mean **−9.5%**, median **−5.3%**. Telemetry: A had 0 native lines; B continuous `native Q8_0 FA … head=64`.

**Verdict:** On M4 Max, **native Q8 FA is slower than stock Q8→F16 FA** even at head=64. The earlier “+18% vs stock” was vs **no-FA** baseline, not a win for the native kernel. Keep `Q8_KV` experimental/off by default. Lab serve should enable **FA + q8_0 KV** without forcing native. P0 still regresses. P2 blocked. MTL FA recognition remains the durable fix.

## Closed (2026-09-01)

Speed track for borrowed m4-prefill kernels is **done** for now:

1. Lab default: FA + `q8_0` KV, **`Q8_KV` unset** → stock kv_f16 (`scripts/phase/m4_prefill_lab_serve.sh`).
2. `Q8_KV=auto|always` and `SWIGLU=1` remain experiment-only (measured regressions on M4 Max).
3. P2 head-major QKV / P3 double-buffer GEMM stay staged; do not wire without a positive same-binary A/B.
4. Reopen only if a new Metal Q8 FA (or fused FFN) kernel beats stock paths under paired FA-on baselines.

