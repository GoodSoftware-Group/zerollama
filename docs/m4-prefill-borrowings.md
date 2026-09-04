# m4-prefill-engine borrowings (Apple Silicon Metal prefill)

**Status:** **closed for speed work** — durable win is **MTL FA recognition** + lab **FA/`q8_0` KV** (stock kv_f16). Native Q8 FA + fused SwiGLU wired opt-in but **regress on M4 Max**; P2–P3 staged only.  
**Upstream:** [mohamedhossammohamed/m4-prefill-engine](https://github.com/mohamedhossammohamed/m4-prefill-engine) (Apache-2.0, © 2026 Mohammed Hossam)  
**Sibling checkout (Mac lab):** `../m4-prefill-engine`  
**Staged / live sources:** [`ml/backend/ggml/ggml/src/ggml-metal/m4-prefill-shipped/`](../ml/backend/ggml/ggml/src/ggml-metal/m4-prefill-shipped/)  
**Findings:** [m4-prefill-borrowings-findings.md](./m4-prefill-borrowings-findings.md)

Cite the upstream project if these ideas ship in docs or releases (author request in their README).

## Why this exists

Research PoC claims **~3.4–3.7×** prefill on a **1B** Q4_0 transformer on **M4 MBA**, measured with Metal GPU timestamps (kernel time only). Wins come from fused Gate/Up SwiGLU, Q8 KV + flash-attn that reads Q8, and head-major QKV (skip transpose) — not from reinventing stock `mul_mm`.

**Why not vendor the whole engine?** Custom host layout, hardcoded `D=64`, not GGUF. Stage kernels under `m4-prefill-shipped/` the same way Eliza shaders live under `eliza-shipped/`.

## Already covered in zerollama (skip)

| Upstream | Ours |
|----------|------|
| Baseline Q4_0 GEMM | `kernels/mul_mm.metal` `kernel_mul_mm_q4_0_*` |
| FP16 FlashAttention | `kernels/fa.metal` / `ggml-metal-embed-fa.metal` |
| Q8_0 KV write + FA templates | `SET_ROWS` + `kernel_flash_attn_ext_q8_0_*` (stock) |
| Standalone SwiGLU | `kernels/unary.metal` `kernel_swiglu_*` |
| ANE FFN SwiGLU fuse | `ane_ffn_swiglu_fuse*` (different backend) |

## Patch backlog

| Patch | Priority | Status | Notes |
|-------|----------|--------|-------|
| **0122** | P0 | wired (opt-in, **regresses** on 0.5B) | PoC fused kernel vs stock simdgroup `mul_mm` |
| **0123/0124** | P1 | **partial (opt-in, +18% on 0.5B)** | Native Q8 FA on prefill; needs MTL FA recognition |
| **0125** | P2 | staged | `pipe_qkv_head_gemm_q4_0.metal` |
| **0126** | P3 | staged | `pipe_gemm_q4_0_32x32.metal` — A/B only |

## Lab A/B (2026-09-01, M4 Max)

Qwen2.5-0.5B **Q4_0**, 512-token prefill on `:11435` (`ZEROLLAMA_UMA_SCHED=off`):

| | Median prefill tok/s |
|--|--:|
| stock | **4649** |
| P0 (fused SwiGLU) | 3884 (−16%) |
| **P1 only** (FA + `q8_0` KV + `Q8_KV=1`) | **5469 (+18%)** |
| P0+P1 | 3532 (−24%) |

**MTL FA gate:** discovery reports `Library=MTL`; older `FlashAttentionSupported` only matched `Metal`, which disabled FA and cleared quantized KV. Fixed via `ml.MetalLikeLibrary` (`discover` reuses it for UMA skip).

### 1.5B Q4_0 (Qwen2.5-1.5B)

| | Median prefill tok/s | Notes |
|--|--:|--|
| stock | **1589** | 1116 / **2070** / **2062** / 604† |
| **P1 only** | **1638 (+3%)** | stable ~1580–1710; FA+q8 telemetry OK |
| P0 only | 1265 (−20%) | still regresses |

† Dropping the 604 outlier → stock median ≈ **2062**, so P1 is ~**−21%** vs hot stock. Native Q8 FA helps small models more; larger prefill often prefers stock Q8→F16 FA (llama.cpp compute-bound tradeoff).

### Enable lab FA+q8 (ports only)

```bash
./scripts/phase/m4_prefill_lab_serve.sh
# or:
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
ZEROLLAMA_M4_PREFILL_TELEMETRY=1 \
ZEROLLAMA_UMA_SCHED=off \
OLLAMA_HOST=127.0.0.1:11435 ./zerollama serve
```

Do **not** set `ZEROLLAMA_M4_PREFILL_Q8_KV` for speed — paired A/B shows native Q8 FA is slower than stock kv_f16 on M4 Max. Use `Q8_KV=always|auto` only for experiments.
Leave `ZEROLLAMA_M4_PREFILL_SWIGLU` unset unless testing P0 (known slower).

Never bind production `:11434` / `:8081`.

**Useful path:** FA on + `q8_0` KV + stock kv_f16 FA (needs MTL FA recognition).  
**P1 native skip:** experimental only.  
**P0 gates:** dense Q4_0 gate+up, F32 act/dst, contiguous, `seq≥32`, ic/hidden % 32 == 0.

## Why not ship the staged Q8 FA `.metal` yet

PoC `flash_attn_q8_0_causal` hardcodes **D=64**, half Q, `[H,M,D]`, causal-only (no ggml mask/sinks/kargs). Stock `kernel_flash_attn_ext_q8_0_*` already covers general heads — the missing win was **not using it on large prefill** (kv_f16 path). That opt-in is what 0123/0124 ship.

## Constraints / closed

- **Durable win:** MTL FA recognition (`ml.MetalLikeLibrary`) so FA + quantized KV actually stick on Apple Silicon.
- **P1 native Q8 FA:** paired FA+q8 A/B on 0.5B shows **−5…−10%** vs stock kv_f16 — experimental (`Q8_KV=always|auto`), **off by default**. Lab script enables FA + `q8_0` only.
- **P0** fused SwiGLU lab-only (regresses vs stock `mul_mm`).
- P2–P3: **deferred** (no wire without a new positive A/B).
- Lab: `./scripts/phase/m4_prefill_lab_serve.sh` — never `:11434` / `:8081`.
- Placeholder stubs `llama/patches/0122–0126-ggml-metal-m4-prefill-*.patch` only touch `.todo` files; real wire is in-tree Metal hook / ops (not a llama.cpp vendor quilt).

## Do not take

- Whole `unified_prefill_engine.mm` host
- Roofline / naive-attn baselines
- Claiming their 3.7× vs **our** ggml baseline without a same-binary A/B (P0 regresses; native Q8 FA loses to stock kv_f16 on M4 Max)
