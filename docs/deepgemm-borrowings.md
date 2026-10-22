# DeepGEMM borrowings (GEMM / MoE kernel ideas)

**Why this doc:** DeepSeek’s [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (CUDA, CUTLASS-inspired) and the Ascend port [DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend) publish a clean menu of **near-peak GEMM + MoE** techniques. Zerollama does **not** run Ascend NPUs and does **not** vendor DeepGEMM. We keep a short bring / watch / skip list so FP8/FP4 and MoE work stays aimed at **ggml / llama-server (CUDA + Metal)**, not a second BLAS stack.

**Related:** [cuda-lanes.md](./cuda-lanes.md) (NVFP4 / MXFP4 / native FP8), [native-fp8-gguf.md](./native-fp8-gguf.md), [flash-moe.md](./flash-moe.md), [freetoken-moe-lab.md](./freetoken-moe-lab.md), [llama-fork-watchlist.md](./llama-fork-watchlist.md), [upstream-siblings.md](./upstream-siblings.md), [deepinfer-borrowings.md](./deepinfer-borrowings.md) (schedule/speculation — complementary, also no DeepEP).

**Findings (rematch):** [deepgemm-borrowings-findings.md](./deepgemm-borrowings-findings.md)

**Sources (read, do not depend on at serve time):**

| Tree | Upstream | Role for us |
|------|----------|-------------|
| DeepGEMM (CUDA) | [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | **Primary** idea donor — NVIDIA GEMM / MoE kernels |
| DeepGEMM-Ascend | [deepseek-ai/DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend) | Same API on Ascend 950; useful as a **design checklist**, not a port target |
| DeepJIT | [deepseek-ai/DeepJIT](https://github.com/deepseek-ai/DeepJIT) | JIT + kernel cache behind DeepGEMM |

**Optional sibling (Mac lab / CT):** `../DeepGEMM` — clone only when reviewing CUDA kernel patterns. **Do not** add Ascend/CANN/`torch_npu` to the product path.

**Last checked:** 2026-10-21 — README tip (Mega MoE, grouped contiguous/masked, FP8×FP4, Ascend news 2026-09-30, locality-domain #462). Host unit: `./scripts/phase/deepgemm_borrowings_unit.sh`.

---

## Non-goals (explicit)

| Item | WHY not in zerollama |
|------|----------------------|
| Ascend MAD / fractal layouts / Bisheng / CANN | No Ascend hardware; production is CUDA 5080 + Mac Metal |
| `pip install` DeepGEMM / DeepGEMM-Ascend into runtime | Inference GEMM stays in ggml/`llama-server`; Python runtime is orchestration |
| Claiming DeepGEMM on sm_89 / sm_120 | Upstream advertises **SM90 / SM100**; our lanes use ggml MMA paths ([findings](./deepgemm-borrowings-findings.md)) |
| Ascend scale-factor packing (`UE8M0` pairs → `int16`, MN-major) | **Different from NVIDIA**; do not copy Ascend SF layout into GGUF/CUDA |
| Mega MoE EP8 multi-rank fuse as a product feature | Multi-node expert parallel + symm memory; we serve single-GPU / UMA, not EP fleets |
| Lightning Indexer MQA-logits kernel | DeepSeek-indexer-specific; only if we ship that module |
| mHC (Manifold-Constrained Hyper-Connections) prenorm GEMM | Model-family specific; watch only if GGUF graphs need it |
| Claiming Ascend “99% of peak” numbers on 5080/Metal | Different ASIC; measure on our lanes |

---

## Useful to have (problem decomposition)

These are the durable takeaways. Delivery path is always **llama.cpp / our patches**, not DeepGEMM sources.

| Idea | Why it matters | Zerollama stance |
|------|----------------|------------------|
| **Shape-selected kernel config + disk JIT cache** | Big GEMMs win when tile/config is picked per `(M,N,K)` and compiled artifacts are reused | **Watch** — ggml CUDA graphs / Metal already cache; keep decode-graph invalidation honest ([decode-graph-invalidation.md](./decode-graph-invalidation.md)). No second JIT runtime. |
| **M-grouped GEMM + M/K alignment** | MoE: pack tokens per expert; pad to HW alignment so grouped matmul stays compute-bound | **Watch in llama.cpp** — MoE `mul_mat` / expert packing; document alignment footguns in MoE smokes when we hit them (`get_mk_alignment_for_contiguous_layout` is the *idea*, not an API we call) |
| **Masked grouped GEMM for decode graphs** | CPU may not know per-expert token counts under CUDA graph | **Watch** — ggml MoE + decode-graph epoch; no DeepEP |
| **Scale-factor (SF) layout as first-class** | FP8/FP4 throughput is layout-bound; packing and majorness differ per ASIC | **Taken directionally** — NVFP4 / MXFP4 / native FP8 GGUF paths ([cuda-lanes.md](./cuda-lanes.md), [native-fp8-gguf.md](./native-fp8-gguf.md)). Treat SF transforms as part of weight load, not afterthoughts. |
| **Fuse MoE path: dispatch → 2× grouped GEMM → SwiGLU → combine** | Cuts HBM round-trips vs discrete kernels | **Watch** — prefer fused MoE/FFN landings via llama.cpp / L2 fork; Flash-MoE covers **SSD expert residency**, not Mega MoE EP fuse ([flash-moe.md](./flash-moe.md)) |
| **Boundedness honesty (compute vs pipe vs HBM)** | DeepGEMM Mega MoE benches report TFLOPS **and** interconnect GB/s | **Taken as measurement habit** — prefill vs decode on 5080/Metal: report which limit you hit, not only TFLOPS |
| **Stable multi-backend API over MAD/CUTLASS** | Same DeepGEMM Python API on CUDA vs Ascend | **Skip as a product** — ggml backends already play this role; do not add a DeepGEMM façade |

---

## Already covered (skip re-implementing)

| DeepGEMM theme | Where we already are |
|----------------|----------------------|
| Dense low-precision GEMM (FP8 / FP4 weights) | GGUF NVFP4 / MXFP4 / FP8_E4M3 / FP8_E5M2 on CUDA; Metal NVFP4 patches; MLX MXFP4/NVFP4 on create path |
| MoE larger than RAM | Flash-MoE sidecar + FreeToken MoE lab (policy sim) — different problem than MegaMoE EP |
| Per-shape CUDA graph / kernel reuse | ggml CUDA graphs; L3 decode-graph epoch + invalidate |
| Expert packing / grouped work | ggml MoE paths + runtime MoE models; not DeepGEMM grouped API |
| Dual-4090 NVFP4 correctness | cuda-lanes **P1 Done** — generic MMQ on sm_89; Blackwell-native remains **P2** |

---

## Watch (upstream / llama.cpp)

| Signal | Action |
|--------|--------|
| Upstream DeepGEMM CUDA APIs for **grouped GEMM** / fused MoE | Rematch against ggml MoE `mul_mat` and any eliza fork kernels — borrow **ideas**, not the Python extension |
| llama.cpp PRs: fused SwiGLU+GEMM, better expert packing, Blackwell FP8/NVFP4 MMA | Prefer pin/patch path ([LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md), [llama-fork-watchlist.md](./llama-fork-watchlist.md)); 5080 sm_120 native MMA remains CUDA-lanes **P2** |
| DeepSeek GGUF families that need Lightning Indexer or mHC | Only then revisit MQA-logits / HC prenorm as **optional** ops — not a default serve dependency |
| Ascend-only releases | Ignore for product; skim README for new **operator names** (grouped SF, paged MQA metadata) that might appear later on CUDA DeepGEMM |

---

## Operator / agent checklist

When someone links DeepGEMM or DeepGEMM-Ascend:

1. Ask: **CUDA DeepGEMM idea** or **Ascend port news**? Only the former can affect our stack.
2. Check arch: if the claim needs **SM90/SM100 DeepGEMM wheels**, stop — we ship ggml on sm_89 / sm_120.
3. Map the claim to a row in **Useful to have** — if it is Ascend MAD/SF packing/EP/Mega MoE, stop.
4. If it is FP8/FP4 or MoE fuse, check [cuda-lanes.md](./cuda-lanes.md) + llama.cpp tip before opening a zerollama feature issue.
5. Never start Ascend toolchains or bind DeepGEMM into `:8080` / `:8081` / production ports.

---

## Citation (if we reference publicly)

Prefer citing **DeepGEMM** for CUDA technique discussion; cite Ascend only when discussing the NPU port itself. Upstream bibtex lives in the DeepGEMM README (MIT).
