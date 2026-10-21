# DeepInfer borrowings (scheduling / speculation / SWA policy)

**Why this doc:** DeepSeek’s [DeepInfer](https://github.com/deepseek-ai/DeepInfer) is the production offline engine for DeepSeek-V4.1-Flash (Apache-2.0, Oct 2026). It is a vLLM-derived shell with a custom C++ scheduler and SM100/Ascend kernel stack. Zerollama does **not** vendor DeepInfer or run it as a backend. We keep a bring / watch / skip list so portable **serving ideas** land in **ggml / llama-server / Python runtime**, not a second safetensors EP fleet.

**Related:** [deepgemm-borrowings.md](./deepgemm-borrowings.md), [vllm-borrowings.md](./vllm-borrowings.md), [gpu-profiles-l3.md](./gpu-profiles-l3.md), [upstream-siblings.md](./upstream-siblings.md), runtime [SPECULATIVE.md](../runtime/docs/SPECULATIVE.md) (ngram / eagle3 / dflash / mtp → llama-server).

**Sources (read, do not depend on at serve time):**

| Tree | Upstream | Role for us |
|------|----------|-------------|
| DeepInfer | [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer) | Scheduler / MTP-varlen / YOCO / EPLB **ideas** |
| DeepInfer deploy Q&A | in-tree `doc/inference_deployment_qa.md` | Architecture narrative (KV store + LBs still closed) |
| Vendored kernels | FlashMLA, DeepGEMM, DeepEP, DeepSelect | Already covered as **skip / ideas-only** via DeepGEMM doc |

**Optional sibling:**

| Host | Path |
|------|------|
| Mac lab | `../DeepInfer` under `~/Sites/inference/` |
| astra | `/mnt/ssd2/src/DeepInfer` (cloned 2026-10-20; Python 3.12 venv at `.venv`) |

**Last checked:** 2026-10-20 — tip shallow clone + vendor submodules; hardware probe on astra 2×4090 → SM89, `is_sm_major_10=False`.

---

## Non-goals (explicit)

| Item | WHY not in zerollama |
|------|----------------------|
| `pip install` DeepInfer / FlashMLA / DeepEP into runtime | Inference stays GGUF + `llama-server` / ollama-engine; Python is orchestration |
| NVIDIA **SM100-only** kernels as product path | Production is 5080 (`sm_120`) + Mac Metal; DeepInfer builds `sm_100a`/`sm_103a` only |
| Serving V4.1-Flash-1017 via DeepInfer | ~510 GB weights, ~614 GB VRAM, default EP=8; wrong scale for local product |
| MegaMoE EP / DeepEP RDMA / Engram cluster | Multi-node expert parallel; we serve single-GPU / UMA |
| Their closed KV store + P/D/VE load balancers | Not in OSS; do not pretend we have 3FS |
| Ascend / CANN path | No Ascend hardware |

---

## Already covered (skip re-implementing)

| DeepInfer theme | Where we already are |
|-----------------|----------------------|
| DeepGEMM / FP8–FP4 / MoE fuse ideas | [deepgemm-borrowings.md](./deepgemm-borrowings.md) |
| Prefix / SWA / hybrid / Radix KV | [vllm-borrowings.md](./vllm-borrowings.md), L3 |
| Speculative decode + accept metrics | llama-server adaptive-dm, ANE dflash, MLX MTP gates, ggml MTP |
| Structured outputs | Ollama `format` / JSON schema (xgrammar-class problem already solved) |
| CUDA graph invalidation after KV change | [decode-graph-invalidation.md](./decode-graph-invalidation.md) |

---

## Bring (portable designs)

Delivery path is always **llama.cpp patches / runtime / Go scheduler**, not DeepInfer sources.

| Idea | DeepInfer locus | Why it matters | Zerollama stance |
|------|-----------------|----------------|------------------|
| **Prefill ALL/ANY cadence** | `DEEPINFER_PREFILL_ATTEMPT_INTERVAL` / `…_ANY_…` in `scheduler.hpp` | Collective prefill when all ranks busy vs starve protection when any rank needs it. | **DI7 done (analogue)** — opt-in *slot-launch* defer while decode-hot (patch **0140**; not DP vote / not exclusive steps). **Not** PA `tick()` admit. |
| **Async schedule discipline (ZOS + sync guard)** | `csrc/engine/zos.*`, `utils.forbid_cuda_sync`, `DeferredRelease` | Overlap CPU `schedule(N+1)` with GPU `forward(N)`; forbid accidental host sync in hot path. | **DI6 done** — inventory in [deepinfer-ladder.md](./deepinfer-ladder.md): `yield_to_queue` already overlaps; logits/sampler syncs required; no ZOS/`forbid_cuda_sync` port. |
| **SWA bounded re-prefill (YOCO compact)** | `DEEPINFER_YOCO_PREFILL_*`, `worker/cache_engine.py` StateBlock, `v4_modules/yoco.py` | On consumer SWA recompute, keep suffix ≥ `window + max_backtrack` instead of full prompt. | **DI4 policy done** — keep-token budget helper; **non-goal:** whole-prompt truncate on hybrid GGUF (YOCO is layer-aware) |
| **Batch-invariant eval mode** | `DEEPINFER_BATCH_INVARIANT`, FlashMLA split-KV off, fixed `seq_seed` | Goldens must not drift with co-batching. | **DI5 done (eval hygiene)** — force greedy+seed; deny cache_prompt/Radix/auto-batch; serialize `generate_batch`; not kernel batch-invariance |
| **Confidence-scheduled draft budget (DSpark / MTP varlen)** | `tilelang_ops/schedule_mtp_varlen.py`, `csrc/engine/mtp_varlen.hpp`, confidence head in `models/mtp/dflash.py` | Fixed draft depth wastes verify; allocate batch `total_q` by predicted accept. | **DI2 design done** — deliver via adaptive-dm; predicted confidence ≠ verified accept; **DI3** spike parked |

---

## Watch

| Signal | Action |
|--------|--------|
| EPLB greedy rebalance (`model_executor/eplb/policy.py`) | Keep algorithm notes for future multi-GPU MoE; **not** product on 5080/UMA |
| Decode perf cache (shape → measured SPS) | Rematch against L1 autotune profiles before adding a second cache |
| `EngineType` PROMPT / COMPLETION | PD-disagg interface shape only; no cluster glue |
| Multimodal shared-memory pool | We already defer covered MM encode; pool sizing is fleet-scale |
| FlopsEstimator at schedule time | Nice for MFU logs; Phase 11 VRAM opinion stays primary admission |
| FlashMLA / DeepGEMM tip changes that add **sm_120** | Only then revisit kernel ideas for 5080; still no vendor-into-runtime |

---

## Skip

| Item | Why |
|------|-----|
| FlashMLA / DeepGEMM / DeepEP / DeepSelect wheels | SM100 (or Ascend); see [deepgemm-borrowings.md](./deepgemm-borrowings.md) |
| Engram / Lightning indexer / MegaMoE / mHC | V4- or fleet-specific |
| DeepInfer as `modality_backends` / runtime engine | Wrong substrate and hardware |
| Online KV store, P/D/VE LBs, deepseek-recipe as serve core | Out of tree or wrong product layer |

---

## Hardware reality (do not “just try serve”)

| Need | DeepInfer | Zerollama hosts |
|------|-----------|-----------------|
| NVIDIA arch | **SM100** (`is_sm_major_10`) | 4090=SM89, 5080=SM120, H200=SM90 — **none match** |
| Model | V4.1-Flash-1017 only (OSS) | Local GGUF zoo |
| Parallelism | EP/DP scale-out (doc: 8-GPU prefill, 64–128 decode) | Single GPU / small multi-GPU, UMA |

---

## Operator / agent checklist

When someone links DeepInfer or the DeepSeek “open-source inference engine” post:

1. Ask: **portable schedule/spec/SWA idea** or **run DeepInfer here**? Only the former can land.
2. Map the claim to **Bring / Watch / Skip** above — if it is SM100 kernels or EP fleet, stop.
3. Prefer delivery via llama-server adaptive-dm, L3 SWA policy, or runtime PA — not a new PyTorch engine.
4. Never bind DeepInfer to production `:11434` / `:8081` or start it on the PVE hypervisor.

---

## Ladder

Ordered rungs **DI0–DI8** (ROADMAP **M29**): [deepinfer-ladder.md](./deepinfer-ladder.md).

| Done | Next |
|------|------|
| DI0–DI1; DI7 (**0140**); DI6; DI4; DI5; DI2 design | **DI3** spike parked; DI4 hybrid wire non-goal |
