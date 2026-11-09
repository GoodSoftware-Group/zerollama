# Core Serving Roadmap

This file holds the detailed backlog for the **core serving path**: local inference convergence, runtime shape, and product-track Apple serving work.

Research-track serving experiments live in [../RESEARCH_ROADMAP.md](../RESEARCH_ROADMAP.md).

---

## Product model: queues, stakeholders, and GPU time

Zerollama is a **GPU-backed batching system** with several request surfaces: local CLI/library use, OpenAI-compatible HTTP, agents and integrations, optional **Eliza cloud** merge, and the local **Python runtime** path when enabled. Those surfaces feed **inference work** that is admitted and scheduled across the **Go scheduler** and the **Python runtime scheduler**.

**Training** remains a **separate job queue** (`/api/train/*`, embedded `training.py`, optional TCP `:9500`) that shares VRAM with inference. The open product work is the **policy** on top of those queues: how inference, training, and optional idle-time background work should coexist on one machine.

**Fleet** and **Slurm** are separate layers on purpose:

- **Fleet** decides **which zerollama node** should take a request.
- **Slurm** remains the external **cluster allocator** at sites that already run it.

The operator rationale is in [../scheduling-vram-policy.md](../design/serving/scheduling-vram-policy.md), [../fleet-scheduling.md](../design/training/fleet-scheduling.md), and [../slurm-integration.md](../design/training/slurm-integration.md).

---

## Technology ladder (north star)

Directional stack, still useful as the architectural north star:

```text
Layer 0 (today)     Go daemon — registry, HTTP, Eliza proxy, embed lifecycle, ggml scheduler
Layer 1 (now→)      Python runtime — admission, PA bookkeeping, batching policy, training
Layer 2 (later)     Native hot path — GGUF decode, KV blocks, in-process llama.cpp / fork kernels
Edge (long-lived)   Thin API + pull + cloud
```

Open design goal: keep converging default GGUF serving toward **Go → llama-server** where that reduces merge pain and operational complexity, while leaving Python in place for admission, training, and experiments that are not yet ready to freeze into the hot path.

---

## Local inference — actionable phases

**History:** closed work for phases **8–10**, **12–14**, and **17** lives in [CHANGELOG](../../CHANGELOG.md) and the linked docs such as [../phase14-inprocess-llama.md](../design/serving/phase14-inprocess-llama.md), [../phase17-llama-server.md](../design/serving/phase17-llama-server.md), and [../upstream-ollama-diff.md](../upstream/ollama-diff.md).

| Phase | Status | Open work | Primary docs |
|-------|--------|-----------|--------------|
| **11** | **Partial** | Admission tuning, evidence-backed threshold polish, and operator clarity around when to keep or change current defaults | [../phase11-runtime-admission.md](../design/serving/phase11-runtime-admission.md), [../scheduling-vram-policy.md](../design/serving/scheduling-vram-policy.md) |
| **13 (polish)** | **Optional** | Clamp-on smoke and tokenizer-exact render cleanup called out as follow-on polish | [../phase13-runtime-vram.md](../design/serving/phase13-runtime-vram.md) |
| **15** | **Partial** | Finish tensor page bind / migration story and stabilize native decode batch across remaining memory types | [../phase15-native-kv.md](../design/serving/phase15-native-kv.md), [../handoff-phase15-native-kv.md](../handoffs/handoff-phase15-native-kv.md) |
| **16** | **Partial** | Finish edge binary cleanup and decide how far "Go gone" should really go for inference control plane | [../phase16-thin-edge.md](../design/serving/phase16-thin-edge.md) |

### Phase 15 — exit criteria (Partial)

Open criteria:

| # | Criterion | Status |
|---|-----------|--------|
| 5 | **Tensor page bind** — PA `block_ids` → llama KV tensor pages | **Partial** — donor buffers, layout probes, migration descriptors, and Metal validation exist; remaining blocker is a stable upstream writable KV page story across memory types |
| 6 | **Native decode batch** in C wired to `kv_forward_plans` | **Partial** — core batch path exists and sign-offs passed on Metal / 5080 / dual-4090, but long-term closure still depends on the page-bind story above |

**Operator gates:** `phase15_kv_native_ci.sh`, `phase15_inprocess_signoff.sh`, `phase15_metal_signoff.sh`.

### Phase 17 — upstream GGUF path alignment

Closed. Core alignment shipped; keep history in [../phase17-llama-server.md](../design/serving/phase17-llama-server.md), [../upstream-ollama-diff.md](../upstream/ollama-diff.md), and [CHANGELOG](../../CHANGELOG.md). Live follow-on work now sits under Phase 16 edge cleanup, L2 fork-profile evaluation, and adjacent llama-server tracks such as Flash-MoE.

### Phase 16 — exit criteria (Partial)

Open criteria:

| # | Criterion | Status |
|---|-----------|--------|
| 7 | Drop in-process ggml from edge binary | **Partial** — ggml CGO is out of the edge dep tree, but Python embed / MLX compile-time footprint remains |
| 8 | Inference control plane "Go gone" | **Not started** — current edge still keeps the API/control plane in Go and routes GGUF into llama-server / MLX |

---

## Python inference runtime (GGUF + PagedAttention)

This section stays as current-shape context, not as shipped history:

- Runtime remains the home for **PagedAttention bookkeeping**, **admission policy**, **training handoff**, and Phase 15 experiments.
- Default GGUF convergence work is now mostly about **how much Python remains in the hot path**, not whether the runtime exists.
- Open roadmap work is represented above under **Phases 11, 15, and 16**.

Reference docs: [../runtime-embed.md](../design/serving/runtime-embed.md), [../testing-smoke.md](../runbooks/testing-smoke.md), [../scheduling-vram-policy.md](../design/serving/scheduling-vram-policy.md).

---

## Apple Silicon & Metal track

**History:** closed milestones **M1–M15**, **M20–M23**, and **M27** live in [CHANGELOG](../../CHANGELOG.md) and the Apple/MLX docs.

Near-term product-track Apple work:

| Milestone | Status | Open work | Primary docs |
|-----------|--------|-----------|--------------|
| **M16** Flash-MoE via llama-server | **Partial** | Vendor pin merge and operational polish for slot-bank / SSD sidecar flow | [../flash-moe.md](../research/flash-moe.md) |
| **M24** GGUF MTP on ggml Metal | **Partial** | Mature the `ZEROLLAMA_GGML_MTP=1` path and decide when it is safe outside targeted labs | [../apple-silicon-metal.md](../design/apple/apple-silicon-metal.md) |

Research-track Apple items live in [../RESEARCH_ROADMAP.md#apple-silicon--metal-experiments](../RESEARCH_ROADMAP.md#apple-silicon--metal-experiments): **M17**, **M18**, **M19**, **M25**, **M26**.
