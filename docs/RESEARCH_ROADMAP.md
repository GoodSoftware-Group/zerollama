# Research Roadmap

This file tracks **exploratory, lab, and borrowings work** that may inform the product roadmap later.

- **Near-term product and operator-facing work** lives in [ROADMAP.md](ROADMAP.md).
- **Detailed product backlogs** live in [roadmaps/](roadmaps).
- **Cross-repo watchlist** lives in [upstream-tracker.md](upstream/tracker.md).
- **Shipped / historical detail** belongs in [CHANGELOG](../CHANGELOG.md).
- Use this file for **upstream sibling rematches**, **lab-only platform bets**, and **future-native parity work** that is not yet the main product focus.

---

## Apple Silicon & Metal experiments

These Apple items remain active, but they are better treated as research-track work than as the main product backlog right now.

| Milestone | Status | Open work | Primary docs |
|-----------|--------|-----------|--------------|
| **M17** ANE probe | **Partial** | Keep subprocess reachability probe useful without turning it into a misleading product promise | [ane-probe.md](research/ane-probe.md) |
| **M18** ANE in-process dflash draft | **Partial (lab)** | Improve Metal draft acceptance and complete deeper ANE draft-path pieces (attn/KV/lm_head) | [ane-draft-inprocess.md](research/ane-draft-inprocess.md) |
| **M19** Optional Mac MLX build (`mlx-cgo`) | **Not started / eval** | Pin policy, smoke against upstream, `libjaccl` bundle story, drift tracking | [mac-dev-setup.md](runbooks/mac-dev-setup.md) |
| **M25** MLX MoE gate/up fuse when it fits | **Not started** | Conditional per-layer fuse when free UMA headroom allows it | [apple-silicon-metal.md](design/apple/apple-silicon-metal.md) |
| **M26** MLX DeepSeek-V4 Flash | **Partial** | Finish confidence around the graph/link path and decide how much operator support belongs in-tree | [mlx-deepseek-v4-flash.md](research/mlx-deepseek-v4-flash.md) |

Product-track Apple items stay in [ROADMAP.md#apple-silicon--metal-track](ROADMAP.md#apple-silicon--metal-track).

### Optional Mac MLX build path (mlx-cgo)

Still evaluation-only. Keep the open questions in **M19** above; do not treat this as the default Mac build path.

---

## Local voice & llama borrowings (eliza-v3)

**History:** L1, L2, L3 and the original voice milestones are closed; see [CHANGELOG](../CHANGELOG.md), [gpu-profiles-l1.md](design/serving/gpu-profiles-l1.md), [gpu-profiles-l2.md](design/serving/gpu-profiles-l2.md), [gpu-profiles-l3.md](design/serving/gpu-profiles-l3.md), and [mlx-agent-prompts.md](design/apple/mlx-agent-prompts.md).

Open follow-ons:

| Item | Status | Open work |
|------|--------|-----------|
| **L2 follow-on** | **Lab** | RotorQuant A/B on 5080 before any thought of a default flip; defaults stay L1 |
| **FL1** | **Next** | Turn-boundary prefix checkpoints under memory pressure |
| **FL2** | **Next** | Clear operator KV-limit mental model (`memory` / `context` / `concurrency`) in existing env/YAML terms |
| **FL3** | **Next** | Better host-vs-GPU prefix reuse metrics when disk/LMCache contributes |

### Radix v2 (L3-R) — product gaps

Core Radix v2 work is closed. Open follow-ons only:

- Optional **NIXL / Mooncake / RDMA** style federation beyond today’s blob pull.
- Optional **non-agent default-on** posture if operator evidence justifies it.

Keep the actual history and product-gap narrative in [radix-prefix-share.md](design/serving/radix-prefix-share.md).

---

## LocalAI control-plane borrowings (Jun 2026)

Closed except the current upstream watch:

- **LA12 / PII** remains an upstream watch item in [localai-borrowings.md](upstream/localai-borrowings.md#candidates-la11--suggested-priority).

The closed LA1–LA22 history stays in [localai-borrowings.md](upstream/localai-borrowings.md) and [CHANGELOG](../CHANGELOG.md).

---

## Option 2 — Narrow the gap without SGLang (in-tree over time)

**History:** the shipped Tier 1 borrowings belong in [CHANGELOG](../CHANGELOG.md) and [sglang-multimodal-borrowings.md](upstream/sglang-multimodal-borrowings.md). This section keeps only the remaining native-gap work.

| Phase | Status | Open work |
|-------|--------|-----------|
| **A** | **Partial** | Decode and sampling policy parity for target models |
| **B** | **Partial** | Renderer / template semantics for clip-vs-image distinction |
| **C** | **Partial** | Token-aware context budgeting and actionable failure modes |
| **D** | **Partial** | Goldens and per-model validation gates |
| **E** | **Partial** | Keep any Python bridge narrow and scoped to decode / preprocess only |

Open follow-ons worth stealing next:

- More complete **video agent infer** ship-hardware sign-off
- Any missing **preprocessed metadata consume** paths on runners that still lag
- **Cross-runner ViT / encoder cache federation** only if real workloads justify it

Reference: [video-parity.md](design/media/video-parity.md), [sglang-multimodal-borrowings.md](upstream/sglang-multimodal-borrowings.md).

---

## Promotion rule

When a research item becomes a **near-term product focus** with clear operator value, move it back into [ROADMAP.md](ROADMAP.md). Keep this file for work that is still primarily exploratory, lab-only, or upstream-borrowing in nature.
