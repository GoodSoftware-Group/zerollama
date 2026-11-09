---
title: Typed decisions
description: Structured triage and System-1 reads across Laya, CLM, and lab decider backends.
---

**Status:** `beta`

Typed decisions are the zerollama surface for **structured triage and System-1 style reads**: route, urgency, yes/no, calibrated choice, score, and noul-style answers.

## API surface

- `POST /v1/decisions`
- Alias: `POST /v1/systemone`

Use this when you want **typed outputs**, not free-form chat.

## Current backends

| Backend | Status | What it is good at | Deep docs |
|---|---|---|---|
| **Laya** | `beta` | One-forward-pass typed choice/score/noul with an act/escalate head | [laya-llama-cpp.md](../design/surfaces/laya-llama-cpp.md), [findings](../design/surfaces/laya-llama-cpp-findings.md) |
| **CLM** | `beta` | Qwen3-8B embeddings + small native Go heads; no Python at serve | [clm.md](../design/surfaces/clm.md) |
| **GLiNER2.5-Decide** | `lab` | Small-encoder zero-shot triage through a Python sibling | [gliner-decide.md](../design/surfaces/gliner-decide.md), [findings](../design/surfaces/gliner-decide-findings.md) |
| **OpenJev / DiffusionGemma** | `lab` | Structured readout through a sibling diffusion path; useful when you explicitly want that model family | [diffusion-gemma-llama-cpp.md](../design/surfaces/diffusion-gemma-llama-cpp.md), [findings](../design/surfaces/diffusion-gemma-llama-cpp-findings.md) |

## What is stable at the capability layer

- The client wire is the same basic shape across the feature: `{model, state, questions}`.
- Go owns the high-level packing/rendering so clients do not learn a different request schema for every backend.
- The surface is for **typed answers**, not a second chat endpoint.

## What varies by backend

- Calibration quality and confidence semantics
- Whether the backend is fully local, a sibling process, or a proxy path
- GPU / VRAM footprint
- How much operator setup is required

## Current guidance

- Prefer **Laya** when you want the cleanest typed-decisions-first path.
- Prefer **CLM** when you want native Go heads over an embeddings model.
- Treat **GLiNER2.5-Decide** and **OpenJev / DiffusionGemma** as opt-in lab surfaces.

## Not this capability

- **Not chat**: if you want free-form text, use chat/generate.
- **Not entity extraction**: use [extract.md](extract.md).
- **Not rerank**: `/v1/decisions` is a typed decision surface, not a generic ranking endpoint.

## Related docs

- Product backlog: [surfaces-and-integrations roadmap](../roadmaps/surfaces-and-integrations.md#structured-inference-surfaces)
- Research context: [RESEARCH_ROADMAP.md](../RESEARCH_ROADMAP.md)
- Upstream watchlist: [upstream/tracker.md](../upstream/tracker.md)
