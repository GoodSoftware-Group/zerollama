---
title: Entity extract
description: Zero-shot span extraction through POST /v1/extract and POST /v1/gliner.
---

**Status:** `beta`

Entity extract is the zerollama surface for **zero-shot span extraction**: given text plus label strings, return entities/spans with scores.

## API surface

- `POST /v1/extract` — abstract, agent-friendly path
- `POST /v1/gliner` — mechanical engine path with more knobs

Use this when you want **NER/extraction**, not typed routing or chat.

## Current backend

| Backend | Status | What it is good at | Deep docs |
|---|---|---|---|
| **GLiNER.cpp** | `beta` | Zero-shot span NER through a C++/ONNX sibling, CPU or CUDA ORT | [gliner-cpp.md](../design/surfaces/gliner-cpp.md), [findings](../design/surfaces/gliner-cpp-findings.md) |

## What is stable at the capability layer

- `/v1/extract` is the portable capability surface for agents.
- `/v1/gliner` exposes more of the underlying engine for tuners/operators.
- Go can proxy the capability when `ZEROLLAMA_GLINER_URL` is set.

## Current guidance

- Prefer `/v1/extract` unless you specifically need GLiNER engine knobs.
- Keep CUDA ORT for when the GPU budget allows it; CPU ORT is still a valid working path.

## Not this capability

- **Not typed decisions**: use [decisions.md](decisions.md) for route/urgency/choice/noul work.
- **Not GLiNER2.5-Decide**: that is a triage/decisions surface, not extraction.

## Related docs

- Product backlog: [surfaces-and-integrations roadmap](../roadmaps/surfaces-and-integrations.md#structured-inference-surfaces)
- Deep docs: [gliner-decide.md](../design/surfaces/gliner-decide.md) if you need the separate triage/decider path
