---
name: batch-inference
description: "Fan out multiple text chat completions to the same model in one call via a zerollama server's batch endpoint."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, batch, chat-completions, throughput, runtime]
    category: mlops
    related_skills: [zerollama-integration, generate-embeddings, distill-and-train]
---

# Batch Inference Skill

Run multiple independent text chat completions against the **same model**
in a single call on a [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server via `POST /v1/chat/completions/batch`. This fans out through the
Python runtime's decode-batching path instead of N separate HTTP round
trips, which is meaningfully faster for many small independent prompts
(classification over a list, generating N variants, grading N answers).

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/chat/completions/batch -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/v1/chat/completions   # 200/400 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- You have several independent, same-model text prompts to run together
  (e.g. classify each item in a list, judge N candidate answers, generate
  short variants) and want one call instead of N sequential ones
- You want higher throughput than sequential `/v1/chat/completions` calls
  for small, tool/vision-free text requests

## Prerequisites

- zerollama server running (default `http://localhost:11434`)
- The target model must be served via the **Python runtime path**
  (`modality_backends.inference: zerollama-runtime` in the manifest, or
  `ZEROLLAMA_RUNTIME=1`) — this endpoint is not available for the legacy
  ggml runner path

## API Contract

`POST /v1/chat/completions/batch`

| Field | Notes |
|---|---|
| `model` | one model for the whole batch — **no mixed models** |
| `requests` | array of `{messages: [...]}` items, **max 8** |

Hard constraints (violating any returns `400`):

- All items share the same `model`
- Max **8** items per batch
- **No tools, no vision, no `think`** in any item
- `stream` must be **false** (batch responses are not streamed)

Response is a wrapper object, **not** a bare array:

```json
{
  "object": "chat.completion.batch",
  "model": "llama3.2",
  "count": 2,
  "completions": [
    {"id": "chatcmpl-...", "object": "chat.completion", "choices": [...], "usage": {...}},
    {"id": "chatcmpl-...", "object": "chat.completion", "choices": [...], "usage": {...}}
  ]
}
```

`completions[i]` corresponds to `requests[i]` in order.

## How to Run

```bash
curl -s http://localhost:11434/v1/chat/completions/batch \
  -H 'content-type: application/json' \
  -d '{
    "model": "llama3.2",
    "requests": [
      {"messages": [{"role": "user", "content": "Summarize: cats are independent."}]},
      {"messages": [{"role": "user", "content": "Summarize: dogs are loyal."}]}
    ]
  }'
```

## Mixed-model fan-out

The server does not fan out across models. If you have items destined for
different models, group them client-side by model and call this endpoint
once per group:

```python
from collections import defaultdict

by_model = defaultdict(list)
for item in items:
    by_model[item["model"]].append({"messages": item["messages"]})

for model, requests in by_model.items():
    for chunk in [requests[i:i+8] for i in range(0, len(requests), 8)]:
        # POST /v1/chat/completions/batch with {"model": model, "requests": chunk}
        ...
```

## Pitfalls

- **`400` on tools/vision/`think`/`stream:true`** — this endpoint is
  text-only, non-streaming, no tool calls. Fall back to sequential
  `/v1/chat/completions` calls for those cases.
- **Silent cap at 8** — split larger batches into chunks of ≤8 client-side;
  don't assume the server will auto-chunk for you.
- **Requires the Python runtime path** — if the model only has a legacy
  ggml manifest, this returns an error; check `modality_backends.inference`
  or use `ZEROLLAMA_RUNTIME=1`.
- **One model per call** — never assume the server will split a
  mixed-model `requests` array for you; it validates and rejects it.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `generate-embeddings` — batching embeddings (different endpoint, same "batch of independent inputs" idea)
- `distill-and-train` — using batch inference to generate synthetic distillation data
