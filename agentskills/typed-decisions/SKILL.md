---
name: typed-decisions
description: "Answer typed System-1 questions (choice / score / noul) in one forward pass via zerollama POST /v1/decisions (Laya), with calibrated probabilities and act/escalate — not chat, not score, not rerank."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, laya, decisions, systemone, classification, routing, calibrated]
    category: mlops
    related_skills: [zerollama-integration, rerank-candidates, download-model]
---

# Typed Decisions (Laya) Skill

Run [Laya](https://github.com/NandhaKishorM/laya)-class **System-1** models on a
[zerollama](https://github.com/GoodSoftware-Group/zerollama) server via
`POST /v1/decisions` (alias `POST /v1/systemone`).

One encoder forward answers every question with **calibrated probabilities** and
an **act vs escalate** head — not free-text chat, not `/api/score` continuations,
not `/v1/rerank` RANK pooling.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/decisions -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/systemone -d '{}'   # alias for /v1/decisions
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- Discrete **routing / triage / classification** where labels are known up front
  (`choice`), ordinal levels (`score`), or boolean (`noul`)
- You need **calibrated probs + confidence + act_probability**, not a sampled string
- You want **many questions in one call** over the same `state` (one encoder pass)

## When NOT to Use

- Open-ended generation → `/api/chat` or `/v1/chat/completions`
- Rank documents with a RANK GGUF → `/v1/rerank`
- Score candidate **continuations** of a chat model → `/api/score`
- No Laya GGUF loaded → you will get **501**

## Prerequisites

- zerollama server with LAYA1–LAYA2 (llama.cpp patches **0127–0128**, Go Decider)
- A **Laya** model created from GGUF (`general.architecture=laya`) — convert via
  `scripts/convert_laya_to_gguf.py` ([docs/laya-llama-cpp.md](../docs/laya-llama-cpp.md))
- Lab smokes: use non-production ports (`11435`, `18082`) — never bind `:11434` /
  `:8081` from agent work

## API Contract

`POST /v1/decisions` (same body as `POST /v1/systemone`)

| Field | Required | Notes |
|---|---|---|
| `model` | yes | Laya tag / alias |
| `state` | yes | string \| object \| array — shared context for all questions |
| `questions` | yes | map of `question_id` → `{type, instructions, criteria?, labels?}` |
| `keep_alive` | no | Keep model loaded after the call |
| `options` | no | Passthrough runner options |

`questions[id].type` ∈ `choice` | `score` | `noul`.

- **choice** — `criteria` object; keys are option ids (preserve JSON key order)
- **score** — `criteria` ordered array of levels; answer is expected value
- **noul** — boolean; optional `labels` for false/true display strings

Response `answers[id]` is discriminated by `type`, each with `probabilities`,
`confidence`, and `action.act_probability`.

## How to Run

```bash
# Triage: which team owns this ticket?
curl -s http://127.0.0.1:11434/v1/decisions -H 'content-type: application/json' -d '{
  "model": "laya",
  "state": {"body": "billed twice, refund please"},
  "questions": {
    "dept": {
      "type": "choice",
      "instructions": "Which team?",
      "criteria": {"billing": "refunds", "tech": "bugs"}
    },
    "urgent": {
      "type": "noul",
      "instructions": "Is this urgent?",
      "criteria": {}
    }
  }
}'
```

Pick `answers.dept.choice` when `action.act_probability` is high enough for your
policy; otherwise escalate to a System-2 chat model.

## Pitfalls

- **Wrong model arch → 501** — chat GGUFs cannot decisions; need `arch=laya` +
  llama-server built with patches 0127–0128 (`--decisions` auto when Laya).
- **Choice key order matters** — criteria object key order aligns logits to labels;
  do not reshuffle keys between pack and decode.
- **Not a chat drop-in** — do not send `messages[]`; send `state` + `questions`.
- **MLX sidecar is parked** — active path is llama.cpp CPU/CUDA; do not expect
  Darwin-managed `laya-mlx` spawn (ROADMAP LAYA4).
- **Agent labs** — never kill production `:11434` / `:8081`; use `OLLAMA_HOST=127.0.0.1:11435`.

## Related

- `rerank-candidates` — `/api/score` + `/v1/rerank` (different surfaces)
- `zerollama-integration` — generic API contract
- `download-model` — pull / create GGUF tags
- Docs: [laya-llama-cpp.md](../docs/laya-llama-cpp.md) · [findings](../docs/laya-llama-cpp-findings.md)
