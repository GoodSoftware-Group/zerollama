---
name: rerank-candidates
description: "Score fixed candidate continuations against a shared prompt via a zerollama server, for classification, routing, or reranking without full generation."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, score, rerank, classification, logprob, routing]
    category: mlops
    related_skills: [zerollama-integration, generate-embeddings]
---

# Rerank / Score Candidates Skill

Score a fixed set of candidate continuations against a shared prompt on a
[zerollama](https://github.com/GoodSoftware-Group/zerollama) server via
`POST /api/score` — get log-probabilities without generating free text.
This is the LocalAI "Score RPC" pattern: cheaper than a full chat
completion when you already know the finite set of possible answers
(classification labels, tool-routing choices, multiple-choice options,
scoring a shortlist as continuations).

Dedicated **cross-encoder rerank** (Jina / llama.cpp `--reranking` RANK GGUF)
is a different route: `POST /v1/rerank` (aliases `/v1/reranking`, `/rerank`,
`/api/rerank`). That needs a reranker model, not a chat continuation score.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/api/score -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/rerank -d '{}'   # 400/422 = Jina rerank exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- Choosing among a small, known set of discrete outputs (label
  classification, intent routing, multiple choice) without paying for
  open-ended generation
- Reranking a shortlist of retrieved documents/snippets by how well they
  continue a query, instead of a separate cross-encoder model
- You need calibrated log-probabilities, not just a generated string

## Prerequisites

- zerollama server running (default `http://localhost:11434`)
- Any local chat/base model pulled — no special "reranker" model is
  required, since scoring works against candidate continuations of any
  loaded model

## API Contract

`POST /api/score`

| Field | Required | Notes |
|---|---|---|
| `model` | yes | Model to score with |
| `prompt` | yes | Shared prefix; each candidate is scored as its continuation |
| `candidates` | yes | Array of continuation strings, at least one |
| `length_normalize` | no | Divide joint log-prob by candidate token count — use this to compare candidates of very different lengths fairly |
| `include_token_logprobs` | no | Return per-token logprobs for each candidate |
| `keep_alive` | no | How long to keep the model loaded after this request |
| `options` | no | Model-specific options |

Response `candidates[i]` matches `candidates[i]` in the request order, each
with `log_prob`, `length_normalized_log_prob` (if requested), `num_tokens`,
and optional `tokens` (per-token logprobs).

`POST /v1/rerank` — Jina body: `model`, `query`, `documents` (or TEI `texts`),
optional `top_n`. Needs a RANK-pooling GGUF; 501 if the runner cannot rerank.
Not a drop-in for `/api/score`.

## How to Run

```bash
# Classify: which label best continues the prompt?
curl -s http://localhost:11434/api/score -d '{
  "model": "llama3.2",
  "prompt": "Classify the sentiment of: \"This movie was a total waste of time.\"\nSentiment:",
  "candidates": [" positive", " negative", " neutral"]
}'

# Rerank retrieved snippets against a query, length-normalized
curl -s http://localhost:11434/api/score -d '{
  "model": "llama3.2",
  "prompt": "Query: how do I reset my password?\nDoes this passage answer the query? Passage: ",
  "candidates": ["Go to Settings > Security > Reset Password.", "Our refund policy allows returns within 30 days."],
  "length_normalize": true
}'
```

Pick the candidate with the highest `log_prob` (or
`length_normalized_log_prob` when comparing candidates of different
lengths).

## Pitfalls

- **Unnormalized log-prob favors short candidates** — joint log-prob is
  more negative the more tokens a candidate has; always set
  `length_normalize: true` when candidates have meaningfully different
  lengths (e.g. reranking documents of varying size).
- **Leading whitespace/punctuation changes tokenization** — `" positive"`
  vs `"positive"` can tokenize differently and skew comparisons; keep a
  consistent prefix convention across all candidates in one call.
- **Not a substitute for a dedicated cross-encoder** — `/api/score` is a
  general-purpose scoring primitive against any local chat model. Use
  `/v1/rerank` when you actually have a RANK GGUF.
- **`include_token_logprobs` adds payload size** — only request it when you
  actually need per-token detail (e.g. debugging why one candidate scored
  low).

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `generate-embeddings` — vector similarity as an alternative/complement to scoring
