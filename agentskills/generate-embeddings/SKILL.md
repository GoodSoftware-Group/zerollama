---
name: generate-embeddings
description: "Generate vector embeddings for text via a zerollama server, for RAG, semantic search, or clustering."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, embeddings, vectors, rag, semantic-search]
    category: mlops
    related_skills: [zerollama-integration, rerank-candidates, download-model]
---

# Generate Embeddings Skill

Generate vector embeddings for text on a [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server for use in RAG pipelines, semantic search, or clustering. Three
equivalent endpoints exist; prefer the native `/api/embed` for batching and
truncation/dimension controls.

## When to Use

- Building or querying a vector index (RAG, semantic search, dedup)
- Computing text similarity locally instead of via a cloud embeddings API
- Choosing an embedding dimension to fit a downstream vector store

## Prerequisites

- zerollama server running (default `http://localhost:11434`)
- An embedding model pulled (e.g. `embeddinggemma`, `nomic-embed-text`) —
  check `GET /api/tags`

## API Contract

| Endpoint | Shape | Notes |
|---|---|---|
| `POST /api/embed` | native | `model`, `input` (string or array of strings), optional `truncate`, `dimensions` |
| `POST /api/embeddings` | legacy native | single-`prompt` predecessor of `/api/embed`; prefer `/api/embed` for new code |
| `POST /v1/embeddings` | OpenAI-compatible | same idea, OpenAI request/response shape |

## How to Run

```bash
# Single input
curl -s http://localhost:11434/api/embed -d '{
  "model": "embeddinggemma",
  "input": "Why is the sky blue?"
}'

# Batch multiple inputs in one call
curl -s http://localhost:11434/api/embed -d '{
  "model": "embeddinggemma",
  "input": ["Why is the sky blue?", "Why is the grass green?"]
}'

# Truncate inputs that exceed the model's context instead of erroring
curl -s http://localhost:11434/api/embed -d '{
  "model": "embeddinggemma",
  "input": "very long document text...",
  "truncate": true
}'

# Request a smaller output dimension (matryoshka-style models)
curl -s http://localhost:11434/api/embed -d '{
  "model": "embeddinggemma",
  "input": "Generate embeddings for this text",
  "dimensions": 128
}'
```

Response includes `embeddings` (array of float arrays, one per input, in
the same order) plus `total_duration`/`load_duration`/`prompt_eval_count`.

## Pitfalls

- **Batch order is preserved** — `embeddings[i]` corresponds to `input[i]`;
  don't re-sort inputs before indexing results.
- **`dimensions` only works on models trained for it** — asking for a
  truncated dimension on a model without matryoshka-style training either
  errors or silently returns the full size; check the model card first.
- **`truncate: false` (default) errors on over-length input** — set
  `truncate: true` if you'd rather silently truncate than fail long inputs.
- **Don't mix embedding models across an index** — vectors from different
  models (or even different `dimensions` settings) are not comparable;
  re-embed the whole corpus if you switch models.
- **`/api/embeddings` (legacy) takes a single `prompt`, not `input`** — use
  `/api/embed` for new integrations; it supports batching.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `rerank-candidates` — score/rank retrieved candidates against a query
- `download-model` — pulling an embedding model before use
