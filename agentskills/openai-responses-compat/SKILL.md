---
name: openai-responses-compat
description: "Talk to a zerollama server using OpenAI's Responses API wire format (/v1/responses) instead of Chat Completions."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, openai, responses-api, compat]
    category: mlops
    related_skills: [zerollama-integration, hermes-provider]
---

# OpenAI Responses API Compat Skill

Point a client built against OpenAI's newer **Responses API**
(`POST /v1/responses`) at a local [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server, as an alternative to the older Chat Completions shape
(`/v1/chat/completions`). Zerollama translates the Responses request into
its native chat pipeline.

## When to Use

- An SDK or agent framework defaults to `client.responses.create(...)`
  (OpenAI Python/JS SDKs `>=1.5x`-era) instead of `chat.completions.create`
- Migrating an existing Responses-API integration to point at a local model
- Deciding whether to use `/v1/responses` vs. `/v1/chat/completions` for a
  new integration — prefer whichever shape the client library already
  speaks natively; both land on the same local model

## Prerequisites

- zerollama server running (default `http://localhost:11434`)
- Client's OpenAI `base_url` set to `http://localhost:11434/v1`

## API Contract

`POST /v1/responses` accepts a gzip/zstd-decompressible JSON body
(`Content-Encoding: zstd` is decoded server-side before parsing) matching
OpenAI's `ResponsesRequest` shape (`model`, `input`, optional `instructions`,
`tools`, `stream`, etc.).

- A malformed body is rejected with a `400` in OpenAI's error envelope
  before any model resolution happens.
- Cloud passthrough applies the same as other `/v1/*` endpoints: certain
  model names route straight to the remote inference path instead of the
  local runner.

## How to Run

```bash
curl -s http://localhost:11434/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3.2",
    "input": "Summarize the plot of Hamlet in one sentence."
  }'
```

## Pitfalls

- **Newer/less common surface than Chat Completions** — expect narrower
  feature parity (e.g. some advanced Responses-API-only features may not be
  fully implemented); fall back to `/v1/chat/completions` if something
  errors that works fine there.
- **`Content-Encoding: zstd` support is specific to this endpoint** —
  don't assume every zerollama endpoint decodes zstd bodies; check before
  compressing requests to other routes.
- **Cloud passthrough still applies** — a `:cloud`-suffixed or known cloud
  model name here is proxied remotely just like on `/v1/chat/completions`;
  it isn't guaranteed to hit your local GPU.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `hermes-provider` — Hermes config wiring (uses Chat Completions, for comparison)
