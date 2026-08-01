---
name: anthropic-messages-compat
description: "Talk to a zerollama server using the Anthropic Messages API wire format (/v1/messages) for tools built against Claude's API."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, anthropic, claude, messages-api, compat, tool-use]
    category: mlops
    related_skills: [zerollama-integration, hermes-provider]
---

# Anthropic Messages API Compat Skill

Point a tool or SDK built against Anthropic's **Messages API**
(`POST /v1/messages`) at a local [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server without rewriting it to the OpenAI shape. Zerollama translates the
Anthropic request/response/streaming format to its native chat pipeline
under the hood.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/messages -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/v1/chat/completions   # 200/400 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- Wiring a Claude-Code-style client, Anthropic SDK, or any tool that only
  speaks the Messages API to a local zerollama model
- Debugging a `/v1/messages` request that errors or streams incorrectly
- Deciding whether to use this vs. `/v1/chat/completions` (OpenAI shape) for
  a new integration

## Prerequisites

- zerollama server running (default `http://localhost:11434`)
- Client configured to point its Anthropic `base_url` at
  `http://localhost:11434` (not `/v1` — the middleware mounts at `/v1/messages` directly, same as Anthropic's real API root)
- A local model that supports tool calling if the client sends `tools`

## API Contract

`POST /v1/messages` — standard Anthropic Messages request shape:
`model`, `messages`, `max_tokens`, optional `system`, `tools`,
`tool_choice`, `stream`.

- **Streaming**: `stream: true` returns Anthropic-style SSE events
  (`content_block_delta`, etc.) with `Content-Type: text/event-stream`,
  translated live from the underlying `api.ChatResponse` stream.
- **Tool use**: tool calls in the underlying model's response are mapped to
  Anthropic `tool_use` content blocks; tool results sent back must use
  Anthropic's `tool_result` message shape.
- **Errors**: mapped to Anthropic's `ErrorResponse` envelope, not a raw
  zerollama error object.

## How to Run

```bash
curl -s http://localhost:11434/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "llama3.2",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

Streaming:

```bash
curl -N http://localhost:11434/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3.2",
    "max_tokens": 1024,
    "stream": true,
    "messages": [{"role": "user", "content": "count to 5"}]
  }'
```

## Pitfalls

- **Cloud passthrough for `:cloud` model suffixes** — a model name ending
  in `:cloud` (or certain known cloud model families) is proxied straight
  through to the remote inference path instead of the local runner; check
  the model name if you expected a local model to handle the request.
- **Invalid JSON body is a hard failure before your model is even
  resolved** — the middleware binds the whole Anthropic request shape
  up front, so a malformed body never reaches routing/model logic.
- **`tools` with certain built-in Anthropic tool types (e.g.
  `web_search_20250305`) may not be supported locally** — check whether the
  target model/backend actually supports the tool type before relying on
  it; unsupported tool types are a client-visible error, not silently
  ignored.
- **Not the same code path as `/v1/chat/completions`** — behavior
  differences (thinking/relax-thinking flags, streaming buffering) are
  possible between the two compat surfaces even against the same model;
  don't assume perfect parity.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `hermes-provider` — Hermes config wiring (uses the OpenAI-compatible path, for comparison)
