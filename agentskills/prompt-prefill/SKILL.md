---
name: prompt-prefill
description: "Draft the next zerollama chat with POST /api/prefill (same body as /v1/prefill) before the last word, so the finished /api/chat continues from that prefix. Not /api/cache/warm, and not a faster cold decode."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, prefill, prompt-cache, llama-server, agent]
    category: mlops
    related_skills: [zerollama-integration, hermes-provider]
---

# Prompt prefill

Store a chat prefix before the last word, then send the finished chat.

**Why this exists:** an agent turn resends the same tools and rules. The slow
part is reading that prefix again. This call leaves the last word for the
follow-up, so the finished `/api/chat` continues from the saved text.

**Why a new prompt is the same speed:** the matmul is unchanged. The win is
the next request that starts with this draft.

**Why not `POST /api/cache/warm`:** warm runs a whole prompt into an L3 slot
and can pin it. This call is the chat draft. The two caches are different
directories.

Docs: [kinetic-prompt-reuse.md](../../docs/design/serving/kinetic-prompt-reuse.md) ·
OpenAPI `PrefillRequest`.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s http://localhost:11434/api/version | jq '.zerollama.capabilities.prefill'   # true means the route exists
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/api/prefill -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/prefill -d '{}'   # 400/422 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

`capabilities.prefill: true` means the route is in this binary. **501** from
the handler means the loaded runner cannot prefill.

## When to Use

- The next user message is known except its last word, and the follow-up
  `/api/chat` will send the same messages, tools, and `think`
- The model is on llama-server (`ZEROLLAMA_LLAMA_SERVER=1`, Linux `auto`, or
  `--edge`)

## When NOT to Use

- A one-shot prompt that will not be sent again
- Warming an L3 slot by name → `POST /api/cache/warm`
- Forcing one later chat to miss → `options.cache_reset` or OpenAI
  `cache_prompt: false`
- MLX, cloud, or the Mac ggml default. Those return **501**
- Messages that contain images, audio, or video. Those return **400**

## API contract

`POST /api/prefill` and `POST /v1/prefill` are the same handler.

| Field | Required | Notes |
|-------|----------|-------|
| `model` | yes | Local tag. Cloud is 501 |
| `messages` | yes | Same array the finished `/api/chat` will send |
| `tools` | no | Same tools as the follow-up, or the saved prefix will not match |
| `think` | no | Same value as the follow-up |
| `options` | no | Sampling fields are stored with the logits so a later top-k still covers |
| `keep_alive` | no | How long the model stays loaded |

The server cuts the last message before its last word. A last message that is
one unfinished word returns `cached_tokens: 0` and `prompt_tokens: 0` and does
not run.

**Response:** `{ "object": "prefill", "cached_tokens": N, "prompt_tokens": N }`

## How to run

```bash
# Draft. The last word is "the" and is not stored.
curl -s http://127.0.0.1:11434/api/prefill -d '{
  "model": "llama3.2:3b",
  "messages": [
    {"role": "user", "content": "Summarize this.\n\nWhat is the"}
  ]
}'

# Finished chat. Same messages, including the last word.
curl -s http://127.0.0.1:11434/api/chat -d '{
  "model": "llama3.2:3b",
  "messages": [
    {"role": "user", "content": "Summarize this.\n\nWhat is the capital?"}
  ],
  "stream": false
}'
```

Use a lab port (`11435`) when the production server on `11434` is the ggml
runner. Do not bind a second server on `11434` or `8081`.

## Pitfalls

- **The follow-up must be a prefix of the same render.** Changing tools,
  `think`, or the system text between the draft and the chat misses.
- **Mac default is ggml.** The route answers 501 until the model is on
  llama-server built from this tree.
- **Do not send `LLAMA_WARM_DISK` or `LLAMA_SHORTLIST` on the request.** Those
  are process env. One request can only force a miss with `cache_reset`.
