---
name: zerollama-integration
description: "Connect any agent harness to a zerollama server."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, ollama, local, inference, qos, kv-cache, batch, prompt-cache, cache-warm]
    category: mlops
    related_skills: [install-zerollama, configure-zerollama-env, hermes-provider, generate-image, generate-video, download-model, text-to-speech, speech-to-text, generate-embeddings, rerank-candidates, batch-inference, fleet-vram-admission, model-authoring, agent-web-tools, anthropic-messages-compat, openai-responses-compat, account-auth, distill-and-train, video-understanding-chat, cloud-model-routing, diagnose-server-health, benchmark-model-speed, launch-agent-integration, fleet-management, gpu-capability-discovery, lmstudio-cache-import, doctor-model, model-suggester]
---

# Zerollama Integration Skill

Connect an agent harness (Hermes, Claude Code, Cursor, or any OpenAI-compatible
agent) to a [zerollama](https://github.com/GoodSoftware-Group/zerollama) server — an
Ollama fork with fleet QoS scheduling, prompt/KV cache pinning, and batch
inference. Covers server verification, context sizing, QoS options, and
pitfalls. Harness-specific wiring (config file keys) lives in the consuming
harness's own skill.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/tags   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/version   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/api/can-load -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/v1/chat/completions   # 200/400 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- Setting up or repairing any agent's connection to a zerollama server
- A local model is slow to reload between turns, or VRAM is being
  over-allocated
- You want QoS scheduling (interactive vs background) or prefix-cache reuse
  across agent turns
- Debugging "invalid option provided", "HOLD_GPU failed", or
  "model not found" errors against zerollama
- Choosing or changing models for the main loop or side tasks

## Prerequisites

- zerollama server running locally (default `http://localhost:11434`)
- Model names exist on the server (`GET /api/tags`)
- The harness points at `http://localhost:11434/v1` (OpenAI-compatible) or
  `http://localhost:11434` (native Ollama API)

## How to Run

Verify the server first, then check the harness config, then adjust:

```bash
# 1. Confirm zerollama is reachable and is the zerollama fork
curl -s http://localhost:11434/api/version

# 2. List available models
curl -s http://localhost:11434/api/tags

# 3. Pre-flight VRAM check for a candidate model
curl -s -X POST http://localhost:11434/api/can-load \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model>:<tag>"}'
```

## API Contract Summary

Zerollama is OpenAI-compatible (`/v1/chat/completions`) plus these extensions:

| Feature | Where | Notes |
|---|---|---|
| QoS scheduling | `options.zerollama.qos_class` | `interactive` \| `auxiliary` \| `background` |
| QoS fulfillment | `options.zerollama.fulfillment` | `complete` \| `benchmark` |
| Project label | `options.zerollama.project_name` | shows in `zerollama ps` |
| KV/prompt cache key | `prompt_cache_key` (top-level or options) | stable key = cache reuse |
| Cache pinning | `POST /api/cache/pin` | keep prefix cache alive |
| Cache warming | `POST /api/cache/warm` | prefill + optionally pin an L3 slot, no generated text |
| Model keep-alive | `keep_alive` (top-level) | pin model in VRAM between turns |
| Batch inference | `POST /v1/chat/completions/batch` | max 8, same model, text-only |
| Thinking control | `think` (bool or `high`/`medium`/`low`) | top-level on /v1 |
| Pre-flight VRAM | `POST /api/can-load` | returns `can_load`, `vram_*`, `tensor_parallel` |
| Per-call deadline | `timeout` (e.g. `"30s"`) | 504 on DeadlineExceeded |

## Cache Warming (L3 prefill)

Warm the prefix cache for a thread/session before real traffic so the first
turn is a cache hit instead of a cold decode. `POST /api/cache/warm` runs a
real prefill (`llama_decode` over the whole prompt) with **no sampling** and
no generated tokens; `pin: true` also creates a `/api/cache/pin` lease in the
same call.

```bash
curl http://localhost:11434/api/cache/warm -d '{
  "model": "llama3.2:3b",
  "prompt": "<system prompt / agent template>",
  "prompt_cache_key": "hermes:thread:42",
  "pin": true,
  "ttl_seconds": 3600
}'
```

Response:

```json
{
  "warmed": true,
  "prompt_cache_key": "hermes:thread:42",
  "kv_decode_steps": 423,
  "pin_id": "cpin_...",
  "expires_at": "2026-07-31T12:00:00Z",
  "notes": "prefix decoded and pinned to a slot; no tokens were generated"
}
```

- Why a dedicated endpoint instead of `/api/generate` with `num_predict=1`:
  callers don't need to discover a throwaway-token convention, pay for
  sampling/logging as if it were a real completion, or guess whether the
  prefix actually landed in a pinned slot.
- **Timeout**: warm does real decode work, so the server allows up to 120s
  (vs the fire-and-forget pin/unpin notifiers). A big prompt on a cold model
  load can take tens of seconds.
- **Key must match real traffic**: use the same `prompt_cache_key` the live
  requests carry (`hermes:thread:42`, `hermes:agent:main:...`, etc.), and the
  same model + `num_ctx`. A mismatch warms a slot nothing will read.
- **Server-side TTL**: without `pin`, the warm slot is still subject to
  normal L3 eviction; pin to hold it across idle gaps.
- The Python runtime also exposes loopback-only `POST /internal/cache/warm`
  for the Go server's own prefill path.

## Sizing Context (num_ctx)

Do not request the model's marketing max. Size from measured use:

```
num_ctx = round_up(p99_prompt_tokens + max_new_tokens + slack, bucket)
buckets: 32k / 48k / 64k / 80k / 96k / 128k
```

- Effective prompt room ≈ `num_ctx − max_new_tokens`. With `num_ctx=131072`
  and `max_tokens=65536` you only get ~65k of prompt — this is the classic
  "context wall" symptom.
- Cap `max_new_tokens` (2k–8k typical for agents; 16k only if you mean it).
  Huge max_tokens is as bad as huge num_ctx for budget math.
- Bump a bucket upward only when measured prompt tokens cross the next bucket.
- Compact/summarize history as you approach the ceiling; don't ask for 256k
  to avoid summarizing.
- Keep the cache key and `num_ctx` stable across turns; only bump allocation
  when the session policy says so (new long job, new model).

## Verification

```bash
# Model loads and responds (OpenAI-compatible path)
curl -s -X POST http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model>","messages":[{"role":"user","content":"hi"}]}'

# Live fleet state / KV allocation
zerollama ps

# Per-model token/scheduling stats
curl -s http://localhost:11434/api/metrics
```

## Pitfalls

- **`invalid option provided` (zerollama logs)** — pass-through keys
  (`zerollama`, `enable_prefix_mm_cache`, `keep_alive`) must be in zerollama's
  suppress list in `api/types.go::FromMap`. `keep_alive` should also be a
  top-level field, not nested inside `options`.
- **`num_predict` wall** — `max_tokens` reserves decode space. `num_ctx −
  max_tokens` is the real prompt ceiling. Cap `max_tokens`, don't inflate
  `num_ctx` to compensate.
- **`HOLD_GPU failed` / `uma lease begin (gpu)`** — transient GPU/lease
  contention on zerollama; retry or free loaded models. Can also mean the
  model doesn't fit at the requested `num_ctx`.
- **`model not found`** — model name mismatch. Names include the tag
  (`qwen3-coder-next:6bit`). Confirm against `GET /api/tags`.
- **Prompt cache not reused** — `prompt_cache_key` must stay stable across
  turns. Changing `num_ctx` or other options invalidates the prefix cache key,
  so bump allocation only when needed.
- **Warm key mismatch** — `/api/cache/warm` only helps if the warmed key +
  model + `num_ctx` exactly match what live requests send. Warm with the same
  key the traffic uses, not a test key.
- **Aux/side-task models over-ask** — if the harness routes side tasks
  (summarization, judging) to a second model, cap its `num_ctx` explicitly;
  otherwise zerollama may allocate its 262144 default.

## Related

- `install-zerollama` — bootstrap/build a fresh checkout before any of this API is reachable
- `doctor-model` — diagnose a specific model's manifest/blob health and config traps
- `model-suggester` — pick the right model for a task by capability/context/VRAM fit
- `configure-zerollama-env` — navigating the ~1000-flag env/YAML configuration surface
- `hermes-provider` — Hermes-specific config wiring for this integration
- `generate-image` — image generation/editing (MLX + ComfyUI)
- `generate-video` — text-to-video generation (Wan, async jobs)
- `download-model` — pulling/registering models before use
- `text-to-speech` / `speech-to-text` — audio synthesis and transcription
- `generate-embeddings` / `rerank-candidates` — vectors and candidate scoring
- `batch-inference` — fan out multiple same-model chat completions in one call
- `fleet-vram-admission` — capacity dry-runs, co-residency plans, eviction pins
- `model-authoring` — create/quantize/copy/repair custom model variants
- `agent-web-tools` — proxied web search / web fetch as agent tools
- `anthropic-messages-compat` / `openai-responses-compat` — alternate wire formats for chat
- `account-auth` — Eliza Cloud identity, sign-out, key management
- `distill-and-train` — LoRA/QLoRA fine-tuning, teacher-to-student distillation via synthetic data, ternary QAT
- `video-understanding-chat` — sending video into chat for a VLM to watch (input, not generation)
- `cloud-model-routing` — routing chat/embeddings to remote Eliza Cloud models via `:cloud`
- `diagnose-server-health` — `zerollama doctor` environment/health diagnostics
- `benchmark-model-speed` — `zerollama bench` throughput measurement + `PERF` cache
- `launch-agent-integration` — wiring other coding-agent CLIs to zerollama via `zerollama launch`
- `fleet-management` — multi-node warm-model routing via `zerollama fleet serve`
- `gpu-capability-discovery` — which GPU backend/profile the server actually picked
- `lmstudio-cache-import` — reusing LM Studio's downloaded models instead of re-pulling
