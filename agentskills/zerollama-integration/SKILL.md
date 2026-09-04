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
| Explicit load/unload | `POST /api/load`, `POST /api/unload` | LocalAI also `POST /backend/load`. Unload key is `model`. Generate `keep_alive: 0` remains the Ollama unload. |
| Batch inference | `POST /v1/chat/completions/batch` | max 8, same model, text-only |
| Thinking control | `think` (bool or `high`/`medium`/`low`/`xhigh`) | top-level on /v1. Qwen 3.8 omitted think is **low** (mlx-serve); `high`/`xhigh` inject the long guidance. `reasoning_budget_tokens` (0 off, &gt;0 on) outranks `reasoning_effort` / `enable_thinking`. Native `think` still wins. |
| Pre-flight VRAM | `POST /api/can-load` | returns `can_load`, `vram_*`, `tensor_parallel` |
| Per-call deadline | `timeout` (e.g. `"30s"`) | 504 on DeadlineExceeded |
| Continue cut-off reply | `continue_final_message` | last message must be assistant text, no tool calls; `/v1` and `/api/chat`. `/v1/messages` infers the same when the last turn is assistant text (mlx-serve). `/v1/responses` does not. |
| Messages stop echo | `stop_sequence` | `/v1/messages` reports `stop_reason: stop_sequence` and the matched string when a request stop fires. |
| Model context on `/v1/models` | `context_length`, `max_model_len`, `model_max_tokens`, `meta.context_length` | top-level so oh-my-pi / OpenCode don't assume 128k |
| Model caps on `/v1/models` | `capabilities`, `input_modalities`, `meta.architecture` | mlx-serve names (`chat`, `tool_use`, `streaming`, `vision`, `reasoning`, `json_schema`, `embeddings`). Mapped from native `/api/tags` capabilities. |
| MTP advertised | `supports_mtp` | true when the tag ships a trained draft head (`mtp/` sidecar, GGUF nextn, or in-weight `mtp.*`). PLD-only and Gemma `drafter/` are false. |
| Chat `n` | `n` | `n>1` is HTTP 400 on chat and `/v1/completions` (no n-best). `n=1` or omit is fine. Extra-body `n` counts. Image gen/edits: `n>1`, `mask`, `response_format=url`, `stream:true` are also 400. |
| `service_tier` | `auto` / `default` / omit | `flex` / `scale` / `priority` are HTTP 400 (no capacity tiers). Extra-body copies fold. |
| Chat/Responses `store` | omit / `false` | `store:true` is HTTP 400 (no response store). Extra-body copies fold. |
| Wrong modality | chat/generate on embed/image | 400 names the kind and the right endpoint (`/v1/embeddings`, `/v1/images/generations`, …). |
| Responses `background` | `background` | `true` is HTTP 400 (no async response store). Omit or `false`. |
| Responses store chain | `previous_response_id` / `conversation` / `truncation` | Set `previous_response_id` or `conversation` → 400. `truncation` must be omit/`disabled` (`auto` is 400). Send the full `input`. |
| Responses `include` | `include` | Any non-empty value is 400 (no encrypted reasoning or file_search result extras). |
| Responses `max_tool_calls` | `max_tool_calls` | Must be omit or ≥1. `1` keeps a single tool call. |
| Responses cache hits | `usage.input_tokens_details.cached_tokens` | Prefix KV reuse (0 on miss), same as Chat Completions. |
| Parallel tools | `parallel_tool_calls` | `false` returns at most one tool call (OpenAI SDK structured-output default). Anthropic `tool_choice.disable_parallel_tool_use` is the same. |
| Named `tool_choice` | object / Anthropic `type: tool` | Keeps only that function in the prompt. `none` omits tools (chat, Responses, Messages). `required` / `any` keep the list and append a last-user “must invoke a tool” line (named function too). Prompt-side only. Unknown name is 400. Legacy OpenAI `functions` / `function_call` map onto `tools` / `tool_choice` when those are empty. |
| Tight output budget | `max_tokens` / `num_predict` &lt; 12288 | Last user gets “Keep the reply concise…”. Omit, −1, or ≥12288 does not. Native `/api/chat` and `/internal/render-chat` too. |
| Completions `best_of` | `best_of` | `best_of>1` is HTTP 400 (same as `n>1`). |
| Truncated tool vs length | `finish_reason` | If generation hits the token cap, `finish_reason` stays `length` even when a tool call was parsed. `/v1/messages` uses `max_tokens`. `/v1/responses` is `status: incomplete` with `incomplete_details.reason=max_output_tokens`. |
| Think-tag leaks | content + thinking | Trailing `</think>` and a leading unclosed `<think>` are stripped. Unparsed `<tool_call>` is cut from thinking as well as content. A `</think>` inside an open tool-call argument is payload (not a think closer). Think-off also closes an open `<think>` on the prompt (LFM2.5). Muse think-off with no tools commits `to=user`. Muse drops `to=self` reasoning from assistant turns before the last user. |
| Qwen XML tools | `<function=` | If a `<tool_call>` body has `<function=`, it is XML (Qwen 3.5+). JSON is not tried first, so a `package.json` in a parameter cannot become the tool name. |
| Tool autocorrect | default on | String `"3"` / `"true"` become integer/boolean from the tool schema. A required top-level key buried in a nested object (e.g. `edit.path`) is hoisted. Truncated Qwen tool JSON keeps the name and ships `{}` (never half-parsed args). OpenAI `arguments` is always a JSON object string. LFM2 pythonic calls cut off mid-argument also ship name + `{}`. A Qwen `<tool_call>` that never gets `</tool_call>` still parses on done. `<tool_calls:` (with colon) is the Hunyuan-style plural wrapper; `<tool_calls>` without colon is not. Laguna bare JSON `{name, arguments}` is only a tool call when that name is in the request's tool list; a tagged `<tool_call>` is never filtered. Gemma 4 dropped `<|"|>` strings keep inner commas/braces until the next argument key. Truncated Gemma `call:name{` ships name + `{}`. Laguna tagged `<tool_call>` JSON that is truncated also ships name + `{}` (bare inferred JSON still must name a declared tool). Glimmer truncated ATEM keeps completed parameters (non-ATEM tool bodies salvage the recipient + `{}`). Harmony truncated tool JSON ships name + `{}`. FunctionGemma missing `<end_function_call>` still parses on done. DeepSeek missing tool-call closers still parse on done (truncated JSON is name + `{}`). Cogito fenced `function` calls do the same. Ministral truncated `[ARGS]` ships name + `{}`. Olmo3 missing `</function_calls>` still parses on done (truncated args are name + `{}`). GLM missing `</tool_call>` still parses on done (truncated XML is name + `{}`). Cohere truncated `<|START_ACTION|>` JSON ships name + `{}`. Empty history tool-call ids become stable `call_N`. Mixed thinking+content streams put `logprobs` on the content chunk, not reasoning; reasoning-only drops them. Completions use the four-array OpenAI logprobs shape. Each logprob entry includes vocab `id` when known. Completions `logprobs: 0` still returns chosen-token logprobs (`1`–`5` add alternatives; outside 0–5 is 400). Unparsed LFM2/Gemma/FunctionGemma/`<function_calls>`/ATEM tool wrappers are cut from assistant text. Muse ATEM invoke names keep the trailing identifier (error-echo prefixes dropped). `ZEROLLAMA_TOOL_AUTOCORRECT=0` disables type coerce/hoist only. |
| Completions `echo` | `echo` | `/v1/completions` prepends the prompt to the completion text. |
| `logit_bias` | map of token id → additive logit | `/v1` chat, completions, Responses. MLX sampler + llama-server. Cap 256 entries. |
| Structured output | `response_format` / Responses `text.format` | `json_object` constrains to JSON on chat **and** `/v1/responses`. `json_schema` accepts nested `json_schema.schema` or a flat `schema` field. |
| FIM | `/v1/completions` `suffix` | Prompt-only templates (typical MLX) wrap Qwen `<|fim_prefix|>` / `<|fim_suffix|>` / `<|fim_middle|>`. Templates that already use `.Suffix` are unchanged. MLX tags list `insert`. Streaming holds an all-whitespace first chunk and prepends it to the first real text so indent matches non-stream. |
| HF sampling defaults | omitted `temperature` / `top_p` / `top_k` | MLX uses `generation_config.json`; Modelfile PARAMETER and the request still win. `/v1` chat, completions, and Responses do not inject 1.0 when omitted. `top_k`, `min_p`, `typical_p`, `repetition_penalty`/`repeat_penalty` map to Ollama options and the MLX sampler. `typical_p` 1.0 is off. `max_completion_tokens` is an alias of `max_tokens`. `stop` is honored on MLX (not only llama-server). FIM/reserved specials are never sampled (think/tool tags still are). |
| Per-request spec | `enable_pld` / `enable_mtp` / `enable_drafter` | mlx-serve flags. Omit = process default. `enable_drafter` aliases `enable_mtp`. MoE models park MTP unless you set `enable_mtp`/`enable_drafter`. Qwen 3.5/Next checkpoints that ship `mtp.*` use that head (not only a sidecar). `logprobs` and `format`/`grammar` park spec; tools do not. |
| Chat compression | `compression` (top-level or `extra_body`) | Agent tool/think threads auto-`placeholder` (elide newest fat tools first). Echo `elide_from` from the previous response: native `compression`, Chat Completions / Responses / Messages `usage.compression_meta` (Messages streaming: `message_delta.usage`). Native `zerollama run` / `--experimental` already echo it. Go SDK: `api.ChatThread`. Hermes/`prompt_cache_key` threads get the same cut server-side if they omit `elide_from`. Hermes `extra_body.compression` is folded on ChatMiddleware, the Python `/v1` chat proxy, `/v1/responses`, and `/v1/messages`. `enabled: false` disables. Summary mode is opt-in. |

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
  to avoid summarizing. Tool loops get in-place placeholder elide by default
  (no extra compressor model). Keep `prompt_cache_key` stable and echo
  `elide_from` so a later `num_ctx` bump cannot restore an already-elided
  tool body and split prefix KV.
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
  model doesn't fit at the requested `num_ctx`. If it persists with free
  memory and no resident runner (`/api/ps` empty), the machine broker may
  hold a wedged lease: query `STATUS` / `JOBS` / `QUEUE` over
  `/tmp/uma_daemon.sock` — a `mlxrunner-load` job stuck in
  `running`/`holding` with a dead owner blocks the whole GPU queue (queued
  waiters time out after ~90s each and the jam is self-sustaining).
  `RELEASE <id>` the dead holder. Note the server-side load cooldown
  (`model load in cooldown: retry after ...`): blind client retries inside
  the window always 503 — wait it out, then try once.
- **`model not found`** — model name mismatch. Names include the tag
  (`qwen3-coder-next:6bit`). Confirm against `GET /api/tags`.
- **Prompt cache not reused** — `prompt_cache_key` must stay stable across
  turns. Changing `num_ctx` or other options invalidates the prefix cache key,
  so bump allocation only when needed. After placeholder compression, echo
  `elide_from` (index into **this request's** messages) or keep a stable
  `prompt_cache_key` so the server remembers the cut. Raising `num_ctx`
  without either can restore a full tool body and miss the cached prefix.
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
