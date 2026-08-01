---
name: model-suggester
description: "Pick the right zerollama model for a task: match required capabilities (tools/vision/embedding/thinking), context length, and available VRAM against local inventory, LM Studio cache, and cloud fallback before recommending a pull or a :cloud route."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, model-selection, recommendation, capabilities, vram, catalog]
    category: mlops
    related_skills: [zerollama-integration, gpu-capability-discovery, fleet-vram-admission, download-model, cloud-model-routing, doctor-model, lmstudio-cache-import]
---

# Model Suggester Skill

Choose which model on a [zerollama](https://github.com/GoodSoftware-Group/zerollama)
host to use for a given task, instead of guessing a name. Decide in this
order: **capability match → context fit → VRAM fit → local-first, cloud
fallback.**

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/tags   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/embed   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/show   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/ps   # 200/400 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- A user or task requires a specific capability (tool calling, vision,
  embeddings, long context, reasoning/thinking) and you're not sure which
  locally-registered model supports it
- Deciding whether to pull a new model or reuse what's already downloaded
- A task needs more context or VRAM than any local model can serve and you
  need to decide between quantizing down, picking a smaller model, or
  routing to `:cloud`
- Setting up `zerollama launch` / an agent integration and need a sane
  default model list

## Step 1 — Inventory what's already local, with capabilities

```bash
curl -s http://127.0.0.1:11434/api/tags | jq '.models[] | {name, size, capabilities, quantization: .details.quantization_level, params: .details.parameter_size, context: .details.context_length, host_max_context}'
```

`capabilities` on each entry is a subset of: `completion`, `tools`,
`insert`, `vision`, `video`, `embedding`, `thinking`, `image`, `video_gen`,
`audio`, `speech` (`types/model/capability.go`). Filter for the capability
the task needs before considering anything else — a model without `tools`
in this list will not reliably emit tool calls, and one without `embedding`
cannot serve `/api/embed`.

`details.context_length` (trained/GGUF ceiling) and `host_max_context`
(largest `num_ctx` estimated to fit current free VRAM/RAM) ride along on
`/api/tags` for **GGUF models only** — populated from the GGUF header at
list time (`server/model_details.go` `enrichModelDetailsFromGGML`), no
per-model `/api/show` round trip needed. **MLX/safetensors models do not
get this** — `enrichModelDetailsFromSafetensors` never sets
`ContextLength`, so those rows show `context_length: null` even though the
model has a real context window; don't read that as "no context support."

For a single candidate's full picture (quant, template presence — and for
MLX models, the only place left to look for context is the model's own
card/docs, since `/api/show` doesn't fill the gap either):

```bash
curl -s http://127.0.0.1:11434/api/show -d '{"model": "<name>"}' | jq '{capabilities, parameters, template: (.template != null)}'
```

## Step 2 — Match capability to task type

| Task | Required capability | Notes |
|---|---|---|
| Plain chat / completion | `completion` | Nearly all local text models |
| Agentic tool calling | `tools` | Verify with `doctorCheckToolCallShape`-class checks (see `doctor-model`) if calls look malformed |
| Code fill-in-the-middle | `insert` | Smaller set of models; check before assuming |
| Image understanding in chat | `vision` | See `video-understanding-chat` skill for the video-specific variant (`video` capability, distinct from `vision`) |
| Text embeddings / rerank prep | `embedding` | Use with `generate-embeddings` / `rerank-candidates` skills |
| Chain-of-thought / reasoning traces | `thinking` | Toggle via `think` request field; see minefield traps 01/03/04/20/29 in `doctor-model` if thinking behaves oddly |
| Image generation | `image` | See `generate-image` skill |
| Video generation | `video_gen` | See `generate-video` skill (Wan) |
| Speech synthesis / transcription | `speech` / `audio` | See `text-to-speech` / `speech-to-text` skills |

If no local model has the required capability, decide between pulling one
(`download-model` skill; check the LM Studio cache first via
`lmstudio-cache-import` to avoid a redundant download) or routing to a
`:cloud` model that has it (`cloud-model-routing` skill).

## Step 3 — Check context fit

Compare the task's expected prompt+completion length against the model's
**trained** context (not the advertised one — see `doctor-model` trap
55/61): `context_length` from the GGUF, surfaced in `/api/show` under
`model_info` or via `/api/ps` `loaded_metadata.train_context_length` once
warm. Don't trust a manifest `num_ctx` above that number to actually work
well; it's a capability claim, not a guarantee.

## Step 4 — Check VRAM fit before recommending a load

Before telling a user "use model X," confirm it will actually fit given
current GPU state — don't recommend a model that will fail admission or
evict something else the user is relying on:

```bash
curl -s http://127.0.0.1:11434/api/can-load -d '{"model": "<name>"}' | jq
```

See the `fleet-vram-admission` and `gpu-capability-discovery` skills for
the full VRAM broker semantics (`can-load` is read-only estimation; loading
still goes through the broker at request time). If `can-load` says no, the
options are: suggest a smaller/more-quantized variant, suggest freeing a
loaded model first, or fall back to a `:cloud` model.

## Step 5 — Default recommendations when nothing else constrains the choice

`zerollama launch` ships two general-purpose local defaults, useful as a
starting point absent other constraints:

| Model | Use case | Approx VRAM |
|---|---|---|
| `gemma4` | Reasoning and code generation | ~16GB |
| `qwen3.5` | Reasoning, coding, and visual understanding (adds `vision`) | ~11GB |

These are the same defaults `zerollama launch`'s picker pre-ranks to the
top of the list (`cmd/launch/models.go` `recommendedModels`); they are not
hardcoded into the picker if better live inventory exists, but they're a
reasonable default suggestion when a user hasn't specified a model and has
no relevant local inventory yet.

## Pitfalls

- **`/api/tags` `details.context_length` and `host_max_context` are
  GGUF-only** — MLX/safetensors models always come back with no context
  info from `/api/tags` (and `/api/show` doesn't fill the gap either); don't
  read a missing/null context field as "this model has no context window,"
  it means check the model's own docs instead.
- **`capabilities` on `/api/tags` reflects the current manifest, not GGUF
  ground truth** — a model missing `tools` in this list may still technically
  parse tool syntax if forced, but zerollama won't treat it as tool-capable
  by default; don't override this without testing.
- **Advertised parameter size ≠ speed or quality guarantee** — always check
  `details.quantization_level` alongside `parameter_size`; a heavily
  quantized large model can be slower and lower quality than a smaller
  model at a higher quant for a given task.
- **Don't recommend a model purely from the catalog without checking
  `can-load`** — a model that looks right on paper can still fail to load
  if VRAM is already committed to other resident models (see
  `fleet-vram-admission`).
- **`:cloud` models bypass local VRAM math entirely** — if `can-load`
  fails for every local candidate, check `cloud-model-routing` before
  concluding "no model works."
- **Recency bias** — `/api/tags` ordering is not a ranking; always inspect
  `capabilities` and `details` per-model rather than assuming the
  most-recently-pulled model is the best fit.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `gpu-capability-discovery` — which GPU backend/profile is active
- `fleet-vram-admission` — VRAM broker semantics behind `can-load`
- `download-model` — pulling a model once you've decided which one
- `cloud-model-routing` — fallback when no local model fits
- `doctor-model` — verifying a candidate model's health/config before relying on it
- `lmstudio-cache-import` — checking for an already-downloaded match before pulling
