---
name: hermes-provider
description: "Wire Hermes to a zerollama server."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, ollama, local, inference, qos, kv-cache, prompt-cache, provider]
    category: mlops
    related_skills: [zerollama-integration]
---

# Zerollama Integration (Hermes) Skill

Wire this Hermes agent to a [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server — an Ollama fork with fleet QoS scheduling, prompt/KV cache pinning,
and batch inference. This is the Hermes-specific half of the integration; the
harness-agnostic zerollama API contract, `num_ctx` sizing, and pitfalls live
in the `zerollama-integration` generic skill (shipped in the zerollama repo).

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
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/can-load   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/cache/warm   # 200/400 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- Setting up or repairing Hermes's connection to a zerollama server
- A local model is slow to reload between turns, or VRAM is being
  over-allocated
- Choosing or changing the main model / auxiliary task models
- Debugging zerollama errors from inside Hermes

## Prerequisites

- zerollama server running locally (default `http://localhost:11434`)
- Model names exist on the server (`GET /api/tags`)
- Hermes profile configured with the `custom` provider

## How to Run

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

## Procedure

1. **Confirm the server** — `GET /api/version` returns `"distribution": "zerollama"`.
2. **Set the provider** — `model.provider: custom`, `providers.custom.base_url`
   to the zerollama endpoint (`/v1` suffix).
3. **Choose models** — main model + separate cheaper models for auxiliary
   tasks (goal_judge, compression) so the scheduler isn't loading one big
   model for every side call.
4. **Size `num_ctx`** per the generic skill's bucket formula. Set
   `context_length` and `ollama_num_ctx` to the same value.
5. **Cap `max_tokens`** to a sane agent decode budget (4k–8k). Never 65536.
6. **Add QoS extras** per model if the fleet schedules: `project_name`,
   `qos_class`, optional `fulfillment: complete`.
7. **Set `keep_alive`** (top-level `model.keep_alive`) to keep the main model
   resident between turns.
8. **Warm the prefix cache at boot** (optional) — prefill the agent's system
   prompt / template into an L3 slot before real traffic so the first turn
   isn't a cold decode:
   ```bash
   hermes cache-warm "$(cat ~/.hermes/agent_template.txt)" \
     --cache-key "hermes:agent:main:discord:dm:<thread>" --pin
   ```
   Or via curl directly against `/api/cache/warm` (see the generic skill for
   the exact contract).
9. **Verify** and restart the gateway after config changes:
   `launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway`.

## Quick Reference — config.yaml

```yaml
model:
  default: qwen3-coder-next:6bit
  provider: custom
  context_length: 81920        # session budget, not model marketing max
  ollama_num_ctx: 81920        # matches context_length (KV allocation)
  max_tokens: 8192             # cap decode; do NOT leave at 65536
  keep_alive: 30m              # pin model in VRAM between turns

providers:
  custom:
    base_url: http://localhost:11434/v1
    default_model: qwen3-coder-next:6bit
    models:
      qwen3-coder-next:6bit:
        context_length: 81920
        extra_body:
          project_name: hermes-lean
          qos_class: background   # interactive | auxiliary | background
          fulfillment: complete   # complete | benchmark
```

Route side-LLM work to a cheaper model with an explicit budget:

```yaml
auxiliary:
  goal_judge:
    provider: custom
    model: qwen3.6:35b-a3b-mlx
    timeout: 15
    max_tokens: 4096
  compression:
    provider: custom
    model: qwen3.6:35b-a3b-mlx
    base_url: http://localhost:11434/v1
    timeout: 120
```

## Verification

```bash
# Hermes is using the custom provider + zerollama
hermes status

# Model loads and responds
hermes batch run "hello" "reply in one word"

# Warm the prefix cache, then confirm the slot landed (requires server rebuild
# with /api/cache/warm — 404 on the running binary means an old build)
hermes cache-warm "You are a helpful assistant." \
  --cache-key "hermes:cache-warm-smoke" --pin --json

# Watch KV allocation / QoS in fleet logs (zerollama side)
zerollama ps
```

## Hermes-Specific Pitfalls

- **Aux models over-ask** — the auxiliary path (`_build_call_kwargs` in
  `auxiliary_client.py`) forwards `num_ctx` from the per-model
  `context_length` in `providers.custom.models.<model>`. Keep those caps in
  sync or 35b-style aux models fall back to zerollama's 262144 default.
- **`keep_alive` placement** — must be a top-level request field. Hermes pops
  it out of the merged options in the custom provider, so don't add it back
  into `extra_body.options`.
- **Warm cache key must match the live key** — `hermes cache-warm
  --cache-key` must equal the `prompt_cache_key` Hermes derives for that
  thread (`hermes:agent:main:<platform>:...`); a mismatched key warms a slot
  nothing will read. `hermes_prompt_cache_key()` in
  `agent/zerollama_local.py` is the source of truth.
- **Gateway restart** — the gateway rebuilds the agent per message, but config
  changes need a gateway restart to take effect.

## Related

- `zerollama-integration` generic skill — API contract, sizing, pitfalls
- `batch` (external, `skills/productivity/batch`, not in this catalog) —
  parallel text generation via the batch endpoint
