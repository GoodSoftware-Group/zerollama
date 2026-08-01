---
name: cloud-model-routing
description: "Route chat/embedding/message requests on a zerollama server to remote Eliza Cloud models using the :cloud model suffix, instead of local GGUF inference."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, eliza-cloud, cloud, remote-inference, proxy]
    category: mlops
    related_skills: [zerollama-integration, account-auth, download-model, model-suggester]
---

# Cloud Model Routing Skill

Call a **remote** model through a [zerollama](https://github.com/GoodSoftware-Group/zerollama)
server instead of local GGUF inference, via the built-in [Eliza Cloud](https://www.elizacloud.ai)
proxy. This lets an agent fall back to a bigger hosted model without a
separate client integration — same base URL, a special model-name suffix.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/v1/models/:model   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/tags   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/v1/models   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/chat/completions -d '{}'   # 400/422 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- The local GPU can't run a task's ideal model size and a cloud fallback is
  acceptable
- The user references a hosted/frontier model (GPT, Claude, etc. via Eliza's
  catalog) by name
- Debugging a `401` on cloud-routed calls, or unexpected raw/unshaped JSON
  from a cloud model's `/v1/models/:model` response

## Prerequisites

- `ELIZACLOUD_API_KEY` set on the server (sent as `X-API-Key` on outbound
  proxied requests) — without it, cloud calls return `401`
- Optional: `OLLAMA_CLOUD_BASE_URL` if pointed at a non-default cloud host
  (default `https://www.elizacloud.ai:443`; legacy `https://ollama.com:443`
  uses a different Ed25519-signing auth flow)
- `OLLAMA_NO_CLOUD` must **not** be set (it disables cloud features
  entirely — catalog merge and proxying both stop working)

## How it works

- Model names ending in **`:cloud`** are routed remotely (e.g.
  `openai/gpt-4o-mini:cloud`). The suffix is a local routing bit; it's
  stripped before the underlying `model` field is sent upstream.
- `GET /api/tags` **merges** local GGUF models with the remote Eliza catalog
  (`GET {base}/api/v1/models`) so a client sees one unified list, cached
  with a TTL + singleflight so concurrent list calls don't stampede Eliza.
- Client-facing routes stay the same (`/v1/chat/completions`, etc.); the
  server rewrites paths internally (`/v1/*` → `/api/v1/*`; `/api/embed*` →
  `/api/v1/embeddings`).

## Which endpoints support cloud models

| Endpoint | Cloud model support |
|---|---|
| `POST /v1/chat/completions` | Yes |
| `POST /v1/embeddings` | Yes |
| `POST /v1/messages` (Anthropic compat) | Yes |
| `POST /api/chat`, `POST /api/generate` | **No** — errors and points you at the `/v1/*` routes |

**Why**: Eliza speaks OpenAI/Anthropic shapes natively; bridging every
native-API feature through the legacy Ollama pipeline would duplicate work
and blur error semantics.

## How to Run

```bash
# List models — local GGUFs and Eliza catalog entries merged
curl -s http://localhost:11434/api/tags | grep -i cloud

# Call a cloud model via the OpenAI-compatible path
curl -s http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-4o-mini:cloud","messages":[{"role":"user","content":"hi"}]}'

# Same idea via Anthropic Messages compat
curl -s http://localhost:11434/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"anthropic/claude-3-5-sonnet:cloud","max_tokens":512,"messages":[{"role":"user","content":"hi"}]}'
```

## Pitfalls

- **`POST /api/chat` / `POST /api/generate` reject `:cloud` models** — use
  `/v1/chat/completions` or `/v1/messages` instead; this is intentional,
  not a bug.
- **Missing `ELIZACLOUD_API_KEY`** — the server logs a one-time warning and
  cloud calls will 401; check server logs, not just the client error, when
  debugging.
- **Some responses are raw upstream JSON, not reshaped** — e.g.
  `GET /v1/models/:model` for a cloud id returns Eliza's JSON as-is rather
  than Ollama's `ShowResponse` shape; don't assume every field you'd get
  from a local model's `/api/show` exists here.
- **`/api/me` / `/api/signout` behave differently against Eliza** — see the
  `account-auth` skill; those routes implement the legacy `ollama.com`
  account contract and only fully apply there.
- **`OLLAMA_NO_CLOUD` silently removes all of this** — if a `:cloud` model
  isn't in `/api/tags` or calls fail outright, check whether cloud is
  disabled server-side before debugging the request itself.
- **Dedup by lowercased model id** — a manually added `:cloud` entry and
  the same id appearing in the live Eliza catalog collapse into one row;
  don't expect duplicates in `/api/tags`.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `account-auth` — checking cloud account identity (`/api/me`, `/api/signout`)
- `download-model` — pulling/registering local models as the non-cloud alternative
- `model-suggester` — deciding when to fall back to `:cloud` in the first place
