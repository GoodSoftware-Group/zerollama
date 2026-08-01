---
name: model-authoring
description: "Create custom model variants, quantize, copy, and repair manifests on a zerollama server (Modelfile-style /api/create, /api/copy, /api/repair)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, ollama, create, modelfile, quantize, copy, repair, manifest]
    category: mlops
    related_skills: [zerollama-integration, download-model, doctor-model]
---

# Model Authoring Skill

Create custom model variants (system prompt presets, quantized copies),
duplicate models, and repair manifest metadata on a
[zerollama](https://github.com/GoodSoftware-Group/zerollama) server. This is
distinct from `download-model`: authoring derives a **new local model** from
an existing one instead of pulling from a registry.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/create   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/copy   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/repair   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/blobs/:digest   # 200/400 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- Baking a system prompt / persona into a named model so callers don't have
  to pass it every request
- Quantizing an existing fp16 model to a smaller quant (e.g. Q4_K_M) to fit
  VRAM
- Duplicating a model under a new name (backup before overwriting, or
  branching a variant)
- Fixing manifest metadata after a partial/corrupted pull or upstream
  registry drift

## API Contract

| Endpoint | Method | Notes |
|---|---|---|
| `/api/create` | `POST` | `{model, from, system?, quantize?, ...}` — build a new model from an existing base. Streams NDJSON status by default. |
| `/api/blobs/:digest` | `POST` | Upload a raw layer blob (used internally by `create` for local file-based Modelfiles; rarely called directly by agents) |
| `/api/copy` | `POST` | `{source, destination}` — duplicate a model under a new name |
| `/api/repair` | `POST` | `{model?, models?, all?, write?}` — detect/fix manifest metadata drift. `write:false` (default-safe) is a dry run; `write:true` applies fixes |

## How to Run

```bash
# 1. Create a variant with a baked-in system prompt
curl -s http://localhost:11434/api/create -d '{
  "model": "alpaca",
  "from": "gemma3",
  "system": "You are Alpaca, a helpful AI assistant. You only answer with Emojis."
}'

# 2. Quantize an existing model down
curl -s http://localhost:11434/api/create -d '{
  "model": "llama3.1:8b-instruct-Q4_K_M",
  "from": "llama3.1:8b-instruct-fp16",
  "quantize": "q4_K_M"
}'

# 3. Copy/backup a model before modifying it
curl -s http://localhost:11434/api/copy -d '{
  "source": "gemma3",
  "destination": "gemma3-backup"
}'

# 4. Dry-run repair to see what would change (all models)
curl -s http://localhost:11434/api/repair -d '{"all": true, "write": false}'

# 5. Apply the repair for one model once verified
curl -s http://localhost:11434/api/repair -d '{"model": "gemma3", "write": true}'
```

## Pitfalls

- **`/api/create` streams progress by default** — expect NDJSON status
  lines (quantizing, verifying, writing manifest, success), not a single
  instant response, especially for `quantize` jobs on large models.
- **Quantizing needs the source at a higher precision** — you generally
  quantize from an `fp16`/`fp32` base, not from an already-quantized model;
  check `/api/show` on the source first if unsure.
- **`repair` defaults to a dry run** — always check the returned
  `RepairChange` list with `write: false` first; only pass `write: true`
  once you've confirmed the proposed changes are correct. Don't skip
  straight to `write: true` on `all: true` against a production host.
- **`/api/copy` doesn't duplicate the underlying blobs on disk** — it's a
  cheap manifest-level copy (shared layers), not a full disk-space clone.
- **Naming collisions** — `/api/create`/`/api/copy` overwrite an existing
  model at the destination name without a confirmation prompt; check
  `/api/tags` first if the name might already be in use.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `doctor-model` — diagnose which models need `repair` and why
- `download-model` — pulling the base model this skill derives variants from
