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
