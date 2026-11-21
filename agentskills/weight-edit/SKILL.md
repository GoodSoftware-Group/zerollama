---
name: weight-edit
description: "Inspect, apply, or restore a geodesic-patch v1 weight edit on a linked safetensors model via POST /api/weights/edits. Not /api/create, and not a GGUF quantize."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, weights, geodesic-patch, safetensors, edit]
    category: mlops
    related_skills: [model-authoring, zerollama-integration, download-model]
---

# Weight edit

Apply a small claimed change to a linked safetensors checkpoint and register
the result as a new tag.

**Why not `/api/create`:** create bakes a Modelfile or a quant. A weight edit
decodes a patch, checks every claim against the source bytes, writes a new
directory, and clones the manifest onto `destination`.

**Why a linked directory:** the handler reads `source_dir` on the manifest.
A blob-only GGUF tag has no directory to bind, and the call is **400**.

Docs: [weight-patch.md](../../docs/design/serving/weight-patch.md) · OpenAPI
`WeightEditApplyRequest`.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/api/weights/edits/inspect -d '{}'   # 400/415/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/api/weights/edits -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/api/weights/edits/restore -d '{}'   # 400/422 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- You have a geodesic-patch v1 file and a local safetensors tag whose manifest
  records `source_dir`
- You want to read the patch first (`inspect`) before writing a tag

## When NOT to Use

- Quantizing or baking a system prompt → `/api/create`
- Copying a tag without changing weights → `/api/copy`
- A GGUF blob with no linked checkpoint directory

## API contract

| Endpoint | Body | Result |
|----------|------|--------|
| `POST /api/weights/edits/inspect` | raw patch bytes, or multipart field `patch` | format, id, per-tensor change counts. No checkpoint is opened |
| `POST /api/weights/edits` | `{model, destination, patch}` JSON (`patch` is base64) or multipart | New directory and a new tag. Source is unchanged |
| `POST /api/weights/edits/restore` | same body; `model` is the patched tag | Inverse write, checked against the result fingerprint |

HTTP caps the body at 32 MiB of patch bytes. The CLI streams without that cap.

| Status | Meaning |
|--------|---------|
| 400 | Model missing, remote, or not a linked safetensors directory |
| 409 | `destination` tag or output directory already exists |
| 413 | JSON or raw body over the cap |
| 415 | Patch is not geodesic-patch v1 |
| 422 | A claim does not match the checkpoint bytes, or restore fingerprint mismatches |

## How to run

```bash
# Read the patch. No model is touched.
curl -s --data-binary @edit.gpatch \
  http://127.0.0.1:11434/api/weights/edits/inspect

# Apply. patch is base64 of the file.
curl -s http://127.0.0.1:11434/api/weights/edits -d '{
  "model": "my-safetensors:latest",
  "destination": "my-safetensors:edited",
  "patch": "'"$(base64 < edit.gpatch)"'"
}'

# Restore from the edited tag back to a third tag.
curl -s http://127.0.0.1:11434/api/weights/edits/restore -d '{
  "model": "my-safetensors:edited",
  "destination": "my-safetensors:restored",
  "patch": "'"$(base64 < edit.gpatch)"'"
}'
```

The same calls exist as `zerollama weight inspect`, `zerollama weight apply`,
and `zerollama weight restore`. The CLI does not use the 32 MiB HTTP cap.

## Pitfalls

- **Apply does not overwrite.** Pick a free `destination`. 409 leaves both
  tags as they were.
- **Identity is checked before any write.** A wrong base model is 422, not a
  silent partial file.
- **Restore reads the patched tag**, not the original. The patch's result
  fingerprint has to match that directory.
