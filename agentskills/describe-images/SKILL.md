---
name: describe-images
description: "Catalog pictures in a folder with zerollama describe or POST /api/describe (SigLIP tags by default; optional vision caption and OWLv2 boxes). The caller keeps the JSON; the server never opens a folder path."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, describe, images, siglip, catalog, vision]
    category: mlops
    related_skills: [zerollama-integration, generate-embeddings, object-detect, video-understanding-chat]
---

# Describe Images Skill

Build a content-hash catalog of pictures on a
[zerollama](https://github.com/GoodSoftware-Group/zerollama) host. Default
tags are SigLIP cosine matches against a short built-in phrase list
(`a photograph`, …). Empty tag lists stay empty — there is no forced
winner.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/api/describe -d '{}'   # 400/422 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.


## When to Use

- The user wants every picture in a directory tagged, captioned, or boxed
- Resume a catalog after adding or deleting files
- HEIC on Darwin (converted with `sips` for the model call)

## How to Run

The CLI talks to `OLLAMA_HOST`. It does not start serve. The server never
receives a folder path — the CLI sends image bytes.

```bash
# Tags only (default). Writes DIRECTORY.json beside the folder.
zerollama describe ~/Pictures/trip

# Own labels, one-sentence captions, object boxes
zerollama describe ~/Pictures/trip \
  --labels "cat,dog,document,screenshot" \
  --caption qwen2.5vl \
  --objects \
  -o ~/Pictures/trip.json
```

| Flag | Role |
|---|---|
| `-o` / `--output` | Catalog path (default `DIRECTORY.json`) |
| `--labels` | Comma-separated tag words |
| `--tag-model` | SigLIP embed model (default: the only local siglip) |
| `--caption MODEL` | Vision model, one sentence per picture |
| `--objects` | OWLv2 boxes; needs `ZEROLLAMA_DETECT_URL` |
| `--min-score` | Drop tags below this cosine |
| `--fields` | `tags,caption,objects` |

Re-running the same command skips unchanged hashes, adds new files, and
drops missing paths. The JSON is rewritten after each finished picture.

`POST /api/describe` is the same contract for a harness that already holds
bytes: `images[]` with `id` and base64 `data`, optional prior `catalog`,
`fields`, `labels`, `min_score`, `tag_model`, `caption_model`, `objects`.
The response `images[].action` says what changed. Keep the catalog on the
client.

## Pitfalls

- **Not video understanding and not image generation** — this catalogs
  stills. Use `video-understanding-chat` or `generate-image` for those.
- **`--objects` without a detect sidecar does nothing useful** — see
  `object-detect` and `ZEROLLAMA_DETECT_URL`.
- **Builtin labels are coarse** — pass `--labels` when the default phrase
  list is the wrong vocabulary.
- **Do not point this at production ports from an agent lab** — use the
  user's existing serve, or a lab port they asked for.

## Related

- `generate-embeddings` — SigLIP tag models are embedders
- `object-detect` — the box backend behind `--objects`
- `video-understanding-chat` — clips, not a folder of stills
