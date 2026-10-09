---
name: entity-extract
description: "Zero-shot span NER via zerollama POST /v1/extract (portable) or POST /v1/gliner (GLiNER.cpp knobs) — C++/ONNX sibling; not decisions, not chat."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, gliner, ner, extract, entities, onnx]
    category: mlops
    related_skills: [zerollama-integration, typed-decisions]
---

# Entity Extract (GLiNER) Skill

Run **zero-shot NER** on [zerollama](https://github.com/GoodSoftware-Group/zerollama):

| Path | Use when |
|------|----------|
| `POST /v1/extract` | Portable agents — `{text, labels, threshold?}` |
| `POST /v1/gliner` | Need GLiNER.cpp knobs — `max_width`, `max_length`, `model_type`, `device_id`, `flat_ner`, `multi_label` |

Requires `ZEROLLAMA_GLINER_URL` → sibling `gliner-server` (lab `:18094`). Docs: [gliner-cpp.md](../../docs/gliner-cpp.md).

**Why not `/v1/decisions`:** spans ≠ choice/score/noul.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/extract -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/gliner -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/v1/decisions   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/chat   # 200/400 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.


## Compatibility

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:11437/v1/extract -d '{}'
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:11437/v1/gliner -d '{}'
```

## When to Use

- Extract people/orgs/locations/custom labels from short documents
- PII-ish label sets without training a CRF
- Need engine knobs (`flat_ner`, `model_type=token`) → `/v1/gliner`

## When NOT to Use

- Typed triage → `/v1/decisions` (Laya/CLM/OpenJev)
- Open-ended generation → `/api/chat`
- Production binds on `:11434` / `:8081` from agent labs

## Example

```bash
# Abstract
curl -s http://127.0.0.1:11437/v1/extract -H 'content-type: application/json' -d '{
  "model": "gliner",
  "text": "Kyiv is the capital of Ukraine.",
  "labels": ["city", "country"],
  "threshold": 0.3
}'

# Mechanical (span — default product path)
curl -s http://127.0.0.1:11437/v1/gliner -H 'content-type: application/json' -d '{
  "model": "gliner",
  "text": "Kyiv is the capital of Ukraine.",
  "labels": ["city", "country"],
  "threshold": 0.3,
  "flat_ner": true,
  "max_width": 12,
  "model_type": "span"
}'

# Token multitask (GL5a) — serve with --model-type token + multitask ONNX
curl -s http://127.0.0.1:11438/v1/gliner -H 'content-type: application/json' -d '{
  "model": "gliner",
  "text": "Kyiv is the capital of Ukraine.",
  "labels": ["city", "country", "river", "person", "car"],
  "threshold": 0.3,
  "flat_ner": true,
  "model_type": "token",
  "max_width": 12,
  "max_length": 768
}'
```

## Pitfalls

- Missing `ZEROLLAMA_GLINER_URL` → 400
- Abstract path strips engine keys; they are not errors, just ignored
- `model_type` is fixed at sibling process start — request echo must match the loaded ONNX
- Token multitask needs rebuild after Finding **10** patch (channel-first logits); empty spans ⇒ stale binary
- VRAM: CUDA ORT (GL4) adds ~1 GiB for `gliner_small`; unload OpenJev/emb on 16 GB first (Finding 7) or keep CPU ORT
- Lab ports only (`18094` CPU, `18095` CUDA, `18097` token e2e, `11437`/`11438` Go)
- RelEx / bi-encoder **not** in GLiNER.cpp yet (Finding 9)

## Related

- [gliner-cpp.md](../../docs/gliner-cpp.md) · [findings](../../docs/gliner-cpp-findings.md) · ROADMAP **GL\***
