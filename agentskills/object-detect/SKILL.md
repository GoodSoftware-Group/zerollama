---
name: object-detect
description: "Open-vocabulary boxes via zerollama POST /v1/detect (OWLv2 sibling). Not the camera chat model."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, detect, owlv2, vision, robot]
    category: mlops
    related_skills: [zerollama-integration, object-segment]
---

# Object detect

`POST /v1/detect` with `image` (base64 or data URL) and `queries`. Boxes are pixel `[x1, y1, x2, y2]`.

`status=wire_only` means the route exists and OWLv2 did not run. Set `ZEROLLAMA_DETECT_URL` to `detect/server/app.py` and `ZEROLLAMA_OWL_MODEL` to a local weight directory. Docs: [detect.md](../../docs/detect.md).

This is not `/v1/chat/completions`. The camera chat model describes a frame. This route returns boxes.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/detect -d '{}'   # 400/422 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/v1/chat/completions   # 200/400 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

