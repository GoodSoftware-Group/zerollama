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
