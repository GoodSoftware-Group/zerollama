---
name: object-segment
description: "Mask request via zerollama POST /v1/segment. grasp_ready is false; a mask is not a grasp."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, segment, sam2, vision, robot]
    category: mlops
    related_skills: [zerollama-integration, object-detect]
---

# Object segment

`POST /v1/segment` takes the same image fields as detect plus an optional pixel `box`.

The current sibling answers `status=wire_only` and `grasp_ready=false`. Do not treat a future mask as a grasp. `capabilities.segment.grasp_ready` stays false. Docs: [segment.md](../../docs/segment.md).
