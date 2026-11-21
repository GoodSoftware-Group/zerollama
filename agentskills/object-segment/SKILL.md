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

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/segment -d '{}'   # 400/422 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

