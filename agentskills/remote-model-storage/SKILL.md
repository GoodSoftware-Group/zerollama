---
name: remote-model-storage
description: "Keep one canonical OLLAMA_MODELS tree on a big disk and fetch blobs into a capped local cache with zerollama storage serve (lab :18090). Not an inference server and not NFS."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, storage, models, blobs, rdma, cache]
    category: mlops
    related_skills: [zerollama-integration, download-model, install-zerollama]
---

# Remote Model Storage Skill

Inference disks fill with model weights before VRAM does. `zerollama
storage` serves one canonical `$OLLAMA_MODELS` tree and lets other nodes
fetch a blob on miss into a local LRU cache.

[zerollama](https://github.com/GoodSoftware-Group/zerollama) lab listen is
**`:18090`**. Never bind storage on **11434**, **8081**, or **8080**.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
zerollama storage --help                 # missing subcommand = this build predates storage
# storage serve is a separate listener (lab :18090), not :11434
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.


## When to Use

- Models should live on a bigger box; GPU nodes only cache what they load
- Migrating blobs with `storage push` / `storage pull` instead of copying
  the whole tree
- Checking a peer with `storage discover` / `storage doctor`

## How to Run

Auth is HMAC-SHA256 (`ZEROLLAMA_STORAGE_SECRET` on both sides). Data plane
is RDMA READ when built `-tags rdma`, otherwise HTTP Range GET.

```bash
# On the disk host (lab port)
ZEROLLAMA_STORAGE_SECRET=... zerollama storage serve --listen 127.0.0.1:18090

# Point an inference node at it, then chat as usual — GetModel fetches
# missing blobs. Scheduler refcounts pins while loaded.

# Registry → storage host
zerollama storage pull gemma3

# This machine ← storage host, then run
zerollama storage run gemma3

zerollama storage ls
zerollama storage verify
zerollama storage gc
zerollama storage discover    # ZEROLLAMA_MDNS=1 → _zerollama-storage._tcp
zerollama storage doctor
```

`storage push --reclaim` deletes a blob only after every referencing
manifest has been pushed. Incomplete `PUT /v1/blob/{digest}` with
`Content-Range` returns **202** and `X-Zerollama-Partial` so a dropped
transfer resumes.

Guide: `docs/design/surfaces/remote-model-storage.md`.

## Pitfalls

- **Not `zerollama serve`** — storage does not do chat, image, or video.
- **Do not bind inference ports** — `:18090` is the lab default.
- **Secret must match** — a missing `ZEROLLAMA_STORAGE_SECRET` refuses to
  serve.
- **RDMA is optional** — without it, peers fall back to TCP Range GET.
  mlx4 bounce throughput is not line-rate 40GbE.
- **Reclaim is not `rm -rf`** — blobs still referenced by a manifest stay.

## Related

- `download-model` — gallery / Hugging Face / Pirate Face pulls into a
  local store
- `install-zerollama` — building the binary that contains `storage`
