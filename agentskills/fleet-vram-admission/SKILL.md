---
name: fleet-vram-admission
description: "Inspect and dry-run zerollama's model admission/scheduling: capacity checks, co-residency planning, pins, and live fleet status before loading a model."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, vram, admission, scheduling, fleet, pin, capacity]
    category: mlops
    related_skills: [zerollama-integration, download-model, distill-and-train, fleet-management, gpu-capability-discovery, model-suggester]
---

# Fleet / VRAM Admission Skill

Inspect and dry-run [zerollama](https://github.com/GoodSoftware-Group/zerollama)'s
model admission and scheduling **without loading anything** — capacity
checks, multi-model co-residency plans, eviction-blocking pins, and a live
snapshot of what's currently loaded/queued. Use this before submitting
inference to a shared/multi-tenant host to avoid surprise 503s or thrash.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/status   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/ps   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/can-load   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/propose-load   # 200/400 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- Deciding whether a model will fit before running/generating with it
- Planning to run 2+ models concurrently and need to know if they'll
  co-reside or serialize
- Preventing a model from being evicted mid-session (long-running agent
  turn, benchmark run)
- Debugging "HOLD_GPU failed", unexpected model eviction, or queue backups
  on a shared host

## API Contract

| Endpoint | Method | Notes |
|---|---|---|
| `/api/status` | `GET` | Live ggml/runtime queue snapshot, `inference.config` (NUM_PARALLEL, MAX_LOADED_MODELS, MAX_QUEUE, keep-alive policy), `inference.pins` |
| `/api/ps` | `GET` | Models currently loaded, with `size_vram`, `expires_at`, `pending` request counts |
| `/api/can-load` | `POST` | Single-model capacity dry-run. Always `200` with structured fields — never actually loads |
| `/api/propose-load` | `POST` | Multi-model capacity **plan** — tells you if models can co-reside or must serialize |
| `/api/pin` | `POST` | Block eviction for listed model keys for a TTL (does not load) |
| `/api/pin/{id}` | `DELETE` | Release a pin lease early |
| `/api/metrics` | `GET` | Per-model token/scheduling stats |

## How to Run

```bash
# 1. What's currently loaded / queued?
curl -s http://localhost:11434/api/ps
curl -s http://localhost:11434/api/status

# 2. Will this model fit without evicting anything?
curl -s http://localhost:11434/api/can-load -d '{"model":"llama3.2:latest","options":{"num_ctx":8192}}'

# 3. Planning to run two models together — will they co-reside?
curl -s http://localhost:11434/api/propose-load -d '{
  "models": [{"model":"llama3.2:latest"},{"model":"qwen2.5:latest"}]
}'

# 4. Pin a model so it survives idle eviction during a long agent session
curl -s http://localhost:11434/api/pin -d '{"models":["llama3.2:latest"],"ttl_seconds":1800}'

# 5. Release the pin early when done
curl -s -X DELETE http://localhost:11434/api/pin/<pin_id>
```

## Reading `can-load` responses

- **`needs_eviction: true`** — admitting this model would evict something
  else first. Thrash-sensitive callers (interactive agents, benchmarks)
  should treat this as a soft "wait" signal, not proceed blindly.
- **`already_loaded`** — requires an exact GGUF path match, not just "some
  model is loaded." A different quant/tag of the "same" model still counts
  as not loaded.
- **Confidence varies by path** — the Python runtime path gives an exact
  VRAM estimate; the legacy ggml path is a coarser count/group heuristic.
  Fails closed (reports can't-load) when the estimate is unavailable.

## Reading `propose-load` responses

- **`plan.co_resident: false` + `serialize_required: true`** — the batch
  spans 2+ distinct runtime GGUFs; the Python runtime holds only one GGUF
  at a time, so these models cannot be warm simultaneously. Run them
  sequentially instead of assuming parallel warmth.
- This never loads models — it's purely advisory for planning a batch of
  agent tasks against different models.

## Pitfalls

- **Pins don't load models** — `/api/pin` only blocks eviction of an
  *already-loaded* model; pin after the model is warm, not as a substitute
  for pulling/loading it.
- **Pin conflicts with exclusive fulfillment** — `409` means the pin set
  would create two or more distinct runtime GGUFs while another exclusive
  lease is active; the server fails closed rather than lying about
  co-residency.
- **`can-load` is a dry run, not a guarantee** — GPU state can change
  between the check and your actual generate call on a busy shared host;
  treat it as a strong hint, not a hard reservation (use `/api/pin` for an
  actual reservation).
- **Don't poll `/api/status` in a tight loop** — it's a snapshot for
  planning/debugging, not a readiness gate; back off between checks.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `download-model` — pulling a model before checking if it can load
- `distill-and-train` — checking GPU headroom before submitting a training job
- `fleet-management` — multi-node routing built on top of this single-host status
- `model-suggester` — using `can-load` to filter candidate models before recommending one
- `gpu-capability-discovery` — which GPU backend is actually behind this capacity data
