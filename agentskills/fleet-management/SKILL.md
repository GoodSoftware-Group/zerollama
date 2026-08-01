---
name: fleet-management
description: "Run a zerollama fleet management node that routes agent requests to the best of several zerollama peers by warm-model status, via zerollama fleet serve."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, fleet, multi-node, routing, mdns, cluster]
    category: mlops
    related_skills: [zerollama-integration, fleet-vram-admission]
---

# Fleet Management Skill

Run a lightweight management node that polls several
[zerollama](https://github.com/GoodSoftware-Group/zerollama) peers and
assigns agent requests to whichever node already has the requested model
warm, via `zerollama fleet serve`. This is **multi-node** cluster routing —
different from `fleet-vram-admission`, which is single-host capacity
planning against one server's `/api/*` admission endpoints.

## When to Use

- You have more than one zerollama instance (multiple GPU boxes) and want
  agents to hit one stable endpoint instead of hardcoding a node
- Warm-model affinity matters (avoid cold-loading a model on node B when
  it's already resident on node A)
- Setting up LAN auto-discovery of zerollama nodes instead of maintaining a
  static peer list

## Prerequisites

- 2+ zerollama servers reachable over the network, each exposing
  `GET /api/status`
- Static peer list, or mDNS enabled on all nodes for auto-discovery

## How to Run

```bash
# Static peer list
zerollama fleet serve --peers http://gpu-a:11434,http://gpu-b:11434

# LAN auto-discovery (browse for _zerollama._tcp)
zerollama fleet serve --mdns

# Advertise this node's fleet endpoint on the LAN too (for other tools to find)
zerollama fleet serve --mdns --mdns-advertise

# Custom listen address / poll interval
zerollama fleet serve --listen 0.0.0.0:9090 --peers http://gpu-a:11434 --poll-interval 5s
```

Env equivalents exist for scripting/systemd: `ZEROLLAMA_FLEET_PEERS`,
`ZEROLLAMA_FLEET_LISTEN`, `ZEROLLAMA_FLEET_MDNS`,
`ZEROLLAMA_FLEET_MDNS_ADVERTISE`.

## How it works

- The fleet node polls each peer's `GET /api/status` on `--poll-interval`
  to learn what's currently loaded/warm.
- Incoming agent requests are assigned to the best peer for the requested
  model (filter-then-score: nodes that already have it warm win over
  cold-load candidates).
- This is a **thin** management layer — it does **not** remotely trigger a
  load on an idle node; it polls and assigns, it doesn't push model loads
  across the network.

## Pitfalls

- **Requires at least one of `--peers` or `--mdns`** — `fleet serve` with
  neither refuses to start (`fleet peers required` error); you must supply
  a static list, enable discovery, or both.
- **This is routing, not remote administration** — don't expect
  `fleet serve` to pull/load models on peers for you; it assumes each peer
  already manages its own models (use `download-model` / `fleet-vram-admission`
  per node for that).
- **Poll interval trades staleness vs. load** — a longer interval means
  routing decisions can lag behind a peer's actual state (e.g. just evicted
  a model); tighten it for latency-sensitive multi-tenant routing, loosen
  it to reduce polling overhead on a large fleet.
- **mDNS discovery is LAN-scoped** — it won't find nodes across routed
  subnets/VPNs without multicast relay; use a static `--peers` list for
  those topologies instead.
- **Not the same thing as `zerollama fleet` alone** — the fleet management
  node itself only has the `serve` subcommand; there's no separate "fleet
  client" CLI — agents just point their HTTP client at the fleet node's
  `--listen` address like any other zerollama endpoint.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `fleet-vram-admission` — single-host capacity/admission checks that the fleet node's polling relies on (`/api/status`)
