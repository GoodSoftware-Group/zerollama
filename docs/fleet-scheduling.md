# Fleet scheduling (multi-node)

This document describes the **directional** plan for running **many zerollama nodes** behind **agents and integrations**, without wasting GPU work on constrained hardware.

It complements:

- [ROADMAP.md](./ROADMAP.md) — fleet track milestones **F1–F6**
- [scheduling-vram-policy.md](./scheduling-vram-policy.md) — **per-node** queues, VRAM broker, training coordination
- [testing-smoke.md](./testing-smoke.md) — single-host smoke

---

## Why a management node

Each zerollama instance already has a **local scheduler** (Go ggml queue + optional Python runtime queue). That is correct for **one GPU (or one machine)** but agents talking to a **fleet** need a layer above it:

| Layer | Role | Why separate |
|-------|------|----------------|
| **Management / fleet scheduler** | Discovery, warm-model routing, assign node + short token | Cross-node view; no eviction on the wrong box |
| **Zerollama node** | Admit, queue, load, generate | Owns VRAM and runner lifecycle |
| **Agent** | Pick model, deadline, cancel policy | Cannot see other nodes’ queues without the fleet layer |

**What we are not building inside zerollama first:** a global optimizer that preempts in-flight decode across nodes, or long-lived **60s reservations** that oscillate when agents cancel. GPU nodes are too constrained for scatter-gather or quote storms.

---

## Design principles (from fleet UX work)

1. **Warm-model first** — Route to a node that **already has the model loaded** when possible. Cold load + eviction on a busy fleet is the expensive path; the management node should prefer residency over “any idle GPU.”
2. **Status over speculation** — Agents need **honest, low-latency signals** from the node they chose: queue depth, loading, generating. Prefer **commit-then-stream** with clear progress over multi-node probes that go stale.
3. **Cheap cancel only while queued** — On a node, cancellation before `loading` removes the request from the pending FIFO (`dropPendingOnCancel`). After load/eviction starts, cancel wastes work; agents should treat `status: loading` as commit.
4. **Short assignment window** — When the fleet layer grants a slot, use a **short TTL token** (order of seconds), not a long quote. Reduces thundering herd when agents change their mind.
5. **No scatter-gather on constrained GPUs** — Firing the same request at N nodes and cancelling N−1 can trigger evictions and loads on multiple boxes; avoid for large / cold models.

---

## Target architecture

```text
Agents / integrations
        │
        ▼
Fleet management node          ← discovery, model→node map, assign token
        │
        ├── zerollama A (:11434)  ggml_pending / loaded / stream status
        ├── zerollama B (:11434)
        └── zerollama C (:11434)

Discovery (directional): mDNS / DNS-SD (_zerollama._tcp) or static config + heartbeat
```

**Registration:** Each node **advertises** itself (host, port, capabilities, loaded models snapshot). The management node maintains a **ready-list**: e.g. “node B has `llama3` loaded, queue depth 0.”

**Assignment flow (directional):**

1. Agent: “I need model M, interactive latency.”
2. Management: pick node with M warm and lowest queue; issue **assignment token** (TTL ~5–10s).
3. Agent: `POST /api/chat` (or generate) **to that node** with token header (future) or direct URL.
4. First stream chunks: `accepted` → `queued` (optional) → `loading` → `generating` → tokens.
5. If `queued` and `position` exceeds agent threshold → **cancel** (HTTP context cancel) and request a new assignment **only while still `queued`.**

---

## Status contract (node ↔ agent)

### Shipped (fleet polling)

**`GET /api/status`** returns a point-in-time snapshot for management nodes and agents:

```json
{
  "cloud": { "disabled": true, "source": "env" },
  "inference": {
    "ggml": {
      "pending": 0,
      "active": 1,
      "loaded": 1,
      "loads_paused": false,
      "loading": false,
      "loaded_models": ["llama3:latest"]
    },
    "runtime": {
      "enabled": true,
      "available": true,
      "waiting": 0,
      "running": 0,
      "llama_loaded": true,
      "state": "idle"
    }
  }
}
```

When the sidecar is configured but `/health` fails, runtime queue fields are **omitted** (not zero):

```json
"runtime": { "enabled": true, "available": false }
```

| Field | Meaning |
|-------|---------|
| `inference.ggml.pending` | Requests waiting in the Go scheduler FIFO |
| `inference.ggml.active` | In-flight ggml work (refs + active load) |
| `inference.ggml.loaded` | Resident runner count (same snapshot as `loaded_models`) |
| `inference.ggml.loaded_models` | Short names of loaded models (warm routing); `len` equals `loaded` when names resolve |
| `inference.ggml.loading` | Scheduler is loading or evicting for a pending request |
| `inference.runtime.enabled` | `ZEROLLAMA_RUNTIME_URL` (or sidecar) is configured |
| `inference.runtime.available` | Runtime `/health` probe succeeded — **check this before using queue fields** |
| `inference.runtime.waiting` / `running` | Python runtime scheduler queues (omitted when `available` is false) |
| `inference.runtime.llama_loaded` | Runtime has an active llama-server (omitted when `available` is false) |

Poll interval for fleet management: **1–5s** is typical; combine with stream progress for in-request updates.

### Shipped (F3 management node v0)

**Operator guide:** [fleet-management.md](./fleet-management.md) — quick start, API, env, agent pattern, and **why** the manager stays thin.

Run a thin management process that polls peers and assigns agents to a node:

```bash
ZEROLLAMA_FLEET_PEERS=http://192.168.1.10:11434,http://192.168.1.11:11434 \
  zerollama fleet serve --listen 0.0.0.0:11450
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness |
| `/api/fleet/status` | GET | All peer snapshots + **warm_models** map |
| `/api/fleet/assign` | POST | Pick `{url, node_id}` for a model |

**Assign request:**

```json
{
  "model": "llama3:latest",
  "prefer_warm": true,
  "warm_only": false,
  "exclude": ["192.168.1.10:11434"]
}
```

**Assign response:**

```json
{
  "url": "http://192.168.1.11:11434",
  "node_id": "192.168.1.11:11434",
  "warm": true,
  "queue_depth": 0,
  "generated_at": "2026-06-12T20:00:00Z"
}
```

**Routing (v0):** Prefer nodes with the model in `inference.ggml.loaded_models` and lowest combined queue (`pending + active` + runtime `waiting + running` when runtime probe is available). Cold route picks lowest queue among available peers. Management **does not** load models or evict on remote nodes.

**Env:** `ZEROLLAMA_FLEET_PEERS`, `ZEROLLAMA_FLEET_LISTEN` (default `0.0.0.0:11450`), `ZEROLLAMA_FLEET_POLL_INTERVAL` (default `3s`).

### Shipped (streaming progress)

On **`/api/chat`** and **`/api/generate`** with `stream: true` (default), nodes emit NDJSON progress before content:

| Field | Meaning |
|-------|---------|
| `status` | `accepted`, `queued`, `loading`, `generating` |
| `position`, `queue_depth` | FIFO position when queued (ggml path) |
| `detail` | Human-readable line for UI |
| `done: false`, empty content | Progress-only chunk |

Agents should treat **`status` + empty content** as progress UI, not model output.

### Planned (fleet-facing)

| Surface | Purpose |
|---------|---------|
| **Assignment token** | Optional header validated by node; short TTL; one slot held |
| **mDNS advertisement** | LAN discovery with loaded-model hints in TXT records |

Per-node backlog internals (for reference):

- Go: `InferenceBacklog()`, `schedLoadedModelNames()`
- Python runtime: `/health` `waiting`, `running`, `llama_server`

---

## Discovery: mDNS and alternatives

**Directional default for LAN / homelab:** **mDNS** (DNS-SD) service type e.g. `_zerollama._tcp.local` with TXT records for version, tags, optional `models_loaded` hash.

| Mechanism | Best for | Notes |
|-----------|----------|-------|
| **mDNS / Bonjour** | Mac, Linux LAN, agents on same subnet | Zero config; management node browses and watches |
| **Static config** | K8s, cloud, fixed IPs | `ZEROLLAMA_FLEET_PEERS` or management YAML |
| **K8s Service / headless** | Cluster deployments | DNS replaces mDNS |

**Non-goal:** Zerollama nodes pulling work from a central Redis queue (pattern 5) as the **first** fleet story—that adds infra before warm routing and status are solid. Can follow for batch throughput later.

---

## Warm-model routing policy

Management node keeps a **model residency map**:

```text
llama3:latest  → [node-A (loaded, q=0), node-C (loaded, q=2)]
qwen2.5:14b    → [node-B (loaded, q=0)]
```

**Routing rules (directional):**

1. Prefer **loaded + lowest queue** for requested model.
2. If none loaded, pick node with **most free VRAM headroom** (from health) and accept cold-load cost—or reject if SLA requires warm only.
3. Optional **sticky shard**: pin heavy models to dedicated nodes to reduce churn (documented operator choice, not required in v1).

This aligns with existing per-node behavior: `GetRunner` **fast path** when model already loaded (ticket `0`); fleet layer maximizes how often agents hit that path.

---

## Anti-patterns (explicit non-goals)

| Pattern | Problem on constrained GPUs |
|---------|-------------------------------|
| Scatter-gather (same request to N nodes) | Multiple loads/evictions; cancel after dequeue wastes scheduling |
| Long quote / 60s reservation | Stale estimates; cancel storms; queue oscillation |
| Probe-only routing without stream feedback | Race between probe and request; no cheap bail-out |
| Fleet layer starting loads remotely | Only the **target node** should load/evict; management assigns, does not execute |

---

## Relationship to single-node scheduling

| Concern | Where it lives |
|---------|----------------|
| FIFO, eviction, `keep_alive` | Go `server/sched.go` |
| Runtime admission, PA queue | Python `runtime/` |
| Training vs inference VRAM | Go broker + T6 policy — [scheduling-vram-policy.md](./scheduling-vram-policy.md) |
| **Which node for model M** | **Fleet management** (new) |
| **Agent cancel while queued** | Client policy + HTTP cancel → `dropPendingOnCancel` |

Phase **11+ / T6** unified **policy on one GPU** remains independent of fleet routing. A management node does not replace the VRAM broker; it **chooses a node** that already runs that broker.

---

## Milestones

See [ROADMAP.md — Fleet scheduling track](./ROADMAP.md#fleet-scheduling-multi-node).

| Milestone | Summary |
|-----------|---------|
| **F1** | Stream progress contract documented; agents can implement cancel-while-queued |
| **F2** | **`GET /api/status`** inference snapshot for management polling |
| **F3** | Management node v0: static peer list + warm-model map + assign URL | **Shipped** |
| **F4** | mDNS registration/browse for LAN discovery |
| **F5** | Short-TTL assignment token (optional header) |
| **F6** | Operator docs: sticky shards, SLA classes, when to reject cold route |

---

## Agent integration sketch

```python
# Directional — not a shipped SDK
import requests

fleet_base = "http://127.0.0.1:11450"
assignment = requests.post(
    f"{fleet_base}/api/fleet/assign",
    json={"model": "llama3", "prefer_warm": True},
    timeout=5,
).json()
stream = client.chat(model="llama3", messages=..., base_url=assignment["url"])
for chunk in stream:
    if chunk.get("status") == "queued" and chunk.get("position", 0) > MAX_QUEUE:
        stream.cancel()  # cheap: still in pending FIFO
        assignment = requests.post(
            f"{fleet_base}/api/fleet/assign",
            json={"model": "llama3", "exclude": [assignment["node_id"]]},
            timeout=5,
        ).json()
        ...
    elif chunk.get("status") == "loading":
        break  # commit — do not cancel
```

---

## How to contribute

Open an issue with: deployment shape (LAN vs K8s), model sizes, agent SLA (interactive vs batch), and whether **warm-only** routing is required. Fleet work should stay **thin**—heavy scheduling remains on each zerollama node.
