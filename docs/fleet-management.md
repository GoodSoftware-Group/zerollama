# Fleet management node (F3)

**Audience:** operators running **multiple zerollama hosts** and agents that need warm-model routing.

**Related:** [fleet-scheduling.md](./fleet-scheduling.md) (design principles, anti-patterns), [ROADMAP.md](./ROADMAP.md#fleet-scheduling-multi-node), [testing-smoke.md](./testing-smoke.md) (single-node).

---

## Why not route inside each zerollama?

Per-node schedulers (`server/sched.go`, Python `runtime/`) are correct for **one GPU**. They know pending FIFO depth, loaded runners, and when eviction is safe. They cannot see other machines.

Putting fleet logic **inside** every node would require either:

- **Scatter-gather** — same request to N nodes; cancel N−1 after one wins → multiple loads/evictions on constrained GPUs.
- **Quote/reservation markets** — 60s holds that go stale when agents change their mind → cancel storms and queue oscillation.

F3 is deliberately **thin**: a separate process polls status and **assigns** a URL. The target node still owns admission, load, and generate.

---

## Quick start

### 1. Run zerollama on each GPU host

```bash
# On each box (ports can differ)
OLLAMA_HOST=0.0.0.0:11434 zerollama serve
```

### 2. Start the management node

```bash
export ZEROLLAMA_FLEET_PEERS=http://192.168.1.10:11434,http://192.168.1.11:11434
zerollama fleet serve --listen 0.0.0.0:11450
```

**Why `0.0.0.0:11450` default:** agents on other LAN hosts must reach the manager; bind loopback-only only when everything is local.

### 3. Assign and call the chosen node

```bash
# Pick a node
curl -s -X POST http://127.0.0.1:11450/api/fleet/assign \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.2:3b","prefer_warm":true}' | jq .

# Generate on the assigned URL (example)
curl -s http://192.168.1.11:11434/api/generate -d '{"model":"llama3.2:3b","prompt":"hi","stream":false}'
```

---

## HTTP API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness |
| `/api/fleet/status` | GET | All peer snapshots + `warm_models` map |
| `/api/fleet/assign` | POST | Return `{url, node_id, warm, queue_depth}` for a model |

### Assign request

```json
{
  "model": "llama3:latest",
  "prefer_warm": true,
  "warm_only": false,
  "exclude": ["192.168.1.10:11434"]
}
```

| Field | Why it exists |
|-------|----------------|
| `prefer_warm` | Default true — cold load + eviction on a busy fleet is the expensive path; prefer residency. |
| `warm_only` | SLA gate — reject cold route when no node has the model loaded (HTTP 404). |
| `exclude` | Retry after cancel-while-queued (F1 stream `status: queued`) without picking the same overloaded node. |

### Assign response

```json
{
  "url": "http://192.168.1.11:11434",
  "node_id": "192.168.1.11:11434",
  "warm": true,
  "queue_depth": 0,
  "loading": false,
  "generated_at": "2026-06-12T20:00:00Z"
}
```

**Why `warm` on cold route:** when `prefer_warm` is false, the manager still reports whether the chosen node already had the model loaded — useful for metrics, not just routing policy.

---

## Routing policy (v0)

1. Filter to **available** peers (recent `/api/status` probe succeeded).
2. If `prefer_warm`: among nodes with model in `inference.ggml.loaded_models`, pick **lowest queue**.
3. Else if not `warm_only`: pick lowest queue among all available peers (cold route).
4. Tie-break: prefer node not `loading`, then stable sort by `node_id`.

**Queue depth** = `ggml.pending + ggml.active` plus runtime `waiting + running` when runtime probe is `available` (F2 omits runtime queue fields when probe fails — manager respects that).

**Model matching:** case-insensitive exact name, or base name before `:` (`llama3` matches `llama3:latest`).

**Non-goals (v0):** remote load/evict, global preemption, assignment tokens (F5), mDNS (F4).

---

## Environment

| Variable | Default | Why |
|----------|---------|-----|
| `ZEROLLAMA_FLEET_PEERS` | *(required)* | Static peer list — K8s headless DNS or mDNS (F4) can replace later; explicit config avoids surprise LAN traffic. |
| `ZEROLLAMA_FLEET_LISTEN` | `0.0.0.0:11450` | Separate port from zerollama `:11434` so one host can run both node + manager. |
| `ZEROLLAMA_FLEET_POLL_INTERVAL` | `3s` | Balance freshness vs load; 1–5s typical per [fleet-scheduling.md](./fleet-scheduling.md). |

CLI flags override env: `zerollama fleet serve --peers ... --listen ... --poll-interval 5s`.

---

## Agent integration pattern

```python
import requests

fleet = "http://mgr:11450"
assignment = requests.post(
    f"{fleet}/api/fleet/assign",
    json={"model": "llama3", "prefer_warm": True},
    timeout=5,
).json()

# Call zerollama directly on assigned node
stream = ollama_client.chat(
    model="llama3",
    messages=[...],
    host=assignment["url"],
)

for chunk in stream:
    if chunk.get("status") == "queued" and chunk.get("position", 0) > MAX_QUEUE:
        stream.cancel()  # cheap: still in pending FIFO (F1)
        assignment = requests.post(
            f"{fleet}/api/fleet/assign",
            json={"model": "llama3", "exclude": [assignment["node_id"]]},
        ).json()
        ...
    elif chunk.get("status") == "loading":
        break  # commit — do not cancel; load/evict already started
```

**Why stream progress matters:** assignment is a point-in-time snapshot; queue depth can change before your request arrives. F1 `status` chunks let agents bail out cheaply only while `queued`.

---

## Local dev (single machine)

```bash
# Terminal 1 — node A
OLLAMA_HOST=127.0.0.1:11434 zerollama serve

# Terminal 2 — node B (optional second port for testing)
OLLAMA_HOST=127.0.0.1:11435 zerollama serve

# Terminal 3 — manager
ZEROLLAMA_FLEET_PEERS=http://127.0.0.1:11434,http://127.0.0.1:11435 \
  zerollama fleet serve --listen 127.0.0.1:11450

curl -s http://127.0.0.1:11450/api/fleet/status | jq .
```

---

## macOS runtime stack (related)

For Apple Silicon CI/dev, `./scripts/serve_mac_runtime.sh` starts sidecar + Go proxy. **Why logs were quiet:** both processes background to files so CI stays clean. The script now prints startup progress and log paths (`MACOS_RT_LOG`, `MACOS_GO_LOG`). Daily use remains `zerollama serve` (Darwin bootstrap).

---

## What's next (fleet track)

| Milestone | Why |
|-----------|-----|
| **F4 mDNS** | Zero-config LAN discovery instead of static `ZEROLLAMA_FLEET_PEERS`. |
| **F5 assignment token** | Short TTL hold (~5–10s) so two agents don't race the same queue slot after assign. |
| **F6 playbooks** | Sticky shards, warm-only SLA, documented cancel policy for operators. |

See [fleet-scheduling.md](./fleet-scheduling.md) for anti-patterns (scatter-gather, long quotes).
