# Fleet management node (F3)

**Audience:** operators running **multiple zerollama hosts** and agents that need warm-model routing.

**Related:** [fleet-scheduling.md](./fleet-scheduling.md) (design principles, anti-patterns), [fleet-playbooks.md](./fleet-playbooks.md) (F6 sticky / warm-only / cancel), [ROADMAP.md](./ROADMAP.md#fleet-scheduling-multi-node), [testing-smoke.md](./testing-smoke.md) (single-node).

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
| `/api/fleet/assign` | POST | Return `{url, node_id, warm, queue_depth, assignment_token?}` for a model |
| `/api/fleet/assign-hold` | POST | **On each node** — register F5 soft hold from a minted token |

### Assign request

```json
{
  "model": "llama3:latest",
  "prefer_warm": true,
  "warm_only": false,
  "exclude": ["192.168.1.10:11434"],
  "session_key": "agent-thread-1",
  "prefix_block_hashes": ["abc…", "def…"]
}
```

| Field | Why it exists |
|-------|----------------|
| `prefer_warm` | Default true — cold load + eviction on a busy fleet is the expensive path; prefer residency. |
| `warm_only` | SLA gate — reject cold route when no node has the model loaded (HTTP 404). |
| `exclude` | Retry after cancel-while-queued (F1 stream `status: queued`) without picking the same overloaded node. |
| `session_key` / `prompt_cache_key` | L3 affinity + soft radix residency score when hashes are absent. |
| `prefix_block_hashes` | L3-R9 / LA13 — ordered hashes from token 0; longest leading match against peer `inference.runtime.radix.block_hashes`. Build with Go `prefixblock.Hashes` (L3-R11). |

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

**Queue depth** = `ggml.pending + ggml.active` plus runtime `waiting + running` when runtime probe is `available` (F2 omits runtime queue fields when probe fails — manager respects that). Soft F5 holds are folded into `ggml.pending` (and exposed as `ggml.assign_holds`).

**Model matching:** case-insensitive exact name, or base name before `:` (`llama3` matches `llama3:latest`).

**Non-goals (v0):** remote load/evict, global preemption, long reservation markets.

---

## Assignment tokens (F5)

**Why:** two agents can get the same “queue 0” assign within one poll interval and race the warm slot. A **short** hold (~5–10s) bumps reported queue depth until the agent’s chat/generate arrives or the TTL expires — not a 60s quote.

### Enable (same secret on fleet manager + nodes)

```bash
export ZEROLLAMA_FLEET_ASSIGN_SECRET='shared-fleet-hmac'
# optional: ZEROLLAMA_FLEET_ASSIGN_TTL=8s  ZEROLLAMA_FLEET_ASSIGN_PUSH=1
```

### Assign response (when secret set)

```json
{
  "url": "http://192.168.1.11:11434",
  "node_id": "192.168.1.11:11434",
  "warm": true,
  "queue_depth": 0,
  "assignment_token": "…",
  "expires_at": "2026-07-17T17:45:08Z",
  "expires_in": 8
}
```

Fleet best-effort `POST {url}/api/fleet/assign-hold` with the token so peers see `assign_holds` immediately. Agents still send:

```http
X-Zerollama-Assignment-Token: <assignment_token>
```

on `/api/chat` or `/api/generate`. Invalid → **401**; expired → **409**. Missing header remains allowed (non-breaking).

| Env | Default | Meaning |
|-----|---------|---------|
| `ZEROLLAMA_FLEET_ASSIGN_SECRET` | (empty) | HMAC key; empty disables tokens |
| `ZEROLLAMA_FLEET_ASSIGN_TOKEN` | on if secret | `0` kill-switch |
| `ZEROLLAMA_FLEET_ASSIGN_TTL` | `8s` | clamp 2–30s |
| `ZEROLLAMA_FLEET_ASSIGN_PUSH` | on | fleet→node hold register after mint |

---

## LAN discovery (F4 mDNS)

**Why opt-in:** multicast on shared networks is surprising; homelab operators enable explicitly. K8s and fixed-IP deployments keep static peers.

### Inference nodes

```bash
ZEROLLAMA_MDNS=1 OLLAMA_HOST=0.0.0.0:11434 zerollama serve
```

Advertises **`_zerollama._tcp`** on the listen port with TXT `role=node`, `version=…`.

### Fleet manager — browse only (no static peers)

```bash
zerollama fleet serve --mdns --listen 0.0.0.0:11450
# or: ZEROLLAMA_FLEET_MDNS=1 zerollama fleet serve
```

### Fleet manager — static + mDNS merge

```bash
ZEROLLAMA_FLEET_PEERS=http://10.0.0.5:11434 \
  ZEROLLAMA_FLEET_MDNS=1 \
  zerollama fleet serve
```

Static peers are always polled; mDNS adds LAN hosts as they appear.

### Advertise fleet endpoint for agents

```bash
zerollama fleet serve --mdns --mdns-advertise --listen 0.0.0.0:11450
```

Registers **`_zerollama-fleet._tcp`** so agents can find the manager without a fixed IP. Warm-model details still come from F2 polling, not TXT records (v0).

---

## Environment

| Variable | Default | Why |
|----------|---------|-----|
| `ZEROLLAMA_FLEET_PEERS` | *(optional with mDNS)* | Static peer list — required unless `ZEROLLAMA_FLEET_MDNS=1`. K8s headless DNS stays static-only. |
| `ZEROLLAMA_FLEET_LISTEN` | `0.0.0.0:11450` | Separate port from zerollama `:11434` so one host can run both node + manager. |
| `ZEROLLAMA_FLEET_POLL_INTERVAL` | `3s` | Balance freshness vs load; 1–5s typical per [fleet-scheduling.md](./fleet-scheduling.md). |
| `ZEROLLAMA_FLEET_ASSIGN_SECRET` | *(empty)* | F5 HMAC secret on **manager + nodes**; empty disables tokens. |
| `ZEROLLAMA_FLEET_ASSIGN_TTL` | `8s` | Soft-hold window (clamped 2–30s). |
| `ZEROLLAMA_FLEET_ASSIGN_PUSH` | `1` | Manager POSTs `/api/fleet/assign-hold` after mint. |
| `ZEROLLAMA_MDNS` | `0` | On inference nodes: advertise `_zerollama._tcp` when `zerollama serve` starts. |
| `ZEROLLAMA_FLEET_MDNS` | `0` | Fleet manager: browse LAN for `_zerollama._tcp` peers (merged with static list). |
| `ZEROLLAMA_FLEET_MDNS_ADVERTISE` | `0` | Fleet manager: advertise `_zerollama-fleet._tcp` for agent discovery. |

CLI flags override env: `zerollama fleet serve --peers ... --listen ... --poll-interval 5s --mdns --mdns-advertise`.

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

For Apple Silicon CI/dev, `./scripts/serve/serve_mac_runtime.sh` starts sidecar + Go proxy. **Why logs were quiet:** both processes background to files so CI stays clean. The script now prints startup progress and log paths (`MACOS_RT_LOG`, `MACOS_GO_LOG`). Daily use remains `zerollama serve` (Darwin bootstrap).

---

## What's next (fleet track)

| Milestone | Why |
|-----------|-----|
| **F5 assignment token** | **Done (Jul 2026)** — see [Assignment tokens (F5)](#assignment-tokens-f5). |
| **F6 playbooks** | **Done (Jul 2026)** — [fleet-playbooks.md](./fleet-playbooks.md) (sticky shards, warm-only SLA, cancel policy). |

See [fleet-scheduling.md](./fleet-scheduling.md) for anti-patterns (scatter-gather, long quotes).
