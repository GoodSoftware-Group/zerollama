# Fleet playbooks (F6)

**Audience:** operators pinning models across a LAN fleet, and agents that call `POST /api/fleet/assign` then chat/generate on the returned URL.

**Related:** [fleet-management.md](./fleet-management.md) (F3–F5 how-to), [fleet-scheduling.md](./fleet-scheduling.md) (design + anti-patterns), [ROADMAP.md](./ROADMAP.md#fleet-scheduling-multi-node).

**Status:** **Done (Jul 2026)** — documentation milestone. No new control-plane API in F6; these playbooks codify sticky shards, warm-only SLA, and cancel policy on top of F1–F5 / F7.

---

## Quick decision table

| Situation | Do this |
|-----------|---------|
| Interactive agents, latency SLA | Sticky shards + `warm_only: true` (or `prefer_warm` default) |
| Overnight batch, cold load OK | `prefer_warm: false` (or omit `warm_only`) on assign |
| Stream says `queued` and position too deep | **Cancel** HTTP request → re-assign with `exclude` |
| Stream says `loading` or `generating` | **Do not cancel** — work already committed on that GPU |
| Two agents race the same warm slot | Enable F5 assign secret / soft-hold (~5–10s), not a 60s quote |

---

## Playbook A — Sticky model shards (operators)

**Goal:** Heavy models stay resident on dedicated nodes so agents hit Go’s loaded-runner fast path more often and churn (evict ↔ load) stays low.

### Pattern

1. Pick **shard nodes** by VRAM class (e.g. 70B on A/B, 8B–14B on C/D).
2. On each shard, **pre-warm** the pinned model(s) once after boot:

```bash
# On the shard that owns llama3.1:70b
curl -s http://127.0.0.1:11434/api/generate -d '{
  "model": "llama3.1:70b",
  "prompt": " ",
  "stream": false,
  "options": { "num_predict": 1 }
}'
```

3. Keep residency with a long `keep_alive` on warm traffic (or a small cron generate) so idle unload does not undo the shard.
4. Point the fleet manager at all peers (`ZEROLLAMA_FLEET_PEERS` or mDNS). Assign still **chooses**; it does not load remotely — the shard must already be warm for `warm_only` to succeed.

### Optional: agent-side pin

When you know the shard URL, skip assign for that model and call the node directly. Prefer assign + `warm_only` when shards can fail over (second node also warm).

### What sticky is not

- Not a fleet API field.
- Not remote load/evict from the manager.
- Not a reason to scatter-gather “whoever warms first.”

---

## Playbook B — Warm-only vs prefer-warm SLA (operators + agents)

Fleet assign already implements both policies ([`fleet/types.go`](../fleet/types.go)):

| Field | Behavior |
|-------|----------|
| `prefer_warm` (default **true** when omitted) | Prefer loaded peers; may still cold-route if no warm peer (with cold score penalty). |
| `warm_only: true` | **Reject** (no assignment) when no peer has the model loaded — interactive SLA. |

### Interactive / agent fleet

```bash
curl -s -X POST http://127.0.0.1:11450/api/fleet/assign \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.1:70b","warm_only":true}'
```

- **200 + `url`:** generate on that host; treat as warm path.
- **No warm peer:** fail closed — surface “model not warm” to the user or queue for a shard to finish loading; do **not** silently cold-load a 70B onto a busy box.

Combine with sticky shards so `warm_only` usually succeeds.

### Batch / overnight

```bash
curl -s -X POST http://127.0.0.1:11450/api/fleet/assign \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.1:70b","prefer_warm":false}'
```

Cold route is allowed; expect load latency and possible eviction on the chosen node. Monitor `GET /api/fleet/status` / per-node `/api/status` for queue and `loaded_models`.

### Prefix / Radix hint (when enabled)

If L3 Radix / content-hash scoring is on, pass `prefix_block_hashes` (or rely on fleet soft scores) so assign prefers peers that already hold the shared prefix — still subject to warm policy above. See [radix-prefix-share.md](./radix-prefix-share.md) and fleet F7 / L3-R9.

---

## Playbook C — Cancel policy (agents)

**Contract (F1):** streaming `/api/chat` and `/api/generate` emit `status`: `accepted` → `queued` → `loading` → `generating` (plus `position` / `queue_depth` while queued).

| Status | Cancel? | Why |
|--------|---------|-----|
| `accepted` / `queued` | **Yes** | Node still in pending FIFO (`dropPendingOnCancel`); cheap to retry elsewhere. |
| `loading` | **No** | Eviction/load started; cancel wastes VRAM work. |
| `generating` | **No** | Decode in flight; cancel only for user abort, not for re-routing. |

### Recommended agent loop

```python
# Directional sketch — not a shipped SDK
import requests

FLEET = "http://127.0.0.1:11450"
MAX_QUEUE = 3

def assign(model, exclude=None, warm_only=True):
    body = {"model": model, "warm_only": warm_only}
    if exclude:
        body["exclude"] = exclude
    r = requests.post(f"{FLEET}/api/fleet/assign", json=body, timeout=5)
    r.raise_for_status()
    return r.json()

def chat_with_bail(model, messages, client):
    tried = []
    while True:
        a = assign(model, exclude=tried or None, warm_only=True)
        tried.append(a["node_id"])
        headers = {}
        if tok := a.get("assignment_token"):
            headers["X-Zerollama-Assignment-Token"] = tok
        stream = client.chat(
            model=model,
            messages=messages,
            host=a["url"],
            headers=headers,
            stream=True,
        )
        for chunk in stream:
            st = chunk.get("status")
            if st == "queued" and chunk.get("position", 0) > MAX_QUEUE:
                stream.cancel()
                break  # re-assign
            if st in ("loading", "generating") or chunk.get("done"):
                return  # committed or finished
        else:
            return
```

### With F5 assignment tokens

When `ZEROLLAMA_FLEET_ASSIGN_SECRET` is set on manager + nodes:

1. Assign returns `assignment_token` / `expires_in` (~5–10s).
2. Send `X-Zerollama-Assignment-Token` on the first chat/generate to the node (soft-hold bumps pending until arrival or TTL).
3. Still **do not** treat the token as a long quote — if you cancel while queued, drop the token and re-assign; do not hold for minutes.

Details: [fleet-management.md — Assignment tokens (F5)](./fleet-management.md#assignment-tokens-f5).

---

## Playbook D — Operator checklist (new fleet)

1. [ ] Each GPU host: `zerollama serve` reachable from the manager.
2. [ ] Manager: `ZEROLLAMA_FLEET_PEERS=…` or `--mdns`; `zerollama fleet serve --listen …`.
3. [ ] Sticky shards: pre-warm heavy models; document which node owns which tag.
4. [ ] Agents: `warm_only` for interactive; cancel only while `queued`.
5. [ ] Optional: shared `ZEROLLAMA_FLEET_ASSIGN_SECRET` for F5 soft-holds.
6. [ ] Confirm anti-patterns below are not in client code or runbooks.

---

## Explicit non-goals (do not build / do not document as supported)

| Anti-pattern | Why it hurts constrained GPUs |
|--------------|-------------------------------|
| **Scatter-gather** (same request to N nodes, cancel N−1) | Multiple loads/evictions; cancel after dequeue still wastes scheduling |
| **Long quotes / ~60s reservations** | Stale holds; cancel storms; queue oscillation — use F5 short TTL instead |
| **Fleet remote load/evict** | Manager **assigns** URL only; the target node owns admission |
| **Cancel after `loading`** to “try another node” | Eviction/load already paid; re-route amplifies thrash |
| **Global preemption across nodes** | Out of scope; per-node schedulers + T6 stay local |

These match [fleet-scheduling.md — Anti-patterns](./fleet-scheduling.md#anti-patterns-explicit-non-goals).

---

## Where the code lives

| Concern | Location |
|---------|----------|
| Assign + `prefer_warm` / `warm_only` / `exclude` | `fleet/assign.go`, `fleet/types.go` |
| Filter-then-score (F7) | `fleet/score.go` |
| Assignment token + soft-hold (F5) | `fleet/assigntoken.go`, `server/assign_hold.go` |
| Stream `status` (F1) | Go chat/generate streaming paths |
| Node snapshot (F2) | `GET /api/status` → `inference.ggml` / `inference.runtime` |
