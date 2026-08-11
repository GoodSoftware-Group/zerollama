# OpenAI harness QoS wire shapes — findings & learnings

**Audience:** contributors wiring `/v1/chat/completions` clients; anyone debugging `unknown field: project_name, qos_class`.

**Related:** [agent-qos-and-project-tracking.md](./agent-qos-and-project-tracking.md), [mlx-agent-prompts.md](./mlx-agent-prompts.md), [model-serving-minefield.md](./model-serving-minefield.md) (trap 77).

**Status:** Shipped Jul 2026 (M15c) — allowlist + fold on bind; same fold on runtime v1 proxy options forwarded to Python.

---

## Why this doc exists

Hermes (and similar harnesses) send Tier 2 QoS via OpenAI SDK `extra_body`:

```python
client.chat.completions.create(
    model="gemma4:26b-optiq",
    messages=[...],
    extra_body={
        "qos_class": "auxiliary",
        "project_name": "discord:dm:123",
        "project_id": "hermes-lean",
    },
)
```

**What operators saw:** native `/api/chat` with nested `options.zerollama` worked; the auxiliary OpenAI path returned **HTTP 400** `unknown field: project_name, qos_class`. Main-model traffic sometimes “worked” because its provider config never actually flattened those keys onto the wire.

**Why that is worse than a soft ignore:** trap 77 deliberately 400s unknown top-level keys so typos do not silently measure the lane default. Flat harness keys were treated as typos until we allowlisted and folded them.

---

## Finding 1 — OpenAI Python SDK flattens `extra_body`

**What:** The official OpenAI Python SDK merges `extra_body` onto the **HTTP JSON root**. Nested `extra_body.zerollama` becomes top-level `"zerollama": {…}`; flat `extra_body.qos_class` becomes top-level `"qos_class"`.

**Why it matters:** Server code that only read `options.zerollama` or nested `extra_body` never saw the harness metadata. Allowlisting alone would stop the 400 but leave QoS/scheduling blind unless we **fold** into `options.zerollama`.

**Learning:** Treat three wire shapes as one contract:

| Wire shape | Typical producer |
|------------|------------------|
| `options.zerollama.{qos_class,…}` | Native `/api/chat`, careful OpenAI clients |
| Top-level `"zerollama": {…}` | SDK flatten of `extra_body.zerollama` |
| Flat top-level / flat `extra_body` keys | SDK flatten of `extra_body.qos_class` etc. |

---

## Finding 2 — Trap 77 must stay loud

**What:** `CheckUnknownChatCompletionFields` / `rejectUnknownChatCompletionFields` (minefield trap 77) reject invented top-level keys with 400.

**Why keep it:** Silent fail-open turns a mistyped `qos_clas` into “looks like interactive default” and poisons latency/QoS measurements. Allowlisting known harness aliases is the escape hatch; invented keys still fail.

**Learning:** Grow `chatCompletionPassthroughFields` and `chatCompletionZerollamaFlatFields` **together**. Passthrough alone = accept-and-drop; flat-fields alone = 400 again. Two hand-maintained lists — add a test if you add a key.

---

## Finding 3 — Folding for Go is not enough for the Python proxy

**What (audit Jul 2026):** `BindChatCompletionRequest` correctly folded flat keys for the **native** OpenAI middleware path. `proxyOptsFromV1Body` folded for Go-side routing (`resolveRuntimeProxy`, `runtimeForceUnload`). But `runtimeV1ProxyOptions` — which builds the `options` object **forwarded to Python** — only copied `body["options"]` and never called `FoldFlatZerollamaMap`.

**Why it hurt:** Models routed through `runtimeV1ChatCompletionsProxy` would:

- stop 400ing ✅
- influence exclusive-fulfillment / force-unload on the Go side ✅
- **never** deliver `options.zerollama` to Python admission/scheduling ❌

**Learning:** Any path that synthesizes a new `options` map for another process must reuse the same fold helper (`proxyOptsFromV1Body` → `runtimeV1ProxyOptions`). “Parity with bind” means parity on the **wire out**, not only on in-process Go structs.

**Fix:** `runtimeV1ProxyOptions` seeds client opts from `proxyOptsFromV1Body(body)` before GGUF/`num_predict` injection.

---

## Finding 4 — Precedence must be explicit

**What:** When multiple shapes appear in one request, conflicts need a rule.

**Why nested wins:** `options.zerollama` is the explicit Ollama-shaped contract. Top-level `zerollama` is usually an SDK flatten of the same intent. Flat aliases are the weakest (SDK flatten of a flat `extra_body`). Preferring nested avoids “SDK flatten overwrote my careful options.”

| Strength | Source |
|----------|--------|
| Strongest | `options.zerollama` (and `extra_body.options.zerollama` after options merge) |
| Middle | Top-level / `extra_body.zerollama` object (underlays nested) |
| Weakest | Flat `qos_class`, `project_*`, … |

Underlay merge = fill gaps from weaker sources; stronger keys win on conflict.

---

## Finding 5 — Workspace hygiene vs feature diffs

**What:** While verifying this work, unrelated half-applied WIP (envconfig deletions, truncated `api/types.go`, inconsistent llama vocab / mlxrunner cache) blocked `go test ./openai ./server`.

**Why call it out:** A green feature branch can still look broken if the working tree carries unrelated incomplete edits. Restore compile blockers from HEAD before blaming the feature; keep intentional fixes (e.g. MTLB Metal embed load, `json.NewDecoder` for large MLX responses).

---

## Code map (M15c)

| Area | Path | Why |
|------|------|-----|
| Trap 77 + allowlist | `openai/chat_unknown_fields.go` | Known SDK/harness keys must not 400 |
| Bind + fold + precedence | `openai/chat_extras.go` | Single bind path for middleware |
| Exported fold helper | `FoldFlatZerollamaMap` | Shared with server proxy |
| Proxy extract | `server/runtime_proxy.go` `proxyOptsFromV1Body` | Go routing + force-unload QoS |
| Forward to Python | `server/runtime_manifest.go` `runtimeV1ProxyOptions` | Sidecar must see folded zerollama |
| Version advertisement | `server/mlx_qos.go` `zerollamaVersionQoS().openai` | Clients probe flat aliases once |
| Bind tests | `openai/chat_extras_test.go` | Flat / object / nested / precedence |
| Proxy tests | `server/runtime_proxy_options_test.go`, `runtime_v1_proxy_options_test.go` | Fold + HTTP forward |

---

## Operator checklist

1. Prefer `options.zerollama` or `extra_body: { "zerollama": {…} }` on new clients.
2. If you must send flat `extra_body` keys (legacy Hermes), expect fold — not 400.
3. Probe `GET /api/version` → `.zerollama.qos.openai.extra_body` for advertised flat aliases.
4. Typos of unknown top-level keys should still 400 (trap 77).
5. On runtime-proxied models, confirm Python sees QoS via forwarded `options.zerollama` (not only Go logs).

---

## Non-goals

- Accepting arbitrary top-level keys (defeats trap 77).
- Changing native `/api/chat` field names (already nested `options.zerollama`).
- Removing the preferred nested shape — flat aliases are compatibility, not the recommended API.

---

## Follow-on (M15e)

Allowlisting alone was also insufficient for native **`think`** (accept-and-drop). M15e binds `think` / `timeout`, adds wait-abort `preempted_reason`, cache-pin, batch chat, and can-load topology. Findings: [hermes-gap-closure-findings.md](./hermes-gap-closure-findings.md).
