# Launch model inventory (`LaunchModel`)

**Audience:** contributors working on `zerollama launch` integrations (OpenCode, OpenClaw, Pi, Droid, OMP, Cline, …).

**Related:** [upstream-ollama-diff.md](./upstream-ollama-diff.md#cherry-pick-status-jun-2026-upstream-07ed7523--v03010), [phase17-llama-server.md](./phase17-llama-server.md), [ROADMAP.md](./ROADMAP.md#phase-17--upstream-gguf-path-alignment-directional).

---

## Why this exists

Before upstream v0.30.10, each launch integration called **`/api/show` per model** when writing agent config files — to learn vision/thinking capabilities, context length, and cloud limits. That pattern had three problems:

1. **Latency** — configuring five models meant five sequential Show calls (often 5–25 s with timeouts).
2. **Drift** — `/api/tags` and `/api/show` could disagree; the picker and the config writer saw different metadata.
3. **Fragility** — slow or unreachable Show responses produced half-empty configs (missing `contextWindow`, wrong `input` modalities).

Upstream consolidated metadata into **one inventory load per launch run**: list tags once, resolve selected names to rich structs, pass **`[]LaunchModel`** to every `Edit` / `ConfigureWithModels` / `Run` path.

Zerollama ports that pattern so agent integrations stay mergeable with upstream Ollama without reintroducing N× Show at config time.

---

## Data flow

```text
Client / zerollama launch
        │
        ▼
  GET /api/tags  ──►  modelInventory.Load()
        │                  │
        │                  ▼
        │            []LaunchModel  (capabilities, context, remote, tools, …)
        │                  │
        ▼                  ▼
  model picker      inventory.Resolve(selected names)
  (buildModelList)           │
                             ▼
                    integration.Edit([]LaunchModel)
                    integration.ConfigureWithModels(primary, []LaunchModel)
                    integration.Run(model, []LaunchModel, args)
```

**Why one load per run:** inventory is cached on `launcherClient` for the duration of a single launch invocation. Re-listing on every integration method would duplicate work; refreshing is explicit when a **local** model name is missing from the first list (user may have just finished `pull`).

---

## `LaunchModel` fields

| Field | Source | Why integrations care |
|-------|--------|------------------------|
| `Name` | `/api/tags` | Config file model id |
| `Capabilities` | tags + server enrichment | vision → image input; thinking → reasoning flags |
| `ContextLength` | GGUF kv via `ModelDetails` | `contextWindow` / OpenCode `limit.context` |
| `MaxOutputTokens` | cloud limit map (when populated) | cloud output caps without Show |
| `Remote` | `RemoteModel` on list response | cloud routing + `isCloud` in OpenClaw |
| `ToolCapable` | `CapabilityTools` | tool-aware agent configs |
| `Details` | `/api/tags` family/format | OpenCode gpt-oss reasoning level heuristics |

Cloud models missing from tags still resolve via **`fallbackLaunchModel`** (name + `:cloud` suffix heuristics). **Why:** Eliza Cloud models may not appear in local blob inventory but remain valid launch targets.

---

## API / server support

List responses must carry enough metadata for inventory to avoid Show:

- **`ListModelResponse.Capabilities`** — populated in `server/routes.go` from loaded model capabilities.
- **`ModelDetails.ContextLength` / `EmbeddingLength`** — enriched from GGUF headers in `server/model_details.go`.

**Why in list, not only show:** launch runs before any model is loaded into a runner; tags is the only cheap bulk endpoint.

---

## Integration changes (summary)

| Integration | Before | After | Why |
|-------------|--------|-------|-----|
| OpenCode | `resolveOpenCodeModels` + Show | `buildInlineConfig(LaunchModel, …)` | inline JSON from inventory |
| OpenClaw | `openclawModelConfig(ctx, client, id)` | `openclawModelConfig(LaunchModel)` | no per-model HTTP |
| Pi | `createConfig(ctx, client, id)` | `createConfig(LaunchModel)` | same |
| Droid | `updateDroidSettings(…, []string)` | `[]LaunchModel` + `MaxOutputTokens` | cloud limits from inventory |
| OMP | single-model `Configure` | `ConfigureWithModels` + multi-model YAML | catalog from inventory |
| Cline | (unchanged Edit signature) | `Edit([]LaunchModel)` | dual-write uses `.Name` |

**Explicitly not ported:** Kimi launch, desktop app launchers (`claude-desktop`, `codex-app`, `hermes-desktop`) — out of scope for zerollama operator targets.

---

## Launch drift guard (`liveConfigMatches`)

`launchEditorIntegration` compares **saved** model names with **`editor.Models()`** (on-disk live config). If they diverge, launch re-runs `Edit` even when saved integration config matches.

**Why:** stale agent config files caused silent wrong-model sessions after the user switched providers inside the agent UI.

### Cline caveat (fixed)

When `providers.json` exists and `lastUsedProvider != ollama`, **`Models()` returns nil** — it does **not** fall back to legacy `globalState.json`.

**Why:** users who switched Cline to OpenAI/Anthropic still had old Ollama model ids in legacy state; `liveConfigMatches` falsely thought Ollama was active and skipped reconfigure.

---

## Managed single-model catalog (OMP)

Integrations implementing **`ManagedModelListConfigurer`** (OMP) receive the **installed inventory** from `/api/tags`, not the full selectable picker list.

**Why not `loadSelectableModels`:** that API merges **recommendations** (`gemma4`, `qwen3.5`, …) with "(not downloaded)" suffixes. Writing those into `models.yml` would advertise models the operator never pulled.

Code: `launcherClient.managedSingleConfigureModels` → `modelInventory().Load()` → name list → `inventory.Resolve` → `ConfigureWithModels`.

---

## Vendor pin coupling (`Makefile.sync`)

Bumping `FETCH_HEAD` in `Makefile.sync` must **rsync vendor → in-tree** before regenerating `llama/build-info.cpp`.

**Why:** an earlier rule only re-stamped build-info when `.in` changed, so binaries could report `b9781` while `llama/llama.cpp` still reflected an older pin — confusing Phase 17 smoke and native KV debugging. **`make sync` no longer runs `git checkout` on vendor** for the same reason.

Correct order: `checkout` → rsync `llama/llama.cpp` + `ml/backend/ggml/ggml` → sed `build-info.cpp`. Prefer **`./scripts/vendor/sync_vendor_llama.sh`** for the full gated workflow.

---

## Code map

| File | Role |
|------|------|
| `cmd/launch/model_inventory.go` | `LaunchModel`, load/refresh/resolve |
| `cmd/launch/launch.go` | wires inventory into editor + managed paths |
| `cmd/launch/models.go` | `prepareEditorIntegration`, `buildModelList` |
| `api/types.go` | list response capabilities + details |
| `server/routes.go` | tags handler enrichment |
| `server/model_details.go` | GGUF context/embedding length |

Tests: `cmd/launch/model_inventory_test.go`, integration tests using `testLaunchModels()`.

---

## Related upstream commits

Cherry-pick reference: upstream Ollama **v0.30.10** (`07ed7523`) — `cmd/launch/model_inventory.go` and integration signature refactor.
