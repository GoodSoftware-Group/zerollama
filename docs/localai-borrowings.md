# LocalAI control-plane borrowings

**Why this doc exists:** [LocalAI](https://github.com/mudler/LocalAI) optimizes **lifecycle, metadata, and routing** at a thin HTTP core. Zerollama keeps its own inference engines (ggml Metal, Python runtime, Phase 15 KV, training) but adopts **operational patterns** that reduce operator pain without becoming a gallery+gRPC zoo.

**What we are not doing:** 60+ OCI backend images, full `backend.proto`, NATS cluster, or replacing Ollama registry pull with a curated YAML gallery.

**Cross-links:** [scheduling-vram-policy.md](./scheduling-vram-policy.md) (per-node VRAM/queues) · [fleet-scheduling.md](./fleet-scheduling.md) (multi-node routing) · [ROADMAP.md](./ROADMAP.md#localai-control-plane-borrowings-jun-2026) · [upstream-ollama-diff.md](./upstream-ollama-diff.md) (engine convergence via Phase 17)

---

## Mental model

```text
Keep (zerollama engines)          Borrow (LocalAI control plane)
─────────────────────────         ───────────────────────────────
ggml / llama-server / runtime     Fast GGUF metadata read
Phase 15 KV / L3 slot cache       Manifest guess hooks (arch, num_ctx, parser)
Training queue                    Scheduler watchdog (LRU, VRAM reclaim, busy timeout)
Ollama API / Modelfiles           Fleet filter-then-score + prefix affinity
Eliza Cloud default              Post-load metadata probe → /api/status
```

LocalAI’s lesson: **cheap metadata + honest lifecycle** at the daemon boundary—not replacing your hot path.

---

## Shipped (Jun 2026)

### Fast GGUF metadata (`fs/ggml`, `llm`)

**Why:** Large GGUFs on slow disks or Docker volumes used to walk tokenizer vocab + full tensor regions just to read `general.architecture` and `llama.context_length` ([LocalAI #9790](https://github.com/mudler/LocalAI/issues/9790) pattern).

- `DecodeMetadata()` / `LoadModelMetadata()` — metadata-only read (tensor headers, no weight walk)
- Used by scheduler load, `ggml_num_ctx`, manifest parse, fleet probes

### GGUF guess hooks (`server/gguf_guess.go`)

**Why:** Operators paste train-context into manifests (`num_ctx: 262144`) and hang at load; missing `parser` breaks tool/thinking streams on new families.

| Field | Behavior |
|-------|----------|
| `ModelFamily`, `ModelFamilies`, `ContextLen`, `EmbedLen`, `FileType` | Filled when empty |
| `num_ctx` in params | Capped at **8192** (`defaultManifestNumCtxCap`) — train ctx stays in GGUF KV for `/api/show` |
| `spec_type` | `draft-mtp` when MTP heads detected |
| `stop` | Arch-specific (gemma4, qwen35) |
| `parser` | Guessed from arch + chat-template hints (`gguf_guess_parser.go`) |

**When it runs:** `create` (via `guessFromBaseLayers`), `GetModel` / show (in-memory via `applyGGUFGuessToModel`), **`pull`** (on-disk via `EnrichManifestAfterPull` → repair), and `zerollama repair`.

**Kill-switch:** `ZEROLLAMA_DISABLE_GGUF_GUESS=1` or `LOCALAI_DISABLE_GUESSING=true`

**Important:** `GetModel` guess is in-memory only. On-disk manifests are updated on **pull** (enrich), **create**, **`repair --write`**, or re-pull.

### Scheduler watchdog (`server/sched_watchdog.go`)

**Why:** Multi-model workloads need LRU under VRAM pressure and recovery from stuck runners without manual `stop`.

| Feature | Env | Default |
|---------|-----|---------|
| Memory reclaimer (idle LRU when GPU VRAM ≥ threshold) | `ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD` | off (`0`) — try `0.95` on tight GPUs |
| Busy timeout (force unload stuck runner) | `ZEROLLAMA_RUNNER_BUSY_TIMEOUT` | off — e.g. `30m` |
| Tick interval | `ZEROLLAMA_SCHED_WATCHDOG_INTERVAL` | `30s` |

Session scripts: `scripts/sched_watchdog_env.sh`, `scripts/gpu_5080_session.sh`, `scripts/gpu_metal_session.sh`.

**Also:** true LRU victim selection via `lastUsedAt`; load coalescing + `waitUntilReady` race fix; pull `singleflight` (`server/images.go`).

### Concurrency groups (`server/concurrency_groups.go`)

**Why:** On 16 GB cards, imagegen + chat + vision cannot safely stay resident together—`OLLAMA_MAX_LOADED_MODELS` alone does not express **mutual exclusion**.

```dockerfile
# Modelfile example
PARAMETER concurrency_groups ["vram-heavy"]
```

Or in manifest config: `"concurrency_groups": ["vram-heavy", "vision"]`

Before loading a model, the scheduler evicts any **resident** runner that shares a group (idle LRU among conflicts).

### Post-load metadata probe (`server/runner_metadata.go`)

**Why:** Manifest `num_ctx` and Modelfile boilerplate drift from ground truth; fleet and `/api/ps` need effective values after load.

Probed after `syncRunnerLoadOptions`, cached on `runnerRef.loadedMeta`, exposed on:

- `GET /api/ps` → `loaded_metadata`
- `GET /api/status` → `inference.ggml.loaded_model_details`

| Field | Meaning |
|-------|---------|
| `num_ctx` | Effective context (runner / llama) |
| `manifest_num_ctx` | From manifest params |
| `train_context_length` | GGUF KV train ctx |
| `num_parallel`, `num_gpu`, `backend` | Runner shape |
| `parser` | Resolved builtin parser |
| `supports_thinking`, `supports_tools` | From parser capabilities |
| `has_chat_template` | GGUF has `tokenizer.chat_template` |
| `probed_at` | UTC timestamp |

Disk read for probe runs **outside** `refMu` when possible.

### Fleet routing (filter-then-score)

**Why:** Warm-model + queue depth alone ignores session stickiness and loading state; agents need deterministic, testable routing.

| Piece | Role |
|-------|------|
| `fleet/score.go` | Score weights (lower wins): warm −10k, affinity −5k, queue +100, loading +500, cold +2k when `prefer_warm` |
| `fleet/prefixcache.go` | `(model, session_key)` → node_id TTL index |
| `fleet/probecache.go` | Coalesce peer `/health` probes (`ZEROLLAMA_FLEET_PROBE_CACHE_TTL`, default 1s) |
| `POST /internal/score` | Loopback-only ranked peers (management node) |
| Capacity weights | Penalize nodes with other residents / high effective `num_ctx` (`loaded_model_details`) |

**Note:** This is **node** scoring, not LocalAI’s logprob `Score` RPC for model routing — see **LA9** (`POST /api/score`) below.

### Logprob score API (LA9)

| Piece | Role |
|-------|------|
| `POST /api/score` | Joint log-probability of candidate continuations given a shared prompt (agent routing without generation) |
| `llm.Scorer` | Runner contract; `llmServer` forwards to subprocess `POST /score` |
| `runner/llamarunner/score.go` | Legacy CGO ggml decode + `LogprobForTokenID` on slot 0 |
| `runner/ollamarunner/score.go` | Go ollama-engine decode + `LogprobForTokenID` on slot 0 |
| `llm/llama_server_score.go` | llama-server path: `cache_prompt` + `id_slot` + `n_probs` top-128 lookup per token |

Request: `{ "model", "prompt", "candidates": ["a","b"], "length_normalize?", "include_token_logprobs?" }`. MLX/runtime-only/imagegen runners return **501** until a score path exists.

**Status semantics:** `inference.ggml.loaded` counts **ready** runners (matches `loaded_models`). Training idle-wait uses **resident** count (`len(scheduler map)`) via `InferenceBacklog()` — see [scheduling-vram-policy.md](./scheduling-vram-policy.md).

### Hugging Face pull (`server/hf_import.go`)

**Why:** Many GGUF quants live only on Hugging Face; registry pull is not always available.

| URI | Example |
|-----|---------|
| `huggingface://` | `huggingface://TheBloke/phi-2-GGUF/phi-2.Q8_0.gguf` |
| `hf://` | alias for `huggingface://` |
| `source` field | `{"model":"phi2","source":"huggingface://org/repo/file.gguf"}` |

Downloads via Hub API + resolve URL, stages GGUF blob, runs `createModel` (guess + template). Local tag derived from repo/filename when URI-only.

---

## Manifest hygiene (existing tags)

**Problem:** Tags pulled **before** guess hooks still have stale on-disk params (`num_ctx`, missing `parser`, no template layer). Runtime guess fixes **inference** but `/api/show` and fleet snapshots can look wrong until the manifest is rewritten.

| Action | Rewrites manifest? | Re-downloads weights? |
|--------|-------------------|----------------------|
| **Use model** (`run`, `chat`) | No (in-memory guess only) | No |
| **`pull` same tag** | Yes | No (if blob digests unchanged) |
| **`create` FROM tag** | Yes | No (reuses blobs) |
| **Startup prune** | No | Deletes orphan blobs >1h (`PruneLayers`) |
| **`repair`** | Yes (metadata layers only) | No |

### `zerollama repair`

**Why:** One-shot sweep for libraries with dozens of tags—no registry round-trip.

```bash
zerollama repair llama3.2          # dry-run: show proposed changes
zerollama repair llama3.2 --write  # rewrite manifest JSON only
zerollama repair --all --write     # every local tag
```

Behavior:

1. Metadata-only GGUF read per tag
2. Diff params/config (cap `num_ctx`, fill `parser`, arch, renderer)
3. Add missing template layer from `tokenizer.chat_template` (reuse `detectChatTemplate`)
4. `--write` rewrites manifest JSON only; **never** changes weight blob digests

Respects `ZEROLLAMA_DISABLE_GGUF_GUESS` / `LOCALAI_DISABLE_GUESSING`. Does not require a running server.

### Hugging Face pull (LA8)

Pull GGUF directly from Hugging Face Hub (LocalAI-compatible URI):

```bash
zerollama pull huggingface://TheBloke/phi-2-GGUF/phi-2.Q8_0.gguf
zerollama pull hf://meta-llama/Llama-3.2-1B-Instruct-GGUF
zerollama pull mymodel --source huggingface://org/repo/file.gguf   # API: explicit local name
```

- Without a filename, picks the largest non-`mmproj` `.gguf` in the repo (errors if ambiguous small set)
- Gated models: set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`
- Also accepts `https://huggingface.co/.../resolve/...` URLs in `source`

**Workaround (still valid):** `zerollama pull <model>` for registry tags, or `create` FROM existing tag.

### Operator checklist

1. **`/api/show <model>`** — check `num_ctx` in parameters; prefer ≤8192 in manifest, raise per-request via `options.num_ctx`
2. **`/api/ps`** — compare `loaded_metadata.num_ctx` vs manifest
3. **Tight GPU pairs** — add `concurrency_groups` on imagegen + chat models
4. **Agent fleets** — pass `session_key` / `prompt_cache_key` on assign; poll `loaded_model_details`
5. **Disk** — avoid `OLLAMA_NOPRUNE=1` in production unless debugging; prune reclaims failed pulls

---

## Environment reference

| Variable | Purpose |
|----------|---------|
| `ZEROLLAMA_DISABLE_GGUF_GUESS` | Disable manifest guess hooks |
| `LOCALAI_DISABLE_GUESSING` | Same (LocalAI-compatible name) |
| `ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD` | Watchdog VRAM ratio eviction (0–1) |
| `ZEROLLAMA_RUNNER_BUSY_TIMEOUT` | Max busy duration before forced unload |
| `ZEROLLAMA_SCHED_WATCHDOG_INTERVAL` | Watchdog tick |
| `ZEROLLAMA_FLEET_PREFIX_CACHE` | Fleet session affinity (`1` default on manager) |
| `ZEROLLAMA_FLEET_PREFIX_CACHE_TTL` | Affinity entry TTL |
| `ZEROLLAMA_FLEET_PROBE_CACHE_TTL` | Peer health probe cache (default `1s`, `0`=off) |
| `ZEROLLAMA_TRAINING_WAIT_GGML_LOADED` | Block training submit while ggml runners resident |
| `OLLAMA_NOPRUNE` | Skip orphan blob prune on startup |
| `OLLAMA_MAX_LOADED_MODELS` | Max resident runners (Ollama-compatible) |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Hugging Face Hub auth for gated models |

---

## Deferred (not adopted)

| Item | Why deferred |
|------|----------------|
| **Radix prefix cache (fleet)** | Session-key affinity + L3 slots cover most agent threads |
| **Full gallery + gRPC backends** | Architecture mismatch with Phase 15/17 |

---

## What zerollama already does better (do not regress)

- Ollama API/CLI parity and Phase 17 upstream merge path
- L1/L2/L3 GPU profiles and slot-pinned prompt cache
- Dual scheduler + VRAM broker + embedded training
- ggml Metal default on Mac (~+7% vs upstream llama-server on M4 Max)
- LM Studio cache import

---

## Tests

- `go test ./server -run 'GgufGuess|Watchdog|Concurrency|RunnerMetadata|InferenceFleet|InferenceBacklog|Score'`
- `go test ./fleet/...`
- `go test ./fs/ggml -run DecodeMetadata`
