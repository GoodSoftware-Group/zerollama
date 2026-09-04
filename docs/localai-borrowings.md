# LocalAI control-plane borrowings

**Why this doc exists:** [LocalAI](https://github.com/mudler/LocalAI) optimizes **lifecycle, metadata, and routing** at a thin HTTP core. Zerollama keeps its own inference engines (ggml Metal, Python runtime, Phase 15 KV, training) but adopts **operational patterns** that reduce operator pain without becoming a gallery+gRPC zoo.

**What we are not doing:** 60+ OCI backend images, full `backend.proto`, NATS cluster, or replacing Ollama registry pull with a curated YAML gallery.

**Cross-links:** [scheduling-vram-policy.md](./scheduling-vram-policy.md) (per-node VRAM/queues) · [fleet-scheduling.md](./fleet-scheduling.md) (multi-node routing) · [ROADMAP.md](./ROADMAP.md#localai-control-plane-borrowings-jun-2026) · [upstream-ollama-diff.md](./upstream-ollama-diff.md) (engine convergence via Phase 17)

**Local reference tree:** `~/Sites/inference/LocalAI` — upstream [mudler/LocalAI](https://github.com/mudler/LocalAI), checked out at **`v4.9.0`** for diffing control-plane patterns (router, score, watchdog, distributed). Update with `git fetch origin && git checkout v4.9.0` (or `master` for tip).

---

## Mental model

```text
Keep (zerollama engines)          Borrow (LocalAI control plane)
─────────────────────────         ───────────────────────────────
ggml / llama-server / runtime     Fast GGUF metadata read (LA1)
Phase 15 KV / L3 slot cache       Manifest guess + repair (LA2, LA7)
Training queue                    Scheduler watchdog + concurrency groups (LA3, LA4)
Ollama API / Modelfiles           Fleet filter-then-score + prefix affinity (LA6)
Eliza Cloud default              Post-load probe → /api/status (LA5)
                                  HF pull URIs (LA8) · logprob score API (LA9)
                                  Operator bench cache → ls TOK/S (LA10)
```

LocalAI’s lesson: **cheap metadata + honest lifecycle** at the daemon boundary—not replacing your hot path.

**Shipped track:** **LA1–LA11** (incl. **LA11b**), **LA14–LA15**, **LA17–LA21**. **Upstream watch:** remaining **LA12+**.

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

Session scripts: `scripts/runtime/sched_watchdog_env.sh`, `scripts/gpu/gpu_5080_session.sh`, `scripts/gpu/gpu_metal_session.sh`.

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

### Model bench cache (LA10)

**Why:** Operators need *on-this-box* throughput hints in daily workflow—not a separate benchmark suite.

| Piece | Role |
|-------|------|
| `zerollama bench` | Warmup + timed `/api/generate` epochs; avg decode tok/s |
| `~/.ollama/bench.json` | Cache keyed by manifest **digest** (re-pull invalidates stale rows) |
| `zerollama ls` | `TOK/S` column from cache without re-running inference |

Doc: [bench-cache.md](./bench-cache.md). **Not** a LocalAI API port—operator UX in the same spirit as LocalAI’s focus on lifecycle at the CLI boundary.

### Intelligent router (LA11)

**Why:** Agents want one client-facing name (`agent`) that picks a specialist model from the prompt, without a full generation round-trip. Score classifier (LA11) or labelled **KNN corpus** (LA11b).

Config file (`ZEROLLAMA_ROUTER_CONFIG`, default `~/.ollama/router.yaml`):

```yaml
routers:
  agent:
    classifier: llama3.2:1b   # must support POST /api/score
    fallback: llama3.2:3b
    activation_threshold: 0.15
    policies:
      - label: code
        description: programming, debugging, code review
      - label: general
        description: everyday questions
    candidates:
      - model: qwen2.5-coder:7b
        labels: [code]
      - model: llama3.2:3b
        labels: [code, general]
```

| Piece | Role |
|-------|------|
| `POST /api/router/decide` | `{router, prompt}` → labels, softmax, chosen model, fallback flag |
| Score classifier | LA9 joint log-prob of policy **labels** as continuations; softmax; labels ≥ threshold are active |
| Candidate match | First candidate whose `labels` **cover** the active set (order small→large) |
| In-band rewrite | `/api/chat` and `/api/generate` treat a router name as the client model; headers `X-Zerollama-Router*` |

`ZEROLLAMA_ROUTER_REWRITE=0` keeps decide-only. Missing YAML = no routers. Nested routers (candidate is also a router name) are rejected.

**LA11b KNN** — `classifier: knn` plus an embedding model and labelled exemplars. No score LM. Neighbours below `similarity_threshold` (default 0.80) do not vote; labels need a similarity-weighted majority (`vote_threshold` default 0.5). Out of corpus range → empty labels → fallback.

```yaml
routers:
  agent:
    classifier: knn
    embedder: nomic-embed-text
    fallback: llama3.2:3b
    knn:
      k: 3
      similarity_threshold: 0.80
      vote_threshold: 0.5
      corpus:
        - text: "fix this rust compile error"
          labels: [code]
        - text: "what is the weather"
          labels: [general]
    candidates:
      - model: qwen2.5-coder:7b
        labels: [code]
      - model: llama3.2:3b
        labels: [code, general]
```

**LA16 rerank / ColBERT-style router** — `classifier: rerank` or `colbert` plus a RANK-pooling GGUF. Policy **descriptions** (not labels) are scored against the user prompt via llama.cpp `/v1/rerank`. Labels with `relevance_score` ≥ `activation_threshold` (default 0.15) are active. This is llama.cpp RANK pooling, not LocalAI’s Python `bge-m3` ColBERT MaxSim backend.

```yaml
routers:
  agent:
    classifier: rerank
    reranker: qwen3-reranker
    fallback: llama3.2:3b
    activation_threshold: 0.15
    policies:
      - label: code
        description: programming, compilers, and rust
      - label: general
        description: chitchat and general questions
    candidates:
      - model: qwen2.5-coder:7b
        labels: [code]
      - model: llama3.2:3b
        labels: [code, general]
```

Public API: `POST /v1/rerank` (Jina `{query, documents, top_n}`; TEI `texts` alias). RANK GGUFs start llama-server with `--reranking`.

| Piece | Role |
|-------|------|
| `GET /api/router/corpus?router=` | Counts only (no exemplar text) |
| `POST /api/router/corpus` | Session overlay entries (lost on process restart; YAML is durable) |

### Model aliases (LA17)

**Why:** OpenAI-shaped clients send `gpt-4` / `gpt-4o-mini`. `zerollama cp` copies a full manifest; a live alias is a one-hop name redirect with no extra blobs.

`~/.ollama/aliases.yaml` (`ZEROLLAMA_ALIASES_CONFIG`):

```yaml
aliases:
  gpt-4: llama3.2:3b
  gpt-4o-mini: llama3.2:1b
```

One hop only (no alias→alias). Applied before LA11 router rewrite on generate/chat/embed/score/rerank/show. Headers: `X-Zerollama-Alias`, `X-Zerollama-Alias-Target`. `GET /api/aliases` lists; `POST /api/aliases` `{name,target}` is a session overlay (YAML is durable).

### Context compression (LA22)

**Why:** Long agent threads hit `num_ctx` and then truncate or error. LocalAI v4.9 compresses older complete turns with a second model.

**Default (no flags):** if the thread already has **tool** messages, **tool_calls**, or **thinking**, `/api/chat` auto-elides old tool bodies and thinking in place when the prompt crosses ~75% of `num_ctx`. No env, no `compression` object, no second model. `enabled: false` turns that off.

**Summary mode** (second model) stays opt-in: `ZEROLLAMA_CHAT_COMPRESSION=1` or `"compression": {"enabled": true}`. Omit `mode` unless you need to force `summary` on an agent thread or `placeholder` on a non-agent one.

Optional knobs when you need them: `trigger_at_ratio`, `compressor_model` (summary only), `keep_tail_tokens` (**summary** only — placeholder always keeps the last turn then peels). Echo `elide_from` from the previous meta, or send a stable `prompt_cache_key` so the server remembers the cut (Go: `api.ChatThread`). `cache_reset` drops that memory. Native `compression` on the **done** chat response (JSON and NDJSON stream, including the Python runtime proxy). OpenAI Chat Completions / Responses and Anthropic Messages use `usage.compression_meta` (Chat Completions SSE needs `stream_options.include_usage`). Header `X-Zerollama-Compressed: 1`. Cloud chat is still rejected.

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

### Outbound URL SSRF (LA15)

**Why:** User-supplied `video_url`, experimental `web_fetch` URLs, Hugging Face redirects, and registry blob `Location` hops must not reach loopback, RFC1918, link-local, or cloud metadata.

- Shared `internal/ssrf` (`ValidateExternalURL`, redirect re-check) — LocalAI `ValidateExternalURL` / `IsPublicIP`
- Video fetches already had host checks; they now use the shared helper (also blocks `0.0.0.0` / unspecified)
- HF tree + GGUF download: validate URL, follow redirects only to public IPs; `http://huggingface.co` upgrades to HTTPS
- Registry part downloads: validate the blob redirect URL before GET
- `POST /api/experimental/web_fetch`: reject private `url` before cloud proxy

DNS rebinding after the lookup is not fully pinned on the subsequent dial (same limitation as LocalAI).

### Explicit prewarm (LA21)

**Why:** Agents and fleet assign used a dummy `/api/generate` or empty embeddings call to pay cold-start. LocalAI `POST /backend/load` blocks until resident.

```bash
curl http://127.0.0.1:11434/api/load -d '{"model":"llama3.2:3b","keep_alive":"30m"}'
```

- Also LocalAI `POST /backend/load` (same handler). Unload is `POST /api/unload` or generate `keep_alive: 0`.
- Aliases apply; `:cloud` rejected; does **not** run LA11 router rewrite
- `already_loaded` is true when `/api/ps` already listed the tag
- Host mem guard and assignment-token middleware match generate/chat

### Resumable peer GGUF (LA14)

**Why:** `zerollama storage push` truncated `.partial` on every PUT, so a dropped 20 GiB transfer restarted from byte 0.

- `PUT /v1/blob/{digest}` with `Content-Range: bytes start-end/total` appends to `{digest}.partial`
- Incomplete → **202** + `X-Zerollama-Partial: 1` (not 308 — HTTP clients would follow that as a redirect)
- Wrong offset → **416** with `X-Zerollama-Partial-Size`
- HEAD of a partial does **not** count as a complete blob (`storage push` will resume, not skip)
- `PushBlob` HEADs then sends only the remainder
- GET Range-GET was already advertised; tests cover 206

Doc: [remote-model-storage.md](./remote-model-storage.md).

### Operator checklist

1. **`/api/show <model>`** — check `num_ctx` in parameters; prefer ≤8192 in manifest, raise per-request via `options.num_ctx`
2. **`/api/ps`** — compare `loaded_metadata.num_ctx` vs manifest
3. **Tight GPU pairs** — add `concurrency_groups` on imagegen + chat models
4. **Agent fleets** — pass `session_key` / `prompt_cache_key` on assign; poll `loaded_model_details`
5. **Disk** — avoid `OLLAMA_NOPRUNE=1` in production unless debugging; prune reclaims failed pulls
6. **Prewarm** — `POST /api/load` before a latency SLA; do not use empty generate

---

## Environment reference

| Variable | Purpose |
|----------|---------|
| `ZEROLLAMA_DISABLE_GGUF_GUESS` | Disable manifest guess hooks |
| `LOCALAI_DISABLE_GUESSING` | Same (LocalAI-compatible name) |
| `ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD` | Watchdog VRAM ratio eviction (0–1) |
| `ZEROLLAMA_RUNNER_BUSY_TIMEOUT` | Max busy duration before forced unload |
| `ZEROLLAMA_ROUTER_CONFIG` | Router YAML path (default `~/.ollama/router.yaml`; `0`=off) |
| `ZEROLLAMA_ROUTER_REWRITE` | Rewrite chat/generate when `model` is a router name (default on) |
| `ZEROLLAMA_ALIASES_CONFIG` | Aliases YAML (default `~/.ollama/aliases.yaml`; `0`=off) |
| `ZEROLLAMA_LOAD_COOLDOWN_MAX` | Cooldown cap (default `5m`) |
| `ZEROLLAMA_BACKEND_PARENT_WATCH` | Linux: SIGKILL runner children if parent dies (`0`=off) |
| `ZEROLLAMA_FLEET_PREFIX_CACHE` | Fleet session affinity (`1` default on manager) |
| `ZEROLLAMA_FLEET_PREFIX_CACHE_TTL` | Affinity entry TTL |
| `ZEROLLAMA_FLEET_PROBE_CACHE_TTL` | Peer health probe cache (default `1s`, `0`=off) |
| `ZEROLLAMA_TRAINING_WAIT_GGML_LOADED` | Block training submit while ggml runners resident |
| `OLLAMA_NOPRUNE` | Skip orphan blob prune on startup |
| `OLLAMA_MAX_LOADED_MODELS` | Max resident runners (Ollama-compatible) |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Hugging Face Hub auth for gated models |

---

## Upstream watch (LocalAI v4.9.0, checked Aug 2026)

Periodic scan of [mudler/LocalAI](https://github.com/mudler/LocalAI) via GitHub releases and the local tree at **`~/Sites/inference/LocalAI`** (`v4.9.0`). Last checked **2026-08-21**. Sibling map: [upstream-siblings.md](./upstream-siblings.md).

### LocalAI v4.6–v4.9 highlights (since last watch at v4.5.6)

| Area | What LocalAI shipped | Zerollama today | Borrow? |
|------|----------------------|-----------------|---------|
| **Model-load cooldown** (v4.7) | Failed load → cooldown (10s→5m geometric); clients get `503 + Retry-After` instead of respawning crash loops | **LA18 Done** — `ZEROLLAMA_LOAD_COOLDOWN` | **Shipped** |
| **VRAM budget** (v4.8) | `LOCALAI_VRAM_BUDGET=80%` or `12GB`; hard per-process + distributed placement ceiling | **LA19 Done** — `ZEROLLAMA_VRAM_BUDGET` on GPUDevices + runtime free probes | **Shipped** |
| **Parent-death backend watch** (v4.6) | Backend self-terminates if parent PID reparented (`LOCALAI_BACKEND_PARENT_WATCH`) | **LA20 Done** — Linux `Pdeathsig` SIGKILL | **Shipped** |
| **Eager warm + load API** (v4.6) | `POST /backend/load`; realtime pipeline block-warm at session start | **LA21 Done** — `POST /api/load` (+ `/backend/load`) | **Shipped** |
| **KNN router classifier** (v4.9) | `classifier: knn` over labelled corpus; no classifier LM; undecidable → fallback | **LA11b Done** — cosine KNN + `/api/router/corpus` | **Shipped** |
| **Global HTTP admission** (v4.9) | Process-wide in-flight bounds beyond per-backend | Phase 11 + Go pending queue | **Watch** (overlap Phase 11) |
| **Context compression** (v4.9) | Opt-in per-model: compress older turns via local model before infer | **LA22 Done** — `compression` on chat + `ZEROLLAMA_CHAT_COMPRESSION` | **Shipped** |
| **Gallery SSRF / auth default-deny** (v4.6/v4.9) | `ValidateExternalURL`; deny-by-default HTTP auth | **LA15 Done** — `internal/ssrf` on video, HF, blob redirect, web_fetch | **Shipped** |
| **Parallel HF downloads** (v4.9) | Concurrent whole-file snapshot transfers | LA8 sequential | **Maybe — LA8 polish** |
| **Durable cold-load jobs** (v4.9) | Advisory lock ≠ multi-GB transfer lifetime | Go load coalescing + singleflight | **Watch** (we already coalesce) |
| **`context_size: -1`** (v4.7) | Resolve to GGUF `n_ctx_train` with VRAM warn | Guess **caps** at 8192 (intentional) | **No** — opposite of LA2 footgun fix |
| **Interleaved thinking + tools** (v4.7) | `reasoning` survives tool loop; Anthropic `thinking` blocks | Parser/tool path exists; verify parity | **Watch** (Hermes/parser track) |
| **vllm.cpp / audio.cpp / 3D / gallery variants** | New engines + gallery | Phase 15/17 stay in-tree | **Not goals** |
| **Voice library / LongCat / MiniMax-H3** | Modalities | Voice L5–L8 / video track | **Different tracks** |

Refs: [v4.9.0](https://github.com/mudler/LocalAI/releases/tag/v4.9.0) · [v4.8.0](https://github.com/mudler/LocalAI/releases/tag/v4.8.0) · [v4.7.0](https://github.com/mudler/LocalAI/releases/tag/v4.7.0) · [v4.6.0](https://github.com/mudler/LocalAI/releases/tag/v4.6.0)

### Older highlights (v4.4–v4.5, still relevant)

| Area | What LocalAI shipped | Zerollama today |
|------|----------------------|-----------------|
| **Intelligent middleware** | Capability router + score/colbert; `/api/router/*` | **LA9** score primitive only — **LA11** |
| **PII filtering** | NER + regex; reversible pseudonyms (v4.9) | Cloud passthrough — **LA12** |
| **Distributed prefix routing** | Radix + xxhash across replicas | **LA13 Done** via L3-R8/R9 |
| **Model aliases** | Live name redirect | **LA17 Done** |
| **Resumable GGUF transfer** | `Content-Range` | **LA14 Done** |
| **Multi-user defaults** | Prefix cache on; VRAM-scaled `n_parallel` | L1/L3 manual profiles |

### Parity matrix (control plane)

| LocalAI pattern | Zerollama | Milestone |
|-----------------|-----------|-----------|
| Fast GGUF metadata read | `DecodeMetadata` | **LA1** ✓ |
| Manifest guess / cap `num_ctx` | `gguf_guess` + pull enrich | **LA2** ✓ |
| Memory reclaimer / stuck runner | `sched_watchdog` | **LA3** ✓ |
| Mutual exclusion groups | `concurrency_groups` | **LA4** ✓ |
| Effective load metadata | `runner_metadata` | **LA5** ✓ |
| Fleet warm + affinity scoring | `fleet/score`, `/internal/score` | **LA6** ✓ |
| Manifest repair without re-pull | `zerollama repair`, `POST /api/repair` | **LA7** ✓ |
| `huggingface://` pull | `hf_import` | **LA8** ✓ |
| `Score` RPC / `POST /api/score` | `llm.Scorer`, 3 ggml backends | **LA9** ✓ |
| Operator throughput cache | `zerollama bench`, `TOK/S` in `ls` | **LA10** ✓ |
| Failed-load cooldown + Retry-After | `load_cooldown`, `ZEROLLAMA_LOAD_COOLDOWN` | **LA18** ✓ |
| Parent-death orphan backend kill | Linux `Pdeathsig` on runners | **LA20** ✓ |
| Intelligent router (score/colbert/knn) | YAML + `/api/router/decide` (score + knn + rerank) | **LA11** ✓ / **LA11b** ✓ / **LA16** ✓ |
| PII middleware (NER + secrets + reverse) | — | **Candidate LA12** (enterprise) |
| Fleet radix prefix routing | **L3-R8 + L3-R9** | **LA13** ✓ |
| Cross-node KV blob pull | **L3-R10 + L3-R11** | ✓ (NIXL still open) |
| Resumable peer GGUF transfer | `storage` PUT `Content-Range` | **LA14** ✓ |
| Outbound HTTP / SSRF / auth harden | `internal/ssrf` | **LA15** ✓ |
| `POST /v1/rerank` (colbert routing tier) | llama-server `--reranking` + `llm.Reranker` | **LA16** ✓ |
| Model aliases (live name redirect) | `aliases.yaml`, `/api/aliases` | **LA17** ✓ |
| Absolute / % VRAM budget | `ZEROLLAMA_VRAM_BUDGET` | **LA19** ✓ |
| Explicit prewarm / load API | `POST /api/load`, `/backend/load` | **LA21** ✓ |
| Server-side context compression | `compression` on chat; `ZEROLLAMA_CHAT_COMPRESSION` | **LA22** ✓ |
| Backend gallery + gRPC zoo / vllm.cpp | — | **Not goals** |
| NATS cluster / ds4 layer-split | — | **Not goals** |

### Candidates (LA11+) — suggested priority

**Lower priority / different tracks:**

- **LA12 PII** (NER + reversible pseudonyms) — enterprise/cloud-proxy.
- Gallery / vllm.cpp / audio.cpp / 3D / voice library — stay on Phase 15/17 + modality tracks.
- `context_size: -1` → train ctx — **do not adopt**; conflicts with LA2 cap.

---

## Deferred (not adopted)

| Item | Why deferred |
|------|----------------|
| **Full gallery + gRPC backends / vllm.cpp** | Architecture mismatch with Phase 15/17 |
| **NATS distributed cluster** | Fleet F-track uses HTTP peers + optional mDNS |
| **Train-context as default (`context_size: -1`)** | LA2 deliberately caps manifest `num_ctx`; raise per-request |

---

## What zerollama already does better (do not regress)

- Ollama API/CLI parity and Phase 17 upstream merge path
- L1/L2/L3 GPU profiles and slot-pinned prompt cache
- Cross-slot Radix + fleet content-hash routing (L3-R8…R11 / LA13)
- Dual scheduler + VRAM broker + embedded training
- ggml Metal default on Mac (~+7% vs upstream llama-server on M4 Max)
- LM Studio cache import

---

## Tests

- `go test ./server -run 'GgufGuess|Watchdog|Concurrency|RunnerMetadata|InferenceFleet|InferenceBacklog|Score|Rerank|DecideRouter|CompressChat'`
- `go test ./fleet/...`
- `go test ./fs/ggml -run DecodeMetadata`
- `go test ./llm -run TestLlamaServer`
- `go test ./cmd/benchcache/...` (LA10 digest cache)
