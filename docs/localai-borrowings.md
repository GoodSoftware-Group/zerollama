# LocalAI control-plane borrowings

**Why this doc exists:** [LocalAI](https://github.com/mudler/LocalAI) optimizes **lifecycle, metadata, and routing** at a thin HTTP core. Zerollama keeps its own inference engines (ggml Metal, Python runtime, Phase 15 KV, training) but adopts **operational patterns** that reduce operator pain without becoming a gallery+gRPC zoo.

**What we are not doing:** 60+ OCI backend images, full `backend.proto`, NATS cluster, or replacing Ollama registry pull with a curated YAML gallery.

**Cross-links:** [scheduling-vram-policy.md](./scheduling-vram-policy.md) (per-node VRAM/queues) · [fleet-scheduling.md](./fleet-scheduling.md) (multi-node routing) · [ROADMAP.md](./ROADMAP.md#localai-control-plane-borrowings-jun-2026) · [upstream-ollama-diff.md](./upstream-ollama-diff.md) (engine convergence via Phase 17)

**Local reference tree:** `~/Sites/inference/LocalAI` — upstream [mudler/LocalAI](https://github.com/mudler/LocalAI), checked out at **`v4.5.6`** for diffing control-plane patterns (router, score, watchdog, distributed). Update with `git fetch origin && git checkout v4.5.6` (or `master` for tip).

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

**Shipped track:** **LA1–LA10** (see [ROADMAP](./ROADMAP.md#localai-control-plane-borrowings-jun-2026)). **Upstream watch:** [below](#upstream-watch-localai-v44-jul-2026) — what LocalAI shipped recently and what might become **LA11+**.

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

## Upstream watch (LocalAI v4.5.6, checked Jul 2026)

Periodic scan of [mudler/LocalAI](https://github.com/mudler/LocalAI) via GitHub releases and the local tree at **`~/Sites/inference/LocalAI`** (`v4.5.6`). Last checked **2026-07-03**. Use this to decide **LA11+** candidates. Sibling map: [upstream-siblings.md](./upstream-siblings.md).

### LocalAI v4.5 highlights (since v4.4)

| Area | What LocalAI shipped | Zerollama today |
|------|----------------------|-----------------|
| **PII (NER tier)** | `privacy-filter.cpp` backend + regex secret detector; UI editor ([v4.5.0](https://github.com/mudler/LocalAI/releases/tag/v4.5.0)) | Eliza cloud passthrough; no request-side PII — **LA12 candidate** |
| **Multi-user defaults** | Prefix caching **on by default**; VRAM-scaled `n_parallel`; Blackwell batch 2048 | **L1** profiles + **L3** slot cache; not automatic VRAM→`n_parallel` |
| **Model aliases** | Live redirect/rename of model names without client reconfig | `zerollama cp` / manifest tags only — **new candidate LA17** |
| **Realtime voice** | Speaker-aware sessions, summarize-then-drop compaction, semantic VAD EOU | Piper/Whisper subprocess; duplex **L7** deferred |
| **New modality backends** | `depth-anything`, `ced` (527 sound classes), `supertonic` TTS | Different tracks (video/multimodal/voice) — not gallery ports |
| **`swa_full` default** | Sliding-window models default `swa_full:true` ([v4.5.6](https://github.com/mudler/LocalAI/releases/tag/v4.5.6)) | **L3** `prefix_cache_policy` + hybrid Radix gate — overlap, verify parity |
| **Distributed hardening** | `LOCALAI_DISTRIBUTED_SHARED_MODELS` (skip staging on shared volumes); `SyncedMap` cross-replica state; detached cold-load staging | Fleet HTTP peers; no NATS/shared-volume staging flag |
| **Import fixes** | `file://` strip, GGUF-derived names for repo-root URIs | **LA8** HF pull — align edge cases if operators report mismatches |

### LocalAI v4.4 highlights (still relevant)

| Area | What LocalAI shipped | Zerollama today |
|------|----------------------|-----------------|
| **Intelligent middleware** | Capability router: rewrite `input.Model` to smallest capable downstream model; **score** or **colbert** classifiers; `/api/router/*` decision log | **LA9** `POST /api/score` primitive only—no fan-out router policies yet |
| **Distributed v4** | **Prefix-cache-aware routing** (radix tree + xxhash chain, NATS sync); JWT/TLS on NATS; resumable GGUF `Content-Range` transfers | **LA6** fleet warm/affinity + **L3** per-node prefix/Radix; cross-node KV donor still [deferred](./radix-prefix-share.md#product-gaps) |
| **Security** | `pkg/httpclient` blocks cross-host credential-leaking redirects (GHSA) | Standard Go client; no dedicated audit pass |
| **Multimodal backends** | parakeet.cpp, CrispASR, rfdetr-cpp, llama.cpp video input, LTX-2 video gen | Phase 17 llama-server video; Piper/Whisper subprocess; Wan queue—different integration shape |
| **Operator UX** | `local-ai chat` interactive CLI; RAG source citations in agent UI | `zerollama run` / Ollama CLI; **LA10** bench cache in `ls` |
| **Backends** | 60+ OCI gallery backends, gRPC `backend.proto` | **Not goals** — Phase 15/17 engines stay in-tree |

Refs: [v4.5.6](https://github.com/mudler/LocalAI/releases/tag/v4.5.6) · [v4.5.0](https://github.com/mudler/LocalAI/releases/tag/v4.5.0) · [v4.4.0](https://github.com/mudler/LocalAI/releases/tag/v4.4.0) · [middleware docs](https://localai.io/features/middleware/index.html)

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
| Intelligent router (score/colbert policies) | — | **Candidate LA11** |
| PII middleware (NER + secrets regex) | — | **Candidate LA12** (enterprise) |
| Fleet radix prefix routing | **L3-R8 + L3-R9 / LA13** status mirror, residency soft score, content-hash longest-prefix | — |
| Cross-node KV blob pull | **L3-R10 + L3-R11** HTTP digest fetch; auto peers from `FLEET_PEERS`/coordination; Go `prefixblock` | NIXL/Mooncake RDMA still open |
| Resumable peer GGUF transfer | — | **Candidate LA14** |
| Outbound HTTP redirect credential guard | — | **Candidate LA15** |
| `POST /v1/rerank` (colbert routing tier) | llama.cpp has `/reranking`; not wired in zerollama API | **Candidate LA16** |
| Model aliases (live name redirect) | manifest tags / `cp` only | **Candidate LA17** |
| VRAM-scaled `n_parallel` default | L1 profiles manual | **Watch** (overlap L1) |
| Backend gallery + gRPC zoo | — | **Not goals** |
| NATS cluster / ds4 layer-split | — | **Not goals** |

### Candidates (LA11+) — suggested priority

**High fit (control plane, reuses LA9 or fleet):**

1. **LA11 Intelligent router** — Single client-facing model name → policy table of downstream candidates; classify with `POST /api/score` (or optional rerank/colbert later); log decisions (`correlation_id`, chosen model, scores). LocalAI reference: [middleware routing](https://localai.io/features/middleware/index.html). Zerollama already has score + fleet assign—this is **per-node request middleware**, not NATS.

2. **LA13 Fleet prefix-cache routing** — **Done (L3-R8 + L3-R9):** `/api/status.inference.runtime.radix` (+ `block_hashes`); fleet soft residency score; assign/score accept `prefix_block_hashes` for longest leading-hash match (`ZEROLLAMA_FLEET_RADIX_HASH_SCORE`).

**Medium fit:**

3. **LA14 Resumable GGUF transfer** — `Content-Range` resume for fleet peer model staging (flaky WAN). LocalAI #10109.

4. **LA15 Outbound HTTP hardening** — Refuse cross-host redirect credential leaks on cloud proxy and HF pull clients (LocalAI GHSA-3mj3-57v2-4636 class).

5. **LA16 Rerank API** — Expose llama-server `/v1/rerank` for document ranking and as a **colbert** classifier tier for LA11 when score distributions are flat.

6. **LA17 Model aliases** — Server-side name → target model redirect (LocalAI v4.5.0); lighter than full gallery; useful for agent configs that pin one router name.

**Lower priority / different tracks:**

- **LA12 PII middleware** — Valuable for cloud-proxy operators; overlaps Eliza Cloud path; large surface (NER models, admin UI, streaming filter).
- **Gallery / extra backends** — Stay on Phase 17 + runtime modality map.
- **ASR/TTS backend explosion** — See [eliza-v3 L5–L8](./ROADMAP.md#local-voice--llama-borrowings-eliza-v3), not LocalAI gallery ports.

---

## Deferred (not adopted)

| Item | Why deferred |
|------|----------------|
| **Radix prefix cache (fleet)** | Session-key affinity + L3 slots cover most agent threads; cross-node donor needs L3-R4 — see [radix-prefix-share.md — Product gaps](./radix-prefix-share.md#product-gaps). LocalAI v4.4 shipped fleet radix routing; same gap analysis applies. |
| **Full gallery + gRPC backends** | Architecture mismatch with Phase 15/17 |
| **NATS distributed cluster** | Fleet F-track uses HTTP peers + optional mDNS; not adopting LocalAI NATS control plane |

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
- `go test ./llm -run TestLlamaServer`
- `go test ./cmd/benchcache/...` (LA10 digest cache)
