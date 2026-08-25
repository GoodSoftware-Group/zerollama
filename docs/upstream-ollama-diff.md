# Upstream Ollama comparison (zerollama vs `ollama/ollama`)

**Audience:** contributors deciding what to cherry-pick from vanilla Ollama without rebasing zerollama.

**Related:** [ROADMAP.md](./ROADMAP.md#phase-17--upstream-gguf-path-alignment-directional), [llama-cpp-backend.md](./llama-cpp-backend.md), [python-migration.md](./python-migration.md), [development.md](./development.md#compare-with-upstream-ollama).

---

## Why this doc exists

Zerollama forked Ollama and added a **Python runtime**, **GPU training**, **Eliza cloud**, and **native KV experiments**. Upstream Ollama took a different bet: **drop in-process ggml for GGUF** and route all text inference through **`llama-server`**. Both repos still share MLX safetensors routing and much of the Go API surface.

This document captures **architecture deltas**, **pin gaps**, and **actionable cherry-picks** so roadmap work stays intentional—not accidental full merges.

Zerollama is **downstream of `ollama/ollama`**. We path-filter cherry-picks of useful commits. We do **not** rebase onto `main`.

---

## Local upstream checkout

Clone beside zerollama (no merge into this repo):

```bash
./scripts/gpu/clone_upstream_ollama.sh
# default: ../ollama-upstream
```

Build and run on a different port for A/B:

```bash
./scripts/build/build_upstream_ollama_mac.sh   # llama-server (Metal) + go binary — required, not just go build
OLLAMA_HOST=127.0.0.1:11435 ../ollama-upstream/ollama serve
```

Zerollama default: `:11434`. Experimental llama.cpp routing: [llama-cpp-backend.md](./llama-cpp-backend.md).

---

## Architecture (today)

### Upstream Ollama

```text
Client → Go :11434 → sched.go → llm.NewLlamaServer → llama-server (loopback) → libllama
                              └→ mlxrunner / imagegen (safetensors only)
```

- **No** Python runtime sidecar.
- **No** `runner/ollamarunner` — ggml in-process runner removed for GGUF text.
- **No** `OLLAMA_NEW_ENGINE` feature flag (marketing “new engine” = scheduling/memory work, not a toggle).
- GGUF fixes ship via **`llama/compat/`** overlay at CMake fetch time, not a large vendored patch quilt.

### Zerollama

```text
Client → Go :11434 → sched.go → ollamarunner (ggml Metal/CUDA subprocess)     ← Mac default GGUF
                              └→ Python :8081 → llama-server or inprocess     ← Phase 12–15 / --llama-cpp-backend
                              └→ mlxrunner (safetensors)
                              └→ training.py (/api/train, pyembed)
```

- **Two schedulers** (Go ggml + Python runtime) plus VRAM broker (Phase 8).
- **`--llama-cpp-backend`** / `ZEROLLAMA_LLAMA_CPP_BACKEND=1` — routes eligible text GGUF through Python runtime instead of ggml (test harness toward upstream shape).
- **Unique:** training API, Eliza cloud default, Phase 15 native KV pool, Apple M1–M6 operator track.

### Side-by-side

| Concern | Upstream | Zerollama |
|---------|----------|-----------|
| Default GGUF path | Go → **llama-server** | Go → **ggml runner** (Metal on Mac) |
| Go llama integration | `llm/llama_server.go` | **Ported (opt-in)** — `--llama-server-backend`; ggml runner remains Mac default |
| Python runtime | None | `runtime/` FastAPI sidecar/embed |
| Training | None | `/api/train/*`, `training.py`, pyembed |
| Remote cloud | ollama.com | **Eliza Cloud** default |
| llama.cpp pin | `LLAMA_CPP_VERSION` = **`b9888`** (ggml-org) | **`86d86ed4`** (ggml-org master) via `vendor/llama-cpp-86d86ed4` + **79** patch commits | [ggml-b9509-migration.md](./ggml-b9509-migration.md) |
| Ollama-specific llama fixes | `llama/compat/` + CMake `PATCH_COMMAND` | `llama/patches/` (**79** on 86d86ed4) + compat/kv-ext/seq-copy |
| GPU discovery | `discover/llama_server.go` probe | **Hybrid** — llama-server when Linux auto or `ZEROLLAMA_LLAMA_SERVER=1`; ggml `/info` bootstrap otherwise (**why:** Mac default stays ggml; upstream sched inputs on Linux) |
| MLX MTP / speculation | Draft-cache token-pair trie, flush 256, host speculate | Pin `MLX_VERSION=de7b4ed`; M15a live-session retained |

---

## Pin and integration gaps

| Artifact | Upstream | Zerollama | Notes |
|----------|----------|-----------|-------|
| Ollama release | **v0.32.15** (`b7871fc`) | v0.30.11 base + selective cherry-picks through **v0.32.1**, then path-filtered ports through **v0.32.15** (Qwen 3.8 renderer/parser, repeat_penalty default, skipVerify, parse-error cancel) | Fetch: `./scripts/gpu/clone_upstream_ollama.sh`; compare at `../ollama-upstream` |
| llama.cpp tag | `b9888` (upstream v0.32.1) | **`8f114a9b`** (ggml-org master tip; past b10064) | Vendor sync via `./scripts/vendor/sync_vendor_llama.sh`; patch doctor: `./scripts/vendor/llama_patch_doctor.sh` |
| Compat layer | `llama/compat/` | **Partial** — in-tree `llama/compat/` + patches 0015–0017 | Full CMake overlay adoption still incremental; see [ggml-b9509-migration.md](./ggml-b9509-migration.md) |
| llama-server build | `cmake -S llama/server --preset cpu` (or GPU preset) | `./scripts/build/build_llama_server.sh` on sibling tree | Align presets when porting |
| MLX | `de7b4ed` / `fba4470b` | **Matched** (`MLX_VERSION=de7b4ed`) / mlx-c still `fba4470b` | Local overrides: `OLLAMA_MLX_SOURCE`, `OLLAMA_MLX_C_SOURCE` |

**Phase 15 blocker context:** native tensor page bind depends on llama.cpp APIs; staying on an old pin widens the gap. Bumping toward upstream’s pin is prerequisite work, not optional polish.

---

## What to adopt (high value)

Cherry-pick **by package**, not wholesale rebase.

| Target | Upstream paths | Why |
|--------|----------------|-----|
| Go llama-server wrapper | `llm/llama_server.go`, `llm/llama_binary.go` | Direct Go → llama-server; removes Python hop for default GGUF |
| Compat overlay | `llama/compat/*`, `llama/server/CMakeLists.txt` | Maintainability vs `llama/patches/` |
| Root pin file | `LLAMA_CPP_VERSION`, `llama/README.md` runbook | Single source of truth |
| Discovery | `discover/llama_server.go` | GPU probe via llama-server |
| MLX MTP / cache | Recent `x/mlxrunner` commits | Speculative decode on safetensors path |

## How we cherry-pick

Sibling checkout: `./scripts/gpu/clone_upstream_ollama.sh` → `../ollama-upstream`.

```bash
git -C ../ollama-upstream fetch --tags origin
git -C ../ollama-upstream log --oneline v0.32.1..v0.32.15 -- model/renderers model/parsers x/create llm/llama_server.go server/images.go api/types.go

# Path-filtered patch (preferred). Hand-port when server/sched.go or routes.go diverge.
git -C ../ollama-upstream format-patch -1 <sha> -- model/renderers model/parsers
git am --3way --exclude='server/sched.go' --exclude='runtime/*' <that.patch>
```

**Take:** renderer/parser/API bugfixes, llama-server media/compat, small `images.go`/`api` correctness, model-family import selection.  
**Skip:** desktop `app/`, `agent/` TUI, ollama.com cloud, deleting ggml/Python, llama.cpp pin bumps that fight `llama/patches/`, `sched.go` wholesale.

Hotspots to never blind-apply: `server/sched.go`, `llm/server.go`, `discover/`, `runtime/`, `x/trainingworker/`, `runner/ollamarunner/`.

---

---

## What to keep (zerollama differentiators)

Do **not** drop these to “match upstream”:

- **`runtime/`** — PA bookkeeping, admission (Phases 11–13), tools proxy (Phase 12), inprocess forward (Phase 14)
- **Phase 15 native KV** — C block pool + Go snapshot; upstream delegates KV to llama-server
- **Training track T1–T6** — pyembed, `/api/train`, VRAM broker handoff
- **Eliza cloud** — default remote upstream
- **Apple Silicon M1–M6 track** — unified memory admission, MPS LoRA, Metal sign-off scripts
- **Wan T2V, video Option 2, dual-scheduler product model**

---

## Technology ladder (reconciled view)

Zerollama’s ladder ([ROADMAP](./ROADMAP.md#technology-ladder-north-star)) assumed Python owns scheduler experiments before a native hot path. Upstream **already** put the hot path in llama-server (C++):

```text
Upstream default:     Go control plane → llama-server (libllama) → optional MLX
Zerollama default:    Go control plane → ggml runner OR Python runtime → llama
Zerollama north star: Go thin edge → native runtime (Python shrinks → C/Rust KV)
```

**Reconciliation:**

1. **Default GGUF chat** should converge toward **upstream shape** (Go → llama-server) — [Phase 17](./ROADMAP.md#phase-17--upstream-gguf-path-alignment-directional).
2. **Python runtime** remains the lab for **PA, training handoff, admission policy**, and Phase 15 experiments—not necessarily the permanent middleman for every chat token.
3. **`--llama-cpp-backend`** validates runtime routing today; **`llm/llama_server.go`** is the long-term default path for plain GGUF.

---

## Compare / benchmark workflow

### 1. Architecture diff (read-only)

```bash
ROOT=~/Sites/inference/zerollama
UP=~/Sites/inference/ollama-upstream

diff -ruN "$UP/llm/server.go" "$ROOT/llm/server.go" | less
diff -ruN "$UP/server/sched.go" "$ROOT/server/sched.go" | less

# Upstream-only
ls "$UP/llm/llama_server.go"

# Zerollama-only
ls "$ROOT/runtime/server.py" "$ROOT/x/trainingworker/"
```

### 2. Side-by-side serve

```bash
# Terminal A — upstream
cd ../ollama-upstream && go build -o ollama .
OLLAMA_HOST=127.0.0.1:11435 ./ollama serve

# Terminal B — zerollama ggml default
cd ../zerollama && ./zerollama serve

# Terminal C — zerollama llama.cpp backend (Python runtime)
cd ../zerollama && ./scripts/serve/serve_llama_cpp_backend.sh
```

### 3. Throughput smoke

```bash
go run ./cmd/bench -host 127.0.0.1:11435 -model llama3.2:3b -epochs 3 -format csv -output upstream.csv
# zerollama ggml: restart serve with ZEROLLAMA_LEGACY_RUNNER=1 first
go run ./cmd/bench -model llama3.2:3b -epochs 3 -format csv -output zerollama-ggml.csv
# with llama-cpp backend script running:
go run ./cmd/bench -model llama3.2:3b -epochs 3 -format csv -output zerollama-runtime.csv
```

Runtime health (zerollama Python path): `curl -s http://127.0.0.1:8081/health | jq '.llama_backend, .llama_backend_source'`

### 4. Mac Metal note

On Apple Silicon, compare three GGUF arms:

| Arm | Command | Backend |
|-----|---------|---------|
| ggml Metal | `./zerollama serve` | In-process ggml via runner |
| Python runtime | `./scripts/serve/serve_llama_cpp_backend.sh` or default sidecar + runtime routing | `inprocess` or `llama-server` Metal |
| Upstream | `OLLAMA_HOST=127.0.0.1:11435 ./ollama serve` | Go → llama-server Metal |

See [apple-silicon-metal.md](./apple-silicon-metal.md#compare-with-upstream-ollama).

---

## Roadmap mapping

| Zerollama phase | Upstream status | Action |
|-----------------|-----------------|--------|
| **12** Runtime default for text | N/A (no Python runtime) | Keep for tools/admission; don’t require Python for all GGUF long-term |
| **14** In-process llama | Subprocess llama-server only at Go layer | Keep ctypes path for PA/KV experiments |
| **15** Native KV | Delegated to llama-server | Keep; blocked on llama.cpp APIs — bump pin |
| **16** Thin edge daemon | Partially there (no Python) | Target: Go + llama-server + training embed, not “Python forever” |
| **17** Upstream GGUF alignment | **Done upstream** | Port `llama_server.go`, compat, pin bump — [ROADMAP](./ROADMAP.md#phase-17--upstream-gguf-path-alignment-directional) |
| **T1–T6** Training | N/A | Keep |
| **M1–M6** Apple track | N/A | Keep; add **M7** upstream-shape Metal benchmark |

---

## What upstream is investing in now (directional)

From recent `ollama-upstream` history (not exhaustive):

- MLX **MTP / speculation** — draft-cache token-pair trie keys, flush cap 256, host speculative polish
- **`x/create` rewrite** — pipeline/plan/quantize/writer split (zerollama keeps `imagegen.go`; Qwen3.5 parser/renderer selection ported)
- **Agent harness + TUI** — new top-level `agent/` + `cmd/tui/chat` (not ported; product call)
- **llama.cpp bumps** — compat docs, gemma4 projector offload
- **`cmd/launch`** — third-party agent integrations; deprecated-model warn (not ported)

Absent: Python runtime, training API, native KV experiments, Eliza.

---

## Cherry-pick status (Aug 2026, scouted upstream `v0.32.15` / `b7871fc`)

Additive ports that **do not** change zerollama architecture (Mac ggml default, Python sidecar, fleet/training, FIFO scheduler policy):

| Area | Status | Notes |
|------|--------|-------|
| **MLX pin `de7b4ed`** | **Done (Jul 2026)** | Matched upstream; sibling `../mlx` checked out; rebuild dylibs with `BUILD_MLX=1` when ready |
| **MLX load timeout** | **Done** | `WaitUntilRunning` uses `envconfig.LoadTimeout()`; keep tokenize cache |
| **MTP flush cap 256** | **Done** | `mtpPendingFlushTokens` 32→256 |
| **MTP drafter/session split** | **Done** | Persistent `mtpDrafter` + per-request `mtpDraftSession`; sets `draftLookahead=1` |
| **Draft-cache token-pair trie** | **Done** | `trieKey` + `kvCache.key()`; keep M15a live-session / `promptCacheKey` / `fastPath` |
| **Recurrent / GDN cache fixes** | **Done** | Single-pass conv boundary states; stop pinning forward buffer; `CausalConv1D` drops bare weight |
| **Gemma4 chat template** | **Done** | Renderer/parser + Jinja testdata + tool-message thinking Init |
| **Qwen 3.8 renderer/parser (#17745, #17749, #17757, #17855)** | **Done (Aug 2026)** | Dedicated `qwen3.8` variant (effort preamble, preserve thinking, developer fold); import picks it from `resolved_reasoning_effort`; ThinkValue `max`; conv1d reshape. GGUF `draft-mtp` stays **off** on qwen35 (ours; SWA desync). |
| **repeat_penalty default 1.0 (#6a261db)** | **Done (Aug 2026)** | Stop stacking 1.1 on models that omit it (qwen3.5/3.8, gemma4, …). |
| **skipVerify AND on duplicate digest (#15504)** | **Done (Aug 2026)** | `server/images.go` — always verify if any download of that digest was a cache miss. |
| **Parser error cancel (#17883)** | **Done (Aug 2026)** | Chat/generate: record parse error, cancel completion, report once (no wedge). |
| **WebP → PNG for llama-server (#17755)** | **Next** | `llm/llama_server.go` transcode; skip the integration image swap if noisy. |
| **Model metadata cache (#17752)** | **Next** | Per-request overhead; port if `/api/show` is hot. |
| **`x/create` rewrite** | **Deferred** | Upstream removes imagegen create path; keep zerollama `imagegen.go` until surgical split |
| **Agent harness + TUI** | **Skipped** | Product call — not ported |
| **Launch deprecated-model warn** | **Skipped** | Product call — not ported |
| **llama.cpp pin `b9781`** | **Done (Jun 2026)** | 16 patches on `vendor/llama-cpp-b9781`; manual apply for 0010/0012/0015 on b9781 layout; [ggml-b9509-migration.md](./ggml-b9509-migration.md) |
| **v0.30.11 Go delta** | **Done (partial)** | Native chat generate, CUDA 550+ compat, Vulkan Windows, Ornith/Qwen35, MLX speculate refactor, imagegen compile — **skipped** Claude/OpenCode auto-install, Kimi, desktop launchers |
| **llama-server MTP (GGUF)** | Done | `appendMTPDraftArgs`, draft GGUF layers, `DraftNumPredict`, opt-in `draft_num_predict`; **`appendSpecDraftBackendSamplingArg`** probes `--help` — **why:** eliza fork / older llama-server reject `--spec-draft-backend-sampling` |
| **Context shift (llama-server only)** | Done | `resolveContextShift`, `req.Shift` → sched → `LlamaServerConfig.ContextShift`; 400 when disabled |
| **DisableJinja (llama-server)** | Done | `usesOllamaRenderedChat` → `LlamaServerConfig.DisableJinja` for parser/renderer/harmony models |
| **LeadingBOS (llama-server)** | Done | `LeadingBOSForRenderer` + generate/chat `CompletionRequest.LeadingBOS` |
| **llama-server GPU discovery** | Done | Hybrid: Linux auto or `ZEROLLAMA_LLAMA_SERVER=1`; else ggml bootstrap. **Why Mac gated:** avoid spawning llama-server when ggml is default; tests were timing out when binary existed on disk |
| **Pre-tokenized `PromptTokens`** | Done | `chatPrompt` tail-truncate → runner. **Why:** re-tokenize after front-drop diverges; MLX MTP needs exact IDs |
| **MLX agent prompt hardening (M15)** | Done | Context cap, tail truncate, tokenize LRU, keep-alive floor, SSE keepalive, operator logs. **Why:** agent megaprompts on safetensors; see [mlx-agent-prompts.md](./mlx-agent-prompts.md) |
| **MLX agent live-session (M15a)** | Done | `fast_path` live KV + rotating snapshot rewind; prompt-chain on `messages_dropped`; 1× sliding_window trie snapshots. **Why:** 99% trie cached still ~75s on OptiQ without live rewind; see [mlx-agent-prompts.md#m15a](./mlx-agent-prompts.md#m15a-live-session--restore-jul-2026) |
| **CGO `-lc++` (llama.go)** | Done | **Why:** `go test ./discover/` links jinja C++ without production build env |
| **OpenCode thinking / launch drift** | Done | `cmd/launch/opencode.go`, `liveConfigMatches` |
| **LFM2 optional thinking** | Done | parser + renderer |
| **PreservedTokens wiring** | Done | parser interface + harmony + routes |
| **MLX MTP / Gemma4 assistant** | Done | `x/mlxrunner/mtp.go`, `x/models/gemma4/assistant.go`, safetensors draft create |
| **GGUF create `DRAFT`** | Done | `DraftFiles`/`DraftQuantize`, parser `DRAFT`, `server/create.go` draft layers |
| **Safetensors create `DRAFT`** | Done | `x/create/client`, `--draft-quantize` |
| **Upstream llama-server tests** | Done | `llm/llama_server_test.go` ported (MTP tests stay in `llama_server_mtp_test.go`) |
| **llama.cpp pin `b9672`** | Superseded | Replaced by **`b9781`** (v0.30.11) — see row above |
| **Native `gpu-discover` probe** | Done | Hidden subcommand + Linux/Windows CGO probes; enriches llama-server discovery with PCI/CC/gfx |
| **Integrated GPU policy + gfx1151** | Done | Strix Halo 8060S allowlist; `OLLAMA_IGPU_ENABLE`; Metal tensor retry on discovery |
| **llama-server unit tests** | Done | Upstream `llama_server_test.go` (minus MTP dupes); SSE ping, load stall, context shift |
| **OpenAI models list tags** | Done | `ToListCompletion` uses `model` field when set (#16556) |
| **CUDA FA + env log redaction** | Done | `cudaFlashAttentionSupported`; `filteredEnv` secret redaction |
| **Cohere2 MoE MLX (#16670)** | Done | `cohere` parser/renderer, `x/models/cohere2_moe`, safetensors import transform |
| **OMP launch (#16410)** | Done | `cmd/launch/omp.go` — ManagedSingleModel + web search plugin |
| **Launch provider drift (#16683)** | Done | `liveConfigMatches` rewrites editor config when on-disk models diverge |
| **Integration hf CLI (#16765)** | Done | Prefer `hf` over deprecated `huggingface-cli` in integration tests |
| **Cline providers.json (#16402)** | Done | Dual-write `providers.json` + legacy `globalState.json`; npm install prompt |
| **Qwen Code launch** | Done | `cmd/launch/qwen.go` — OpenAI-compatible `/v1` provider config + auto-install |
| **Pool launch** | Done | `cmd/launch/poolside.go` — `POOLSIDE_STANDALONE_BASE_URL` → zerollama `/v1` |
| **Context shift integration test (#16764)** | Done | Accept llama-server 4xx on oversized initial prompts |
| **Phase 17 E2E smoke** | Done | `phase17_llama_server_smoke.sh` — pulled tag via `P17_MODEL`; thinking models OK |
| **Launch model inventory** | Done | `LaunchModel` + `/api/tags` resolve; integration `Edit`/`ConfigureWithModels`; OMP catalog from inventory not picker; Cline stale-state fix; [launch-model-inventory.md](./launch-model-inventory.md) |
| **Makefile.sync pin guard** | Done | `make sync` → `sync_vendor_llama.sh` only; **why:** `checkout` on sync used to reset vendor and ship unpatch trees |
| **x/create test alignment** | Done | FP8 error message + Qwen35 MTP preserve expectations match upstream |
| **Remove CGO runners (#16031)** | **Skipped** | Would break Mac ggml default |
| **Wholesale `sched.go` replace** | **Skipped** | Keep FIFO / VRAM broker / darwin sidecar gates |
| **Mac default → llama-server** | **Skipped** | Phase 17 opt-in only |

**Explicitly not ported:** upstream Mac-default llama-server routing, Python runtime removal, ollama.com cloud default, Laguna MLX (not in zerollama imports), agent harness/TUI, launch deprecated-model warn.

---

- **Full rebase** onto upstream `main` — conflict surface too large; loses training/runtime work.
- **Deleting Python runtime** to match upstream — different product; shrink its role on the critical path instead.
- **Replacing Eliza** with ollama.com cloud.
- **Expecting `--llama-cpp-backend`** to be the final architecture — it is a **zerollama-specific** bridge (Go → Python → llama), not upstream’s Go → llama-server.

---

## Suggested next steps (ordered)

1. ~~Bump sibling `../llama.cpp` toward upstream **`b9509`**~~ — **done**; see [ggml-b9509-migration.md](./ggml-b9509-migration.md).
2. Port **`llama/compat/`** CMake overlay; retire overlapping `llama/patches/` over time (**done** — 0007 retired; 0016 hooks unified).
3. ~~Port **`llm/llama_server.go`**~~ — **scaffold done** (Phase 17 opt-in); wire as default after parity sign-off.
4. ~~Benchmark ggml vs Go-llama-server vs Python runtime~~ — **done** (M7); see [apple-silicon-metal.md](./apple-silicon-metal.md).
5. ~~Cherry-pick MLX MTP commits~~ — **done** (cache snapshots + prefill offsets in `x/mlxrunner/`).
6. ~~Phase 17 E2E smoke~~ — **done** — `phase17_llama_server_smoke.sh` PASS (Jun 2026); vision opt-in: `phase17_llama_server_vision_smoke.sh`.
7. ~~Deprecate **`OLLAMA_NEW_ENGINE`** / **`runner/ollamarunner`** for plain text GGUF~~ — **partial**; explicit `--llama-server-backend` now routes vision/thinking GGUF; Linux auto + Mac default vision still ggml.
8. Path-filter **v0.32.12–v0.32.15**: Qwen 3.8 renderer, repeat_penalty 1.0, skipVerify, parse-error cancel — **done (Aug 2026)**. Next: WebP transcode (#17755), metadata cache (#17752).
9. Skip llama.cpp pin bumps from those tags until `llama/patches/` rebases; our GGUF pin is independent.

---

## Code map (quick reference)

| Piece | Upstream | Zerollama |
|-------|----------|-----------|
| GGUF load/generate | `llm/llama_server.go` | `llm/server.go` → `runner/ollamarunner` |
| Scheduler | `server/sched.go` | `server/sched.go` (+ runtime defer) |
| Runtime routing flag | — | `server/runtime_inference_routing.go`, `envconfig/config.go` |
| Python sidecar | — | `runtime/server.py`, `server/darwin_sidecar.go` |
| llama.cpp integration | `llama/server/`, `llama/compat/` | `llama/patches/`, sibling `../llama.cpp` |
| Training | — | `training.py`, `x/trainingworker/pyembed/` |
| Clone helper | — | `scripts/gpu/clone_upstream_ollama.sh` |
