# Upstream Ollama comparison (zerollama vs `ollama/ollama`)

**Audience:** contributors deciding what to cherry-pick from vanilla Ollama without rebasing zerollama.

**Related:** [ROADMAP.md](./ROADMAP.md#phase-17--upstream-gguf-path-alignment-directional), [llama-cpp-backend.md](./llama-cpp-backend.md), [python-migration.md](./python-migration.md), [development.md](./development.md#compare-with-upstream-ollama).

---

## Why this doc exists

Zerollama forked Ollama and added a **Python runtime**, **GPU training**, **Eliza cloud**, and **native KV experiments**. Upstream Ollama took a different bet: **drop in-process ggml for GGUF** and route all text inference through **`llama-server`**. Both repos still share MLX safetensors routing and much of the Go API surface.

This document captures **architecture deltas**, **pin gaps**, and **actionable cherry-picks** so roadmap work stays intentional—not accidental full merges.

---

## Local upstream checkout

Clone beside zerollama (no merge into this repo):

```bash
./scripts/clone_upstream_ollama.sh
# default: ../ollama-upstream
```

Build and run on a different port for A/B:

```bash
./scripts/build_upstream_ollama_mac.sh   # llama-server (Metal) + go binary — required, not just go build
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
| Go llama integration | `llm/llama_server.go` | **Missing** — uses `runner/ollamarunner` |
| Python runtime | None | `runtime/` FastAPI sidecar/embed |
| Training | None | `/api/train/*`, `training.py`, pyembed |
| Remote cloud | ollama.com | **Eliza Cloud** default |
| llama.cpp pin | `LLAMA_CPP_VERSION` = **`b9509`** (repo root) | **Aligned** — runtime sibling + in-tree ggml @ b9509; [ggml-b9509-migration.md](./ggml-b9509-migration.md) |
| Ollama-specific llama fixes | `llama/compat/` + CMake `PATCH_COMMAND` | `llama/patches/` (**16** on b9611) + in-tree CGO deltas; compat hooks via **0016**, ggml deltas via **0017** |
| GPU discovery | `discover/llama_server.go` probe | ggml-centric + runtime probes |
| MLX MTP / speculation | Recent commits (draft tokens, KV file split) | Pin behind `MLX_VERSION`; cherry-pick as needed |

---

## Pin and integration gaps

| Artifact | Upstream | Zerollama | Notes |
|----------|----------|-----------|-------|
| llama.cpp tag | `b9509` | `b9509` (`512882ac`) | In-tree ggml synced via `./scripts/sync_vendor_b9509.sh` |
| Compat layer | `llama/compat/` | Not adopted | Replaces many manual patches; see upstream `llama/compat/README.md` |
| llama-server build | `cmake -S llama/server --preset cpu` (or GPU preset) | `./scripts/build_llama_server.sh` on sibling tree | Align presets when porting |
| MLX | `MLX_VERSION` / `MLX_C_VERSION` in CMake | Same pattern | Local overrides: `OLLAMA_MLX_SOURCE`, `OLLAMA_MLX_C_SOURCE` |

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

**Merge conflict hotspots (avoid blind rebase):** `server/sched.go`, `llm/server.go`, `discover/`, entire `runtime/`, `x/trainingworker/`, `runner/ollamarunner/` (we have it; upstream deleted it).

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
cd ../zerollama && ./scripts/serve_llama_cpp_backend.sh
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
| Python runtime | `./scripts/serve_llama_cpp_backend.sh` or default sidecar + runtime routing | `inprocess` or `llama-server` Metal |
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

- MLX **MTP / speculation** — draft tokens, streaming, KV cache file layout
- **llama.cpp bumps** — compat docs, gemma4 projector offload
- **`cmd/launch`** — third-party agent integrations
- Prompt caching decoupled from context shift

Absent: Python runtime, training API, native KV experiments, Eliza.

---

## Non-goals for this track

- **Full rebase** onto upstream `main` — conflict surface too large; loses training/runtime work.
- **Deleting Python runtime** to match upstream — different product; shrink its role on the critical path instead.
- **Replacing Eliza** with ollama.com cloud.
- **Expecting `--llama-cpp-backend`** to be the final architecture — it is a **zerollama-specific** bridge (Go → Python → llama), not upstream’s Go → llama-server.

---

## Suggested next steps (ordered)

1. ~~Bump sibling `../llama.cpp` toward upstream **`b9509`**~~ — **done**; see [ggml-b9509-migration.md](./ggml-b9509-migration.md).
2. Port **`llama/compat/`** CMake overlay; retire overlapping `llama/patches/` over time (**done** — 0007 retired; 0016 hooks unified).
3. ~~Port **`llm/llama_server.go`**~~ — **scaffold done** (Phase 17 opt-in); wire as default after parity sign-off.
4. Benchmark **ggml vs Go-llama-server vs Python runtime** on ship hardware ([apple-silicon-metal.md](./apple-silicon-metal.md), [testing-smoke.md](./testing-smoke.md)).
5. Cherry-pick MLX MTP commits after MLX pin alignment (`./scripts/ensure_mlx_sources.sh`).
6. Deprecate **`OLLAMA_NEW_ENGINE`** / **`runner/ollamarunner`** for GGUF once Go → llama-server is default (keep for vision/thinking until parity).

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
| Clone helper | — | `scripts/clone_upstream_ollama.sh` |
