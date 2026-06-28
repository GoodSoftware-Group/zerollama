# Phase 16 — thin edge daemon

**Status:** **Partial (v0 ops, v1 runner stub, v2 CGO drop, Jun 2026)** — operator flag + env + compile tag for upstream-shaped deployments. Full “Go gone” for inference control plane remains directional.

**Related:** [Phase 17](./phase17-llama-server.md) (Go → llama-server) · [upstream-ollama-diff.md](./upstream-ollama-diff.md) · [ROADMAP.md](./ROADMAP.md#local-inference--actionable-phases)

---

## Why Phase 16 exists

Zerollama today runs a **full Go daemon**: registry pull, HTTP API, Eliza cloud proxy, ggml scheduler, Python runtime embed/sidecar, training worker, fleet hooks. Upstream Ollama already matches part of the north star — **Go control plane → llama-server (libllama)** with no Python chat middleman.

Phase 16 shrinks the Go process to a **thin edge** that keeps zerollama differentiators (training, Eliza, fleet, launch integrations) while routing **all local GGUF chat/generate** through upstream-shaped paths (Phase 17) instead of ggml runner + Python runtime by default.

```text
Today (default Mac):     Client → Go → ggml Metal (+ sidecar :8081 for PA/admission)
Today (Linux serve):     Client → Go → llama-server (auto)  OR  Python runtime (if enabled)
Phase 16 edge (--edge):  Client → Go (thin) → llama-server → libllama
                         training / Eliza / fleet unchanged in Go
Phase 16 north star:     Same API surface; inference hot path not owned by ggml runner or Python chat proxy
```

**Non-goals (v0):** separate binary name, Rust rewrite, deleting `runtime/` (still used for training + Phase 15 experiments), removing ggml on Mac default.

---

## Status (v0)

| Item | State | Why |
|------|--------|-----|
| `zerollama serve --edge` | **Done** | Sets `ZEROLLAMA_EDGE=1` |
| `ZEROLLAMA_EDGE=1` | **Done** | Forces Go → llama-server, `ZEROLLAMA_LEGACY_RUNNER=1`, `ZEROLLAMA_RUNTIME=0` |
| Linux auto (`ZEROLLAMA_LLAMA_SERVER=auto`) | **Done** | Serve sets `auto` when llama-server on disk; routes **all** GGUF (text + vision) |
| Mac default unchanged | **Done** | M7 bench: ggml Metal still faster; edge is explicit opt-in |
| Separate edge binary / build tag | **Done (v1 marker + v2 CGO drop)** — `./scripts/build_zerollama_edge.sh` (`-tags edge`); v2 excludes `llm/server.go` ggml CGO from edge link |
| Drop ggml runner from build | **Partial (v2)** — edge build: no in-process ggml CGO; subprocess `zerollama runner` stubbed (v1); Python embed/MLX CGO remain in link (v3 disables runtime embed/sidecar at compile time via `EdgeBuildTag`) |
| Training embed without runtime chat | **Partial** | Edge disables runtime chat; pyembed training path unchanged |

---

## Enable edge mode

```bash
# Build llama-server (Linux CUDA or Mac Metal)
./scripts/build_llama_server.sh   # or build_ollama_llama_server_darwin.sh

# Upstream-shaped serve: llama-server for GGUF, no Python runtime chat
./scripts/serve_edge.sh
# or:
./zerollama serve --edge

# Edge-marked binary (compile marker; serve defaults to edge unless ZEROLLAMA_EDGE=0)
./scripts/build_zerollama_edge.sh
./zerollama-edge serve
```

**What `--edge` sets (when unset):**

| Env | Value | Why |
|-----|--------|-----|
| `ZEROLLAMA_LLAMA_SERVER` | `1` | Explicit Go → llama-server for all GGUF |
| `ZEROLLAMA_LEGACY_RUNNER` | `1` | Skip Python runtime proxy for tagged models |
| `ZEROLLAMA_RUNTIME` | `0` | Disable Python runtime chat middleman |
| `ZEROLLAMA_RUNTIME_EMBED` | `0` | Disable Linux CGO runtime embed |
| `ZEROLLAMA_RUNTIME_DARWIN_SIDECAR` | `0` | Disable Mac uv sidecar bootstrap |

Training (`/api/train/*`), Eliza cloud, fleet, and launch integrations **stay in Go**. Phase 15 in-process / PA experiments still available via explicit `--llama-cpp-backend` (not edge default).

---

## Layer map (what stays vs shrinks)

| Layer | Phase 16 edge | Full zerollama default |
|-------|---------------|------------------------|
| Pull / registry / manifest | **Keep (Go)** | Keep |
| OpenAI + Ollama HTTP API | **Keep (Go)** | Keep |
| Eliza cloud proxy | **Keep (Go)** | Keep |
| Fleet / mDNS / score | **Keep (Go)** | Keep |
| GGUF text chat/generate | **llama-server** | Mac: ggml; Linux: auto llama-server |
| Python runtime chat | **Off** | Linux default-on when sidecar configured |
| Training pyembed | **Keep** | Keep |
| ggml ollama-engine runner | **Fallback only** | Mac default + legacy archs |
| Phase 15 native KV (Python) | Opt-in (`--llama-cpp-backend`) | Sidecar / embed |

---

## Exit criteria (directional)

| # | Criterion | Owner | State |
|---|-----------|--------|--------|
| 1 | `--edge` + env disable runtime chat; route GGUF via llama-server | Go | **Done (v0)** |
| 2 | Linux serve auto (`auto`) routes all GGUF without explicit flag | Go | **Done (Jun 2026)** |
| 3 | Operator doc + env table | Docs | **Done** — this file |
| 4 | Edge smoke: pull tag → generate via llama-server, no runtime on :8081 | Repo | **Done (Mac Jun 2026)** — `./scripts/phase16_edge_smoke.sh` (`--edge`); `./scripts/phase16_edge_binary_smoke.sh` (`-tags edge` artifact); Linux CUDA via `RUN_E2E_UPSTREAM_GGUF=1` pending |
| 5 | `/api/status` `inference.backend` policy snapshot | Go | **Done (Jun 2026)** |
| 5b | `/api/version` `edge_build` compile marker | Go | **Done (Jun 2026)** |
| 6 | Build tag excluding ggml runner subprocess | Go | **Done (v1)** — `-tags edge` sets `GgmlRunnerLinked=false`, stubs `zerollama runner`, fail-fast without llama-server |
| 7 | Drop in-process ggml from edge binary | Go | **Partial (v2)** — `server.go`/`server_score.go` excluded with `//go:build !edge`; edge `NewLlamaServer` is llama-server-only; no `llama`/`model` in edge dep tree |
| 8 | Inference control plane “Go gone” | Go | **Not started** — sched loads llama-server + MLX only |

Mark **v0 Done** when 1–3 pass and criterion 4 smoke passes on ship hardware (`./scripts/phase16_edge_smoke.sh`). Mac sign-off **Done** Jun 2026 (`driaforall/tiny-agent-a-0.5b:q8_0`, isolated port). Edge smoke asserts `/api/status` `inference.backend` (`edge`, `runtime_chat=off`, `gguf_path=llama-server`) and `/api/version` `edge_build`. Compile marker: `./scripts/phase16_edge_build_smoke.sh` (CI regression; no GPU).

**Scheduler guard (v1):** when edge policy is active without llama-server routing, `schedSkipGgmlRunnerLoad` returns HTTP 400 instead of spawning ggml. Edge builds (`-tags edge`) fail fast with `ErrGgmlRunnerUnlinked` unless `ZEROLLAMA_LLAMA_SERVER` is on.

---

## Build tags vs runtime `--edge`

| Mechanism | What it does | Why both exist |
|-----------|--------------|----------------|
| **`zerollama serve --edge`** | Runtime env: llama-server on, runtime chat off | Operator toggle on any binary; no rebuild |
| **`go build -tags edge`** | Compile-time: no ggml CGO in `llm/server.go`, runner subprocess stubbed, `ggml_linked=false` | Smaller edge artifact; link-time guarantee GGUF cannot load ggml paths |
| **`version.EdgeBuild=true`** (ldflags) | CLI/API marker: `zerollama -v`, `/api/version` `edge_build` | CI and fleet can verify which artifact is deployed |

**Typical edge deploy:** `./scripts/build_zerollama_edge.sh` then `./zerollama-edge serve` (edge defaults apply automatically; set `ZEROLLAMA_EDGE=0` to opt out).

---

## Code map (WHY-oriented)

| File | Role |
|------|------|
| `envconfig/serve_backend.go` | `ApplyEdgeModeDefaults`, `EdgeMode()`, llama-server env parsing |
| `envconfig/ggml_runner*.go` | `GgmlRunnerLinked()` — compile-time false for `-tags edge` |
| `server/runtime_inference_routing.go` | Edge mode never routes chat/generate to Python runtime |
| `server/edge_ggml_policy.go` | Scheduler rejects ggml loads when edge/unlinked without llama-server |
| `llm/server_shared.go` | Shared API types + `LoadModel` (pure Go GGUF header decode) |
| `llm/server.go` (`!edge`) | Full ggml/ollama-engine subprocess path |
| `llm/server_edge.go` (`edge`) | llama-server-only `NewLlamaServer`; stub `StartRunner` |
| `runner/runner_edge.go` | Stub `zerollama runner` subprocess entry |
| `discover/gpu_discovery_upstream.go` | Skip ggml bootstrap when runner unlinked |
| `server/inference_backend_policy.go` | `/api/status` `inference.backend` policy snapshot |
| `server/version_handler.go` | `/api/version` `edge_build` marker |

---

## Relationship to Phase 17

Phase 17 ports upstream integration (`llm/llama_server.go`, discovery, LeadingBOS, pin **`b9781`**). Phase 16 **uses** that path as the default local inference shape when `--edge` is set (or on Linux via `auto`).

| Mode | GGUF path | Python runtime chat |
|------|-----------|---------------------|
| Mac default | ggml Metal | Sidecar optional |
| Linux serve | llama-server (`auto`) | On when sidecar configured |
| `--llama-server-backend` | llama-server (explicit) | Usually off in smokes |
| `--edge` | llama-server (explicit) | **Off** |
| `--llama-cpp-backend` | Python → inprocess/llama-server | **On** (Phase 12–15 lab) |

---

## Related docs

- [phase17-llama-server.md](./phase17-llama-server.md)
- [upstream-ollama-diff.md](./upstream-ollama-diff.md)
- [python-migration.md](./python-migration.md) — Phases 0–7 history
- [scheduling-vram-policy.md](./scheduling-vram-policy.md) — VRAM broker (still Go on edge)
