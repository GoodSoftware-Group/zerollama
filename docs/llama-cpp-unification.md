# llama.cpp backend unification

**Status:** Unified (Jul 2026). Runtime + `llama-server` + in-process Go ggml vendor share **ggml-org @ `LLAMA_CPP_COMMIT` (`86d86ed4`)** + **79** Ollama/zerollama patches. Eliza QJL deferred.

**Related:** [runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md), [gpu-profiles-l2.md](./gpu-profiles-l2.md), [phase17-llama-server.md](./phase17-llama-server.md), [ggml-b9509-migration.md](./ggml-b9509-migration.md).

---

## Why unify

Operators were maintaining **multiple llama.cpp trees** with different pins and flags:

| Checkout | Typical use | Problem |
|----------|-------------|---------|
| `../llama.cpp` (stock ggml-org) | Runtime, CUDA graphs, L3 | Missing eliza fork KV / dflash flags |
| `../eliza-llama.cpp` | L2 QJL/Polar experiments | Stale pin; rejects newer argv (e.g. `--spec-draft-backend-sampling`) |
| `llama/vendor/llama-cpp-b9781/` + patches | Go **ollama-engine** Metal/CUDA | Different commit than runtime sibling |
| Flash-MoE / anemll builds | MoE sidecar | Separate binary search path |

**Symptom:** `qwen3.6-mtp` fails when `LLAMA_SERVER_BIN` points at an old eliza checkout while zerollama passes flags the binary does not know.

**Goal:** **One runtime source tree**, **one `llama-server` binary**, **profile flags instead of fork siblings**. Converge in-process ggml vendor onto the same commit when L2 gates justify rebase.

---

## Target architecture

```text
         ┌─────────────────────────────────────┐
         │  ggml-org/llama.cpp @ LLAMA_CPP_COMMIT │
         │  + llama/patches/ (25 Ollama deltas) │
         └─────────────────────────────────────┘
                           │
              vendor/llama-cpp-86d86ed4 + sync
                           │
         ┌─────────────────┴──────────────────────────────┐
         │ ../llama.cpp sibling    in-tree ggml + llama.cpp │
         │ llama-server build      Go CGO ollama-engine     │
         └──────────────────────────────────────────────────┘
```

---

## What is unified today

| Surface | Tree | Pin |
|---------|------|-----|
| `llama-server` (Go + Python subprocess) | `vendor/llama-cpp-<pin>` (patched; fallback `../llama.cpp`) | `LLAMA_CPP_COMMIT` |
| Python in-process ctypes | Same `libllama` from unified build | Same |
| L2 stock vs fork gates | **Same binary**; `ZEROLLAMA_LLAMA_FORK` toggles profile argv | Same |
| Flag probing (`--spec-draft-backend-sampling`, `--ctx-checkpoints`) | Per-binary `--help` cache | Avoids argv crashes on stale builds |

**Deprecated:** `../eliza-llama.cpp`, `build_eliza_llama_server.sh` (wraps unified build), dual `STOCK_LLAMA_CPP_ROOT` / `ELIZA_LLAMA_CPP_ROOT` in L2 scripts (aliases only).

---

## What is not unified yet

| Surface | Current | Next step |
|---------|---------|-----------|
| Flash-MoE llama-server | Optional separate anemll build | Fold behind unified tree or explicit `ZEROLLAMA_FLASH_MOE` override only |
| Upstream Ollama pin | `b9509` reference | Track via [upstream-ollama-diff.md](./upstream-ollama-diff.md); zerollama leads on unified eliza base |

---

## Operator contract

### One clone, one build

```bash
./scripts/vendor/rebase_vendor_unified.sh --sync   # vendor + in-tree ggml sync (U3)
./scripts/build/build_llama_server.sh             # Metal or CUDA — one llama-server + libllama
```

### Environment (prefer unset overrides)

| Variable | Role |
|----------|------|
| `LLAMA_CPP_ROOT` | Override unified sibling (default `../llama.cpp`) |
| `LLAMA_CPP_COMMIT` | Pin file in repo root — do not hand-edit checkout |
| `LLAMA_SERVER_BIN` | **Override only when needed**; doctor warns if outside unified tree |
| `ZEROLLAMA_LLAMA_FORK` | `0` = L1 q8_0 profiles; unset/`1` = fork KV argv when binary supports it |

### Migrate from legacy siblings

```bash
./scripts/vendor/migrate_llama_cpp_unified.sh
MIGRATE_BUILD=1 ./scripts/vendor/migrate_llama_cpp_unified.sh   # clone pin + build if needed
```

**Automatic redirect:** `zerollama serve` and the Darwin runtime sidecar call `ApplyUnifiedLlamaCppEnv()` — if `LLAMA_SERVER_BIN` points at `eliza-llama.cpp` but unified `../llama.cpp` is built, env is rewritten at startup (logged).

**Health:** `curl -s :8081/health | jq .llama_cpp_unified`

---

## Discovery order (`FindLlamaServer`)

1. `LLAMA_SERVER_BIN` (explicit override)
2. `$LLAMA_CPP_ROOT/build/bin/llama-server` (unified sibling)
3. Flash-MoE binary when `ZEROLLAMA_FLASH_MOE=1`
4. Packaged / cmake build artifact search

**Why unified second:** fresh builds at `../llama.cpp` win over stale PATH or legacy sibling env.

---

## Doctor

`zerollama doctor` includes **`llama.cpp unified`**:

- Unified root + pin vs git HEAD
- Whether `LLAMA_SERVER_BIN` lives under unified tree
- Legacy checkout name warning (`eliza-llama.cpp`)
- In-process vendor pin drift (`Makefile.sync FETCH_HEAD` vs `LLAMA_CPP_COMMIT`)

Pin status script: `./scripts/phase/phase17_l2_pin_status.sh`

---

## Implementation map

| Path | Role |
|------|------|
| `LLAMA_CPP_COMMIT` | Unified runtime pin |
| `scripts/vendor/ensure_llama_cpp_sibling.sh` | Clone + checkout |
| `scripts/build/build_llama_server.sh` | Single build entry |
| `llm/llama_cpp_unified.go` | Root resolution + doctor report |
| `llm/llama_server_flags.go` | Per-binary `--help` probe |
| `runtime/LLAMA_CPP_PIN.md` | Runtime operator pin doc |
| `server/darwin_sidecar.go` | Sets `LLAMA_CPP_ROOT` / `LLAMA_SERVER_BIN` from sibling |

---

## Phased rollout

| Phase | Deliverable | Exit |
|-------|-------------|------|
| **U1** (done) | One runtime tree + profile-based L2 | `build_llama_server.sh` only; L2 scripts use one root |
| **U2** (done) | Discovery + doctor + argv probe | No startup crash on stale fork; doctor warns legacy paths |
| **U3** (done) | Vendor rebase | `Makefile.sync` → elizaOS @ `LLAMA_CPP_COMMIT`; `./scripts/vendor/rebase_vendor_unified.sh` |
| **U4** (next) | Phase 17 default | Linux/Mac plain GGUF → Go→unified llama-server without Python hop where policy allows |
| **U5** (deferred) | Flash-MoE in unified tree | MoE flags in same binary or documented override only |

---

## Non-goals

- Replacing MLX / safetensors path (separate MLX pins)
- HTTP to vLLM/SGLang as llama.cpp substitute
- Deleting Go ggml runner before vendor rebase completes
