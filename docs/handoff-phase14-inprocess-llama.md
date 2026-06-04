# Handoff: Phase 14 in-process llama forward

**Audience:** Another engineer picking up zerollama runtime/Go work without this thread.

**Purpose:** Capture **why** Phase 14 exists, **what shipped**, **code locations**, **operator knobs**, **smokes**, and **known gaps** for in-process GGUF forward and Go render tokenize.

**Status (Jun 2026):**

| Item | State | Evidence |
|------|--------|----------|
| **ROADMAP Phase 14** | **Done** | ctypes GPU + wheel CPU smokes; `phase14_5080_signoff.sh` |
| **ROADMAP Phase 15** | **Partial (v0–v8 ops)** | Native KV pool, bind, forward plans — [phase15-native-kv.md](./phase15-native-kv.md), [handoff-phase15-native-kv.md](./handoff-phase15-native-kv.md) |
| **ctypes `inprocess` (GPU)** | **Shipped** | `phase14_inprocess_smoke.sh` PASS on 5080-class host |
| **`llama-cpp-python` (CPU default)** | **Shipped** | `phase14_both_backends.sh` PASS (~10 min); GPU opt-in via env (**wheel GPU aborts on 5080**) |
| **Render `truncate_mode=tokenize`** | **Shipped** | Go `/internal/render-chat` + Python `/internal/tokenize` |
| **Sampling parity** | **Shipped** | `sampler_options.py`; subprocess + in-process + wheel |
| **Heap-batch decode fix** | **Shipped** | `libllama_ctypes.py` `_batch_from_tokens`; `test_batch_from_tokens_sets_pos` |
| **CI** | **Green** | `go test ./server/...`; runtime pytest 410+; `check_gpu_scripts.sh`; `phase12_golden_ci.sh` |

**Prerequisite:** Rebuild `zerollama` from this repo before smokes. Stale serve omits `llama_backend` in `/health` and returns 404 on `/internal/tokenize`.

---

## Documentation index (read these first)

| Doc | Why |
|-----|-----|
| **[phase14-inprocess-llama.md](./phase14-inprocess-llama.md)** | Operator guide: backends, env, health fields, limits |
| [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md) | Phase 14 sign-off checklist on 5080 |
| [testing-smoke.md](./testing-smoke.md) | Script table; `RUN_E2E_PHASE14`; proxy header behavior |
| [handoff-phase12-runtime-tools.md](./handoff-phase12-runtime-tools.md) | Tools render/parse (Phase 12); truncation now uses Phase 14 tokenize |
| [scheduling-vram-policy.md](./scheduling-vram-policy.md) | VRAM broker unchanged; in-process `stop()` hooks same as subprocess |
| [CHANGELOG.md](../CHANGELOG.md) | Unreleased Phase 14 summary + fixes |
| [ROADMAP.md](./ROADMAP.md) | Phase 14 **Done**; Phase 15 native KV in progress |

---

## What we were solving

**Before Phase 14:**

```text
Client → Go :8080 → Python :8081 → HTTP → llama-server (child) → libllama
```

| Pain | Why it mattered |
|------|-----------------|
| Loopback HTTP per completion | Latency; awkward streaming |
| Two processes on one GPU | Phase 8 handoff + debugging harder |
| Tools render without ggml runner | Go used **heuristic** truncation (`len/4`) → wrong drops |
| Phase 15 blocked | Native KV needs forward in-process first |

**After Phase 14 (opt-in):**

```text
Client → Go :8080 → Python :8081 → libllama (ctypes or wheel) in same process
                └─ POST /internal/tokenize (vocab-only) for render-chat
```

**What did not move:** Python scheduler/admission (Phase 11), VRAM estimates (Phase 13), Go templates/parsers (Phase 12), subprocess default.

---

## Three forward backends

| Backend | Env | Production role |
|---------|-----|-----------------|
| **`subprocess`** | (default) | Stable; full llama-server; speculative decode |
| **`inprocess`** | `ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess` | **5080 GPU sign-off** — pinned `libllama.so` |
| **`llama-cpp-python`** | `…=llama-cpp-python` | Dev/CI without local build; **CPU default** |

**Selection:** `runtime/runtime/worker/factory.py` → `create_llama_worker()`. Env wins over YAML `llama_backend` (loaded in `runtime/config.py` `_from_mapping`).

**Why wheel defaults to CPU:** On tested cu124 wheels, `n_gpu_layers >= 1` can abort on `create_completion` (`free(): invalid pointer`) while ctypes GPU works. See `llama_cpp_n_gpu_layers()` in `llama_cpp_python.py`.

---

## Architecture

### Forward path

```text
InferenceEngine
  ├─ model_swap / admission / VRAM (unchanged)
  └─ LlamaForwardWorker
        ├─ LlamaServerProcess      (subprocess, default)
        ├─ LlamaInprocessWorker    (ctypes libllama.so)
        └─ LlamaCppPythonWorker    (pip wheel)
```

**Protocol:** `start(extra_args)` / `stop()` / `completion` / `completion_stream` / `tokenize_text` (when loaded).

**In-process constraints (v1):**

- Global `RLock` — streaming serialized on one GPU product default.
- Per-request `llama_context` — model stays loaded; ctx freed after completion/stream ends.
- No speculative/draft on in-process — use subprocess.
- ctypes struct sizes pinned to llama.cpp — bump requires binding refresh.

### Render tokenize (Phase 12 + 14)

```text
POST /internal/render-chat (Go)
  1. tokenizeForLoadedModel     → ggml runner if loaded
  2. tokenizeForRuntimeModel   → POST /internal/tokenize (Python, vocab_only)
  3. renderChatPromptHeuristic  → len/4 last resort
```

**Critical Go fix:** `tokenizeForRuntimeModel` uses `runtimeProxyConfigured()` (embed `runtimeworker.BaseURL()` **or** `ZEROLLAMA_RUNTIME_URL`). **Why:** embed leaves URL unset; checking env only left `truncate_mode=heuristic`.

**Python:** `engine.tokenize_gguf_text()` — reuse loaded worker if GGUF matches; else vocab cache (max 4 sessions).

**Failure policy:** Only `ErrRuntimeTokenizeUnavailable` → heuristic. HTTP/model errors from tokenize **fail** the request.

### Sampling

`runtime/runtime/worker/sampler_options.py`:

| Client options | In-process / wheel | Subprocess |
|----------------|-------------------|------------|
| None | Greedy | llama-server defaults |
| Any sampling key | Ollama defaults for omitted fields | Merged into `/completion` JSON |
| `temperature: 0` | Greedy | `temperature: 0` in JSON |

---

## Operator knobs (serve process)

| Env | Default | Why |
|-----|---------|-----|
| `ZEROLLAMA_RUNTIME_LLAMA_BACKEND` | `subprocess` | Opt-in in-process |
| `LLAMA_CPP_LIB` / `LLAMA_CPP_ROOT` | auto | ctypes `libllama.so` path |
| `LLAMA_SERVER_BIN` | — | Subprocess only |
| `ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS` | unset → CPU | Wheel GPU layers; negative → CPU + warning |
| `ZEROLLAMA_RUNTIME_EMBED` | on if URL unset | Embed Python on :8081 |
| `ZEROLLAMA_RUNTIME_URL` | unset for embed | **If set in shell, Go does not embed** |

**Confusion table:**

| Env | What operators think | Reality |
|-----|-------------------|---------|
| `ZEROLLAMA_RUNTIME=1` | Proxy everything | Phase 12 default-on for **eligible** models only |
| `OLLAMA_RUNTIME_ALL=1` | Proxy everything | Yes — all local generate/chat |
| `X-Zerollama-Runtime: 1` | Production default | **Smoke/integration** override |
| `RUN_E2E_PHASE14=1` | Production | Sets proxy header in e2e only |

### `/health` fields (Phase 14 backends)

| Field | In-process / wheel | Subprocess |
|-------|-------------------|------------|
| `llama_backend` | `inprocess` / `llama-cpp-python` | `subprocess` |
| `llama_backend_source` | `env`, `config` (explicit YAML key), or `default` | same |
| `llama_cpp` | wheel only: `gpu_mode`, `n_gpu_layers`, `loaded`, `env_n_gpu_layers` | key absent |
| `llama_server` | `false` | `true` |
| `llama_model` | loaded GGUF path | same |

**Why `llama_server: false`:** Means no **child** `llama-server` process, not “no model loaded.”

---

## Smoke scripts (why each exists)

| Script | Why |
|--------|-----|
| `scripts/phase14_serve_env.sh` | Unsets `ZEROLLAMA_RUNTIME_URL`; enables embed — **#1 smoke footgun** |
| `scripts/phase14_backend_smoke.sh` | One backend on running serve; preflight `llama_backend` + `/internal/tokenize` 404 |
| `scripts/phase14_inprocess_smoke.sh` | 5080 ctypes GPU sign-off (`RUN_E2E_INPROCESS=1`, `llama_backend_source=env`) |
| `scripts/phase14_yaml_config_smoke.sh` | Backend smoke with `llama_backend_source=config`; infers `inprocess` or `llama-cpp-python` from `/health` |
| `scripts/phase14_subprocess_default_smoke.sh` | Backend smoke with `llama_backend_source=default` (packaged subprocess default) |
| `scripts/phase14_wheel_cpu_smoke.sh` | Wheel CPU sign-off (`RUN_E2E_LLAMA_CPP_PYTHON=1`, `llama_backend_source=env`) |
| `scripts/phase14_wheel_gpu_smoke.sh` | Optional wheel GPU (`llama_cpp.gpu_mode=gpu` after generate) |
| `scripts/phase14_enable_yaml_inprocess.sh` | Uncomment `llama_backend: inprocess` in `single_gpu.yaml` after env smoke passes |
| `scripts/phase14_both_backends.sh` | Restart serve per backend; `env -u` URL and stale `RUN_E2E_*`; fails if zero backends ran |
| `scripts/phase14_5080_signoff.sh` | One-shot 5080 gate: both backends + YAML config full + Phase 15 multi-seq |
| `scripts/phase14_yaml_config_full_smoke.sh` | Temp YAML `llama_backend: inprocess` without editing repo `single_gpu.yaml` |
| `scripts/phase15_inprocess_signoff.sh` | One-shot Phase 15 GPU gate: KV decode hook + multi-seq |
| `scripts/phase15_inprocess_kv_smoke.sh` | Self-contained inprocess serve + `kv_decode_steps` on generate and `/health` |
| `scripts/phase15_inprocess_multiseq_smoke.sh` | Temp YAML `llama_parallel_slots: 2`; self-contained serve restart |
| `RUN_E2E_PHASE14=1` in `e2e_runtime_smoke.sh` | Forces `X-Zerollama-Runtime` on Go proxy — **sign-off only** |
| `RUN_E2E_LLAMA_BACKEND_SOURCE=config\|env\|default` | Assert `/health` provenance (YAML key vs env override vs packaged default) after serve restart |

### Quick start — inprocess via YAML (5080 GPU)

```bash
# Terminal A — uncomment llama_backend: inprocess in runtime/configs/single_gpu.yaml first
source ./scripts/phase14_serve_env.sh
export LLAMA_MODEL=/path/to/small.q8_0.gguf
export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
# omit ZEROLLAMA_RUNTIME_LLAMA_BACKEND when testing YAML default
./zerollama serve

# Terminal B
export LLAMA_MODEL=/path/to/same.gguf
export RUN_E2E_PROXY_MODEL=<pulled-local-tag>
./scripts/phase14_yaml_config_smoke.sh
```

Or enable YAML in one step after env inprocess smoke passes:

```bash
./scripts/phase14_enable_yaml_inprocess.sh
# restart serve without ZEROLLAMA_RUNTIME_LLAMA_BACKEND
./scripts/phase14_yaml_config_smoke.sh
```

### Quick start — inprocess (5080 GPU)

```bash
# Terminal A
source ./scripts/phase14_serve_env.sh
export LLAMA_MODEL=/path/to/small.q8_0.gguf
export LLAMA_CPP_LIB=$HOME/llama.cpp/build/bin/libllama.so
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess
./zerollama serve

# Terminal B
export LLAMA_MODEL=/path/to/same.gguf
export RUN_E2E_PROXY_MODEL=<pulled-local-tag>
./scripts/phase14_inprocess_smoke.sh
```

**Pass:** `PASS: phase14_backend_smoke`, `render-chat truncate_mode: tokenize`.

### Quick start — both backends

```bash
export LLAMA_MODEL=... RUN_E2E_PROXY_MODEL=...
./scripts/phase14_both_backends.sh
# wheel CPU ~10 min; inprocess GPU faster
```

---

## Code map

### Python (`runtime/`)

| Path | Role |
|------|------|
| `runtime/worker/factory.py` | Backend selection; `canonical_llama_backend()`; `llama_backend_source()` |
| `runtime/config.py` | YAML `llama_backend` + `llama_backend_from_file` |
| `runtime/vram_recommendations.py` | Shared Phase 13 skip-global-factor rules (health + snapshot) |
| `runtime/worker/libllama_ctypes.py` | ctypes bind; sampler chain; stream `finally` UAF fix |
| `runtime/worker/llama_inprocess.py` | In-process worker |
| `runtime/worker/llama_cpp_python.py` | Wheel worker; CPU-default GPU layers |
| `runtime/worker/sampler_options.py` | Ollama options → three backends |
| `runtime/worker/llama_server.py` | Subprocess (default) |
| `runtime/engine.py` | Worker lifecycle; `tokenize_gguf_text`; vocab cache |
| `runtime/server/app.py` | `POST /internal/tokenize` |

### Go (`server/` + `internal/`)

| Path | Role |
|------|------|
| `server/runtime_tokenize.go` | `tokenizeForRuntimeModel`, `memoizeTokenize` |
| `server/runtime_chat_prompt.go` | Truncation order ggml → runtime → heuristic |
| `server/runtime_url.go` | `effectiveRuntimeURL`, `runtimeProxyConfigured` |
| `internal/runtimeclient/tokenize.go` | HTTP client to `/internal/tokenize` |
| `x/runtimeworker/client.go` | Embed base URL; `SetBaseURLForTest` |

### Tests / hygiene

| Path | Why |
|------|-----|
| `server/sched_test.go` `TestMain` | Unsets runtime env from operator shells |
| `server/runtime_tokenize_test.go` | Embed + sidecar tokenize tests |
| `runtime/tests/conftest.py` | Clears `LLAMA_MODEL`, `RUN_E2E_*` between unit tests |
| `runtime/tests/test_llama_inprocess.py` | ctypes unit + optional GPU e2e |
| `runtime/tests/test_llama_cpp_python.py` | Wheel policy tests + optional GPU e2e |

---

## Bugs fixed during Phase 14 (learn from these)

| Symptom | Root cause | Fix |
|---------|------------|-----|
| `truncate_mode=heuristic` with embed on | Go checked `ZEROLLAMA_RUNTIME_URL` only | `runtimeProxyConfigured()` |
| ggml `runner terminated` in Phase 14 proxy smoke | Pulled tag hit legacy path | `RUN_E2E_PHASE14=1` → `X-Zerollama-Runtime` (smoke only) |
| `:8081` not listening | `ZEROLLAMA_RUNTIME_URL` set → embed off | `phase14_serve_env.sh` unsets URL |
| `go test ./server` sched failures | Shell exported runtime URL | `sched_test.go` TestMain unsets env |
| Stream segfault in-process | `finally` ran before consuming generator | `libllama_ctypes.py` stream `finally` after `yield` |
| Wheel smoke `RUN_E2E_INPROCESS` mismatch | Stale env in shell | `phase14_both_backends.sh` clears flags |
| pytest abort after wheel crash | `LLAMA_MODEL` in shell | `conftest.py` autouse `delenv` |
| `free(): invalid pointer` wheel GPU | pip wheel + GPU layers on host | Default wheel to CPU; env opt-in GPU |

---

## Known gaps / watch list

| Item | Severity | Notes |
|------|----------|--------|
| Subprocess still default | Product | Opt in after card-specific smoke |
| Wheel GPU on cu124 | High on some hosts | Use inprocess for 5080 GPU; wheel CPU OK for sign-off |
| ctypes struct drift | Medium | llama.cpp bump → update `libllama_ctypes.py` + CI |
| Grammar/mirostat in-process | Low | Not in v1 sampler chain |
| Multi-slot parallel in-process | Low | Serialized; Phase 15 may change |
| Production default-on without header | Ops | Phase 14 smoke does not prove all pulled tags route to runtime |
| Phase 15 native KV | Next | Do not expand Phase 14 scope |

---

## Suggested next steps

**For the next engineer:**

1. **5080 / production GPU:** Run `phase14_backend_smoke.sh` with `inprocess` on target hardware after every llama.cpp pin bump.
2. **Wheel:** Treat as optional; only enable GPU via `ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS` after verifying `create_completion` on that wheel build.
3. **Phase 15:** Native KV / scheduler — see [ROADMAP.md](./ROADMAP.md); do not add more sampler features to Phase 14 unless blocking.
4. **Docs:** [phase14-inprocess-llama.md](./phase14-inprocess-llama.md) is operator-facing; this file is handoff-facing.

**Do not:**

- Set `ZEROLLAMA_RUNTIME_URL` in systemd and expect embed (use unset URL + `ZEROLLAMA_RUNTIME_EMBED=on`).
- Assume `ZEROLLAMA_RUNTIME=1` equals Phase 14 proxy-all (use `OLLAMA_RUNTIME_ALL` or manifest/runtime-default).
- Block 5080 on `gpt-oss:20b` harmony or wheel GPU.

---

## Related docs

| Doc | Role |
|-----|------|
| [handoff-phase12-runtime-tools.md](./handoff-phase12-runtime-tools.md) | Tools path; render now tokenizes via Phase 14 |
| [handoff-gpu-training-integration.md](./handoff-gpu-training-integration.md) | Training + same-process Python |
| [runtime-embed.md](./runtime-embed.md) | Why embed vs sidecar |
| [python-migration.md](./python-migration.md) | Full migration ladder |

---

## Conversation reference

Transcript: search for `Phase 14`, `truncate_mode=tokenize`, `llama_cpp_n_gpu_layers`, `phase14_both_backends`, `runtimeProxyConfigured`, agent transcript `9b3b028b-5454-4dd1-a89b-5311b3f27819`.
