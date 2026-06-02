# Phase 14 — in-process llama forward

**Status:** Partial (May 2026). Subprocess `llama-server` remains the default; opt in with env. In-process forward + libllama tokenize for Go render-chat shipped. **5080 sign-off:** `inprocess` GPU via `phase14_backend_smoke.sh`; both backends via `phase14_both_backends.sh` (wheel uses CPU unless `ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS` set).

**Upgrade serve first:** Phase 14 needs a **current** `zerollama serve` (rebuild from this repo). Stale runtimes return HTTP 404 on `/internal/tokenize` and omit `llama_backend` in `/health`; `phase14_backend_smoke.sh` fails fast on that.

**Embed vs sidecar:** If `ZEROLLAMA_RUNTIME_URL` is set in the shell, Go will **not** embed Python on `:8081` (expects an external sidecar). For single-process smokes:

```bash
source ./scripts/phase14_serve_env.sh
export LLAMA_MODEL=/path/to/model.gguf
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess
./zerollama serve
```

---

## Why Phase 14 exists

The Python runtime (Phases 1–13) already owns **scheduling**, **admission**, **VRAM estimates**, and the **HTTP API**. Forward still went through a **second process**:

```text
Client → Go :8080 (proxy) → Python :8081 → HTTP → llama-server (loopback) → libllama
```

That design was correct for bootstrap, but it creates real costs:

| Problem | Why it hurts |
|---------|----------------|
| **Loopback HTTP** on every completion | Extra latency and serialization; harder to stream efficiently. |
| **Two processes** holding GPU context | VRAM accounting and handoff (Phase 8) must coordinate two runtimes; OOM debugging is harder. |
| **No tokenizer without ggml** | Phase 12 tools need **token-accurate** prompt truncation in Go `/internal/render-chat`. Without a loaded ggml runner, the runtime path used a **len/4 heuristic** — wrong for tools and long contexts. |
| **Blocks Phase 15** | Native KV / scheduler in C/Rust needs forward in-process first; subprocess HTTP is not a stable foundation. |

Phase 14 moves **only forward + vocab tokenize** into the Python process. Go still owns registry, proxy, render templates, tool parsers, and training coordination.

---

## Design principles (what we did *not* change)

| Layer | Still owned by | Why unchanged |
|-------|----------------|---------------|
| Request queues, priorities | Python `InferenceEngine` + scheduler | Product model: many stakeholders, one admission policy (Phase 11). |
| VRAM pre-check / clamp | Python `gpu_vram`, Go proxy options | Estimates (Phase 13) already run before load; forward backend does not bypass them. |
| Model swap / handoff | `model_swap`, Go `server/vram` | Training and ggml eviction still call the same hooks; in-process worker implements `stop()` like subprocess. |
| Chat templates & tools | Go `/internal/render-chat`, parsers | Templates stay in Go so one Modelfile path serves ggml and runtime. |

**Why subprocess stays default:** Production stability. ctypes binds a **pinned** `libllama.so`; struct layouts can break on llama.cpp bumps. Operators opt in after GPU smoke on their card.

---

## Three forward backends

| Backend | Env | Why it exists |
|---------|-----|----------------|
| **`subprocess`** (default) | (unset) | Battle-tested; speculative decode, full llama-server flags, no ctypes fragility. |
| **`inprocess`** | `ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess` | **5080 GPU sign-off** — same pinned `libllama.so` as `LLAMA_SERVER_BIN`, no loopback HTTP. |
| **`llama-cpp-python`** | `…=llama-cpp-python` | Hosts without a local build; pip wheel only. **CPU default** on many wheels (see below). |

Selection: `runtime/runtime/worker/factory.py` → `create_llama_worker()`. Env wins over `RuntimeConfig.llama_backend`.

---

## Operator knobs

| Env | Default | Role |
|-----|---------|------|
| `ZEROLLAMA_RUNTIME_LLAMA_BACKEND` | `subprocess` | `inprocess`, `llama-cpp-python`, or `subprocess` |
| `LLAMA_CPP_LIB` | auto | Path to `libllama.so` (else `LLAMA_CPP_ROOT/build/bin/libllama.so`) |
| `LLAMA_CPP_ROOT` | sibling `../llama.cpp` | Checkout used to locate `libllama.so` |
| `LLAMA_SERVER_BIN` | — | Required for **subprocess** only |
| `ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS` | (unset → CPU) | **Wheel only** — positive layer count; negative values fall back to CPU with warning |

Related (unchanged from Phase 12):

| Env | Why operators confuse it with Phase 14 |
|-----|----------------------------------------|
| `ZEROLLAMA_RUNTIME=1` | Enables runtime **default-on** for eligible models — not proxy-all. |
| `OLLAMA_RUNTIME_ALL=1` | Forces **every** local generate/chat to the runtime without per-model config. |
| `X-Zerollama-Runtime: 1` | Per-request smoke/integration override (see testing-smoke.md). |

Admission (Phase 11), VRAM estimates (Phase 13), and Go render/parse (Phase 12) are unchanged.

---

## Enable on serve

```bash
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=inprocess
export LLAMA_MODEL=/path/to/model.gguf
# optional: export LLAMA_CPP_LIB=/path/to/llama.cpp/build/bin/libllama.so
zerollama serve
```

`GET :8081/health` when weights are loaded:

| Field | In-process / wheel | Subprocess |
|-------|-------------------|------------|
| `llama_backend` | `"inprocess"` or `"llama-cpp-python"` | `"subprocess"` |
| `llama_server` | `false` (no child process) | `true` |
| `llama_model` | path to loaded GGUF | same |
| `inference_state` | `"running"` | same |

**Why `llama_server: false` for in-process:** The field means “a `llama-server` **child process** is running,” not “weights are loaded.” For Phase 14 backends, use `inference_state` and `llama_model`.

---

## Render tokenize (Phase 12 + 14)

**Why:** Tools chat builds a prompt in Go, then truncates to fit `num_ctx`. With a ggml runner loaded, truncation uses real token counts. Runtime-only inference had no runner → **heuristic** truncation (`truncate_mode: heuristic`) → dropped wrong messages.

**Flow:**

```text
POST /internal/render-chat (Go)
  → ggml runner tokenize?  (if loaded)
  → else POST /internal/tokenize (Python, vocab_only)
  → else len/4 heuristic (last resort)
```

**Why vocab-only:** Render needs tokenizer weights, not full GPU weights. Loading full GGUF per tokenize would destroy VRAM and latency. Python caches up to 4 vocab sessions (`engine._VOCAB_CACHE_MAX`).

**Why `runtimeProxyConfigured()` in Go:** Embedded runtime sets loopback URL via `runtimeworker.BaseURL()`, not `ZEROLLAMA_RUNTIME_URL`. Checking only the env var skipped tokenize when embed was on.

**Failure policy:** HTTP or model errors from `/internal/tokenize` **fail** the render request. Only unreachable runtime → heuristic (so partial outages are visible).

---

## Sampling

**Why separate rules for “no keys” vs “some keys”:**

| Client sends | In-process / wheel | Subprocess |
|--------------|-------------------|------------|
| No sampling keys | **Greedy** (deterministic smokes) | llama-server defaults |
| Any sampling key | Ollama defaults for omitted fields | Merged into `/completion` JSON |
| `temperature: 0` | Greedy chain | `temperature: 0` in JSON |

Implementation: `runtime/runtime/worker/sampler_options.py` — single mapping for HTTP JSON, ctypes sampler chain, and llama-cpp-python kwargs.

---

## Smoke scripts (why each exists)

| Script | Why |
|--------|-----|
| `scripts/phase14_serve_env.sh` | Unsets `ZEROLLAMA_RUNTIME_URL` so Go **embeds** Python; sets `ZEROLLAMA_RUNTIME_EMBED=on`. **Why:** exporting URL in the shell is the #1 reason `:8081` never listens during smokes. |
| `scripts/phase14_backend_smoke.sh` | One backend against **already running** serve; `RUN_E2E_PHASE14=1`; strict `/health` + `/internal/tokenize` preflight. **Why:** catch stale binaries before a 20‑minute GPU run. |
| `scripts/phase14_both_backends.sh` | Restarts serve per backend; embed-safe; clears stale `RUN_E2E_*` env. **Why:** backend is fixed at process start — cannot flip in-process → wheel without restart. |
| `RUN_E2E_PHASE14=1` in `e2e_runtime_smoke.sh` | Sends `X-Zerollama-Runtime: 1` on Go proxy steps. **Why:** sign-off must hit runtime + `truncate_mode=tokenize`, not accidental ggml for pulled tags. **Smoke-only** — not production default-on. |

**5080 checklist:** [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md#phase-14-sign-off-in-process-llama).

---

## Code map

| Path | Role |
|------|------|
| `runtime/runtime/worker/libllama_ctypes.py` | ctypes bind to pinned `libllama.so`; stream `finally` frees ctx after chunks consumed |
| `runtime/runtime/worker/llama_inprocess.py` | `LlamaInprocessWorker` (ctypes) |
| `runtime/runtime/worker/llama_cpp_python.py` | `LlamaCppPythonWorker` (pip wheel); CPU-default GPU layers |
| `runtime/runtime/worker/sampler_options.py` | Ollama options → sampler chain / HTTP / wheel kwargs |
| `runtime/runtime/worker/factory.py` | `create_llama_worker()` |
| `runtime/runtime/worker/llama_server.py` | Subprocess backend (default) |
| `runtime/runtime/engine.py` | `LlamaForwardWorker` protocol; vocab cache; `/health` `llama_backend` |
| `POST /internal/tokenize` | Vocab-only tokenize for Go render |
| `internal/runtimeclient/tokenize.go` | Go HTTP client |
| `server/runtime_tokenize.go` | `tokenizeForRuntimeModel`; `memoizeTokenize` |
| `server/runtime_chat_prompt.go` | Truncation order: ggml → runtime → heuristic |
| `server/runtime_url.go` | `effectiveRuntimeURL` / embed vs sidecar |
| `x/runtimeworker/client.go` | Embedded runtime base URL |

---

## llama-cpp-python backend

For hosts without a pinned `libllama.so` build:

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
export ZEROLLAMA_RUNTIME_LLAMA_BACKEND=llama-cpp-python
export LLAMA_MODEL=/path/to/model.gguf
```

Aliases: `pypi`, `wheel`, `cpp-python`.

**Why CPU default:** On some hosts (including 5080 cu124 wheels), `create_completion` with `n_gpu_layers >= 1` aborts (`free(): invalid pointer`) while ctypes `inprocess` GPU works. Default `n_gpu_layers=0`; opt in with `-ngl` on load args or `ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS=<positive>`.

**Why not fix the wheel here:** Version skew between pip wheel and pinned `libllama.so` is a separate supply-chain problem; 5080 production GPU should use **inprocess**.

---

## Limits (v1)

| Limit | Why deferred |
|-------|----------------|
| Grammar / DRY / XTC / mirostat in-process | Sampler chain parity first; advanced samplers need more ctypes bindings. |
| Multi-slot / parallel in-process | Serialized under `RLock` — matches single-GPU product default; Phase 15 may revisit. |
| Speculative / draft on in-process | Different load path; subprocess keeps full llama-server support. |
| Per-request `llama_context` | Model stays loaded; ctx per completion — simpler than KV broker (Phase 15). |
| ctypes struct sizes (72/144 bytes) | Pinned llama.cpp; bump requires binding refresh + CI. |

---

## Next

- **Phase 15** — native KV block pool + decode loop in C/Rust; Python becomes config/admission layer.
- **Wheel GPU** — revisit when pip wheel matches host CUDA reliably; until then use inprocess for GPU.
