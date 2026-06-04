# Iterative path: moving inference toward Python

Directional plan for shifting **local GPU inference** from the Go + ggml `runner` stack to a **GGUF-first Python runtime** with **PagedAttention (PA)** KV management. Training stays in **`training.py`** (embedded today via `x/trainingworker/pyembed`).

**Targets:** dual **RTX 4090** (2×24 GB), continuous batching, no dependency on running upstream **vLLM** or **SGLang** servers (ideas and kernels may be ported/forked, not HTTP sidecars).

**Non-goals (this track):** RadixAttention in v1; replacing Go entirely in phase 1; reimplementing PyTorch training in C++.

---

## End state (north star)

**Mental model:** many **clients and routes** feed **inference** as queued, batchable GPU work; **training** is a **separate job queue** with its own lifecycle. A future **policy layer** (idle windows, priorities, SLOs) sits above both so one GPU is time-shared deliberately—not only by VRAM accidents.

```text
zerollama (Go) — pull, registry, Eliza cloud, CLI, HTTP; admits & routes work to schedulers
       │
       ├── ggml scheduler (queue, eviction) — legacy / vision / tools path
       │
       ▼
zerollama_runtime (Python) — PA scheduler, continuous batching, llama-server workers
       │
       └── training.py — job queue (/api/train); shares GPU under explicit VRAM policy
```

Go remains valuable for **distribution and API compatibility** until the Python runtime is stable enough to own more of local `/api/generate` and `/api/chat`.

---

## Phase 0 — Foundations (no user-visible change)

**Goal:** Repo layout, pins, and contracts so later phases do not thrash.

| Task | Notes |
|------|--------|
| Add `runtime/` Python package skeleton | `runtime/__init__.py`, `scheduler/`, `kv/`, `worker/`, `server/` stubs |
| Pin **llama.cpp** | Single submodule or path (e.g. `../llama.cpp`); document commit; align with Ollama v0.30 / `llama-server` direction when rebasing |
| Document env matrix | CUDA, Python 3.11+, `python3-dev` for existing embed; optional `uv`/`venv` for runtime |
| Define **GPU mutex** contract | Python API: `pause_inference()`, `unload_all()`, `resume_inference()` — mirror today’s `PauseNewLoads` / `UnloadAllRunners` / `ResumeLoads` |
| CI smoke | Import `runtime`; CPU-only tests for block allocator math (no GPU required) |

**Exit criteria:** `pytest runtime/tests/test_block_pool.py` (or similar) passes; llama.cpp revision recorded in doc.

**Toolchain on this host (CT 1564):** Go is at `/var/lib/vz/private/1564/usr/local/go/bin/go` (go1.24.x). Inside the CT shell, `.bashrc` uses `/usr/local/go/bin`. Workspace is `/var/lib/vz/private/1564/root/zerollama`. Do not look under `.../root/usr/local/go` on the host—that path is empty.

---

## Phase 1 — Sidecar runtime (parallel to production)

**Goal:** Run inference in Python **without** turning off the existing Go `runner`.

| Task | Notes |
|------|--------|
| `runtime serve --port 8081` | Minimal HTTP: `GET /health`, `POST /v1/completions` or Ollama-shaped stub |
| **PA block pool** (CPU logic first) | Block size, `num_blocks`, alloc/free, per-sequence block table |
| **GGUF load** | subprocess `llama-server` (or `llama-cli`) from pinned llama.cpp; localhost RPC |
| Env `ZEROLLAMA_RUNTIME_URL` | Go **opt-in** proxy for one model tag or `%experimental%` header |

**Exit criteria:** Same GGUF model answers a prompt via Go (legacy) and via runtime sidecar; latency logged for comparison on 2×4090.

---

## Phase 2 — Scheduler + single-GPU decode loop

**Goal:** Own **batching** in Python; llama.cpp does forward steps, Python owns **who** is in the batch.

| Task | Notes |
|------|--------|
| Waiting / running queues | Add request, prefill, decode token, finish |
| Map batch → llama-server calls | One decode per scheduling tick initially |
| KV **block accounting** | Reserve blocks from pool before prefill; free on done |
| Metrics | `tokens/s`, batch size, KV utilization |

**Exit criteria:** Continuous batching of N concurrent sequences (N≥4) on **one** 4090; no KV OOM at configured pool size.

---

## Phase 3 — Dual 4090 topology

**Goal:** Use **48 GB** intentionally.

| Task | Notes |
|------|--------|
| **Tensor parallel (TP=2)** | Large GGUF split across GPUs (llama.cpp TP flags / server args) |
| **Block pool per device** | Scheduler aware of `device_id` per block |
| **Optional DP=2** | Two replicas, two ports — later; not required for v1 |
| **Spec decode layout** | Document: target on GPU0, draft on GPU1 when two models loaded |

**Exit criteria:** 70B-class Q4 or dual 8B (target+draft) runs with documented `runtime/config/4090.yaml`.

---

## Phase 4 — Wire Go scheduler ↔ Python runtime

**Goal:** One operator process; training OOM bridge keeps working.

| Task | Notes |
|------|--------|
| Replace `llm.NewLlamaServer` path for tagged models | `sched` calls runtime client instead of `runner` subprocess |
| Implement **GPU mutex** in Python | On train start: drain batches, unload GGUF, signal Go |
| Port OOM hook | `training.py` → runtime unload (same as today → Go sched) |
| `OLLAMA_TRAINING` unchanged | Still embedded `training.py`; only inference moves |

**Exit criteria:** `zerollama serve` with `ZEROLLAMA_RUNTIME=1` serves chat without spawning `runner`; training + inference coexist with passing integration test.

---

## Phase 5 — Feature ports (spec + KV quant)

**Goal:** Bring “good stuff” into **your** tree, not vLLM/SGLang processes.

| Priority | Feature | Source of truth |
|----------|---------|-----------------|
| P0 | **n-gram / draft-simple** spec | llama.cpp `common/speculative` + your scheduler |
| P1 | **DFlash** | llama.cpp PR ecosystem / fork pin; plugin in `runtime/speculative/` |
| P1 | **EAGLE3** | Same |
| P2 | **MTP / NextN** | After llama.cpp MTP lands in pinned revision |
| P2 | **KV quant** | llama.cpp `-ctk`/`-ctv`; TurboQuant-like ops only if ported as kernels |

Each feature: **one active spec plugin per model**; integration test on 2×4090.

**Exit criteria:** Documented flags for at least one spec method + measurable speedup vs no-spec baseline.

---

## Phase 6 — Deprecate legacy inference paths

**Goal:** Single inference story.

| Task | Notes |
|------|--------|
| Default `ZEROLLAMA_RUNTIME=1` | Go runner off by default |
| Remove or gate **mlxrunner** | Mac MLX exception path only if still needed |
| Keep **SGLang proxy** optional | `modality_backends.video_understanding=sglang` until native VLM path is enough |
| Bump vendored `llama/llama.cpp` or drop in favor of external pin | Match Phase 0 pin |

**Exit criteria:** CI green without building old GGML Go backend for Linux CUDA; README points to Python runtime.

---

## Phase 7 — Optional consolidation

**Goal:** Reduce two-language ops burden if desired.

| Option | Tradeoff |
|--------|----------|
| **A. Go daemon + Python runtime** (recommended long-term) | Best compatibility with existing `zerollama` binary |
| **B. `python -m runtime.serve` only** | Go becomes optional CLI wrapper |
| **C. Merge training into runtime package** | Single venv; drop CGO embed for training |

Only choose B/C when Phase 4–6 are stable.

---

## What stays in Go (until proven otherwise)

- Model pull / manifest / `ollama create` GGUF pipeline
- Eliza cloud proxy and catalog merge
- Auth, logging, desktop app IPC (if applicable)
- Public API surface during transition

---

## What stays in Python

- **`training.py`** — LoRA/QLoRA, `Trainer`, job queue
- **Inference runtime** — PA scheduler, block pools, spec plugins
- Future: optional `run_script` training hooks in same package

---

## Risk register

| Risk | Mitigation |
|------|------------|
| llama.cpp API drift | Pin commit; submodule update checklist |
| PA + llama.cpp KV mismatch | Start with server-owned KV; PA tracks logical slots; refine |
| Dual-GPU bugs | TP tests in CI; one test per topology in `runtime/tests/gpu/` |
| VRAM train vs infer | Go broker + handoff; Python coordinator next; document “no silent parallel train + infer without policy” |
| Scope creep (Radix, full vLLM fork) | PA only v1; Radix only if prefix hit rate measured |

---

## Suggested order of work (checklist)

1. [x] Phase 0 — skeleton + llama.cpp pin + block pool unit tests (`runtime/`)  
2. [x] Phase 1 — sidecar HTTP + Go proxy (`ZEROLLAMA_RUNTIME_URL`, `runtime_generate_proxy.go`)
3. [x] Phase 1b — `llama-server` build script (`scripts/build_llama_server.sh`, sm_89); E2E needs `LLAMA_MODEL`  
4. [x] Phase 2 — `InferenceEngine` + scheduler admit + `/api/generate` on runtime; Go proxy uses `/api/generate`  
5. [x] Phase 3 — dual 4090 YAML (`configs/dual_4090.yaml`), multi-pool KV admit, TP llama-server flags  
6. [x] Phase 4 — Go proxy for `/api/generate` + `/api/chat` (`ZEROLLAMA_RUNTIME=1`, per-model `zerollama-runtime`); training handoff wired  
7. [x] Phase 5 — spec plugins (`runtime/speculative/`, ngram + draft flags, parallel slots in `generate_batch`)  
8. [x] Phase 6 — runtime on by default when `ZEROLLAMA_RUNTIME_URL` set; ggml runner skipped for `zerollama-runtime` models (`ZEROLLAMA_LEGACY_RUNNER=1` to override)  
9. [x] Phase 7 — `zerollama-runtime up` + embedded mode (`ZEROLLAMA_RUNTIME_EMBED`, `x/runtimeworker`)  

**Next (see [ROADMAP.md](./ROADMAP.md#local-inference--actionable-phases)):**

10. [x] Phase 8 — Go VRAM broker (automatic eviction; no public unload API)  
11. [x] Phase 9 — manifest → runtime model paths (`options.gguf` from Go proxy)  
12. [x] Phase 10 — CI (`zerollama-regression.yaml`: `go test` + runtime pytest + `check_gpu_scripts.sh`; optional `zerollama-gpu-smoke.yaml`)  
13. [ ] Phase 11+ — VRAM / admission policy in Python (partial); optional **idle-time training** policy (roadmap **T6**); Phase 14 in-process forward (**Done**); Phase 15 native KV (partial)

---

## Related docs

- [gpu-training.md](./gpu-training.md) — embedded training, OOM bridge  
- [handoff-gpu-training-integration.md](./handoff-gpu-training-integration.md)  
- [multimodal-backends.md](./multimodal-backends.md) — pattern for optional backends (cf. future video path)  
- [ROADMAP.md](./ROADMAP.md) — product-level tracks  
