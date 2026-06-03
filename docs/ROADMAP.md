# Roadmap

This file tracks **directional** plans. It is not a commitment schedule.

**Why this file exists:** Large features (video, remote cloud, GPU training) touch **API compatibility**, **security**, and **optional subprocesses / upstreams**. A short roadmap keeps **intent** and **non-goals** visible so contributors do not assume every deployment wants the same tradeoffs.

---

## Product model: queues, stakeholders, and GPU time

Directional (not a shipped scheduler contract): **GPU-backed batching system** fed by many **stakeholders** at once: local CLI and library pulls, OpenAI-compatible HTTP, agents and integrations, optional **Eliza cloud** merge, and (when enabled) the embedded **Python runtime** path. Those surfaces funnel into **inference work** that must be **admitted, queued, and executed** as efficiently as the hardware allows—today split between the **Go scheduler** (ggml runners, eviction, public routing) and the **Python runtime scheduler** (PagedAttention bookkeeping, `llama-server` orchestration).

**Training** is intentionally a **separate job queue** (`/api/train/*`, embedded `training.py`, optional TCP `:9500`): jobs are submitted, listed, and cancelled independently of a single chat FIFO. **VRAM** is shared: Go scaffolding (Phase 8) and Python coordination ensure inference and training do not corrupt each other’s memory; **Phase 11** moves more of that **policy** into Python.

**What is not automatic yet:** a single product-level **orchestrator** that says “maximize inference throughput until the backlog is idle, then drain training jobs” or “night window only.” That is **scheduling policy on top** of the existing queues—priority classes, SLOs, and optional idle-time training—aligned with the training track below and **Phase 11+**. Documenting the target shape here avoids mistaking today’s **two schedulers + one VRAM broker** for a finished global optimizer.

---

## Technology ladder (north star)

Directional stack—not a promise to delete Go or Python on a fixed date.

```text
Layer 0 (today)     Go daemon — registry, HTTP, Eliza proxy, embed lifecycle, ggml scheduler
        ↓ shrink    Why: shipped fast; still owns processes Python cannot see (runners)
Layer 1 (now→)      Python runtime — PA bookkeeping, batching policy, llama-server orchestration, training
        ↓ shrink    Why: best place for scheduler experiments before freezing native APIs
Layer 2 (later)     C / Rust hot path — GGUF decode, KV blocks, in-process llama.cpp (or forked kernels)
        ↓           Why: throughput and memory layout; no GIL/subprocess HTTP on the critical path
Edge (long-lived)   Thin API + pull + cloud — may stay Go *or* move to Rust; “Go gone” means inference
                    control plane gone from Go, not necessarily zero Go in the repo
```

**VRAM policy** follows the same ladder: **Go broker (scaffolding)** → **Python `InferenceGpuCoordinator`** → **native allocator** when Python shrinks.

**Detailed migration log (Phases 0–7, done):** [python-migration.md](./python-migration.md). **Operator smoke:** [testing-smoke.md](./testing-smoke.md).

**Operator guide (WHY-oriented, phases 8–13 + T6):** [scheduling-vram-policy.md](./scheduling-vram-policy.md) — VRAM broker, training defer queue, runtime pre-checks, env tables, code map.

---

## Local inference — actionable phases

Phases **0–7** are **done** (sidecar, embed, Go proxy, spec decode plugins). Work below is **8+**.

| Phase | Goal | Owner | Exit criteria |
|-------|------|--------|----------------|
| **8** | **Automatic VRAM handoff (scaffolding)** | Go | **Done** — `server/vram`: legacy load → `training-handoff`; runtime proxy → `UnloadAllRunners` + `inference/resume`; training submit → both. No public unload API. |
| **9** | **Manifest → runtime** | Go + Python | **Done** — Go proxy sets `options.gguf` from manifest and forwards client `options` (e.g. `num_ctx`); runtime loads/swap per request. `LLAMA_MODEL` remains optional fallback for direct `:8081` / smoke. |
| **10** | **CI regression gate** | Repo | **Done** — `.github/workflows/zerollama-regression.yaml`: `go test` (incl. Golden) + runtime pytest (incl. tools meta) + `check_gpu_scripts.sh`. Local/GPU preflight: `./scripts/phase12_golden_ci.sh`. Optional: `zerollama-gpu-smoke.yaml` (`workflow_dispatch`, self-hosted). |
| **11** | **VRAM + admission policy in Python** | Python | **Partial** — inference-first + VRAM checks; **low** throttling; min-free + training reserve via env or `single_gpu.yaml` `vram:` block. Backpressure thresholds overridable (`ZEROLLAMA_RUNTIME_BACKLOG_BATCH_MIN`, …). **5080:** `gpu_5080_session.sh` PASS; defaults unchanged (gates active, admission fits). |
| **12** | **Runtime default for text local models** | Go + Python | **Done** (tools path) — default-on; streaming proxies; tools via Go render + stateful `parse-tool-output` sessions. Render ctx aligned with load via `resolve_num_ctx_for_request`. v1 proxy injects manifest `options.gguf`. CI goldens: `./scripts/phase12_golden_ci.sh`. **Harmony real-weight:** CI synthetic only; `gpt-oss:20b` needs ~40+ GiB host RAM on runtime path (not required on 5080). |
| **13** | **Single-GPU + host autoconfig** | Python | **Partial** — estimates, autotune, `suggested_max_num_ctx`, clamp default **off** in YAML; persisted autotune catalog on `/health`; `python -m runtime.gpu_snapshot` after session JSON; `vram:` defaults in `single_gpu.yaml`. **5080 gate:** [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md). Doc: [phase13-runtime-vram.md](./phase13-runtime-vram.md). |
| **14** | **In-process llama forward** | Python → C/Rust | **Partial** — ctypes `inprocess` (GPU) + `llama-cpp-python` wheel (CPU default; GPU via `ZEROLLAMA_LLAMA_CPP_N_GPU_LAYERS`), render tokenize, sampling; subprocess default. **5080:** `phase14_backend_smoke.sh` PASS (`inprocess`); `phase14_both_backends.sh` PASS (wheel CPU, ~10 min). Doc: [phase14-inprocess-llama.md](./phase14-inprocess-llama.md), handoff: [handoff-phase14-inprocess-llama.md](./handoff-phase14-inprocess-llama.md). **Next:** Phase 15 native KV; wheel GPU stability when pip wheel allows. |
| **15** | **Native scheduler + KV** | C/Rust | **Partial (v0–v8 ops)** — C pool + tick/decode hooks; logical bind + forward plans; Go loopback KV snapshot; scheduler/proxy fixes; `kv_page_bind` readiness + opt-in live physical health. **Blocked:** tensor page bind + batched decode in C (llama.cpp API). Docs: [phase15-native-kv.md](./phase15-native-kv.md), [handoff-phase15-native-kv.md](./handoff-phase15-native-kv.md). |
| **16** | **Thin edge daemon** | Rust or minimal Go | Pull/registry/cloud only; all local generate/chat through native runtime. **Why:** complete “Go gone” for inference control plane. |

**Deprioritized:** public `POST /api/runtime/unload` or `/resume` — automatic eviction only ([Phase 8](#local-inference--actionable-phases) → [Phase 11](#local-inference--actionable-phases)).

**Non-goals (this ladder):** RadixAttention v1; required vLLM/SGLang servers; rewriting training in C++ (`llama-finetune` WIP); bit-for-bit SGLang parity.

### Phase 8 — shipped

See `server/vram/broker.go` and `server/runtime_manifest.go`. Next: **Phase 11** (admission hardening) and **Phase 14** (in-process llama forward).

---

## Zerollama remote cloud (Eliza)

**Shipped direction:** Default upstream [Eliza Cloud](https://www.elizacloud.ai); API key via `ELIZACLOUD_API_KEY`; path rewrite to Eliza `/api/v1/...`; catalog merge with singleflight + cache TTL; `:cloud` model suffix for routing. See [eliza-cloud.md](./eliza-cloud.md) for full rationale.

### Possible follow-ups

- **Stricter response mapping:** Optional adapters from raw Eliza JSON to OpenAI/Ollama-shaped responses where schemas stabilize, without losing fields today’s clients rely on.
- **Catalog hardening:** Explicit policy when local and remote names collide beyond simple dedupe (e.g. precedence rules in UI).
- **Multipart / image routes:** Clearer behavior when `model` is not in JSON (document or extend passthrough detection).

### Non-goals (for this track)

- **Replacing** Eliza with another host transparently without configuration — different providers have different auth and routes; `OLLAMA_CLOUD_BASE_URL` exists precisely to make that an explicit operator choice.
- **Guaranteeing** bit-for-bit parity with every Eliza API revision — we proxy and merge where Zerollama needs; upstream drift is handled case by case.

**Why a separate subsection from video:** Remote HTTP inference and local ffmpeg/video pipelines share almost no code paths; mixing them in one bullet list would blur ownership and testing strategy.

---

## Python inference runtime (GGUF + PagedAttention)

**Status:** Phases **0–7** complete — see [Local inference — actionable phases](#local-inference--actionable-phases) for **8+**.

**Direction:** GGUF-first **`runtime/`** with PagedAttention block pools; not HTTP to vLLM/SGLang. **Embed:** [runtime-embed.md](./runtime-embed.md). **Spec decode:** [runtime/docs/SPECULATIVE.md](../runtime/docs/SPECULATIVE.md). **Smoke validated:** [testing-smoke.md](./testing-smoke.md) on RTX 5080-class hosts. **Operator ladder (WHY):** [gpu-5080-operator-guide.md](./gpu-5080-operator-guide.md) — one session script proves Phase 10–13 on 16 GB; harmony real-weight is **not** required (CI Go golden + host-RAM limits for `gpt-oss:20b`).

**Still on ggml runner:** vision, logprobs, think, MLX, and models without `zerollama-runtime` backend. **Tools** on runtime-routed text models use Go render/parse (Phase 12 — done). Handoff: [handoff-phase12-runtime-tools.md](./handoff-phase12-runtime-tools.md).

---

## GPU training (fine-tuning)

**Shipped direction:** Go embeds **CPython** (`x/trainingworker/pyembed`), serves **`/api/train/*`** and legacy **TCP `:9500`** (newline JSON), and coordinates VRAM with the scheduler on CUDA OOM (pause new inference loads → unload runners → ack Python). Training logic stays in repo-root **`training.py`**, loaded in-process—not a second public listener on 9500.

**Why this track exists:** Fine-tuning stacks (Transformers, PEFT, bitsandbytes) are Python-native; Ollama’s control plane is Go. Embedding Python keeps **public wire** in Go while avoiding a second process and gRPC for every deployment.

### Phases (training track)

| Phase | Goal | Exit criteria |
|-------|------|----------------|
| **T1** | **Proactive VRAM (with inference Phase 8)** | **Done** — broker before `load_model` and on inference/runtime paths; OOM path remains safety net. |
| **T2** | **Auth on `/api/train` and `:9500`** | Same threat model as main HTTP API. |
| **T3** | **Progress over HTTP** | SSE or WebSocket; reduce poll-only UX. |
| **T4** | **CPU CI smoke** | **Partial** — Go tests register `/api/train/*` and exercise `GET /api/train/status` (502 without embed is OK in CI). Full embedded-python health still needs integration. |
| **T5** | **Native training (optional)** | Rust/libtorch only if Python embed becomes a bottleneck—default stay on PyTorch. |
| **T6** | **Unified queue policy (directional)** | **Partial** — idle-wait; `priority`; defer queue; **allowed window**; **cross-queue FIFO** (global tickets, Go↔Python mirror). **Next:** richer SLO classes, stricter training/inference class separation. |

### Non-goals (for this track)

- **Guaranteeing** mid-training automatic resume after OOM without checkpoints—unsafe for arbitrary `Trainer` loops; v1 notifies Go and may retry **model load** only.
- **Replacing** `training.py` with an empty stub—the library is the reference implementation until a deliberate migration plan exists.

**Why a separate subsection from video / Eliza:** Training touches **subprocess lifecycle**, **GPU memory shared with llama runners**, and **optional TCP**—different failure modes and operators than multimodal decode or remote HTTP.

---

## Video understanding (VLM) — shipped direction

- **Native path:** `video_url` / `videos` → ffmpeg frame sampling → same vision pipeline as images (**frame-list semantics**).
- **Optional SGLang:** Full-body proxy of `POST /v1/chat/completions` when `video_understanding=sglang` and `OLLAMA_SGLANG_URL` is set.

**Why** this split: local users should not need another server; advanced users can delegate decoding and model-specific video handling to SGLang when they already run it.

## Video generation — Wan T2V v1 (shipped)

**Why a separate track from VLM:** Video **understanding** (ffmpeg → vision encoder) and video **generation** (diffusion on GPU for minutes) share almost no code. Mixing them in one roadmap bullet hid ownership, testing, and API limits.

**Shipped (v1):**

- OpenAI async Videos API on Go `:8080` — `POST /v1/videos`, `GET /v1/videos/:id`, `GET /v1/videos/:id/content`.
- Local **Wan** only (`wan2.1-t2v:1.3b`, `wan2.2-ti2v-5b`, 16g manifests); weights via `install_wan_video.sh`, not Ollama blobs.
- Jobs run as training **`run_script`** (wrapper `scripts/wan_video_generate.py` → upstream `generate.py`).
- **VRAM / queue:** training handoff, optional `defer-*` when inference busy; artifacts under `$OLLAMA_MODELS/generated/<job_id>.mp4`.

**Operator guide:** [wan-t2v.md](./wan-t2v.md) (architecture, status mapping, env, troubleshooting).

| Milestone | Goal | Why |
|-----------|------|-----|
| **v1** | Wan T2V via training queue | **Done** — reuse embed + broker instead of new subprocess daemon. |
| **v1.1** | Cancel + TTL + kill running Wan | Operators need to abort long jobs and reclaim disk. |
| **v1.2** | TI2V `input_reference`, 24g/32g tiers | Product parity with Wan2.2 TI2V; headroom for longer clips. |
| **v1.3** | Optional **Wan CPU worker** sidecar | Remote T5 encode / VAE on high-RAM host; 16 GB GPU CT. Plan: [wan-cpu-worker.md](./wan-cpu-worker.md). |
| **Later** | Diffusers / other `runner` values | Other stacks without forking Wan wrapper for each upstream. |

**Not in v1:** list videos, `DELETE /v1/videos/:id`, Eliza `:cloud` passthrough, in-process Diffusers runner.

**Future tracks:** GGUF Wan2.2 if 16 GB OOMs; upstream proxy for non-Wan stacks; CogVideoX / LTX via `runner: diffusers`.

## Option 2 — Narrow the gap without SGLang (in-tree over time)

**Execution checklist:** [video-parity.md](./video-parity.md) (reference workloads + parity matrix).

**Goal:** Keep inference **inside** Ollama/zerollama (ffmpeg + vision runner) and **deliberately port or reimplement** the behaviors that matter, instead of depending on an HTTP proxy to SGLang.

**Why a separate “Option 2” track:** Many users want **native** inference only (no second server). Option 2 spells out **policy** (how frames are chosen), **representation** (how templates see images vs video), and **limits** (context, mllama) as separate layers—so we do not conflate “better ffmpeg” with “SGLang-style scheduling.”

**Reality check:** SGLang is a large Python serving stack; **100% behavioral parity on every model** is not a single milestone. This roadmap is about **closing the gap** where it matters for **your** models and workloads.

### Phase A — Decode and sampling policy

- Align **fps / stride / max frames** with reference behavior for target models (env + per-model options where needed).
- Expand **container support** (what ffmpeg accepts) and document **failure modes** (corrupt input, no keyframes).
- Optional: **deterministic** sampling (fixed seeds / fixed frame indices) for reproducible evals.

**Why first:** Native path quality is dominated by **how** you turn video into frames, not the HTTP boundary.

### Phase B — Renderer and template semantics

- Where a model distinguishes **video spans** vs **unrelated images**, extend templates/renderers with **explicit placeholders** (not only a flat `[img-*]` list).
- Per **model family** (e.g. Qwen3-VL, others): document **expected** ordering and token layout.

**Why:** Frame-list semantics are a **ceiling** until templates express “N frames from one clip.”

### Phase C — Context and limits

- **Token-aware** budgeting: relate frame count to **effective vision tokens** and `num_ctx` so users get **actionable errors** before runtime blowups.
- Tune **mllama** / single-image constraints vs multi-frame video (clear errors or automatic downsample).

**Why:** SGLang’s stack does scheduling/budgeting; native path must encode **policy** explicitly.

### Phase D — Validation and regression

- **Golden tests:** small fixtures (short MP4/WebM) with **expected frame counts** / hashes after sampling.
- **Per-model smoke:** optional CI jobs when ffmpeg + GPU are available.

**Why:** Parity is proven by **tests**, not by matching another repo’s README.

### Phase E — Optional subprocess bridge (still no SGLang HTTP proxy)

- If a **specific** decode or preprocessing step must stay in Python, a **narrow subprocess** contract (stdin/stdout or temp files) can wrap **only that step**, with Ollama owning scheduling and limits—different from proxying full chat.

**Why:** Sometimes parity needs **one** binary without adopting a second full server.

### What this does *not* promise

- Automatic parity with **every** SGLang model and feature.
- Replacing **SGLang’s** distributed scheduler or custom kernels without equivalent work in Ollama.

## Cross-cutting / hardening

| Item | Phase hint | Why |
|------|------------|-----|
| Inference vs training **priority / idle policy** | Training **T6**, inference **Phase 11** | One GPU, many clients—documented target is queued work + policy, not “implicitly fair” |
| Eliza catalog / response mapping | Eliza follow-ups | Operator UX when local + cloud lists collide |
| Video Option 2 A–D | Video track | Native VLM quality without SGLang dependency |
| SSRF hardening | Security | High-assurance deployments |
| ffmpeg / SGLang E2E | Video + hardening | Regression beyond unit tests |

## How to contribute

Open an issue or PR with a concrete use case (API shape, model family, deployment constraints). **Why** matters as much as **what** for multimodal features—resource limits and API compatibility affect everyone.
