# Roadmap

This file tracks **directional** plans. It is not a commitment schedule.

**Why this file exists:** Large features (video, remote cloud, GPU training) touch **API compatibility**, **security**, and **optional subprocesses / upstreams**. A short roadmap keeps **intent** and **non-goals** visible so contributors do not assume every deployment wants the same tradeoffs.

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

## GPU training (fine-tuning)

**Shipped direction:** Go spawns a single Python **`trainingdaemon`** (gRPC on UDS), serves **`/api/train/*`** and legacy **TCP `:9500`** (newline JSON), and coordinates VRAM with the scheduler on CUDA OOM (pause new inference loads → unload runners → ack Python). Training logic stays in repo-root **`training.py`**, imported by the daemon—not a second public listener on 9500.

**Why this track exists:** Fine-tuning stacks (Transformers, PEFT, bitsandbytes) are Python-native; Ollama’s control plane is Go. Splitting **public wire** (Go) from **GPU work** (Python) keeps one upgrade path for clients while avoiding a rewrite of every training primitive in another language.

### Possible follow-ups

- **Proactive VRAM:** Optional “training reservation” or explicit handoff before large loads (today: reactive OOM + `load_model` retry).
- **Auth on `:9500` and `/api/train`:** Same reasons as main HTTP API—training can exfiltrate paths and burn GPU time.
- **Structured progress over HTTP:** SSE or WebSocket for job progress without polling (TCP stream already maps to internal events).
- **Rust or libtorch path:** Only if we need fewer Python processes or stricter ABI isolation—large effort; Python remains the pragmatic default.
- **CI:** Optional pipeline with CPU-only smoke (imports, gRPC round-trip) vs full CUDA jobs behind labels.

### Non-goals (for this track)

- **Guaranteeing** mid-training automatic resume after OOM without checkpoints—unsafe for arbitrary `Trainer` loops; v1 notifies Go and may retry **model load** only.
- **Replacing** `training.py` with an empty stub—the library is the reference implementation until a deliberate migration plan exists.

**Why a separate subsection from video / Eliza:** Training touches **subprocess lifecycle**, **GPU memory shared with llama runners**, and **optional TCP**—different failure modes and operators than multimodal decode or remote HTTP.

---

## Video understanding (VLM) — shipped direction

- **Native path:** `video_url` / `videos` → ffmpeg frame sampling → same vision pipeline as images (**frame-list semantics**).
- **Optional SGLang:** Full-body proxy of `POST /v1/chat/completions` when `video_understanding=sglang` and `OLLAMA_SGLANG_URL` is set.

**Why** this split: local users should not need another server; advanced users can delegate decoding and model-specific video handling to SGLang when they already run it.

## Video generation — not in scope yet

**Why** it is separate: **understanding** (encode video → tokens/frames → text) and **generation** (text/image → video) use different stacks, APIs (`/v1/chat/completions` vs `/v1/videos`), and operational concerns (async jobs, GPUs, safety).

**Possible future tracks:**

1. **Proxy** `POST /v1/videos` to an upstream that implements OpenAI-style video generation (e.g. SGLang multimodal generation or another service).
2. **Native** diffusion / video models inside Ollama — larger milestone (runtime, memory, UX).

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

## Hardening candidates

- **SSRF:** Dial-time pinning to pre-resolved IPs or stricter URL policies for high-security environments.
- **Templates:** Distinct placeholders for “N frames from one video” vs N unrelated images where models support it.
- **E2E tests:** Optional CI with ffmpeg present; optional `E2E_SGLANG_URL` for proxy smoke tests.

## How to contribute

Open an issue or PR with a concrete use case (API shape, model family, deployment constraints). **Why** matters as much as **what** for multimodal features—resource limits and API compatibility affect everyone.
