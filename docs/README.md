# Documentation

### Getting Started
* [Quickstart](https://docs.ollama.com/quickstart)
* [Examples](./examples.md)
* [Importing models](https://docs.ollama.com/import)
* [MacOS Documentation](https://docs.ollama.com/macos)
* [Linux Documentation](https://docs.ollama.com/linux)
* [Windows Documentation](https://docs.ollama.com/windows)
* [Docker Documentation](https://docs.ollama.com/docker)

### Reference

* [API Reference](https://docs.ollama.com/api)
* [Modelfile Reference](https://docs.ollama.com/modelfile)
* [OpenAI Compatibility](https://docs.ollama.com/api/openai-compatibility)
* [Anthropic Compatibility](./api/anthropic-compatibility.mdx)

### Multimodal & video (repo)

These live in-repo (not only on docs.ollama.com) because they explain **design rationale**—API shape, limits, and optional backends:

* [Video understanding (VLM)](./video-understanding.md) — **why** `video_url` / `videos` → ffmpeg → vision pipeline; **why** preflight and `video_spans` exist.
* [SGLang multimodal borrowings](./sglang-multimodal-borrowings.md) — **why** native path adopted agent caches, padded inject, precomputed/processor ingest, usage breakdown, and audit fixes without requiring SGLang.
* [mtmd `grid_thw` handoff](./mtmd-grid-thw-handoff.md) — **why** client patch grids are hints-only until llama.cpp mtmd accepts them; Go seam + operator signals.
* [video-c (Pure-C Wan + H3 stub)](./video-c.md) — Darwin UMA + CUDA twin lab; client-optional runner; [dit-pager](./dit-pager.md); [cuda-uma-toolkit](./cuda-uma-toolkit.md). ([wan-c.md](./wan-c.md) redirect.)
* [music-c (MiniMax Music 3)](./music-c.md) — **why** mlx-audio first (no Comfy GPL runtime, no CUDA Omni to hear); C parked until a WAV; Omni rematch gold. [findings](./music-c-findings.md).
* [H3 MLX borrowings](./h3-mlx-borrowings.md) — [minimax-h3-mlx](https://github.com/mrbizarro/minimax-h3-mlx) as rematch oracle (AdaLN drop, TE truncation, packing); not the product runner.
* [H3 ClipProj](./h3-clipproj.md) — NicoLab28 small-Qwen3-VL → `[seq,5120]` TE map; video-c host load/apply.
* [Wan text-to-video (T2V)](./wan-t2v.md) — **why** `/v1/videos` is async, **why** training `run_script` + wrapper, VRAM/defer queue, artifacts; TI2V keyframes.
* [LTX text-to-video (v1.4)](./ltx-t2v.md) — **why** LTXV distilled+quanto first (not LTX-2/Gemma); Wan2GP runner behind same `/v1/videos`.
* DiT media toolkit (Wan / H3 / LTX, parallel) — Mac lab umbrella in bmtl `uma_toolkit/docs/WISHLIST_DIT_MEDIA.md`; product ROADMAP video section.
* [Media uploads (`/v1/media`)](./media-uploads.md) — **why** session/label PUT + CAS (no client digests, no refcounts); **why** not model `blobs/`; keyframe workflow + `media_missing` recovery.
* [wan-c vs Python MPS speed gap](./wan-c-speed-gap.md) — profile + Phase1 cuts + toolkit `DIT_BLOCK` / flash ATTN / feat_cache asks.
* [MLX image generation (Z-Image Turbo)](./imagegen-zimage-turbo.md) — **why** a fourth VRAM stack (MLX subprocess); staged load on 16 GB CUDA; CPU VAE handoff; scheduler/broker integration; build + troubleshoot.
* [ComfyUI image backend](./comfyui-image-backend.md) — **why** orchestrate Comfy for agent edit/ControlNet/LoRA instead of porting every HF DiT to MLX; bindings, discovery, VRAM handoff, example workflow calibration.
* [Optional multimodal backends](./multimodal-backends.md) — env + manifest; **why** both layers; image drivers (`mlx-imagegen`, `external-image`, `comfyui`).
* [Roadmap — local voice & llama borrowings (eliza-v3)](./ROADMAP.md#local-voice--llama-borrowings-eliza-v3) — **inference first:** GPU autotune profiles (**L1**), fork kernels (**L2**), KV prefix cache (**L3**); voice **L5+** later.
* [L1 GPU profiles (autotune)](./gpu-profiles-l1.md) — **why** batch/parallel/MTP tuning is separate from Phase 13 VRAM estimates; **`l1_cuda_full_gate.sh`**; NVIDIA + Apple tiers; operator env.
* [llama.cpp backend unification](./llama-cpp-unification.md) — **why** one elizaOS tree @ `LLAMA_CPP_COMMIT` replaces stock + eliza-llama siblings; discovery, doctor, vendor rebase plan.
* [L2 unified llama-server profiles](./gpu-profiles-l2.md) — L1 vs fork argv on one binary; **5080 Jun 2026:** L1 q8_0 wins @ 8k (fork profiles opt-in).
* [CUDA lanes (dual-4090 / 5080)](./cuda-lanes.md) — **why** shared CUDA playbook vs 5080-only; NVFP4/MXFP4/FP8 weight roadmap; probes + sign-off.
* [Native FP8 GGUF (E4M3/E5M2)](./native-fp8-gguf.md) — **why** block FP8 types + `--fp8-native` instead of full F16 dequant; patches 0073–0076; MMVQ/MMQ; probes.
* [Runtime env reference](./runtime-env.md) — **why** profiles/YAML/smart defaults beat dozens of `ZEROLLAMA_*` exports; L3, KV, VRAM; `./scripts/runtime/runtime_env_doctor.sh`.
* [L3 prompt cache → slot bridge](./gpu-profiles-l3.md) — **why** Phase 15 dynamic slots discard KV each turn; stable keys → pinned llama-server slots + disk TTL; cuts agent prefill latency (complements L1 tok/s, L2 VRAM). **Audit (Jun 2026):** canonical GGUF hashing, orphan hash-dir sweep, strict batch keys, native bind before slot release; SWA/draft-spec policy; decode graph epoch + CUDA invalidation (in-process + subprocess HTTP).
* [Cross-slot Radix prefix share](./radix-prefix-share.md) — **why** L3 one-slot-per-key leaves duplicate prefills for shared system prompts; donor KV seed + v2 milestones (warm catch-up, ref-count metadata, Redis LMCache, hybrid SWA gate); vendor `POST /kv/seq-copy`; live smoke; **[product gaps](./radix-prefix-share.md#product-gaps)** (v2 vs full RadixAttention).
* [Decode graph invalidation](./decode-graph-invalidation.md) — **why** L3 slot clears must break ggml CUDA graphs; epoch + native invalidate + `POST /cuda-graph/invalidate` for subprocess llama-server.
* [vLLM borrowings (L3)](./vllm-borrowings.md) — **why** slot-level prefix cache vs vLLM block pool; taken vs deferred; env + `cache_salt` / drop-last-block / SWA retention / subprocess graph clear.
* [Upstream sibling checkouts](./upstream-siblings.md) — **why** weekly pull map (`../vllm`, `../LocalAI`, …); agent entry [AGENTS.md](../AGENTS.md).
* [Video parity matrix](./video-parity.md) — **why** reference workloads for native vs SGLang.
* [H3 CUDA port research](./h3-cuda-port.md) — **why** antirez/h3.c Metal MiniMax-H3 → CUDA via `h3_gpu.h`; Wan2GP/SGLang vs native stub; CT 1564 RAM wall.
* [Roadmap](./ROADMAP.md) — **why** Option 2 is phased (policy, templates, context, optional subprocess).
* [Upstream Ollama comparison](./upstream-ollama-diff.md) — **why** vanilla Ollama dropped ggml for GGUF; pin gaps; cherry-pick map; Phase 17 alignment.
* [Phase 17 — Go → llama-server](./phase17-llama-server.md) — **why** upstream GGUF path is cherry-picked for mergeability; Mac keeps ggml default (M7 bench).
* [Flash-MoE (anemll)](./flash-moe.md) — **why** slot-bank + SSD sidecar for MoE models larger than unified RAM; Phase 17 llama-server passthrough (not ggml Metal default).
* [ANE probe (maderix)](./ane-probe.md) — **why** subprocess smoke for private ANE APIs before hybrid inference; not on hot path.
* [ANE dflash in-process (B1–B6)](./ane-draft-inprocess.md) — **why** same-PID IOSurface handoff on llama-server dflash draft decode; lab port 11435; draft tokens still Metal until B7.
* [ANE hybrid path (lab)](./ane-hybrid-path.md) — crossover sweeps, prefill proxy, operator tooling index.
* [ggml IOSurface hook](./ane-ggml-iosurface-hook.md) — Metal buffer API + speculative integration points.
* [Phase 16 — thin edge daemon](./phase16-thin-edge.md) — **why** `--edge` / `-tags edge` for upstream-shaped deploys (runtime chat off, llama-server only) without dropping training/Eliza/fleet.
* [Launch model inventory](./launch-model-inventory.md) — **why** `zerollama launch` loads `/api/tags` once and passes `LaunchModel` metadata to agent configs (no N× Show).
* [Model bench cache](./bench-cache.md) — **why** `zerollama bench` caches decode tok/s by digest and **`zerollama ls`** shows **TOK/S** without re-running inference.
* [ggml @ b9509 migration](./ggml-b9509-migration.md) — **why** vendored ggml/llama.cpp rebased to real upstream tags (**current: b9781 / v0.30.11**); 16 patches; sync workflow; **why `make sync` must not reset vendor**.
* [llama.cpp backend (experimental)](./llama-cpp-backend.md) — route text GGUF through Python runtime + sibling llama.cpp; benchmark vs ggml.

### Apple Silicon (repo)

* [Apple Silicon & Metal operator guide](./apple-silicon-metal.md) — onboarding tiers (M14); unified memory; L1 profiles; GPU bootstrap; sched_reserve; **`metal_signoff.sh` + qwen35 (`eliza-1-2b:latest`)**; manifest vs `/api/ps` context.
* [Qwen 3.5/3.6 on Mac](./qwen35-apple-silicon.md) — **why** compat + Metal embed; Go ollama-engine; **full `metal_signoff.sh` + qwen35** (qwen35 before Phase 15; canonical **`eliza-1-2b:latest`**); manifest `num_ctx` vs request options; thinking-model fields.
* [Mac dev setup](./mac-dev-setup.md) — **`dev_bootstrap.sh`** tier 0–3; **prereqs:** Go **1.24.1+**, full Xcode.app (or Homebrew Python), **cmake**, uv; script map after reorg; **why** `:11434` daily vs `:8080` CI.
* [MLX routing policy](./mlx-routing-policy.md) — ggml Metal vs runtime vs mlxrunner; LM Studio MLX disk summary.
* [UMA admission overview (Darwin)](./uma-admission.md) — M20–M23 surfaces, multi-unit HOLD, `mac_uma_signoff.sh` ladder, disable knobs.
* [MLX UMA broker admission (M20)](./mlx-uma-sched.md) — machine-wide `uma_daemon` gate around mlxrunner `Eval` (`BUILD_UMA=auto`, default `ZEROLLAMA_UMA_SCHED=auto`).
* [GGUF ggml UMA admission (M21)](./ggml-uma-sched.md) — same broker for ollamarunner / llamarunner Metal.
* [llama-server UMA admission (M22)](./llama-server-uma-sched.md) — vendor `graph_compute` HOLD + sync (`BUILD_UMA` → `libuma_llama.a`).
* [MLX agent prompts](./mlx-agent-prompts.md) — **why** context cap, tail truncate, `PromptTokens`, tokenize cache, keep-alive floor, SSE keepalive, **M15a live-session + rotating-KV restore** (`fast_path`, `messages_dropped`), and operator logs for agent megaprompts on safetensors models.
* [Megaprompt tokenize benches (README evidence)](./readme-marketing-benches.md) — **why** cite ~3–7× / hundreds-of-ms legacy; Jul 2026 medians; reproduce command.
* [Faster BPE tokenize](./faster-bpe-tokenize.md) — **why** megaprompt `llama_tokenize` hurt agents; patches `0106–0126`; `mega_1mib_ascii` / `_chat` benches; identity gates.
* [Faster BPE findings](./faster-bpe-tokenize-findings.md) — **why** not Rust gigatoken; measurement traps (bogus 6–22×); three-tree wiring; Qwen vs Gemma speedup shape.
* [Agent QoS and project tracking](./agent-qos-and-project-tracking.md) — **why** session gate TOCTOU fix, multiplex key hot-map / `wait_parent`, session→cache great loop (`cache_reset` / `cache_level`), `project_id` / `zerollama ps`, inference-path branching, and progressive client ladder; keeps Tier 2 options off vanilla Ollama and unkeyed CUDA traffic.
* [Hermes ↔ zerollama gap analysis](./hermes-zerollama-gap.md) — **why** many “blocked/missing” wishlist items are already shipped under different field names or native `/api/*`; real gaps closed in M15e; **§8** documents `POST /v1/chat/completions/batch` wire (`object=chat.completion.batch`, ordered `completions`, client group-by-model).
* [Hermes gap closure findings (M15e)](./hermes-gap-closure-findings.md) — **why** bind≠allowlist, 504≠499, wait-abort≠hard preempt, topology≠TP planner, cache-pin≠model-pin, thin batch proxy + **document-before-extend** for batch schema; OpenAPI map.
* [LM Studio cache import](./lmstudio-import.md) — **why** pull-from-cache, MLX copy vs GGUF symlink, disk policy, env vars, troubleshooting.

### GPU training & scheduling (repo)

* [Scheduling, VRAM, and queue policy](./scheduling-vram-policy.md) — **why** inference and training are separate queues; Phase 8 broker; T6 idle-wait + `defer-*` queue; Phase 11–13 runtime heuristics; **ggml unload / manifest `num_ctx` at load**; **M12 ggml suggest/clamp**; **prompt truncation / context-overflow API fields** (`prompt_truncated`, runtime detect).
* [Inference wishlist — host capacity (Phase A/B)](./inference-wishlist-host.md) — **why** Orient/Decide need capacity APIs; pin/propose with honest single-resident runtime; broker must respect pins; B0 requires ggml-empty; 503 before resume on pin conflicts; `stable_multi_model_swap` still false.
* [T6 unified queue policy (operator guide)](./t6-unified-queue.md) — idle-wait, defer queue, allowed window, cross-queue FIFO, env table, `/api/status` queue_policy, smoke script.
* [Open-source shoutouts](./open-source-shoutouts.md) — Gigatoken, vLLM, SGLang, LocalAI, minefield, Hermes, Ollama, llama.cpp — what we borrowed and why.
* [LocalAI control-plane borrowings](./localai-borrowings.md) — **why** LA1–LA10 (metadata, watchdog, fleet score, repair, HF pull, `/api/score`, bench cache); **upstream watch** for LA11+ candidates; env reference.
* [Fleet scheduling (multi-node)](./fleet-scheduling.md) — **why** a management node above per-node schedulers; warm-model routing; filter-then-score (F7); anti-patterns (scatter-gather, long quotes).
* [Fleet management operator guide](./fleet-management.md) — **why** F3 is thin (poll + assign, no remote load); `zerollama fleet serve`; API, env, agent pattern.
* [Remote model storage](./remote-model-storage.md) — **why** central content-addressed blobs + HMAC LAN auth + fetch-on-miss; RDMA-prefer/TCP fallback; pin/refcount LRU; ephemeral cleanup; tensor catalog language for later streaming (spec-only in v1).
* [Phase 11 runtime admission](./phase11-runtime-admission.md) — **why** opinionated VRAM + inference-first policy; priority classes; enqueue/dequeue flow; `/health` gates; `VRAM_MIN_FREE` / `TRAINING_VRAM_RESERVE`.
* [Phase 13 runtime VRAM estimates](./phase13-runtime-vram.md) — **why** GGUF VRAM heuristics, `suggested_max_num_ctx`, opt-in clamp, autotune, autoconfig, operator CLI. Complements **L1** throughput profiles: [gpu-profiles-l1.md](./gpu-profiles-l1.md).
* [Phase 14 in-process llama](./phase14-inprocess-llama.md) — **why** subprocess HTTP was replaced for forward; three backends; render tokenize; sampling parity; 5080 sign-off scripts.
* [Phase 14 handoff](./handoff-phase14-inprocess-llama.md) — architecture, code map, smoke footguns, bugs fixed during bring-up.
* [Phase 15 native KV](./phase15-native-kv.md) — **why** PA/C block pool + scheduler bind precede tensor KV; v8–v20 page bind + tensor probe; **v26–v30** continuous batch decode; `/health.kv_decode_loop`, `kv_continuous_batch`; loopback `POST /internal/generate-batch`; GPU sign-off `phase15_metal_signoff.sh` (batch step 3/5).
* [Phase 15 llama-kv-ext upstream tracking](./phase15-llama-kv-ext-upstream.md) — **why** patch 0014 + pin check; hybrid/iSWA memory resolve; upstreaming checklist; writable bind still blocked.
* [Phase 15 handoff](./handoff-phase15-native-kv.md) — code map, `/health` fields, gaps; **v0–v31 shipped** (C decode loop, engine resume, L3 gate, batch decode + Metal sign-off PASS Jun 2026).
* [GPU training integration](./gpu-training.md) — **why** Go fronts HTTP + TCP `:9500` while Python holds PyTorch; embedded CPython; inference-first VRAM policy; OOM ordering; env vars and troubleshooting.
* [GPU training handoff (internal)](./handoff-gpu-training-integration.md) — embedded training + Phase 11 VRAM interaction (not a substitute for `gpu-training.md`).
* [Roadmap — training track T7–T11](./ROADMAP.md#gpu-training-fine-tuning) — Unsloth borrowings: train→GGUF (**T7 Done**), efficient SFT (**T8 Done**), stock Trainer polish (**T9**, no second backend), GRPO, lite recipes (not Studio).
* [Phase 12 tools + Phase 11 admission handoff](./handoff-phase12-runtime-tools.md) — runtime tools (Go render/parse), opinionated admission, smokes, code maps.
* [Inference smoke testing](./testing-smoke.md) — **why** runtime (`:8081`) and legacy ggml (`:8080`) share one GPU.
* [Model serving minefield](./model-serving-minefield.md) — trap registry mapped onto `zerollama doctor` (config + live serving checks) and known gaps.
* [Doctor model repair](./doctor-model-repair.md) — **why** Modelfile overlays for empty-`response` / slash-collapse / ChatML stop hygiene (not `doctor --fix`); `--repair-models` / `--apply` / `--all-local`; Qwen3 gate for invasive TEMPLATE rewrites.
* [5080 runbook — start here](./5080-runbook.md) — **`source scripts/gpu/5080_env.sh`** + **`./scripts/gpu/5080_resignoff.sh`**; ordered tiers; CT 1564 status; Radix vendor build.
* [GPU 5080 operator guide](./gpu-5080-operator-guide.md) — extended reference (VRAM, MLX, production serve) — use runbook for daily gates.
* [Embedded Python runtime](./runtime-embed.md) — **why** embed vs sidecar; **remote clients use Go `:8080` only**; port conflicts; log redirect pattern.

### Upstream Ollama (compare, don't merge)

* [Upstream Ollama comparison](./upstream-ollama-diff.md) — architecture deltas vs `../ollama-upstream`; Phase 17; benchmark workflow.
* [llama.cpp backend (experimental)](./llama-cpp-backend.md) — `--llama-cpp-backend` test harness toward upstream GGUF path.

### Remote inference — Eliza Cloud (Zerollama)

* [Eliza Cloud](./eliza-cloud.md) — **why** default upstream is Eliza (not legacy ollama.com), **why** path rewrites and `X-API-Key`, **why** catalog merge + cache, **why** raw JSON on some routes, **why** account stubs off ollama.com.

### Resources

* [Troubleshooting Guide](https://docs.ollama.com/troubleshooting)
* [FAQ](https://docs.ollama.com/faq#faq)
* [Development guide](./development.md)
