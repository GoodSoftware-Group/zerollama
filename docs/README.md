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
* [Wan text-to-video (T2V)](./wan-t2v.md) — **why** `/v1/videos` is async, **why** training `run_script` + wrapper, VRAM/defer queue, artifacts.
* [Optional multimodal backends](./multimodal-backends.md) — env + manifest; **why** both layers.
* [Video parity matrix](./video-parity.md) — **why** reference workloads for native vs SGLang.
* [Roadmap](./ROADMAP.md) — **why** Option 2 is phased (policy, templates, context, optional subprocess).
* [Upstream Ollama comparison](./upstream-ollama-diff.md) — **why** vanilla Ollama dropped ggml for GGUF; pin gaps; cherry-pick map; Phase 17 alignment.
* [Phase 17 — Go → llama-server](./phase17-llama-server.md) — upstream GGUF path scaffold, build/serve scripts, M7 benchmark decision.
* [ggml @ b9509 migration](./ggml-b9509-migration.md) — **why** vendored ggml/llama.cpp rebased to real upstream b9509; patches, sync workflow, Ollama deltas.
* [llama.cpp backend (experimental)](./llama-cpp-backend.md) — route text GGUF through Python runtime + sibling llama.cpp; benchmark vs ggml.

### Apple Silicon (repo)

* [Apple Silicon & Metal operator guide](./apple-silicon-metal.md) — **why** unified memory ≠ CUDA VRAM; three inference paths; runtime autoconfig.
* [Qwen 3.5/3.6 on Mac](./qwen35-apple-silicon.md) — **why** three failure layers (Go engine, compat metadata, Metal embed); rebuild checklist; `num_ctx` guidance.
* [Mac dev setup](./mac-dev-setup.md) — one-command `mac_setup.sh`, CGO, daily serve.
* [LM Studio cache import](./lmstudio-import.md) — **why** pull-from-cache, MLX copy vs GGUF symlink, disk policy, env vars, troubleshooting.
* [MLX routing policy](./mlx-routing-policy.md) — ggml Metal vs runtime vs mlxrunner; LM Studio MLX disk summary.

### GPU training & scheduling (repo)

* [Scheduling, VRAM, and queue policy](./scheduling-vram-policy.md) — **why** inference and training are separate queues; Phase 8 broker; T6 idle-wait + `defer-*` queue; Phase 11–13 runtime heuristics; tight-host env checklist.
* [Fleet scheduling (multi-node)](./fleet-scheduling.md) — **why** a management node above per-node schedulers; warm-model routing; anti-patterns (scatter-gather, long quotes).
* [Fleet management operator guide](./fleet-management.md) — **why** F3 is thin (poll + assign, no remote load); `zerollama fleet serve`; API, env, agent pattern.
* [Phase 11 runtime admission](./phase11-runtime-admission.md) — **why** opinionated VRAM + inference-first policy; priority classes; enqueue/dequeue flow; `/health` gates; `VRAM_MIN_FREE` / `TRAINING_VRAM_RESERVE`.
* [Phase 13 runtime VRAM estimates](./phase13-runtime-vram.md) — **why** GGUF VRAM heuristics, `suggested_max_num_ctx`, opt-in clamp, autotune, autoconfig, operator CLI.
* [Phase 14 in-process llama](./phase14-inprocess-llama.md) — **why** subprocess HTTP was replaced for forward; three backends; render tokenize; sampling parity; 5080 sign-off scripts.
* [Phase 14 handoff](./handoff-phase14-inprocess-llama.md) — architecture, code map, smoke footguns, bugs fixed during bring-up.
* [Phase 15 native KV](./phase15-native-kv.md) — PA/C block pool, scheduler KV bind, seq-position track, forward plans (v0–v8 ops partial).
* [Phase 15 handoff](./handoff-phase15-native-kv.md) — code map, `/health` fields, gaps, v8+ next steps.
* [GPU training integration](./gpu-training.md) — **why** Go fronts HTTP + TCP `:9500` while Python holds PyTorch; embedded CPython; inference-first VRAM policy; OOM ordering; env vars and troubleshooting.
* [GPU training handoff (internal)](./handoff-gpu-training-integration.md) — embedded training + Phase 11 VRAM interaction (not a substitute for `gpu-training.md`).
* [Phase 12 tools + Phase 11 admission handoff](./handoff-phase12-runtime-tools.md) — runtime tools (Go render/parse), opinionated admission, smokes, code maps.
* [Inference smoke testing](./testing-smoke.md) — **why** runtime (`:8081`) and legacy ggml (`:8080`) share one GPU.

### Upstream Ollama (compare, don't merge)

* [Upstream Ollama comparison](./upstream-ollama-diff.md) — architecture deltas vs `../ollama-upstream`; Phase 17; benchmark workflow.
* [llama.cpp backend (experimental)](./llama-cpp-backend.md) — `--llama-cpp-backend` test harness toward upstream GGUF path.

### Remote inference — Eliza Cloud (Zerollama)

* [Eliza Cloud](./eliza-cloud.md) — **why** default upstream is Eliza (not legacy ollama.com), **why** path rewrites and `X-API-Key`, **why** catalog merge + cache, **why** raw JSON on some routes, **why** account stubs off ollama.com.

### Resources

* [Troubleshooting Guide](https://docs.ollama.com/troubleshooting)
* [FAQ](https://docs.ollama.com/faq#faq)
* [Development guide](./development.md)
