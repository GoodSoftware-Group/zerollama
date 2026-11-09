# Documentation

Put a doc in the bucket that matches its **role**, not just its topic:

- [capabilities/](capabilities/README.md) — what works now
- [runbooks/](runbooks/README.md) — how to operate it
- [design/](./design/README.md) — how it works and why
- [research/](research/README.md) — lab-only or future-facing work that is not an upstream borrowing
- [upstream/](upstream/tracker.md) — external repos, borrowings, and watchlists
- [handoffs/](handoffs/README.md) — implementation history for maintainers
- [roadmaps](ROADMAP.md) — open work only; track detail is in [roadmaps/](roadmaps)
- [CHANGELOG](../CHANGELOG.md) — what shipped

Public docs stay in [index.mdx](index.mdx), [docs.json](docs.json), [api/](api), and [integrations/](integrations). A shipped feature gets a capability page, then links out to a runbook or design note. Do not start a new root doc when it belongs in one of the buckets above.

### Getting Started
* [Quickstart](https://docs.ollama.com/quickstart)
* [Examples](examples.md)
* [Importing models](https://docs.ollama.com/import)
* [MacOS Documentation](https://docs.ollama.com/macos)
* [Linux Documentation](https://docs.ollama.com/linux)
* [Windows Documentation](https://docs.ollama.com/windows)
* [Docker Documentation](https://docs.ollama.com/docker)

### Reference

* [API Reference](https://docs.ollama.com/api)
* [Modelfile Reference](https://docs.ollama.com/modelfile)
* [OpenAI Compatibility](https://docs.ollama.com/api/openai-compatibility)
* [Anthropic Compatibility](api/anthropic-compatibility.mdx)

### Roadmaps & Trackers

* [Roadmap](ROADMAP.md) — executive view of what matters next.
* [Research roadmap](RESEARCH_ROADMAP.md) — exploratory work, borrowings, and lab bets.
* [Upstream tracker](upstream/tracker.md) — which repos we follow, why we follow them, and what we may borrow next.
* [Core serving roadmap](roadmaps/core-serving.md) — local inference, runtime, and product-track Apple serving.
* [Surfaces and integrations roadmap](roadmaps/surfaces-and-integrations.md) — Eliza cloud, structured inference surfaces, and LM Studio.
* [Training and scheduling roadmap](roadmaps/training-and-scheduling.md) — queue policy, Slurm, and GPU training.
* [Media roadmap](roadmaps/media.md) — video, music, image, and mesh backlog.

### Capabilities

* [Capabilities index](capabilities/README.md) — front door for working features and API capabilities, including tool calling, streaming, vision, and more.
* [Typed decisions](capabilities/decisions.md) — `POST /v1/decisions` / `/v1/systemone` across Laya, CLM, and lab decider backends.
* [Entity extract](capabilities/extract.md) — `POST /v1/extract` / `/v1/gliner` for zero-shot span extraction.
* [Video](capabilities/video.md) — video understanding in chat plus async video generation.
* [Image](capabilities/image.md) — MLX fast path and ComfyUI utility path.

### Runbooks

* [Runbooks index](runbooks/README.md) — operator and start-here docs.
* [5080 runbook — start here](runbooks/5080-runbook.md) — **`source scripts/gpu/5080_env.sh`** + **`./scripts/gpu/5080_resignoff.sh`**; ordered tiers; CT 1564 status; Radix vendor build.
* [GPU 5080 operator guide](runbooks/gpu-5080-operator-guide.md) — extended reference (VRAM, MLX, production serve) — use runbook for daily gates.
* [Inference smoke testing](runbooks/testing-smoke.md) — **why** runtime (`:8081`) and legacy ggml (`:8080`) share one GPU.
* [Mac dev setup](runbooks/mac-dev-setup.md) — **`dev_bootstrap.sh`** tier 0–3; **prereqs:** Go **1.24.1+**, full Xcode.app (or Homebrew Python), **cmake**, uv; script map after reorg; **why** `:11434` daily vs `:8080` CI.
* [Runtime env reference](runbooks/runtime-env.md) — **why** profiles/YAML/smart defaults beat dozens of `ZEROLLAMA_*` exports; L3, KV, VRAM; `./scripts/runtime/runtime_env_doctor.sh`.
* [Doctor model repair](runbooks/doctor-model-repair.md) — **why** Modelfile overlays for empty-`response` / slash-collapse / ChatML stop hygiene (not `doctor --fix`); `--repair-models` / `--apply` / `--all-local`; Qwen3 gate for invasive TEMPLATE rewrites.
* [Model serving minefield](runbooks/model-serving-minefield.md) — trap registry mapped onto `zerollama doctor` (config + live serving checks) and known gaps.
* [Intel Arc A380 runbook](runbooks/a380-runbook.md) — Vulkan/ANV operator target for the 6 GB Arc A380.

### Research

* [Research index](research/README.md) — lab notes that are not upstream borrowings.
* [wan-c vs Python MPS speed gap](research/wan-c-speed-gap.md) — profile + Phase1 cuts + toolkit `DIT_BLOCK` / flash ATTN / feat_cache asks.
* [Flash-MoE (anemll)](research/flash-moe.md) — **why** slot-bank + SSD sidecar for MoE models larger than unified RAM; Phase 17 llama-server passthrough (not ggml Metal default).
* [FreeToken MoE lab](research/freetoken-moe-lab.md) — offline sim of edge MoE policies (LRU cache, \(q^\star\) CPU/PCIe split, prefill overlap, semantic checkpoints); [arXiv:2608.16157](https://arxiv.org/abs/2608.16157).
* [GLM Flash Lite ladder (GF0–GF7)](research/glm-flash-lite-ladder.md) — **why** EXL3 GLM-5.3-Flash stays a native/venv sidecar (astra `:30000`); FreeToken/Flash-MoE rematch only; do not vendor `nv2`.
* [ANE probe (maderix)](research/ane-probe.md) — **why** subprocess smoke for private ANE APIs before hybrid inference; not on hot path.
* [ANE dflash in-process (B1–B6)](research/ane-draft-inprocess.md) — **why** same-PID IOSurface handoff on llama-server dflash draft decode; lab port 11435; draft tokens still Metal until B7.
* [ANE hybrid path (lab)](research/ane-hybrid-path.md) — crossover sweeps, prefill proxy, operator tooling index.
* [MLX DeepSeek-V4 Flash](research/mlx-deepseek-v4-flash.md) — **why** CSA/HCA not full MHA; **why** `--link` instead of a 90 GiB blob copy. [findings](research/mlx-deepseek-v4-flash-findings.md).

### Handoffs

* [Handoffs index](handoffs/README.md) — implementation snapshots for maintainers.
* [Phase 12 tools + Phase 11 admission handoff](handoffs/handoff-phase12-runtime-tools.md) — runtime tools (Go render/parse), opinionated admission, smokes, code maps.
* [Phase 14 handoff](handoffs/handoff-phase14-inprocess-llama.md) — architecture, code map, smoke footguns, bugs fixed during bring-up.
* [Phase 15 handoff](handoffs/handoff-phase15-native-kv.md) — code map, `/health` fields, gaps; **v0–v31 shipped** (C decode loop, engine resume, L3 gate, batch decode + Metal sign-off PASS Jun 2026).
* [GPU training handoff (internal)](handoffs/handoff-gpu-training-integration.md) — embedded training + Phase 11 VRAM interaction (not a substitute for [gpu-training.md](design/training/gpu-training.md)).

### Design

How it works and why. These are the deep notes; use a capability page for the working surface and a runbook for operator steps.

* [Design index](./design/README.md)
* [Serving](./design/serving/README.md) — runtime, llama.cpp, phases, admission, and GPU profiles.
* [Surfaces](./design/surfaces/README.md) — decisions, extract, Hermes, cloud, routing, and remote storage.
* [Training](./design/training/README.md) — GPU training, queue policy, fleet, and Slurm.
* [Media](./design/media/README.md) — video, music, image, mesh, and voice.
* [Apple](./design/apple/README.md) — Metal, MLX, UMA, and the ANE hooks that are not lab-only research.

### Upstream Ollama (compare, don't merge)

* [Upstream Ollama comparison](upstream/ollama-diff.md) — architecture deltas vs `../ollama-upstream`; Phase 17; benchmark workflow.
* [Open-source shoutouts](upstream/open-source-shoutouts.md) — projects whose ideas shaped zerollama.
* [llama.cpp backend (experimental)](design/serving/llama-cpp-backend.md) — `--llama-cpp-backend` test harness toward upstream GGUF path.

### Resources

* [Troubleshooting Guide](https://docs.ollama.com/troubleshooting)
* [FAQ](https://docs.ollama.com/faq#faq)
* [Development guide](development.md)
