<p align="center">
  <a href="https://ollama.com">
    <img src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7" alt="ollama" width="200"/>
  </a>
</p>

# Ollama

Start building with open models.

## Download

### macOS

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

or [download manually](https://ollama.com/download/Ollama.dmg)

### Windows

```shell
irm https://ollama.com/install.ps1 | iex
```

or [download manually](https://ollama.com/download/OllamaSetup.exe)

### Linux

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

[Manual install instructions](https://docs.ollama.com/linux#manual-install)

### Docker

The official [Ollama Docker image](https://hub.docker.com/r/ollama/ollama) `ollama/ollama` is available on Docker Hub.

### Libraries

- [ollama-python](https://github.com/ollama/ollama-python)
- [ollama-js](https://github.com/ollama/ollama-js)

### Community

- [Discord](https://discord.gg/ollama)
- [𝕏 (Twitter)](https://x.com/ollama)
- [Reddit](https://reddit.com/r/ollama)

## Get started

```
zerollama
```

You'll be prompted to run a model or connect Ollama to your existing agents or applications such as `Claude Code`, `OpenClaw`, `OpenCode` , `Codex`, `Copilot`,  and more.

### Coding

To launch a specific integration:

```
zerollama launch claude
```

Supported integrations include [Claude Code](https://docs.ollama.com/integrations/claude-code), [Codex](https://docs.ollama.com/integrations/codex), [Copilot CLI](https://docs.ollama.com/integrations/copilot-cli), [Droid](https://docs.ollama.com/integrations/droid), and [OpenCode](https://docs.ollama.com/integrations/opencode).

### AI assistant

Use [OpenClaw](https://docs.ollama.com/integrations/openclaw) to turn Ollama into a personal AI assistant across WhatsApp, Telegram, Slack, Discord, and more:

```
zerollama launch openclaw
```

### Chat with a model

Run and chat with [Gemma 3](https://ollama.com/library/gemma3):

```
zerollama run gemma3
```

See [ollama.com/library](https://ollama.com/library) for the full list.

See the [quickstart guide](https://docs.ollama.com/quickstart) for more details.

## REST API

Ollama has a REST API for running and managing models.

```
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [{
    "role": "user",
    "content": "Why is the sky blue?"
  }],
  "stream": false
}'
```

See the [API documentation](https://docs.ollama.com/api) for all endpoints.

### Python

```
pip install ollama
```

```python
from ollama import chat

response = chat(model='gemma3', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response.message.content)
```

### JavaScript

```
npm i ollama
```

```javascript
import ollama from "ollama";

const response = await ollama.chat({
  model: "gemma3",
  messages: [{ role: "user", content: "Why is the sky blue?" }],
});
console.log(response.message.content);
```

## Supported backends

- [llama.cpp](https://github.com/ggml-org/llama.cpp) project founded by Georgi Gerganov.

## Documentation

- [CLI reference](https://docs.ollama.com/cli)
- [REST API reference](https://docs.ollama.com/api)
- [Importing models](https://docs.ollama.com/import)
- [Modelfile reference](https://docs.ollama.com/modelfile)
- [Building from source](https://github.com/ollama/ollama/blob/main/docs/development.md)

### Building zerollama on macOS (ggml @ b9611)

**Fresh clone (any path):** `./scripts/dev_bootstrap.sh` then `./zerollama serve`.

**Why tiers, not one setup script:** sign-off needs pulled GGUFs; CI smokes bind Go **`:8080`** while daily serve uses **`:11434`**; `../llama.cpp` is cloned automatically on tier 0. **`zerollama doctor --fix`** clones the sibling and builds Metal libllama when missing — same order as `mac_setup`. Details: [mac-dev-setup.md](docs/mac-dev-setup.md) · ROADMAP [M14](docs/ROADMAP.md#apple-silicon--metal-track).

### Building zerollama on CUDA (RTX 5080-class / Proxmox CT)

**Why a separate section:** CUDA hosts use discrete VRAM (`single_gpu.yaml`, `nvidia-smi`), **sm_120** needs CUDA **12.8+** `nvcc`, and Proxmox operators often land on the **host** while GPU passthrough lives in an **LXC** — run gates **inside** the CT (`pct exec 1564 -- …`), not on the host with stale CUDA 12.3.

```bash
# Inside CT (e.g. cudallama) — sibling llama.cpp @ b9611 + kv-ext patch 0015
export LLAMA_MODEL=/root/Llama-OuteTTS-1.0-1B-Q8_0.gguf   # ~1B Q8 smoke
export LLAMA_CPP_LIB=/root/llama.cpp/build/bin/libllama.so
export CMAKE_CUDA_ARCHITECTURES=120-real
CMAKE_CUDA_ARCHITECTURES=120-real ./scripts/build_llama_server.sh

# Gate sequence (see gpu-5080-operator-guide.md)
./scripts/phase15_llama_kv_ext_pin_check.sh
./scripts/phase15_inprocess_signoff.sh          # PASS
CUDA_LLAMA_MODEL=$LLAMA_MODEL ./scripts/l2_cuda_full_gate.sh
./scripts/l3_cache_smoke.sh && ./scripts/l3_gate_report.sh /tmp/l3-cache-smoke.json
```

**Jun 2026 sign-off (CT 1564):** Phase 15 **PASS**; L2 **FAIL merge** @ 8k (stock faster); L3 **SOFT PASS**. Full playbook: [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md).

**Why a separate build script:** CGO needs Xcode SDK, embedded Python, and Metal ggml from the **patched in-tree** vendor (`llama/patches/` on `b9611`), not only sibling `../llama.cpp`.

```bash
# Tier 0 — build + serve (recommended first run)
./scripts/dev_bootstrap.sh
./zerollama serve

# Tier 1 — pull a model
./zerollama pull llama3.2:3b

# Optional: vendor sync when hacking ggml patches (not required for first build)
# make -f Makefile.sync clean apply-patches && ./scripts/sync_vendor_llama.sh

# Rebuild ggml binary only
./scripts/build_zerollama_mac.sh   # BUILD_MLX=auto when ../mlx present

# Tier 2 — sign-off (after pull; uses :8080/:8081 smoke layout)
# MAC_SETUP_SIGNOFF=1 MAC_SETUP_GO=0 MAC_SETUP_BUILD=0 ./scripts/mac_setup.sh
# Full gate + qwen35 (M4 Max, Jun 2026):
# RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/metal_signoff.sh

# Optional: qwen35 ggml smoke only (needs :8080/:8081 stack or run via metal_signoff)
# RUN_E2E_QWEN35_MODEL=qwen3.6:latest ./scripts/qwen35_mac_smoke.sh
```

**Daily serve:** `zerollama serve` — Go `:11434` (or `:8080` in dev) + uv sidecar `:8081` with `apple_silicon.yaml` inprocess backend.

**GPU profile autotune (L1):** the runtime sidecar auto-merges tuned llama-server flags (`-b`, `-ub`, `-np`, `-fa`, …) from `runtime/configs/gpu/` based on **unified RAM** on Mac (`apple-silicon-16g` … `128g`). Check `curl -s :8081/health | jq .gpu_profile`. **Why:** Phase 13 estimates fit; L1 picks throughput knobs — a 128 GiB M4 Max should not share the same batch/parallel defaults as a 16 GiB Air. Disable: `ZEROLLAMA_GPU_PROFILE=0`. Doc: [gpu-profiles-l1.md](docs/gpu-profiles-l1.md).

**Prompt cache → slot bridge (L3):** pass a stable session key in request `options` (`eliza.conversationId`, `prompt_cache_key`, …) and the runtime pins llama-server `id_slot` + sets `cache_prompt: true` so repeat system prompts skip full prefill. Check `curl -s :8081/health | jq .llama_cache`. **Why:** agent threads re-send the same system prompt every turn — dynamic Phase 15 slots throw away KV on `complete()`. L3 maps keys → slots (needs `-np > 1` from L1). Batch rows use `prompt_cache_keys[]` (strict per-index — no silent fallback to a flat key). Disk cache lives under `~/.cache/zerollama/llama-cache/<modelHash>/`; hash canonicalizes GGUF paths and includes L2 cache types so fork/stock blobs never mix. Disable: `ZEROLLAMA_LLAMA_CACHE=0`. Sign-off: `./scripts/l3_cache_smoke.sh` + `./scripts/l3_gate_report.sh` (or `RUN_E2E_L3=1` on `m3_metal_signoff.sh`). **5080 Jun 2026:** SOFT PASS on 1B Q8 (bridge wired). Doc: [gpu-profiles-l3.md](docs/gpu-profiles-l3.md).

**Fork evaluation (L2):** build eliza `llama-server` sibling + A/B against stock. `./scripts/l2_full_gate.sh` (Mac) or `./scripts/l2_cuda_full_gate.sh` (CUDA). **Why:** QJL/TurboQuant live in `elizaOS/llama.cpp` — vendor merge is gated on measured wins. Metal Jun 2026: stock wins decode at 8k/27k; fork runs 131k ctx. **CUDA 5080 Jun 2026:** stock wins @ 8k (FAIL merge); long-ctx legs pending. Doc: [gpu-profiles-l2.md](docs/gpu-profiles-l2.md).

**Why rebuild after pull:** Jun 2026 fixed Mac **GPU bootstrap discovery** (`total_vram="0 B"` / CPU-only offload) and **Go ollama-engine sched_reserve** for qwen35moe (`GGML_ASSERT(tensor->buffer == NULL)` abort). Rebuild so `/info` uses `DiscoverBackendDevices()` and graph tensors defer to the scheduler — see [apple-silicon-metal.md](docs/apple-silicon-metal.md#gpu-bootstrap-discovery-jun-2026) and [sched_reserve](docs/apple-silicon-metal.md#go-ollama-engine-sched_reserve-jun-2026).

**Context length on Mac (ggml):** keep manifest `num_ctx` modest (4096); use **`options.num_ctx` per request** for long context — manifest defaults pre-allocate KV at load and very large values can hang. `/api/ps` shows the **loaded** runner, not `/api/show`. After `/api/create` or `zerollama stop`, confirm with empty `/api/ps`. See [qwen35-apple-silicon.md](docs/qwen35-apple-silicon.md#manifest-num_ctx-vs-request-optionsnum_ctx-jun-2026).

**Faster inference-only startup** (skip training embed + blob prune):

```bash
OLLAMA_TRAINING=false OLLAMA_NOPRUNE=1 ./zerollama serve
```

### LM Studio cache (reuse local downloads)

**Why:** LM Studio and zerollama often share a Mac; re-downloading 30–70 GB weights from the registry wastes time and disk when `~/.lmstudio/models` already has them.

```bash
./zerollama list                                    # includes discoverable LM Studio caches
./zerollama pull lmstudio-community/gemma-4-31b-it:q8_0   # registers from cache when matched
OLLAMA_LMSTUDIO_LIST_ALL=1 ./zerollama serve      # list MLX models even when disk is tight
```

- **GGUF:** symlinked into `OLLAMA_MODELS` (near-zero extra space).
- **MLX safetensors** (`config.json` + weights): repacked into zerollama blobs (~full model size free required).
- **Pull** fails early with a clear disk error if MLX import cannot fit.

Full rationale, env vars, and troubleshooting: [docs/lmstudio-import.md](docs/lmstudio-import.md).

**Why runtime header on proxy:** Pulled model names may route to legacy ggml and contend with the sidecar on one Metal device — use `X-Zerollama-Runtime: 1` or runtime-default manifest backend. See [apple-silicon-metal.md](docs/apple-silicon-metal.md#scheduler-errors-http-status).

### In this repository

- [Eliza Cloud / Zerollama remote inference](docs/eliza-cloud.md) — **why** Eliza is the default upstream (OpenAI/Anthropic APIs + API keys), **why** legacy Ed25519 signing is limited to `ollama.com`, path rewrites, catalog merge, and when responses are raw upstream JSON.
- [Video understanding (VLM)](docs/video-understanding.md) — **why** OpenAI `video_url` merges into one message, **why** ffmpeg samples to frames, security (HTTPS, SSRF), native **fps/stride** sampling, context preflight, and optional SGLang proxy.
- [Wan text-to-video (T2V)](docs/wan-t2v.md) — **why** async `/v1/videos` uses the training `run_script` queue (not GGUF chat), **why** checkpoints install separately, defer ids, and artifact paths.
- [Multimodal / video backends](docs/multimodal-backends.md) — **why** env vars and manifest `config.json` both exist; Whisper, Piper, and **OLLAMA_VIDEO_*** for native video.
- [Video parity matrix](docs/video-parity.md) — **why** reference workloads and a comparison table for Option 2 (native vs optional SGLang).
- [Changelog](CHANGELOG.md) — what changed and **why** it matters for operators.
- [Roadmap](docs/ROADMAP.md) — remote Eliza cloud follow-ups, Wan T2V v1 + follow-ups, Option 2 milestones, **Phase 17 upstream GGUF alignment**, **why** each phase exists (policy vs templates vs limits).
- [Upstream Ollama comparison](docs/upstream-ollama-diff.md) — **why** vanilla Ollama uses Go→llama-server for GGUF; pin gaps; cherry-pick map vs zerollama Python runtime and training.
- [llama.cpp backend (experimental)](docs/llama-cpp-backend.md) — `--llama-cpp-backend` routes text GGUF through Python runtime + sibling llama.cpp; benchmark vs ggml and upstream.
- [ggml @ b9611 migration](docs/ggml-b9509-migration.md) — **why** in-process ggml uses a pinned vendor tree + 14 reviewable patches (not overlay snapshots); ahead of vanilla Ollama b9509; sync, Ollama deltas, Mac sign-off checklist.
- [Scheduling, VRAM, and queue policy](docs/scheduling-vram-policy.md) — **why** inference and training are not one FIFO; VRAM broker; T6 `defer-*` queue; runtime VRAM heuristics (NVML, GGUF metadata); **ggml unload / manifest `num_ctx` at load**; **M12 ggml `suggested_max_num_ctx` + opt-in clamp** (parity with Phase 13); prompt truncation fields.
- [Fleet management (multi-node)](docs/fleet-management.md) — **why** a thin manager above per-node schedulers; `zerollama fleet serve`; warm-model assign API (F3); pairs with [fleet scheduling design](docs/fleet-scheduling.md).
- [Phase 11 runtime admission](docs/phase11-runtime-admission.md) — **why** opinionated VRAM + inference-first; priority classes; enqueue before queue; tunable min-free and training reserve.
- [Phase 13 runtime VRAM estimates](docs/phase13-runtime-vram.md) — **why** pre-check and `suggested_max_num_ctx` before load; opt-in context clamp; `runtime_vram_estimate.sh`; autotune on tight GPUs. **Ggml path (M12):** same suggest/clamp idea on Go scheduler — see [scheduling-vram-policy.md](docs/scheduling-vram-policy.md#ggml-vram-suggest-and-opt-in-clamp-m12-jun-2026).
- [Phase 14 in-process llama](docs/phase14-inprocess-llama.md) — **why** loopback `llama-server` added latency, split VRAM, and blocked token-accurate tools truncation; three backends (`subprocess` default, `inprocess` ctypes GPU, `llama-cpp-python` wheel CPU-default); `POST /internal/tokenize` for Go render-chat; sign-off scripts `phase14_inprocess_smoke.sh`, `phase14_wheel_cpu_smoke.sh`, `phase14_yaml_config_smoke.sh`, `phase14_both_backends.sh`.
- [Phase 14 handoff](docs/handoff-phase14-inprocess-llama.md) — engineer handoff: architecture, code map, bugs fixed, 5080 sign-off commands.
- [Phase 15 native KV](docs/phase15-native-kv.md) — **why** PA block pool + scheduler bind precede tensor KV; **v13–v16** C `llama_decode` + GIL release + engine resume via `current_pos`; **v26–v30** continuous batch decode (`run_batch_step`, `generate_batch`, `stream_generate_batch`, per-row C sampling); **GPU sign-off** `phase15_metal_signoff.sh` (Mac) + `phase15_inprocess_signoff.sh` (**PASS RTX 5080 Jun 2026**); loopback `POST /internal/generate-batch`; `/health.kv_decode_loop`, `kv_continuous_batch`. Linked build: `scripts/phase15_runtime_kv_env.sh` (sibling `../llama.cpp` @ b9611 + patch 0015).
- [Phase 12 tools + admission handoff](docs/handoff-phase12-runtime-tools.md) — runtime tools (Go render/parse), GPU code maps, smokes.
- [GPU training integration](docs/gpu-training.md) — **why** Go owns HTTP + TCP `:9500`; embedded CPython; inference-first VRAM on OOM; defer queue env vars. Code map: [`x/trainingworker/pyembed/README.md`](x/trainingworker/pyembed/README.md).
- [Python GGUF runtime (embedded)](docs/runtime-embed.md) — **why** a sidecar/in-process FastAPI runtime fronts `llama-server` while Go keeps registry/API; env `ZEROLLAMA_RUNTIME_EMBED`, `LLAMA_MODEL`, `LLAMA_SERVER_BIN`.
- [Inference smoke testing](docs/testing-smoke.md) — **why** runtime (`:8081`) and legacy ggml (`:8080`) share one GPU; `gpu_smoke_all.sh`, `gpu_health_report.sh`, 5080 build notes.
- [GPU 5080 operator guide](docs/gpu-5080-operator-guide.md) — **why** `gpu_5080_session.sh` is the single-GPU gate; Proxmox CT layout; **`RUN_E2E_PREFLIGHT=0`** when CGO httplib missing; Phase 15 + L2 + L3 sign-off sequence; API unload before VRAM broker; snapshot + autotune; harmony deferred without high host RAM.
- [L2 eliza fork evaluation](docs/gpu-profiles-l2.md) — **why** QJL/Polar/TurboQuant; build `../eliza-llama.cpp`; **5080 Jun 2026:** stock wins @ 8k (FAIL merge); bench gate before vendor merge.
- [L3 prompt cache → slot bridge](docs/gpu-profiles-l3.md) — **why** stable session keys skip repeat prefill; pinned `id_slot` + disk TTL; batch `prompt_cache_keys`; **5080 Jun 2026:** SOFT PASS (bridge wired on 1B).
- [Apple Silicon & Metal](docs/apple-silicon-metal.md) — **why** unified memory ≠ CUDA VRAM; ggml Metal default; runtime `metal-unified` probe; **L1 GPU profiles** (RAM tiers); Darwin Metal contention policy; scheduler 400/503 errors; **GPU bootstrap discovery**; **Go engine sched_reserve** (qwen35moe); **Jun 2026 sign-off** (`metal_signoff.sh`, `qwen35_mac_smoke.sh`).
- [L1 GPU profiles (autotune)](docs/gpu-profiles-l1.md) — **why** Phase 13 ≠ throughput tuning; NVIDIA name/VRAM buckets; Apple RAM tiers; stock llama.cpp safety; operator env.
- [Qwen 3.5/3.6 on Apple Silicon](docs/qwen35-apple-silicon.md) — **why** qwen35 hits compat metadata + Metal embed layers; **Go ollama-engine default on Mac** (Jun 2026); `PrimaryFamily()` for VL; thinking-model API fields; opt-in `qwen35_mac_smoke.sh`.
- [MLX routing policy](docs/mlx-routing-policy.md) — when to use ggml Metal vs runtime vs mlxrunner; `IsMLX()` guards; LM Studio MLX disk import policy.
- [LM Studio cache import](docs/lmstudio-import.md) — **why** pull-from-cache, **why** MLX copies vs GGUF symlinks, disk policy, `OLLAMA_LMSTUDIO_LIST_ALL`, operator troubleshooting.

## Community Integrations

> Want to add your project? Open a pull request.

### Chat Interfaces

#### Web

- [Open WebUI](https://github.com/open-webui/open-webui) - Extensible, self-hosted AI interface
- [Onyx](https://github.com/onyx-dot-app/onyx) - Connected AI workspace
- [LibreChat](https://github.com/danny-avila/LibreChat) - Enhanced ChatGPT clone with multi-provider support
- [Lobe Chat](https://github.com/lobehub/lobe-chat) - Modern chat framework with plugin ecosystem ([docs](https://lobehub.com/docs/self-hosting/examples/ollama))
- [NextChat](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web) - Cross-platform ChatGPT UI ([docs](https://docs.nextchat.dev/models/ollama))
- [Perplexica](https://github.com/ItzCrazyKns/Perplexica) - AI-powered search engine, open-source Perplexity alternative
- [big-AGI](https://github.com/enricoros/big-AGI) - AI suite for professionals
- [Lollms WebUI](https://github.com/ParisNeo/lollms-webui) - Multi-model web interface
- [ChatOllama](https://github.com/sugarforever/chat-ollama) - Chatbot with knowledge bases
- [Bionic GPT](https://github.com/bionic-gpt/bionic-gpt) - On-premise AI platform
- [Chatbot UI](https://github.com/ivanfioravanti/chatbot-ollama) - ChatGPT-style web interface
- [Hollama](https://github.com/fmaclen/hollama) - Minimal web interface
- [Chatbox](https://github.com/Bin-Huang/Chatbox) - Desktop and web AI client
- [chat](https://github.com/swuecho/chat) - Chat web app for teams
- [Ollama RAG Chatbot](https://github.com/datvodinh/rag-chatbot.git) - Chat with multiple PDFs using RAG
- [Tkinter-based client](https://github.com/chyok/ollama-gui) - Python desktop client

#### Desktop

- [Dify.AI](https://github.com/langgenius/dify) - LLM app development platform
- [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) - All-in-one AI app for Mac, Windows, and Linux
- [Maid](https://github.com/Mobile-Artificial-Intelligence/maid) - Cross-platform mobile and desktop client
- [Witsy](https://github.com/nbonamy/witsy) - AI desktop app for Mac, Windows, and Linux
- [Cherry Studio](https://github.com/kangfenmao/cherry-studio) - Multi-provider desktop client
- [Ollama App](https://github.com/JHubi1/ollama-app) - Multi-platform client for desktop and mobile
- [PyGPT](https://github.com/szczyglis-dev/py-gpt) - AI desktop assistant for Linux, Windows, and Mac
- [Alpaca](https://github.com/Jeffser/Alpaca) - GTK4 client for Linux and macOS
- [SwiftChat](https://github.com/aws-samples/swift-chat) - Cross-platform including iOS, Android, and Apple Vision Pro
- [Enchanted](https://github.com/AugustDev/enchanted) - Native macOS and iOS client
- [RWKV-Runner](https://github.com/josStorer/RWKV-Runner) - Multi-model desktop runner
- [Ollama Grid Search](https://github.com/dezoito/ollama-grid-search) - Evaluate and compare models
- [macai](https://github.com/Renset/macai) - macOS client for Ollama and ChatGPT
- [AI Studio](https://github.com/MindWorkAI/AI-Studio) - Multi-provider desktop IDE
- [Reins](https://github.com/ibrahimcetin/reins) - Parameter tuning and reasoning model support
- [ConfiChat](https://github.com/1runeberg/confichat) - Privacy-focused with optional encryption
- [LLocal.in](https://github.com/kartikm7/llocal) - Electron desktop client
- [MindMac](https://mindmac.app) - AI chat client for Mac
- [Msty](https://msty.app) - Multi-model desktop client
- [BoltAI for Mac](https://boltai.com) - AI chat client for Mac
- [IntelliBar](https://intellibar.app/) - AI-powered assistant for macOS
- [Kerlig AI](https://www.kerlig.com/) - AI writing assistant for macOS
- [Hillnote](https://hillnote.com) - Markdown-first AI workspace
- [Perfect Memory AI](https://www.perfectmemory.ai/) - Productivity AI personalized by screen and meeting history

#### Mobile

- [Ollama Android Chat](https://github.com/sunshine0523/OllamaServer) - One-click Ollama on Android

> SwiftChat, Enchanted, Maid, Ollama App, Reins, and ConfiChat listed above also support mobile platforms.

### Code Editors & Development

- [Cline](https://github.com/cline/cline) - VS Code extension for multi-file/whole-repo coding
- [Continue](https://github.com/continuedev/continue) - Open-source AI code assistant for any IDE
- [Void](https://github.com/voideditor/void) - Open source AI code editor, Cursor alternative
- [Copilot for Obsidian](https://github.com/logancyang/obsidian-copilot) - AI assistant for Obsidian
- [twinny](https://github.com/rjmacarthy/twinny) - Copilot and Copilot chat alternative
- [gptel Emacs client](https://github.com/karthink/gptel) - LLM client for Emacs
- [Ollama Copilot](https://github.com/bernardo-bruning/ollama-copilot) - Use Ollama as GitHub Copilot
- [Obsidian Local GPT](https://github.com/pfrankov/obsidian-local-gpt) - Local AI for Obsidian
- [Ellama Emacs client](https://github.com/s-kostyaev/ellama) - LLM tool for Emacs
- [orbiton](https://github.com/xyproto/orbiton) - Config-free text editor with Ollama tab completion
- [AI ST Completion](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) - Sublime Text 4 AI assistant
- [VT Code](https://github.com/vinhnx/vtcode) - Rust-based terminal coding agent with Tree-sitter
- [QodeAssist](https://github.com/Palm1r/QodeAssist) - AI coding assistant for Qt Creator
- [AI Toolkit for VS Code](https://aka.ms/ai-tooklit/ollama-docs) - Microsoft-official VS Code extension
- [Open Interpreter](https://docs.openinterpreter.com/language-model-setup/local-models/ollama) - Natural language interface for computers

### Libraries & SDKs

- [LiteLLM](https://github.com/BerriAI/litellm) - Unified API for 100+ LLM providers
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic_kernel/connectors/ai/ollama) - Microsoft AI orchestration SDK
- [LangChain4j](https://github.com/langchain4j/langchain4j) - Java LangChain ([example](https://github.com/langchain4j/langchain4j-examples/tree/main/ollama-examples/src/main/java))
- [LangChainGo](https://github.com/tmc/langchaingo/) - Go LangChain ([example](https://github.com/tmc/langchaingo/tree/main/examples/ollama-completion-example))
- [Spring AI](https://github.com/spring-projects/spring-ai) - Spring framework AI support ([docs](https://docs.spring.io/spring-ai/reference/api/chat/ollama-chat.html))
- [LangChain](https://python.langchain.com/docs/integrations/chat/ollama/) and [LangChain.js](https://js.langchain.com/docs/integrations/chat/ollama/) with [example](https://js.langchain.com/docs/tutorials/local_rag/)
- [Ollama for Ruby](https://github.com/crmne/ruby_llm) - Ruby LLM library
- [any-llm](https://github.com/mozilla-ai/any-llm) - Unified LLM interface by Mozilla
- [OllamaSharp for .NET](https://github.com/awaescher/OllamaSharp) - .NET SDK
- [LangChainRust](https://github.com/Abraxas-365/langchain-rust) - Rust LangChain ([example](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_ollama.rs))
- [Agents-Flex for Java](https://github.com/agents-flex/agents-flex) - Java agent framework ([example](https://github.com/agents-flex/agents-flex/tree/main/agents-flex-llm/agents-flex-llm-ollama/src/test/java/com/agentsflex/llm/ollama))
- [Elixir LangChain](https://github.com/brainlid/langchain) - Elixir LangChain
- [Ollama-rs for Rust](https://github.com/pepperoni21/ollama-rs) - Rust SDK
- [LangChain for .NET](https://github.com/tryAGI/LangChain) - .NET LangChain ([example](https://github.com/tryAGI/LangChain/blob/main/examples/LangChain.Samples.OpenAI/Program.cs))
- [chromem-go](https://github.com/philippgille/chromem-go) - Go vector database with Ollama embeddings ([example](https://github.com/philippgille/chromem-go/tree/v0.5.0/examples/rag-wikipedia-ollama))
- [LangChainDart](https://github.com/davidmigloz/langchain_dart) - Dart LangChain
- [LlmTornado](https://github.com/lofcz/llmtornado) - Unified C# interface for multiple inference APIs
- [Ollama4j for Java](https://github.com/ollama4j/ollama4j) - Java SDK
- [Ollama for Laravel](https://github.com/cloudstudio/ollama-laravel) - Laravel integration
- [Ollama for Swift](https://github.com/mattt/ollama-swift) - Swift SDK
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/) and [LlamaIndexTS](https://ts.llamaindex.ai/modules/llms/available_llms/ollama) - Data framework for LLM apps
- [Haystack](https://github.com/deepset-ai/haystack-integrations/blob/main/integrations/ollama.md) - AI pipeline framework
- [Firebase Genkit](https://firebase.google.com/docs/genkit/plugins/ollama) - Google AI framework
- [Ollama-hpp for C++](https://github.com/jmont-dev/ollama-hpp) - C++ SDK
- [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) - Julia LLM toolkit ([example](https://svilupp.github.io/PromptingTools.jl/dev/examples/working_with_ollama))
- [Ollama for R - rollama](https://github.com/JBGruber/rollama) - R SDK
- [Portkey](https://portkey.ai/docs/welcome/integration-guides/ollama) - AI gateway
- [Testcontainers](https://testcontainers.com/modules/ollama/) - Container-based testing
- [LLPhant](https://github.com/theodo-group/LLPhant?tab=readme-ov-file#ollama) - PHP AI framework

### Frameworks & Agents

- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT/blob/master/docs/content/platform/ollama.md) - Autonomous AI agent platform
- [crewAI](https://github.com/crewAIInc/crewAI) - Multi-agent orchestration framework
- [Strands Agents](https://github.com/strands-agents/sdk-python) - Model-driven agent building by AWS
- [Cheshire Cat](https://github.com/cheshire-cat-ai/core) - AI assistant framework
- [any-agent](https://github.com/mozilla-ai/any-agent) - Unified agent framework interface by Mozilla
- [Stakpak](https://github.com/stakpak/agent) - Open source DevOps agent
- [Hexabot](https://github.com/hexastack/hexabot) - Conversational AI builder
- [Neuro SAN](https://github.com/cognizant-ai-lab/neuro-san-studio) - Multi-agent orchestration ([docs](https://github.com/cognizant-ai-lab/neuro-san-studio/blob/main/docs/user_guide.md#ollama))

### RAG & Knowledge Bases

- [RAGFlow](https://github.com/infiniflow/ragflow) - RAG engine based on deep document understanding
- [R2R](https://github.com/SciPhi-AI/R2R) - Open-source RAG engine
- [MaxKB](https://github.com/1Panel-dev/MaxKB/) - Ready-to-use RAG chatbot
- [Minima](https://github.com/dmayboroda/minima) - On-premises or fully local RAG
- [Chipper](https://github.com/TilmanGriesel/chipper) - AI interface with Haystack RAG
- [ARGO](https://github.com/xark-argo/argo) - RAG and deep research on Mac/Windows/Linux
- [Archyve](https://github.com/nickthecook/archyve) - RAG-enabling document library
- [Casibase](https://casibase.org) - AI knowledge base with RAG and SSO
- [BrainSoup](https://www.nurgo-software.com/products/brainsoup) - Native client with RAG and multi-agent automation

### Bots & Messaging

- [LangBot](https://github.com/RockChinQ/LangBot) - Multi-platform messaging bots with agents and RAG
- [AstrBot](https://github.com/Soulter/AstrBot/) - Multi-platform chatbot with RAG and plugins
- [Discord-Ollama Chat Bot](https://github.com/kevinthedang/discord-ollama) - TypeScript Discord bot
- [Ollama Telegram Bot](https://github.com/ruecat/ollama-telegram) - Telegram bot
- [LLM Telegram Bot](https://github.com/innightwolfsleep/llm_telegram_bot) - Telegram bot for roleplay

### Terminal & CLI

- [aichat](https://github.com/sigoden/aichat) - All-in-one LLM CLI with Shell Assistant, RAG, and AI tools
- [oterm](https://github.com/ggozad/oterm) - Terminal client for Ollama
- [gollama](https://github.com/sammcj/gollama) - Go-based model manager for Ollama
- [tlm](https://github.com/yusufcanb/tlm) - Local shell copilot
- [tenere](https://github.com/pythops/tenere) - TUI for LLMs
- [ParLlama](https://github.com/paulrobello/parllama) - TUI for Ollama
- [llm-ollama](https://github.com/taketwo/llm-ollama) - Plugin for [Datasette's LLM CLI](https://llm.datasette.io/en/stable/)
- [ShellOracle](https://github.com/djcopley/ShellOracle) - Shell command suggestions
- [LLM-X](https://github.com/mrdjohnson/llm-x) - Progressive web app for LLMs
- [cmdh](https://github.com/pgibler/cmdh) - Natural language to shell commands
- [VT](https://github.com/vinhnx/vt.ai) - Minimal multimodal AI chat app

### Productivity & Apps

- [AppFlowy](https://github.com/AppFlowy-IO/AppFlowy) - AI collaborative workspace, self-hostable Notion alternative
- [Screenpipe](https://github.com/mediar-ai/screenpipe) - 24/7 screen and mic recording with AI-powered search
- [Vibe](https://github.com/thewh1teagle/vibe) - Transcribe and analyze meetings
- [Page Assist](https://github.com/n4ze3m/page-assist) - Chrome extension for AI-powered browsing
- [NativeMind](https://github.com/NativeMindBrowser/NativeMindExtension) - Private, on-device browser AI assistant
- [Ollama Fortress](https://github.com/ParisNeo/ollama_proxy_server) - Security proxy for Ollama
- [1Panel](https://github.com/1Panel-dev/1Panel/) - Web-based Linux server management
- [Writeopia](https://github.com/Writeopia/Writeopia) - Text editor with Ollama integration
- [QA-Pilot](https://github.com/reid41/QA-Pilot) - GitHub code repository understanding
- [Raycast extension](https://github.com/MassimilianoPasquini97/raycast_ollama) - Ollama in Raycast
- [Painting Droid](https://github.com/mateuszmigas/painting-droid) - Painting app with AI integrations
- [Serene Pub](https://github.com/doolijb/serene-pub) - AI roleplaying app
- [Mayan EDMS](https://gitlab.com/mayan-edms/mayan-edms) - Document management with Ollama workflows
- [TagSpaces](https://www.tagspaces.org) - File management with [AI tagging](https://docs.tagspaces.org/ai/)

### Observability & Monitoring

- [Opik](https://www.comet.com/docs/opik/cookbook/ollama) - Debug, evaluate, and monitor LLM applications
- [OpenLIT](https://github.com/openlit/openlit) - OpenTelemetry-native monitoring for Ollama and GPUs
- [Lunary](https://lunary.ai/docs/integrations/ollama) - LLM observability with analytics and PII masking
- [Langfuse](https://langfuse.com/docs/integrations/ollama) - Open source LLM observability
- [HoneyHive](https://docs.honeyhive.ai/integrations/ollama) - AI observability and evaluation for agents
- [MLflow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html#automatic-tracing) - Open source LLM observability

### Database & Embeddings

- [pgai](https://github.com/timescale/pgai) - PostgreSQL as a vector database ([guide](https://github.com/timescale/pgai/blob/main/docs/vectorizer-quick-start.md))
- [MindsDB](https://github.com/mindsdb/mindsdb/blob/staging/mindsdb/integrations/handlers/ollama_handler/README.md) - Connect Ollama with 200+ data platforms
- [chromem-go](https://github.com/philippgille/chromem-go/blob/v0.5.0/embed_ollama.go) - Embeddable vector database for Go ([example](https://github.com/philippgille/chromem-go/tree/v0.5.0/examples/rag-wikipedia-ollama))
- [Kangaroo](https://github.com/dbkangaroo/kangaroo) - AI-powered SQL client

### Infrastructure & Deployment

#### Cloud

- [Google Cloud](https://cloud.google.com/run/docs/tutorials/gpu-gemma2-with-ollama)
- [Fly.io](https://fly.io/docs/python/do-more/add-ollama/)
- [Koyeb](https://www.koyeb.com/deploy/ollama)
- [Harbor](https://github.com/av/harbor) - Containerized LLM toolkit with Ollama as default backend

#### Package Managers

- [Pacman](https://archlinux.org/packages/extra/x86_64/ollama/)
- [Homebrew](https://formulae.brew.sh/formula/ollama)
- [Nix package](https://search.nixos.org/packages?show=ollama&from=0&size=50&sort=relevance&type=packages&query=ollama)
- [Helm Chart](https://artifacthub.io/packages/helm/ollama-helm/ollama)
- [Gentoo](https://github.com/gentoo/guru/tree/master/app-misc/ollama)
- [Flox](https://flox.dev/blog/ollama-part-one)
- [Guix channel](https://codeberg.org/tusharhero/ollama-guix)
