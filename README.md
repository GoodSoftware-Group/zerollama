<p align="center">
  <a href="https://github.com/ollama/ollama">
    <img src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7" alt="Zerollama" width="200"/>
  </a>
</p>

# Zerollama

**Ollama-compatible local inference for agents, GPU training, and multi-node fleets.**

Zerollama is a fork of [ollama/ollama](https://github.com/ollama/ollama). Same CLI shape, REST API, and client libraries — with extra capabilities for operators running agents on Apple Silicon and CUDA.

> **Not upstream Ollama.** Install from this repo (`./zerollama`), not [ollama.com/install.sh](https://ollama.com/install.sh). For vanilla local inference only, use upstream.

---

## How Zerollama differs from upstream Ollama

| | **Upstream Ollama** | **Zerollama** |
|---|---|---|
| **Pitch** | Run open models locally | Run, **train**, route, and **fleet-manage** models for agents |
| **Default GGUF engine** | Go → **llama-server** | Go → **ggml Metal** on Mac (~**+7% decode** vs upstream on M4 Max); Linux plain text auto-defaults to llama-server |
| **Cloud default** | [ollama.com](https://ollama.com) | **[Eliza Cloud](https://www.elizacloud.ai)** — OpenAI/Anthropic APIs, API-key auth, `:cloud` suffix |
| **GPU training** | None | **`/api/train/*`** — LoRA/QLoRA in-process (PyTorch embed) |
| **Scheduler** | Single Go scheduler | Go + **Python runtime** sidecar — VRAM admission, tools, autotune profiles |
| **Mac operator track** | Supported | First-class — Metal sign-off, MPS LoRA, LM Studio import, Qwen 3.5/3.6 fixes |
| **LM Studio** | — | **Import from cache** — GGUF symlink, MLX repack; no re-download |
| **Multimodal extras** | Core vision/audio | Video understanding, Wan T2V, Whisper/Piper backends |
| **Multi-node** | One box | **`zerollama fleet serve`** — warm-model routing, mDNS discovery |
| **Convergence** | — | [Phase 17](docs/phase17-llama-server.md) + **llama.cpp `b9781`** (v0.30.11); Python runtime stays for training/admission |

Full architecture comparison: [docs/upstream-ollama-diff.md](docs/upstream-ollama-diff.md)

### Benchmark vs stock Ollama (Apple M4 Max, `llama3.2:3b`, `num_ctx=4096`)

| Arm | Generate tok/s |
|-----|----------------|
| Stock Ollama (Go → llama-server) | ~155 |
| **Zerollama ggml Metal (Mac default)** | **~166 (+7%)** |
| Zerollama `--llama-server-backend` | ~158 |

Reproduce: `./scripts/m4_upstream_vs_zerollama_bench.sh`

---

## Phase 17 — upstream convergence (optional)

**Why not match upstream’s default engine on Mac?** Zerollama’s ggml Metal path is ~**+7% faster** on ship hardware and shares VRAM bookkeeping with the Darwin sidecar. Phase 17 ports upstream’s **Go → llama-server** integration for **mergeability**, not to replace ggml on day one.

| Goal | How |
|------|-----|
| Test upstream-shaped GGUF on Mac | `./zerollama serve --llama-server-backend` or `ZEROLLAMA_LLAMA_SERVER=1` |
| Linux plain text (auto) | Install/build `llama-server`; default when binary found (`ZEROLLAMA_LLAMA_SERVER=0` to disable) |
| Compare side-by-side | [upstream-ollama-diff.md](docs/upstream-ollama-diff.md) · `./scripts/clone_upstream_ollama.sh` |
| Full operator guide | [phase17-llama-server.md](docs/phase17-llama-server.md) |

**What Phase 17 does *not* change:** Python runtime, `/api/train/*`, Eliza cloud, fleet scheduling, Mac ggml default.

---

## Experimental Mac inference (Flash-MoE + ANE probe + ANE dflash lab)

**Why documented separately from Phase 17:** Phase 17 ports upstream **Go → llama-server** for mergeability. **Flash-MoE** and **ANE** tracks are *research integrations* scouted from the Apple Silicon inference community — they extend what Mac operators can run, but do not replace ggml Metal for in-RAM models.

| Track | Problem | Status | Enable |
|-------|---------|--------|--------|
| **[Flash-MoE (anemll)](docs/flash-moe.md)** | MoE models **larger than unified RAM** (e.g. Qwen3.5-397B experts) | Partial — flag passthrough + fork build + **`flash_moe_smoke.sh`** | `ZEROLLAMA_FLASH_MOE=1` + sidecar + `--llama-server-backend` |
| **[ANE probe (maderix)](docs/ane-probe.md)** | Validate **Apple Neural Engine** via private API bridge | Partial — subprocess smoke only | `./scripts/ane_probe_build.sh`; `zerollama doctor` |
| **[ANE dflash in-process (B1–B6)](docs/ane-draft-inprocess.md)** | **Metal base + ANE draft-step** handoff on eliza `*-dflash` | Partial — lab hook on llama-server; tokens still Metal | `ZEROLLAMA_ANE_DRAFT=1` on **lab port 11435** only; see doc |

**Why Flash-MoE is llama-server-only:** slot-bank streaming and `-fit` VRAM budgeting live in anemll's forked server, not ggml Metal. **Why ANE probe is subprocess:** private `_ANEClient` APIs break on macOS updates; isolated smoke keeps `zerollama` stable. **Why ANE dflash is in-process (not probe subprocess):** ANE IOSurface surface IDs are not visible across PIDs — llama-server must own the compiled kernel for ggml handoff.

ROADMAP: [M16 Flash-MoE](docs/ROADMAP.md#apple-silicon--metal-track), [M17 ANE probe](docs/ROADMAP.md#apple-silicon--metal-track), [M18 ANE dflash in-process](docs/ROADMAP.md#apple-silicon--metal-track).

---

## Phase 16 — thin edge daemon (optional)

**Why a separate mode from Phase 17?** Phase 17 ports upstream’s llama-server integration for **mergeability**; Phase 16 is the **deployment shape** — runtime chat off, GGUF always via llama-server, training/Eliza/fleet still in Go. Use it for upstream-shaped edge nodes without deleting zerollama differentiators.

| Goal | How |
|------|-----|
| Upstream-shaped serve (runtime chat off) | `./scripts/serve_edge.sh` or `./zerollama serve --edge` |
| Edge-marked binary (no in-process ggml CGO) | `./scripts/build_zerollama_edge.sh` → `./zerollama-edge serve` |
| Linux auto (no `--edge` flag needed) | `./scripts/serve_linux_auto.sh` when llama-server on disk |
| Operator visibility | `curl -s localhost:11434/api/status \| jq .inference.backend` |
| Full operator guide | [phase16-thin-edge.md](docs/phase16-thin-edge.md) |

**What `--edge` does *not* remove:** `/api/train/*`, Eliza cloud, fleet, launch integrations, or the Python runtime tree (only **chat/generate** routing through runtime is disabled).

---

## Quick start

**macOS (recommended first path):**

```bash
git clone <this-repo> zerollama && cd zerollama
./scripts/dev_bootstrap.sh
./zerollama serve
./zerollama pull llama3.2:3b
./zerollama run llama3.2:3b
```

**CUDA (5080-class):** [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md)

**Compare with upstream on the same machine:**

```bash
./scripts/clone_upstream_ollama.sh
./scripts/build_upstream_ollama_mac.sh
OLLAMA_HOST=127.0.0.1:11435 ../ollama-upstream/ollama serve   # upstream
./zerollama serve                                              # zerollama :11434
```

---

## Ollama compatibility

Zerollama keeps the Ollama API surface so existing tools keep working:

- **CLI:** `serve`, `pull`, `run`, `list`, `bench`, `launch`, … (binary is `./zerollama`)
- **REST:** `http://localhost:11434/api/*` and OpenAI-compatible `/v1/*`
- **Clients:** [ollama-python](https://github.com/ollama/ollama-python), [ollama-js](https://github.com/ollama/ollama-js), LangChain, Continue, Open WebUI, etc.

Remote cloud models use **Eliza** OpenAI/Anthropic routes (`model: …:cloud`), not legacy ollama.com signing — see [eliza-cloud.md](docs/eliza-cloud.md).

---

## Get started

```
zerollama
```

You'll be prompted to run a model or connect Zerollama to your existing agents or applications such as `Claude Code`, `OpenClaw`, `OpenCode`, `Codex`, `Copilot`, and more.

### Coding

To launch a specific integration:

```
zerollama launch claude
```

Supported integrations include [Claude Code](https://docs.ollama.com/integrations/claude-code), [Codex](https://docs.ollama.com/integrations/codex), [Copilot CLI](https://docs.ollama.com/integrations/copilot-cli), [Droid](https://docs.ollama.com/integrations/droid), and [OpenCode](https://docs.ollama.com/integrations/opencode).

**Model metadata:** launch loads your installed models once from `/api/tags` and passes capabilities (vision, thinking, context length) into each agent’s config — **why:** avoids slow per-model `/api/show` calls and keeps the picker and config writer in sync. See [launch-model-inventory.md](docs/launch-model-inventory.md).

### AI assistant

Use [OpenClaw](https://docs.ollama.com/integrations/openclaw) to turn Zerollama into a personal AI assistant across WhatsApp, Telegram, Slack, Discord, and more:

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

Zerollama exposes the same REST API as Ollama for running and managing models.

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

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF via ggml Metal/CUDA (Mac default) or Go→llama-server (Phase 17, Linux auto-default)
- **Flash-MoE (anemll, experimental)** — forked `llama-server` + sidecar for MoE **larger than RAM**; [flash-moe.md](docs/flash-moe.md)
- **MLX** — safetensors via mlxrunner
- **Python runtime** — admission, in-process llama, training embed (`runtime/` sidecar on `:8081`)

## Documentation

**Upstream Ollama docs** (API/Modelfile shape still applies):

- [CLI reference](https://docs.ollama.com/cli)
- [REST API reference](https://docs.ollama.com/api)
- [Importing models](https://docs.ollama.com/import)
- [Modelfile reference](https://docs.ollama.com/modelfile)

**Zerollama-specific:**

- [Development & upstream compare](docs/development.md) · [Upstream diff](docs/upstream-ollama-diff.md) · [Roadmap](docs/ROADMAP.md) · [Changelog](CHANGELOG.md)

### Building zerollama on macOS (ggml vendor @ b9781)

**Fresh clone (any path):** `./scripts/dev_bootstrap.sh` then `./zerollama serve`.

**Vendor sync (after pin bump or patch edit):** `make -f Makefile.sync clean apply-patches` then `./scripts/sync_vendor_llama.sh`. **Why not only `make sync`:** sync rsyncs whatever is in vendor — patches must be applied first; the script refuses bare tags.

**Phase 17 (optional upstream path):** `./scripts/build_ollama_llama_server_darwin.sh` then `./zerollama serve --llama-server-backend` — [phase17-llama-server.md](docs/phase17-llama-server.md). **Why optional on Mac:** ggml Metal ~+7% decode vs llama-server; Phase 17 is merge parity, not default engine swap.

**Why tiers, not one setup script:** sign-off needs pulled GGUFs; CI smokes bind Go **`:8080`** while daily serve uses **`:11434`**; `../llama.cpp` is cloned automatically on tier 0. **`zerollama doctor --fix`** clones the sibling and builds Metal libllama when missing — same order as `mac_setup`. Details: [mac-dev-setup.md](docs/mac-dev-setup.md) · ROADMAP [M14](docs/ROADMAP.md#apple-silicon--metal-track).

### Building zerollama on CUDA (RTX 5080-class / Proxmox CT)

**Start here:** [5080-runbook.md](docs/5080-runbook.md) + `source scripts/5080_env.sh` + `./scripts/5080_resignoff.sh --build`

```bash
# Inside CT 1564
cd ~/zerollama && source ./scripts/5080_env.sh
./scripts/5080_resignoff.sh --build          # full tiers 0–4
./scripts/5080_resignoff.sh --tier 2 --radix --vendor   # Radix live (optional)
```

Long reference (VRAM, remote serve, MLX): [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md)

**Production serve (remote `:8080`):** `cp scripts/serve_production_wrapper.sh ~/bin/serve.sh && ~/bin/serve.sh` — **WHY wrapper:** `serve_gpu_example.sh` in `~/bin` resolves repo as `$HOME`, not `~/zerollama`. See [5080-runbook — Production serve](docs/5080-runbook.md#production-serve-binserve-sh).

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
L3_SPEC_METHOD=ngram ./scripts/l3_spec_cache_smoke.sh   # optional: prefix-cache × spec policy
```

**Jun 2026 sign-off (CT 1564):** Phase 15 **PASS**; L2 **FAIL merge** @ 8k (stock faster); L3 **SOFT PASS**. Full playbook: [gpu-5080-operator-guide.md](docs/gpu-5080-operator-guide.md).

**MLX image generation (experimental):** build MLX-C once, set `OLLAMA_LIBRARY_PATH` to include `mlx_cuda_v12`, pull `x/z-image-turbo`, stop other models, then `zerollama run x/z-image-turbo "prompt"`. Default **384×384** on 16 GB CUDA — **why:** activations scale with pixels². Guide: [imagegen-zimage-turbo.md](docs/imagegen-zimage-turbo.md).

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
# Full gate + qwen35 (M4 Max, Jun 2026 PASS):
# eliza-1-* is the ship qwen35 family — 2B is the default sign-off tag (fast handoff/resume).
# RUN_E2E_QWEN35=1 RUN_E2E_QWEN35_MODEL=eliza-1-2b:latest ./scripts/metal_signoff.sh
# Alternate: RUN_E2E_QWEN35_MODEL=qwen3.6:latest (heavier; same gate shape)

# Optional: qwen35 ggml smoke only (needs :8080/:8081 stack or run via metal_signoff)
# RUN_E2E_QWEN35_MODEL=eliza-1-2b:latest ./scripts/qwen35_mac_smoke.sh
```

**Daily serve:** `zerollama serve` — Go `:11434` (or `:8080` in dev) + uv sidecar `:8081` with `apple_silicon.yaml` inprocess backend.

**GPU profile autotune (L1):** the runtime sidecar auto-merges tuned llama-server flags (`-b`, `-ub`, `-np`, `-fa`, …) from `runtime/configs/gpu/` based on **unified RAM** on Mac (`apple-silicon-16g` … `128g`). Check `curl -s :8081/health | jq .gpu_profile`. **Why:** Phase 13 estimates fit; L1 picks throughput knobs — a 128 GiB M4 Max should not share the same batch/parallel defaults as a 16 GiB Air. Disable: `ZEROLLAMA_GPU_PROFILE=0`. Doc: [gpu-profiles-l1.md](docs/gpu-profiles-l1.md).

**MLX safetensors (`gemma4`, Hermes agents):** long agent prompts are capped and tail-truncated server-side; pass `options.num_ctx` ≤ model max. If first token takes minutes, check logs for `num_ctx capped`, `prompt tail-truncated`, `prefill complete`. Guide: [mlx-agent-prompts.md](docs/mlx-agent-prompts.md).

**Prompt cache → slot bridge (L3):** pass a stable session key in request `options` (`eliza.conversationId`, `prompt_cache_key`, …) and the runtime pins llama-server `id_slot` + sets `cache_prompt: true` so repeat system prompts skip full prefill. Check `curl -s :8081/health | jq .llama_cache`. **Why:** agent threads re-send the same system prompt every turn — dynamic Phase 15 slots throw away KV on `complete()`. L3 maps keys → slots (needs `-np > 1` from L1). Batch rows use `prompt_cache_keys[]` (strict per-index — no silent fallback to a flat key). Disk cache lives under `~/.cache/zerollama/llama-cache/<modelHash>/`; hash canonicalizes GGUF paths and includes L2 cache types so fork/stock blobs never mix. Disable: `ZEROLLAMA_LLAMA_CACHE=0`. Sign-off: `./scripts/l3_cache_smoke.sh` + `./scripts/l3_gate_report.sh` (or `RUN_E2E_L3=1` on `m3_metal_signoff.sh`). **5080 Jun 2026:** SOFT PASS on 1B Q8 (bridge wired). Doc: [gpu-profiles-l3.md](docs/gpu-profiles-l3.md).

**Cross-slot Radix prefix share (L3 v1):** **Why:** L3 pins one slot per `prompt_cache_key`; two agents with the same system prompt but different keys hash to different slots and repeat prefill without cross-slot seed. **`ZEROLLAMA_L3_PROFILE=agent`** (or `ZEROLLAMA_RADIX_PREFIX_SHARE=1`) copies donor KV into cold targets via hash-chained block pool + vendor `POST /kv/seq-copy` (patch 0017). Requires `n_parallel > 1` and patched vendor llama-server — not bare sibling `../llama.cpp`. Live gate: `L3_RADIX_LIVE=1 ./scripts/l3_radix_prefix_smoke.sh` (Mac validated Jun 2026). **Not full RadixAttention:** no ref-count block DAG, no remote LMCache blobs, hybrid models skip copy, cold targets only — see [radix-prefix-share.md](docs/radix-prefix-share.md#product-gaps) and [ROADMAP — L3-R](docs/ROADMAP.md#radix-v2-l3-r--product-gaps).

**Decode graph invalidation (CUDA, vLLM-inspired):** when L3 clears a KV slot (SWA block, owner change, `cache_prompt=false`, draft drop-last-block fallback on subprocess), the runtime bumps a per-slot decode-graph epoch and clears ggml's captured CUDA graph cache. **In-process:** `llama_context_cuda_graph_invalidate` via native/ctypes on the live context. **Subprocess (default backend):** `POST /cuda-graph/invalidate` on the llama-server child — **why:** the child owns `ctx_tgt`; Python epoch alone cannot reach ggml in another process. **Why invalidation at all:** ggml keys captured graphs by compute topology, not sequence id — prefix reuse without breaking graphs can replay wrong GPU work. Check `curl -s :8081/health | jq .llama_cache.decode_graph`. Rebuild sibling `../llama.cpp` after pull (`./scripts/build_llama_server.sh`; CUDA: `GGML_CUDA_GRAPHS=ON`). Disable ggml clear: `ZEROLLAMA_DECODE_GRAPH_INVALIDATE=0`. On Metal the API is a no-op; epoch + L3 policy still run. Doc: [decode-graph-invalidation.md](docs/decode-graph-invalidation.md).

**Multimodal agent caches (SGLang-inspired, native path):** repeat `video_url` / video blobs on the same thread benefit from three layers — (1) HTTPS body LRU, (2) global ffmpeg expansion LRU, (3) session expansion cache when `prompt_cache_key` matches L3. Pass the key via `/api/chat` `options`, `/v1/chat/completions` `prompt_cache_key` or `options`, or Responses `prompt_cache_key`. OpenAI responses include `usage.prompt_tokens_details` (`image_tokens`, `video_tokens`, `cached_tokens` — heuristic modality counts; `cached_tokens` from L3). Vision preflight runs before ffmpeg; pre-expanded `video_spans` count on the latest user turn only so echoed history does not false-reject follow-ups.

**`padded_input_ids` (SGLang preprocessed clients):** pretokenized layouts on the latest user message are spliced into the rendered template and consumed at the vision runner — not re-tokenized from text. **Why:** SGLang clients already computed vision token positions; re-tokenizing duplicates or misplaces placeholders. On **Mac Metal (ollama-engine default)** all native Go VLM families inject in-process: Qwen3-VL (+ qwen25vl/qwen2vl), Gemma4, mllama, Gemma3, Llama4, LFM2, GLM-OCR, Mistral3, DeepSeek-OCR. Grep `padded_layout_consume=<mode>_runner_inject` or `padded_input_ids runner inject` (`engine=ollama` on ollama-engine). **Why tool-aware:** tool results render as pseudo user blocks — they must not count as splice spans or multi-turn inject breaks. Grep `deferred_multimodal_history` or `padded_input_ids splice failed` if agent VLMs lose images after tool calls.

**`grid_thw` hints (partial):** optional `[T,H,W]` on `video_spans` flows to per-frame `llm.ImageData.GridTHW` for preflight, usage, and post-encode debug compare (`vision grid hints`). llamarunner passes hints into `MultimodalTokenize(..., gridTHW)` — mtmd still derives layout from pixels until upstream C API lands. **Why:** when ffmpeg resize differs from the client processor, embed count ≠ hint → padded inject misaligns; operators need visibility before upstream fixes land.

**`precomputed_embedding` + `processor_output` (SGLang preprocessed ingest — partial):** clients that already ran the HF processor or ViT can send **`padded_input_ids`** plus either post-projector **feature rows** (`precomputed_embedding`) or raw **`pixel_values` + `image_grid_thw`** (`processor_output`) instead of PNG bytes. **Why:** agent threads on edge nodes should not decode PNG and re-encode ViT when SGLang already materialized tensors — server runs only vision-tower + LLM prefill (ollama-engine) or splices embed rows at padded slots. **ollama-engine** covers all native Go VLMs (family matrix in [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md) §7c–§7d); **ggml llamarunner** accepts precomputed embed chunks on padded inject; **llama-server** rejects both (subprocess needs base64 rasters). Grep `precomputed_embedding runner inject` or `processor_output runner inject` (`engine=ollama`).

**`enable_prefix_mm_cache` (SGLang session ViT pin):** set **`prompt_cache_key`** on every agent turn (OpenAI top-level or `/api/chat` `options`). Session ViT overlay defaults **ON** with the key; set **`enable_prefix_mm_cache: false`** to disable session pin and use global LRU only. **Why:** global ViT LRU (4–64 slots) can evict clip frames between turns even when ffmpeg/session expansion caches hit — SGLang keeps encoder outputs hot per conversation; zerollama mirrors that with a per-runner session overlay keyed like L3. Without `prompt_cache_key`, overlay stays off (setting the flag alone logs a hint).

**Why caches + padded inject + hints + preprocessed ingest:** agents re-send the same clip every turn; without caches ffmpeg and CDN dominate latency; without keys session cache does not pin per thread; without scoped preflight multi-turn chats fail incorrectly; without splice tool loops silently drop vision layout; without grid hints layout drift is invisible until quality degrades; without preprocessed paths clients pay duplicate ViT work on every turn.

Smoke: `./scripts/video_expand_cache_smoke.sh` (unit), `./scripts/video_agent_cache_smoke.sh` (expand + session cache, `RUN_E2E_VIDEO_AGENT=1`), `./scripts/video_agent_infer_smoke.sh` (live VLM + turn-2 `cached_prompt_tokens`, `RUN_E2E_VIDEO_AGENT_INFER=1`; optional `VIDEO_AGENT_INFER_PREPROC=1` + `VIDEO_AGENT_GO_LOG` for padded layout restore; optional `VIDEO_AGENT_INFER_PREFIX_MM_WARN=1` for prefix-mm hint grep). Doc: [sglang-multimodal-borrowings.md](docs/sglang-multimodal-borrowings.md), [video-understanding.md](docs/video-understanding.md), [mtmd-grid-thw-handoff.md](docs/mtmd-grid-thw-handoff.md).

**Fork evaluation (L2):** build eliza `llama-server` sibling + A/B against stock. `./scripts/l2_full_gate.sh` (Mac) or `./scripts/l2_cuda_full_gate.sh` (CUDA). **Why:** QJL/TurboQuant live in `elizaOS/llama.cpp` — vendor merge is gated on measured wins. Metal Jun 2026: stock wins decode at 8k/27k; fork runs 131k ctx. **CUDA 5080 Jun 2026:** stock wins @ 8k (FAIL merge); long-ctx legs pending. Doc: [gpu-profiles-l2.md](docs/gpu-profiles-l2.md).

**Why rebuild after pull:** Jun 2026 fixed Mac **GPU bootstrap discovery** (`total_vram="0 B"` / CPU-only offload) and **Go ollama-engine sched_reserve** for qwen35moe (`GGML_ASSERT(tensor->buffer == NULL)` abort). Rebuild so `/info` uses `DiscoverBackendDevices()` and graph tensors defer to the scheduler — see [apple-silicon-metal.md](docs/apple-silicon-metal.md#gpu-bootstrap-discovery-jun-2026) and [sched_reserve](docs/apple-silicon-metal.md#go-ollama-engine-sched_reserve-jun-2026).

**Context length on Mac (ggml):** keep manifest `num_ctx` modest (4096); use **`options.num_ctx` per request** for long context — manifest defaults pre-allocate KV at load and very large values can hang. `/api/ps` shows the **loaded** runner, not `/api/show`. After `/api/create` or `zerollama stop`, confirm with empty `/api/ps`. See [qwen35-apple-silicon.md](docs/qwen35-apple-silicon.md#manifest-num_ctx-vs-request-optionsnum_ctx-jun-2026).

**GGUF guess + scheduler hardening (LocalAI borrowings):** new pulls and creates auto-fill arch, capped manifest `num_ctx` (8192), parser, and stops from GGUF headers; pull also rewrites manifest metadata after download. A background watchdog can reclaim VRAM and evict stuck runners. **Why:** train-context manifests and multi-model agents were the top operator footguns. Existing tags: `zerollama repair MODEL --write` or `POST /api/repair`. Agent routing without generation: `POST /api/score` (joint log-prob of candidate continuations). Tight GPUs: `PARAMETER concurrency_groups ["vram-heavy"]` on conflicting models (e.g. imagegen + chat). Doc: [localai-borrowings.md](docs/localai-borrowings.md). Env: `ZEROLLAMA_MEMORY_RECLAIM_THRESHOLD=0.95`, `ZEROLLAMA_DISABLE_GGUF_GUESS=1`.

**Faster inference-only startup** (skip training embed + blob prune):

```bash
OLLAMA_TRAINING=false OLLAMA_NOPRUNE=1 ./zerollama serve
```

**GPU training venv (when `OLLAMA_TRAINING=true`):** packages live in `$REPO/.venv-training/lib/pythonX.Y/site-packages` where **X.Y must match** the libpython linked into your `zerollama` binary (`ldd $(which zerollama) | grep libpython`). **Why:** embedded CPython loads torch from `PYTHONPATH`, not the venv interpreter — ABI mismatch fails at startup with `training worker not started`. **Linux 5080:** prefer **3.11** embed + venv (same as `runtime/.venv`) via [`scripts/training_embed_build_env.sh`](scripts/training_embed_build_env.sh) before `go build`. Setup: [`scripts/training_uv_venv.sh`](scripts/training_uv_venv.sh) (`--embed-py`, `--verify`); **production serve:** `cp scripts/serve_production_wrapper.sh ~/bin/serve.sh` (do **not** copy `serve_gpu_example.sh` to `~/bin` — breaks repo root). After migration, remove legacy `venv-training/` (~7 GiB). Details: [gpu-training.md](docs/gpu-training.md#installing-python-deps-embedded-interpreter).

### LM Studio cache (reuse local downloads)

**Why:** LM Studio and zerollama often share a Mac; re-downloading 30–70 GB weights from the registry wastes time and disk when `~/.lmstudio/models` already has them.

```bash
./zerollama list                                    # includes discoverable LM Studio caches; TOK/S when bench cache exists
./zerollama bench                                   # measure decode tok/s → ~/.ollama/bench.json
./zerollama pull lmstudio-community/gemma-4-31b-it:q8_0   # registers from cache when matched
OLLAMA_LMSTUDIO_LIST_ALL=1 ./zerollama serve      # list MLX models even when disk is tight
```

- **GGUF:** symlinked into `OLLAMA_MODELS` (near-zero extra space).
- **MLX safetensors** (`config.json` + weights): repacked into zerollama blobs (~full model size free required).
- **Pull** fails early with a clear disk error if MLX import cannot fit.

Full rationale, env vars, and troubleshooting: [docs/lmstudio-import.md](docs/lmstudio-import.md).

### Model throughput in `list` (`zerollama bench`)

**Why:** Disk size and parameter labels do not predict decode speed on your GPU. **`zerollama bench`** runs a short generate benchmark per local model and caches tok/s to **`~/.ollama/bench.json`** (keyed by digest so re-pulls reset stale numbers). **`zerollama ls`** shows a **TOK/S** column — `--` until you bench.

```bash
./zerollama bench              # all local text models (skips embed/image-only tags)
./zerollama bench llama3.2     # prefix filter
./zerollama bench --force      # re-bench cached models
./zerollama ls                 # NAME … TOK/S … MODIFIED
```

Numbers reflect **this machine** (backend, VRAM, serve flags), not cloud models. For CI-grade A/B use `cmd/bench/bench.go` or Phase 17 / L1 scripts. Doc: [docs/bench-cache.md](docs/bench-cache.md).

**Why runtime header on proxy:** Pulled model names may route to legacy ggml and contend with the sidecar on one Metal device — use `X-Zerollama-Runtime: 1` or runtime-default manifest backend. See [apple-silicon-metal.md](docs/apple-silicon-metal.md#scheduler-errors-http-status).

### In this repository

- [Eliza Cloud / Zerollama remote inference](docs/eliza-cloud.md) — **why** Eliza is the default upstream (OpenAI/Anthropic APIs + API keys), **why** legacy Ed25519 signing is limited to `ollama.com`, path rewrites, catalog merge, and when responses are raw upstream JSON.
- [Video understanding (VLM)](docs/video-understanding.md) — **why** OpenAI `video_url` merges into one message, **why** ffmpeg samples to frames, security (HTTPS, SSRF), native **fps/stride** sampling, context preflight, expansion caches, and optional SGLang proxy.
- [SGLang multimodal borrowings](docs/sglang-multimodal-borrowings.md) — **why** native path adopted pooled fetch, expansion LRU, session cache, padded inject, precomputed/processor ingest, OpenAI usage breakdown, `cached_tokens`, multi-turn preflight scoping, and OpenAI `prompt_cache_key` without requiring SGLang.
- [Wan text-to-video (T2V)](docs/wan-t2v.md) — **why** async `/v1/videos` uses the training `run_script` queue (not GGUF chat), **why** checkpoints install separately, defer ids, and artifact paths.
- [MLX image generation (Z-Image Turbo)](docs/imagegen-zimage-turbo.md) — **why** diffusion uses an MLX subprocess (not ggml/runtime); 16 GB CUDA staged VRAM; `zerollama run x/z-image-turbo`; build `libmlxc.so` + `patch_mlx_cuda_vram.sh`.
- [Multimodal / video backends](docs/multimodal-backends.md) — **why** env vars and manifest `config.json` both exist; Whisper, Piper, and **OLLAMA_VIDEO_*** for native video.
- [Video parity matrix](docs/video-parity.md) — **why** reference workloads and a comparison table for Option 2 (native vs optional SGLang).
- [Changelog](CHANGELOG.md) — what changed and **why** it matters for operators.
- [Phase 17 llama-server path](docs/phase17-llama-server.md) — **why** upstream Go→llama-server is ported but Mac keeps ggml default
- [Phase 16 thin edge daemon](docs/phase16-thin-edge.md) — **why** `--edge` / `-tags edge` for upstream-shaped deploys without dropping training/Eliza
- [Upstream Ollama comparison](docs/upstream-ollama-diff.md) — **why** vanilla Ollama uses Go→llama-server for GGUF; pin gaps; cherry-pick map vs zerollama Python runtime and training.
- [llama.cpp backend (experimental)](docs/llama-cpp-backend.md) — `--llama-cpp-backend` routes text GGUF through Python runtime + sibling llama.cpp; benchmark vs ggml and upstream.
- [ggml @ b9611 migration](docs/ggml-b9509-migration.md) — **why** in-process ggml uses a pinned vendor tree + 14 reviewable patches (not overlay snapshots); ahead of vanilla Ollama b9509; sync, Ollama deltas, Mac sign-off checklist.
- [Scheduling, VRAM, and queue policy](docs/scheduling-vram-policy.md) — **why** inference and training are not one FIFO; VRAM broker; T6 `defer-*` queue; runtime VRAM heuristics (NVML, GGUF metadata); **ggml unload / manifest `num_ctx` at load**; **M12 ggml `suggested_max_num_ctx` + opt-in clamp** (parity with Phase 13); prompt truncation fields.
- [LocalAI control-plane borrowings](docs/localai-borrowings.md) — **why** fast GGUF metadata, manifest guess, scheduler watchdog, concurrency groups, fleet score, and **`zerollama repair`** for existing tags.
- [Model bench cache](docs/bench-cache.md) — **why** `zerollama bench` persists decode tok/s to `~/.ollama/bench.json` and surfaces **TOK/S** in `zerollama ls`.
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
- [5080 runbook — what to run](docs/5080-runbook.md) — **ordered CUDA gate tiers** after pull (base → L1/L3 → Phase 15 → `RUN_E2E_UPSTREAM_GGUF=1`); CT 1564 status; Mac counterpart `metal_signoff.sh`.
- [GPU 5080 operator guide](docs/gpu-5080-operator-guide.md) — **why** `gpu_5080_session.sh` is the single-GPU gate; Proxmox CT layout; **`OLLAMA_HOST=0.0.0.0:8080`** for remote clients; **CGO `cpp-httplib` vendoring**; **`RUN_E2E_PREFLIGHT=0`** when httplib missing; L1/L3 full gates; Phase 15 + L2 sign-off sequence.
- [L2 eliza fork evaluation](docs/gpu-profiles-l2.md) — **why** QJL/Polar/TBQ via **one** `../llama.cpp` @ `LLAMA_CPP_COMMIT`; L1 vs fork = profile argv (`ZEROLLAMA_LLAMA_FORK`), not separate siblings. Doc: [llama-cpp-unification.md](docs/llama-cpp-unification.md).
- [L3 prompt cache → slot bridge](docs/gpu-profiles-l3.md) — **why** stable session keys skip repeat prefill; pinned `id_slot` + disk TTL; batch `prompt_cache_keys`; SWA/draft-spec policy; **5080 Jun 2026:** STRICT PASS @ 8k + production gate PASS @ 27k on eliza-1 9B.
- [Cross-slot Radix prefix share](docs/radix-prefix-share.md) — **why** same system prompt across different cache keys; donor slot KV seed + block pool verification; vendor `POST /kv/seq-copy`; live smoke `l3_radix_prefix_smoke.sh`; **[product gaps](docs/radix-prefix-share.md#product-gaps)** (v1 vs full RadixAttention).
- [Decode graph invalidation](docs/decode-graph-invalidation.md) — **why** L3 slot clears must break ggml CUDA graphs; epoch scaffold + in-process invalidate + subprocess `POST /cuda-graph/invalidate`; rebuild sibling llama-server; Metal no-op note.
- [vLLM borrowings (L3)](docs/vllm-borrowings.md) — **why** slot-level prefix cache vs vLLM block pool; taken vs deferred; `cache_salt`, drop-last-block, SWA retention, subprocess graph HTTP.
- [Apple Silicon & Metal](docs/apple-silicon-metal.md) — **why** unified memory ≠ CUDA VRAM; ggml Metal default; runtime `metal-unified` probe; **L1 GPU profiles** (RAM tiers); Darwin Metal contention policy; scheduler 400/503 errors; **GPU bootstrap discovery**; **Go engine sched_reserve** (qwen35moe); **Jun 2026 sign-off** (`metal_signoff.sh` + **`eliza-1-2b:latest`** qwen35; `qwen35_mac_smoke.sh`).
- [L1 GPU profiles (autotune)](docs/gpu-profiles-l1.md) — **why** Phase 13 ≠ throughput tuning; **`l1_cuda_full_gate.sh`** concurrent **PASS** on 5080 (+~16–20%); NVIDIA buckets; Apple RAM tiers.
- [Qwen 3.5/3.6 on Apple Silicon](docs/qwen35-apple-silicon.md) — **why** qwen35 hits compat metadata + Metal embed layers; **Go ollama-engine default on Mac** (Jun 2026); `PrimaryFamily()` for VL; thinking-model API fields; opt-in `qwen35_mac_smoke.sh`.
- [MLX routing policy](docs/mlx-routing-policy.md) — when to use ggml Metal vs runtime vs mlxrunner; `IsMLX()` guards; LM Studio MLX disk import policy.
- [MLX agent prompts](docs/mlx-agent-prompts.md) — **why** agent megaprompts need context cap, tail truncate, single tokenize, tokenize cache, keep-alive floor, and SSE keepalive; operator log field guide.
- [LM Studio cache import](docs/lmstudio-import.md) — **why** pull-from-cache, **why** MLX copies vs GGUF symlinks, disk policy, `OLLAMA_LMSTUDIO_LIST_ALL`, operator troubleshooting.

## Community integrations

Zerollama is API-compatible with Ollama — the ecosystem below works against `./zerollama serve` on `:11434`.

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
