# Agents — zerollama

Short map for Cursor / coding agents. Deep environment detail lives in `.cursor/rules/zerollama-bootstrap.mdc` and production ports in `.cursor/rules/ollama-default-port.mdc`.

## Ollama gallery compatibility stamp

**`version.OllamaCompat`** in [`version/version.go`](version/version.go) is the semver we send to `registry.ollama.ai` (`User-Agent: ollama/<OllamaCompat>`). Bump it when syncing upstream Ollama or when pulls 412 for “newer version”. Git identity stays in **`version.Commit`** only. Agent rule: [`.cursor/rules/ollama-compat-version.mdc`](.cursor/rules/ollama-compat-version.mdc).

## Sibling upstream trees (weekly pull)

Canonical paths and the weekly ritual: **[docs/upstream/siblings.md](docs/upstream/siblings.md)**.

Strategic watchlist: **[docs/upstream/tracker.md](docs/upstream/tracker.md)**.

On the Mac lab host (`~/Sites/inference/zerollama`):

| Need | Checkout |
|------|----------|
| vLLM patterns | `../vllm` → [docs/upstream/vllm-borrowings.md](docs/upstream/vllm-borrowings.md) |
| LocalAI control plane | `../LocalAI` → [docs/upstream/localai-borrowings.md](docs/upstream/localai-borrowings.md) |
| SGLang multimodal | `../sglang` → [docs/upstream/sglang-multimodal-borrowings.md](docs/upstream/sglang-multimodal-borrowings.md) |
| MiniMax Music 3 rematch (optional clone) | `../sglang-omni` — **not** in sibling `sglang`; Mac hear path is mlx-audio ([docs/design/media/music-c.md](docs/design/media/music-c.md), [findings](docs/design/media/music-c-findings.md)). Do **not** vendor ComfyUI. |
| llama.cpp pin / fork watch | `../llama.cpp`, `../eliza-llama.cpp` → [runtime/LLAMA_CPP_PIN.md](runtime/LLAMA_CPP_PIN.md) |
| Gigatoken / BMTL techniques (do **not** vendor Rust) | `../bmtl` → [docs/upstream/faster-bpe-tokenize.md](docs/upstream/faster-bpe-tokenize.md) + [findings](docs/upstream/faster-bpe-tokenize-findings.md) (unicode/BPE pretok in `llama/patches/` through **0105** port of islands/SWAR; must reach **`llama/llama.cpp/`** via sync — note **0122–0126** filenames are **m4-prefill stubs**, not BPE) |
| M4 Metal prefill kernels (do **not** vendor host engine) | `../m4-prefill-engine` → [docs/upstream/m4-prefill-borrowings.md](docs/upstream/m4-prefill-borrowings.md) + [findings](docs/upstream/m4-prefill-borrowings-findings.md) (durable: MTL FA fix; FA+q8_0 KV useful; native Q8 FA / fused SwiGLU regress on M4 Max; lab: `scripts/phase/m4_prefill_lab_serve.sh`) |
| Vanilla Ollama diff | `../ollama-upstream` → [docs/upstream/ollama-diff.md](docs/upstream/ollama-diff.md) |
| DeepGEMM (optional; CUDA ideas only) | `../DeepGEMM` → [docs/upstream/deepgemm-borrowings.md](docs/upstream/deepgemm-borrowings.md) · [findings](docs/upstream/deepgemm-borrowings-findings.md) — **do not** vendor; skip Ascend / Mega MoE EP; host unit `./scripts/phase/deepgemm_borrowings_unit.sh` |
| DeepInfer (optional; SM100 engine; ideas only) | `../DeepInfer` (astra: `/mnt/ssd2/src/DeepInfer`) → [docs/upstream/deepinfer-borrowings.md](docs/upstream/deepinfer-borrowings.md) · ladder [docs/upstream/deepinfer-ladder.md](docs/upstream/deepinfer-ladder.md) (**DI0–DI8**) — **do not** vendor; skip FlashMLA/DeepEP wheels |
| Flare (optional; L3 policy ideas only) | `../flare-local-inference` → [docs/upstream/flare-borrowings.md](docs/upstream/flare-borrowings.md) — **do not** vendor Helios/Ignition/FlashInfer; SM120 agent-cache rematch |
| MiniMax-H3 Metal speed (optional) | `../h3-metal` → [docs/upstream/h3-metal-borrowings.md](docs/upstream/h3-metal-borrowings.md) — FastH3 sigma table is in video-c; **do not** replace the ConvRot host path or Wan2GP |

## Do not touch

- Production listeners on **11434** / **8081** — lab ports only (`11435`, `18081`, …).
- **PVE hypervisor (ryzen7950x0):** never start or bind `zerollama` / `llama-server`. Serve only inside **CT 1564** (`pct exec 1564`, clients `192.168.255.164:8080`).
- Remote model storage serve defaults to **`:18090`** (lab). Never point `storage serve` at inference ports. LAN discover: `ZEROLLAMA_MDNS=1` → `_zerollama-storage._tcp`; clients `zerollama storage discover` / `storage doctor`. Guide: [docs/design/surfaces/remote-model-storage.md](docs/design/surfaces/remote-model-storage.md).

## Mac setup (fresh clone)

Canonical guide: **[docs/runbooks/mac-dev-setup.md](docs/runbooks/mac-dev-setup.md)**.

**Prerequisites (once per machine):**

| Need | Why |
|------|-----|
| **Go ≥ 1.24.1** | `go.mod` — docs that say 1.22+ are wrong |
| **Full Xcode.app** (or Homebrew `python@3.12` + `pkg-config`) | CGO needs `python3-embed`; `mac_cgo_env.sh` looks under `/Applications/Xcode.app/...` — CLI tools alone often fail |
| **cmake** | Default bootstrap builds sibling Metal `libllama` / `llama-server` |
| **uv** | `runtime/.venv` |

**Tier 0 (no models required):**

```bash
./scripts/runtime/dev_bootstrap.sh   # host deps + ggml build + ~/.local/bin install
zerollama serve                      # :11434 + sidecar :8081 on Darwin
zerollama pull llama3.2:3b
```

| Script layout (post-reorg) | Use this |
|----------------------------|----------|
| Bootstrap | `scripts/runtime/dev_bootstrap.sh` / `mac_setup.sh` |
| Host deps (no Homebrew required) | `scripts/runtime/mac_host_deps.sh` |
| Install to PATH | `scripts/install_mac.sh` (`PREFIX=~/.local`) |
| Mac CGO build | `scripts/build/build_zerollama_mac.sh` |
| Runtime venv | `scripts/runtime/runtime_uv_venv.sh` |
| CGO vendor headers | `scripts/vendor/ensure_llama_cgo_vendor.sh` |
| Sibling llama.cpp | `scripts/vendor/ensure_llama_cpp_sibling.sh` |
| Metal sign-off | `scripts/gpu/metal_signoff.sh` (CI `:8080`, not production `:11434`) |

**Pin:** ggml-org **`b12119`** (`6f1e6543704bc5b080c7a98e7b6f707b31f6ed51`) — [runtime/LLAMA_CPP_PIN.md](runtime/LLAMA_CPP_PIN.md) · L6 ladder [docs/design/serving/llama-cpp-pin-ladder.md](docs/design/serving/llama-cpp-pin-ladder.md). Rollback **b11870**. Sibling clone defaults to elizaOS for fork kernels; public pin checkout: `LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git`. Patch series **0001–0163** (163 files) is rebased onto this tag. **0162** cherry-picks ggml-org `#30541`. **0163** loads the full gemma4-assistant draft stack. Production `llama-server` is `vendor/llama-cpp-b12119/build/bin/llama-server`.

**Optional (skip cleanly):** `../mlx` (safetensors), `../bmtl/.../uma_toolkit` (UMA), training venv (`MAC_SETUP_TRAINING=1`). DeepSeek-V4 Flash mlx-lm packs: [docs/research/mlx-deepseek-v4-flash.md](docs/research/mlx-deepseek-v4-flash.md) — `create --experimental --link`, do not blob-copy.

**Ggml-only (skip llama.cpp build):** `MAC_SETUP_BUILD=0 MAC_SETUP_LLAMA_OPTIONAL=1 ./scripts/runtime/dev_bootstrap.sh` — chat on `:11434` works; runtime inprocess on `:8081` wants `libllama.dylib`.

## Roadmap / status

- [docs/ROADMAP.md](docs/ROADMAP.md)
- [docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md)
- Detailed track roadmaps: [core serving](docs/roadmaps/core-serving.md) · [surfaces/integrations](docs/roadmaps/surfaces-and-integrations.md) · [training/scheduling](docs/roadmaps/training-and-scheduling.md) · [media](docs/roadmaps/media.md)
- [docs/README.md](docs/README.md) — index of in-repo guides
- Training follow-ons (T2/T5/T6 + T9–T11): [ROADMAP § GPU training](docs/ROADMAP.md#gpu-training-fine-tuning) · [t10-grpo.md](docs/design/training/t10-grpo.md) (+ [findings](docs/design/training/t10-grpo-findings.md)) · [t11-recipes.md](docs/design/training/t11-recipes.md) (presets + `POST /api/train/recipes`) · [gpu-training.md](docs/design/training/gpu-training.md)
- Doctor Modelfile repair (empty `response` / slash-collapse / ChatML hygiene): `zerollama doctor --repair-models [--all-local] [--apply]` — **why** not `doctor --fix`: [docs/runbooks/doctor-model-repair.md](docs/runbooks/doctor-model-repair.md)
- Hermes `/v1` gaps (M15e/M15f): [docs/design/surfaces/hermes-zerollama-gap.md](docs/design/surfaces/hermes-zerollama-gap.md) (§8 batch wire) · [findings](docs/design/surfaces/hermes-gap-closure-findings.md) · OpenAPI `server/openapi/openapi.yaml` (`ChatCompletionsBatchResponse`)
- MultiDecode (forest decode): patches **0127–0129**; `POST /v1/multidecode` + `capabilities.multidecode`; silent LCP pack on Hermes batch — [docs/design/surfaces/multidecode-llama-cpp.md](docs/design/surfaces/multidecode-llama-cpp.md) · [findings](docs/design/surfaces/multidecode-llama-cpp-findings.md) · skill `agentskills/multidecode/`
- Typed decisions (Laya + CLM + OpenJev/DiffusionGemma): `POST /v1/decisions` — Laya GGUF patches **0125–0126**; CLM native heads; DiffusionGemma sibling PR `#24427` (**DG0–DG9** choice+noul+score marker slots; opt-in `CALIBRATED` / `NOUL_CALIBRATED`) — [docs/design/surfaces/laya-llama-cpp.md](docs/design/surfaces/laya-llama-cpp.md) · [docs/design/surfaces/clm.md](docs/design/surfaces/clm.md) · [docs/design/surfaces/diffusion-gemma-llama-cpp.md](docs/design/surfaces/diffusion-gemma-llama-cpp.md) · [findings](docs/design/surfaces/diffusion-gemma-llama-cpp-findings.md) · skill `agentskills/typed-decisions/`
- Entity extract (GLiNER): `POST /v1/extract` + `POST /v1/gliner` — C++/ONNX sibling; dual abstract/mechanical wire; **GL0–GL5a** (CUDA ORT + token multitask; RelEx Parked) — [docs/design/surfaces/gliner-cpp.md](docs/design/surfaces/gliner-cpp.md) · [findings](docs/design/surfaces/gliner-cpp-findings.md) · skill `agentskills/entity-extract/`
- GLiNER2.5-Decide (triage): `POST /v1/decisions` + `POST /v1/gliner-decide` — Python `gliner2` sibling; **not** NER; `ZEROLLAMA_GLINER_DECIDE_URL` — [docs/design/surfaces/gliner-decide.md](docs/design/surfaces/gliner-decide.md) · [findings](docs/design/surfaces/gliner-decide-findings.md)
- Robot perception: `POST /v1/detect` (`ZEROLLAMA_DETECT_URL`, OWLv2, `wire_only` when unset) and `POST /v1/segment` (`ZEROLLAMA_SEGMENT_URL`, `grasp_ready` false) — [docs/robot-perception-ladder.md](docs/robot-perception-ladder.md) · [docs/detect.md](docs/detect.md) · [docs/segment.md](docs/segment.md) · skills `object-detect` / `object-segment`
- Product diffs vs Ollama (README hero): **megaprompts** (Gigatoken-inspired tokenize + L3) + visuals + harness — [README.md § Why](README.md#2-why-zerollama) · [Tour](README.md#4-tour--what-makes-us-different)
- MiniMax Music 3 (mlx hear → C later): [docs/design/media/music-c.md](docs/design/media/music-c.md) — **why** not Comfy / not H3 VAE / not blocking TTS
- GLM Flash Lite (EXL3 sidecar): **GF0–GF7** — astra `127.0.0.1:30000` + tag `glm-5.3-flash` (`openai-remote` `/v1`+`/api/chat`); GF6 parked; rematch [freetoken-moe-lab.md](docs/research/freetoken-moe-lab.md); **do not** vendor EXL3 — [docs/research/glm-flash-lite-ladder.md](docs/research/glm-flash-lite-ladder.md) · ROADMAP **M28**
- DeepInfer borrowings: **DI0–DI8** — **DI7** (**0140**); **DI6**; **DI4**; **DI5**; **DI2** design; **DI3** (**0141** verified `accept_by_pos`) — [docs/deepinfer-ladder.md](docs/upstream/deepinfer-ladder.md)
