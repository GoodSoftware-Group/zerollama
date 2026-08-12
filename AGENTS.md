# Agents — zerollama

Short map for Cursor / coding agents. Deep environment detail lives in `.cursor/rules/zerollama-bootstrap.mdc` and production ports in `.cursor/rules/ollama-default-port.mdc`.

## Sibling upstream trees (weekly pull)

Canonical paths and the weekly ritual: **[docs/upstream-siblings.md](./docs/upstream-siblings.md)**.

On the Mac lab host (`~/Sites/inference/zerollama`):

| Need | Checkout |
|------|----------|
| vLLM patterns | `../vllm` → [docs/vllm-borrowings.md](./docs/vllm-borrowings.md) |
| LocalAI control plane | `../LocalAI` → [docs/localai-borrowings.md](./docs/localai-borrowings.md) |
| SGLang multimodal | `../sglang` → [docs/sglang-multimodal-borrowings.md](./docs/sglang-multimodal-borrowings.md) |
| llama.cpp pin / fork watch | `../llama.cpp`, `../eliza-llama.cpp` → [runtime/LLAMA_CPP_PIN.md](./runtime/LLAMA_CPP_PIN.md) |
| Gigatoken / BMTL techniques (do **not** vendor Rust) | `../bmtl` → [docs/faster-bpe-tokenize.md](./docs/faster-bpe-tokenize.md) + [findings](./docs/faster-bpe-tokenize-findings.md) (patches **0106–0126** in `llama/patches/`; must reach **`llama/llama.cpp/`** via sync) |
| Vanilla Ollama diff | `../ollama-upstream` → [docs/upstream-ollama-diff.md](./docs/upstream-ollama-diff.md) |

## Do not touch

- Production listeners on **11434** / **8081** — lab ports only (`11435`, `18081`, …).
- Remote model storage serve defaults to **`:18090`** (lab). Never point `storage serve` at inference ports. Operator guide + WHYs: [docs/remote-model-storage.md](./docs/remote-model-storage.md).

## Mac setup (fresh clone)

Canonical guide: **[docs/mac-dev-setup.md](./docs/mac-dev-setup.md)**.

**Prerequisites (once per machine):**

| Need | Why |
|------|-----|
| **Go ≥ 1.24.1** | `go.mod` — docs that say 1.22+ are wrong |
| **Full Xcode.app** (or Homebrew `python@3.12` + `pkg-config`) | CGO needs `python3-embed`; `mac_cgo_env.sh` looks under `/Applications/Xcode.app/...` — CLI tools alone often fail |
| **cmake** | Default bootstrap builds sibling Metal `libllama` / `llama-server` |
| **uv** | `runtime/.venv` |

**Tier 0 (no models required):**

```bash
./scripts/runtime/dev_bootstrap.sh
./zerollama serve          # :11434 + sidecar :8081 on Darwin
./zerollama pull llama3.2:3b
```

| Script layout (post-reorg) | Use this |
|----------------------------|----------|
| Bootstrap | `scripts/runtime/dev_bootstrap.sh` / `mac_setup.sh` |
| Mac CGO build | `scripts/build/build_zerollama_mac.sh` |
| Runtime venv | `scripts/runtime/runtime_uv_venv.sh` |
| Sibling llama.cpp | `scripts/vendor/ensure_llama_cpp_sibling.sh` |
| Metal sign-off | `scripts/gpu/metal_signoff.sh` (CI `:8080`, not production `:11434`) |

**Pin:** ggml-org **`5f55650a`** (past tag **b10199**) — [runtime/LLAMA_CPP_PIN.md](./runtime/LLAMA_CPP_PIN.md). Sibling clone defaults to elizaOS for fork kernels; public pin checkout: `LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git`.

**Optional (skip cleanly):** `../mlx` (safetensors), `../bmtl/.../uma_toolkit` (UMA), training venv (`MAC_SETUP_TRAINING=1`).

**Ggml-only (skip llama.cpp build):** `MAC_SETUP_BUILD=0 MAC_SETUP_LLAMA_OPTIONAL=1 ./scripts/runtime/dev_bootstrap.sh` — chat on `:11434` works; runtime inprocess on `:8081` wants `libllama.dylib`.

## Roadmap / status

- [docs/ROADMAP.md](./docs/ROADMAP.md)
- [docs/README.md](./docs/README.md) — index of in-repo guides
- Training borrowings (Unsloth → T7–T11): [ROADMAP § GPU training](./docs/ROADMAP.md#gpu-training-fine-tuning)
- Doctor Modelfile repair (empty `response` / slash-collapse / ChatML hygiene): `zerollama doctor --repair-models [--all-local] [--apply]` — **why** not `doctor --fix`: [docs/doctor-model-repair.md](./docs/doctor-model-repair.md)
- Hermes `/v1` gaps (M15e/M15f): [docs/hermes-zerollama-gap.md](./docs/hermes-zerollama-gap.md) (§8 batch wire) · [findings](./docs/hermes-gap-closure-findings.md) · OpenAPI `server/openapi/openapi.yaml` (`ChatCompletionsBatchResponse`)
- Product diffs vs Ollama (README hero): **megaprompts** (Gigatoken-inspired tokenize + L3) + visuals + harness — [README.md § Why](./README.md#2-why-zerollama) · [Tour](./README.md#4-tour--what-makes-us-different)
