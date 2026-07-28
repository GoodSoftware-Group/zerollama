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
| Vanilla Ollama diff | `../ollama-upstream` → [docs/upstream-ollama-diff.md](./docs/upstream-ollama-diff.md) |

## Do not touch

- Production listeners on **11434** / **8081** — lab ports only (`11435`, `18081`, …).

## Roadmap / status

- [docs/ROADMAP.md](./docs/ROADMAP.md)
- [docs/README.md](./docs/README.md) — index of in-repo guides
