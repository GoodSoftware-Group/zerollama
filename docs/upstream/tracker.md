# Upstream Tracker

This file is the **strategic tracker** for upstream repos we follow, diff, and selectively borrow from.

Use it together with:

- [upstream-siblings.md](siblings.md) for **checkout paths**, **cadence**, and the **last-checked log**
- [open-source-shoutouts.md](open-source-shoutouts.md) for **credit** to the projects we borrowed from
- [RESEARCH_ROADMAP.md](../RESEARCH_ROADMAP.md) for **active exploratory bets**
- [ROADMAP.md](../ROADMAP.md) and [roadmaps/](../roadmaps) for **product priorities**
- [CHANGELOG](../../CHANGELOG.md) for what actually landed

## Status legend

- `dependency` — upstream directly shapes shipped zerollama behavior or pins
- `active borrow` — currently a meaningful source of patterns / ideas / selective cherry-picks
- `watch` — follow periodically or when a nearby roadmap item needs it
- `oracle` — useful for parity, debugging, or reference behavior; not a target runtime
- `parked` — known source, but not worth active attention right now

## Scope

This tracker is for **code and engine repos** we actively follow. It is not the place for model-weight inventories, private secrets, or host-specific path notes.

---

## Core inference and control plane

| Upstream / checkout | Mode | Why watch it | Next candidate | Hard no / constraint | Primary docs |
|---|---|---|---|---|---|
| **Ollama upstream** `../ollama-upstream` | `dependency` | Mergeability, GGUF serving shape, registry compatibility, low-diff upstream improvements | Keep `OllamaCompat` current and keep taking low-risk serve/control-plane deltas | Do not full-rebase away zerollama product deltas just to look upstream-clean | [upstream-ollama-diff.md](ollama-diff.md) |
| **llama.cpp** `../llama.cpp` | `dependency` | Core GGUF engine, llama-server behavior, Metal/CUDA runtime changes | Pin bumps, writable KV/page-bind prerequisites, MTP/Metal wins that survive sign-off | Do not chase tip blindly or reset vendor state outside the pin workflow | [LLAMA_CPP_PIN.md](../../runtime/LLAMA_CPP_PIN.md), [llama-cpp-pin-ladder.md](../design/serving/llama-cpp-pin-ladder.md) |
| **eliza-llama.cpp** `../eliza-llama.cpp` | `watch` | L2 kernel / profile ideas and fork-only experiments worth A/Bing against stock | RotorQuant / BeeLlama style fork labs when L2 evidence justifies it | Do not make fork paths default without live sign-off evidence | [llama-fork-watchlist.md](llama-fork-watchlist.md), [gpu-profiles-l2.md](../design/serving/gpu-profiles-l2.md) |
| **vLLM** `../vllm` | `active borrow` | KV/prefix cache policy, scheduling heuristics, cache semantics, operator mental models | Additional FL1–FL3-style ideas and any durable cache/prefix behaviors that fit zerollama’s architecture | Do not vendor the whole block-manager / distributed runtime design | [vllm-borrowings.md](vllm-borrowings.md), [gpu-profiles-l3.md](../design/serving/gpu-profiles-l3.md) |
| **LocalAI** `../LocalAI` | `watch` | Control-plane lifecycle, metadata, routing, and operator ergonomics | Keep watching **LA12 / PII** and any thin control-plane ideas that reduce operator pain | No OCI backend zoo, no gallery+gRPC re-architecture, no NATS cluster | [localai-borrowings.md](localai-borrowings.md) |
| **SGLang** `../sglang` | `active borrow` | Multimodal parity, media budgeting, processor/precompute handoff, native video/image semantics | Remaining **Option 2** gaps, especially request budgeting and multimodal routing semantics | Do not require an SGLang serve path just to get product parity | [sglang-multimodal-borrowings.md](sglang-multimodal-borrowings.md), [video-parity.md](../design/media/video-parity.md), [RESEARCH_ROADMAP.md](../RESEARCH_ROADMAP.md#option-2--narrow-the-gap-without-sglang-in-tree-over-time) |

---

## Apple / Metal / MLX technique watch

| Upstream / checkout | Mode | Why watch it | Next candidate | Hard no / constraint | Primary docs |
|---|---|---|---|---|---|
| **bmtl** `../bmtl` | `active borrow` | Gigatoken/BPE tricks plus toolkit-side ideas that matter for tokenization and media work | Additional tokenizer wins and media-toolkit ideas that survive in-tree integration | Do not vendor Rust gigatoken or treat BMTL as a product dependency | [faster-bpe-tokenize.md](faster-bpe-tokenize.md), [faster-bpe-tokenize-findings.md](faster-bpe-tokenize-findings.md) |
| **m4-prefill-engine** `../m4-prefill-engine` | `watch` | ggml-metal prefill ideas, especially durable FA / `q8_0` observations | Only bring wins that survive M4 Max regression checks and fit the shipped stack | Do not vendor the host engine wholesale | [m4-prefill-borrowings.md](m4-prefill-borrowings.md), [m4-prefill-borrowings-findings.md](m4-prefill-borrowings-findings.md) |
| **mlx-serve** `../mlx-serve` | `watch` | MLX serving ergonomics, PLD/spec-gate ideas, Metal cookbook patterns | Reuse small serving/ergonomics patterns if **M19** or **M26** needs them | Do not vendor Zig or fork the product around mlx-serve’s host shape | [mlx-serve-borrowings.md](mlx-serve-borrowings.md), [RESEARCH_ROADMAP.md](../RESEARCH_ROADMAP.md#apple-silicon--metal-experiments) |

---

## CUDA / kernel / policy research

| Upstream / checkout | Mode | Why watch it | Next candidate | Hard no / constraint | Primary docs |
|---|---|---|---|---|---|
| **DeepGEMM** `../DeepGEMM` | `watch` | Grouped GEMM / layout / MoE-fuse ideas that might help ggml CUDA lanes | Reusable grouped-GEMM or MoE-fuse lessons that fit ggml on current NVIDIA lanes | Do not vendor DeepGEMM; skip Ascend and Mega MoE EP | [deepgemm-borrowings.md](deepgemm-borrowings.md), [deepgemm-borrowings-findings.md](deepgemm-borrowings-findings.md) |
| **DeepInfer** `../DeepInfer` | `watch` | Serving-policy, MTP-varlen, YOCO, and speculation ideas; not the engine itself | Only new policy-level steals; the core DI ladder is mostly landed already | Do not vendor the SM100 engine or claim 5080/H200 compatibility | [deepinfer-borrowings.md](deepinfer-borrowings.md), [deepinfer-ladder.md](deepinfer-ladder.md) |
| **Flare** `../flare-local-inference` | `watch` | Agent-cache/L3 policy ideas such as FL1–FL3 and reuse metrics | Turn-boundary checkpoints, KV-limit docs, host×GPU reuse metrics when they justify product work | Do not vendor Helios, Ignition, or FlashInfer | [flare-borrowings.md](flare-borrowings.md), [RESEARCH_ROADMAP.md](../RESEARCH_ROADMAP.md#local-voice--llama-borrowings-eliza-v3) |

---

## Media and multimodal runners

| Upstream / checkout | Mode | Why watch it | Next candidate | Hard no / constraint | Primary docs |
|---|---|---|---|---|---|
| **Wan2GP** `../Wan2GP` (`/root/Wan2GP` on CT) | `active borrow` | `mmgp` VRAM bridge for Wan TI2V and product video backends | Keep the `mmgp` bridge healthy; borrow more only if it unlocks shipped `/v1/videos` behavior | Do not absorb the Gradio app or the full multi-model zoo | [wangp-borrowings.md](wangp-borrowings.md), [wan-t2v.md](../design/media/wan-t2v.md) |
| **sglang-omni** `../sglang-omni` | `parked` | MiniMax Music 3 rematch/oracle, only if the current hear path needs another reference | Revisit only if mlx-audio stops being a sufficient oracle for music work | Do not vendor ComfyUI / Omni runtime as the product path | [music-c.md](../design/media/music-c.md), [music-c-findings.md](../design/media/music-c-findings.md) |
| **h3.c** `../h3.c` | `watch` | H3 parity and schedule ideas for `video-c --family h3` | Host/device schedule ideas that help the H3 family path | Not the product runner | [video-c.md](../design/media/video-c.md), [h3-cuda-port.md](../design/media/h3-cuda-port.md) |
| **h3-metal** `../h3-metal` | `watch` | Mac speed lab for MiniMax-H3 | Safe schedule/build ideas only | Do not replace `video-c` or Wan2GP with the sibling binary | [h3-metal-borrowings.md](h3-metal-borrowings.md) |
| **minimax-h3-mlx** `../minimax-h3-mlx` | `oracle` | Python oracle for packing, AdaLN cache, TE behavior, and VAE parity | Use only for parity/debugging and dump comparison when the Mac path needs it | Not a product serve path | [h3-mlx-borrowings.md](h3-mlx-borrowings.md), [video-c.md](../design/media/video-c.md) |

---

## How to use this tracker

When an upstream idea becomes active work, it should appear in one of these places too:

- **Product priority:** [ROADMAP.md](../ROADMAP.md) or a detailed doc under [roadmaps/](../roadmaps)
- **Exploratory bet:** [RESEARCH_ROADMAP.md](../RESEARCH_ROADMAP.md)
- **Landed work:** [CHANGELOG](../../CHANGELOG.md)

Keep `upstream-siblings.md` as the operational companion for checkout paths, cadence, and the last-checked log.
