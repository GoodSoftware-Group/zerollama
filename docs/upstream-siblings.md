# Upstream sibling checkouts

**Why this doc:** Weekly “pull X and decide what to bring into zerollama” needs a stable map of local trees. Paths are relative to this repo (`zerollama/`) unless noted. On the Mac lab host, the parent is `~/Sites/inference/`.

**Agent entry:** [AGENTS.md](../AGENTS.md) (repo root).

---

## Sibling trees (Mac lab: `~/Sites/inference/`)

| Tree | Relative path | Upstream | Borrowings / watch doc | Weekly? |
|------|---------------|----------|------------------------|---------|
| **vLLM** | `../vllm` | [vllm-project/vllm](https://github.com/vllm-project/vllm) `main` | [vllm-borrowings.md](./vllm-borrowings.md) | **Yes** (KV / prefix / scheduler) |
| **LocalAI** | `../LocalAI` | [mudler/LocalAI](https://github.com/mudler/LocalAI) | [localai-borrowings.md](./localai-borrowings.md) | **Yes** (control plane) |
| **SGLang** | `../sglang` | [sgl-project/sglang](https://github.com/sgl-project/sglang) | [sglang-multimodal-borrowings.md](./sglang-multimodal-borrowings.md) | As needed (video / multimodal) |
| **llama.cpp** | `../llama.cpp` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | [LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md), [llama-fork-watchlist.md](./llama-fork-watchlist.md) | Pin bumps only |
| **Ollama upstream** | `../ollama-upstream` | [ollama/ollama](https://github.com/ollama/ollama) | [upstream-ollama-diff.md](./upstream-ollama-diff.md) | As needed (mergeability) |
| **eliza llama fork** | `../eliza-llama.cpp` | elizaOS/llama.cpp | [llama-fork-watchlist.md](./llama-fork-watchlist.md) | L2 kernel watch |
| **ANE / MLX / toolkit** | `../ane`, `../mlx`, `../mlx-c`, `../bmtl` | various | [ane-draft-inprocess.md](./ane-draft-inprocess.md); **bmtl gigatoken techniques** → [faster-bpe-tokenize.md](./faster-bpe-tokenize.md) (patches **0106–0126**, do **not** vendor Rust) | Mac-only / technique watch |
| **Wan2GP** | `../Wan2GP` (CT: `/root/Wan2GP`) | [deepbeepmeep/Wan2GP](https://github.com/deepbeepmeep/Wan2GP) | [wangp-borrowings.md](./wangp-borrowings.md) (`mmgp` VRAM) | As needed (video VRAM / zoo watch) |
| **h3.c** | `../h3.c` | [antirez/h3.c](https://github.com/antirez/h3.c) | Metal MiniMax-H3 **C rematch** for [video-c.md](./video-c.md) (`--family h3`); not the product runner | As needed (H3 parity) |
| **minimax-h3-mlx** | `../minimax-h3-mlx` | [mrbizarro/minimax-h3-mlx](https://github.com/mrbizarro/minimax-h3-mlx) | MLX MiniMax-H3 **Python rematch oracle** (packing, AdaLN cache, TE layer-50, VAEs) for video-c Darwin | As needed (H3 parity / dumps) |
| **ClipProj-MiniMax-H3** | `~/.zerollama/third_party/h3/clipproj/` | [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3) | TE projection matrices (4B/8B → `[seq,5120]`); [h3-clipproj.md](./h3-clipproj.md) | Optional (Darwin TE shrink) |
| **MiniMax-H3 (partial)** | `~/.zerollama/models/MiniMax-H3/` | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | Mac lab: `FL2VA/audio_vae` + `video_vae` only (~10 GiB); DiT/TE (~62 GiB each) need more free disk | Operator-supplied |

Other checkouts under the same parent (`ggml/`, `shard/`, rotorquant labs, …) are optional labs — not part of the weekly scan unless a ROADMAP item points at them.

**Other hosts:** CT 1564 / cudallama may use `~/zerollama` or `/var/lib/vz/.../zerollama` without the full `Sites/inference` layout. Prefer relative `../vllm` when present; otherwise clone once under a documented sibling and note the absolute path in that host’s notes — do not invent paths.

---

## Weekly ritual (15–30 min)

1. **Pull** the watch tree(s):
   ```bash
   cd ../vllm && git fetch origin && git checkout main && git pull --ff-only
   # optional same week:
   cd ../LocalAI && git fetch origin && git pull --ff-only
   ```
2. **Diff since last note** (example for vLLM — replace `OLD` with the SHA recorded in [vllm-borrowings.md](./vllm-borrowings.md)):
   ```bash
   cd ../vllm
   git log --oneline OLD..HEAD -- 'vllm/v1/core/kv_cache*' 'vllm/distributed/kv*' 'vllm/v1/kv_cache*'
   ```
3. **Triage** into bring / watch / skip (same buckets as the last vLLM scan). Update the borrowings doc **Last checked** line + tip SHA.
4. **Do not** start production servers on `:11434` / `:8081` while scanning — lab ports only if you need a smoke.

---

## Last checked (operators update this)

| Upstream | Tip / tag noted | Date | Notes |
|----------|-----------------|------|-------|
| vLLM `main` | `118bcde44` | 2026-07-28 | **Brought:** #48123 tier filter, #48596/#49671 defer blob finalize, #48535 cache creation tokens, #48911 SWA store filter — [vllm-borrowings.md](./vllm-borrowings.md) |
| SGLang `main` | `4e5a05148a` | 2026-07-28 | **Brought:** #31417 / #31438 / #31832 / #29436 (`session_id`) — [sglang-multimodal-borrowings.md](./sglang-multimodal-borrowings.md) |
| LocalAI | v4.5.6 tree | 2026-07-03 | LA11+ candidates in [localai-borrowings.md](./localai-borrowings.md) |
| Wan2GP `main` | `7e45fe7e2110` | 2026-08-11 | **Brought:** `mmgp==3.7.12` attach for 16g TI2V — [wangp-borrowings.md](./wangp-borrowings.md) |

---

## What not to put here

- Host-specific secrets, API keys, or production `OLLAMA_HOST`
- Full ROADMAP status (keep in [ROADMAP.md](./ROADMAP.md))
- Vendor pin SHAs for llama.cpp (keep in [LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md))
