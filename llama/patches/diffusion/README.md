# DiffusionGemma sibling patches

Applied by `scripts/vendor/ensure_diffusion_llama.sh` onto
`vendor/llama-cpp-diffusion-<sha>/` (PR `#24427` tip from `DIFFUSION_LLAMA_COMMIT`).

**Why a separate series (not `llama/patches/0127+`):** those patches land on prod
`vendor/llama-cpp-b10615` (chat / Laya / CLM emb). Diffusion is a draft-PR sibling;
merging kernels into the prod pin before marker readout would risk the live stack.

**Not** applied to prod `b10615`.

| Patch | Purpose | Why |
|-------|---------|-----|
| `0001-diffusion-gemma-systemone-prompt-only.patch` | `POST /v1/systemone` accepts `{prompt\|messages}` → `{answer}` | LA16: Go packs/parses; C++ must not invent `answers` + fake confidence |
| `0002-diffusion-gemma-systemone-gather-tokenize.patch` | `/tokenize` + `readout.slots` / `gather` + hybrid fill | DG3a/b marker logits (prefer) + DG2c answer-span gather for missing markers; GPU sampling off during score |
