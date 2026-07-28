# PrismML ternary (Q2_0) on zerollama

**Product format:** Hugging Face `*-Q2_g64.gguf` (group size **64**, ~2.25 bpw).  
**Not product:** `*-Q2_0.gguf` (Prism **g128**) and `*-PQ2_0.gguf` — fork-only / future type id.

## One llama-server

Ternary CUDA `Q2_0` already lives on the vendor pin (`QK2_0=64` + patch **0082**). Do **not** keep a second TTS-only binary for chat.

| Piece | Status |
|-------|--------|
| Q2_0 CUDA | `llama/patches/0082-…25707.patch` (already on pin) |
| `GGML_CUDA_FORCE_CUBLAS` getenv | **0099** (5080 serve contract) |
| Kokoro `/v1/audio/speech` | **0100** + **0101** (replaces `run/llama-server-tts` for Kokoro) |
| OmniVoice monolith | Deferred (ROADMAP L7/L8) |
| Native `GGML_OP_ISTFT` | Optional follow-up; Kokoro uses CPU `istft_hann` until then |

Apply **0099–0101**, rebuild vendor `llama-server`, point serve at that binary only:

```bash
./scripts/vendor/sync_vendor_llama.sh   # or your usual apply-patches path
# rebuild vendor llama-server with -DLLAMA_BUILD_KOKORO=ON (default in 0101)
export LLAMA_SERVER_BIN=/root/zerollama/vendor/llama-cpp-86d86ed4/build/bin/llama-server
# unset ZEROLLAMA_KEEP_LLAMA_SERVER_BIN so ApplyUnifiedLlamaCppEnv can keep vendor
```

## Model

```bash
hf download prism-ml/Ternary-Bonsai-27B-gguf Ternary-Bonsai-27B-Q2_g64.gguf \
  --local-dir ~/.zerollama/models/bonsai
# zerollama create ternary-bonsai-27b -f Modelfile  (FROM …Q2_g64.gguf)
```

## Explicit non-goals

- Dual llama-server routing / TTS+vendor fallback in Go  
- Prism **g128** / `PQ2_0` on the pin
