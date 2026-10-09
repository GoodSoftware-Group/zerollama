# Strands pointer head (L6 CUDA)

Apache-2.0 pointer-head scoring for System One **Strands** on tip llama-server.
MLX path remains `x/models/strands/`; this directory is the CUDA/GGUF twin of that algorithm.

| Piece | Role |
|-------|------|
| `strands.h` / `strands.cpp` | CPU pointer head — LayerNorm + Q/K + scaled dots |
| Tip wire | `COMMON_DECISION_TYPE_STRANDS` + `/embedding` `pointer_rows` |
| Graft | `scripts/phase/l6_strands_graft_head.py` |

## Product pack

```bash
# Backbone (merged Qwen3.5-2B Q8) + official head
hf download fabricant451/strands-decider-2B-hobson-v19-GGUF \
  --local-dir /root/models/strands-hobson-v19-gguf
hf download StrandsAgents/strands-decider-2B-hobson-v19 \
  head.safetensors hobson_config.json \
  --local-dir /root/models/strands-hobson-v19

python3 scripts/phase/l6_strands_graft_head.py \
  -i /root/models/strands-hobson-v19-gguf/strands-q8_0.gguf \
  --head /root/models/strands-hobson-v19/head.safetensors \
  --config /root/models/strands-hobson-v19/hobson_config.json \
  -o /root/models/strands-hobson-v19-gguf/strands-ollama-q8_0.gguf
```

GGUF meta: `{arch}.decision.type=strands`, `pointer_dim`, float temperatures; tensors `strands.{norm,q,k}.*` (FP32). Loader skips `strands.*` via ollama-compat.

## Lab e2e

```bash
CUDA_HOME=/usr/local/cuda-12.8 CMAKE_CUDA_ARCHITECTURES=120-real \
  ./scripts/build/build_llama_server.sh   # links llama/strands when present
# Rebuild tip Go binary (scorePointerRows client), then:
STRANDS_E2E_ZEROLLAMA=/tmp/zerollama-lab \
  ./scripts/phase/l6_strands_decisions_e2e.sh
```
