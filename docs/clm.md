# Contrastive-LM (CLM) — operator notes

**ROADMAP:** Typed decisions — **CLM1** (optional URL proxy) + **CLM2** (native Go heads, **no Torch at serve**) shipped.  
**Related:** [laya-llama-cpp.md](./laya-llama-cpp.md) · skill `agentskills/typed-decisions/`

## Why this exists

[Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) is a **System-1** model: frozen **Qwen3-8B** last-token embeddings + small contrastive projection heads (`choice` / `score` / `noul`). Public wire: `POST /v1/decisions` / `/v1/systemone`.

**Why not chat / RANK / Laya GGUF:** different graph (dual embed + MLP heads).  
**Why native Go heads (CLM2):** serve path must not require Python/Torch/`clm-serve`. Convert `.pt` → GGUF once; runtime is embeddings HTTP + GEMM in Go.

## Architecture (CLM2 — default)

```
client → zerollama POST /v1/systemone (model=clm)
           │  LoadCLMHeads(ZEROLLAMA_CLM_HEADS)
           │  POST ZEROLLAMA_CLM_EMB_URL/v1/embeddings
           ▼
        llama-server Qwen3-8B GGUF (--embeddings, last-token pooling)
```

Optional **CLM1 fallback:** set `ZEROLLAMA_CLM_URL` to a running `clm-serve` (wins over native).

## One-shot convert (Python only for convert)

```bash
# Download reference head once
hf download Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt --local-dir ~/.cache/clm

PYTHONPATH=../llama.cpp/gguf-py python3 scripts/convert_clm_heads_to_gguf.py \
  --ckpt ~/.cache/clm/CLM_v0.1-8B.pt \
  --outfile ~/.cache/clm/CLM_v0.1-8B.gguf
```

## Env

| Var | Role |
|-----|------|
| `ZEROLLAMA_CLM_HEADS` | Path to heads GGUF (required for native) |
| `ZEROLLAMA_CLM_EMB_URL` | Embeddings base/URL (llama-server `--embeddings`) |
| `ZEROLLAMA_CLM_EMB_MODEL` | Model name in embed requests (default `qwen3-8b`) |
| `ZEROLLAMA_CLM_URL` | Optional `clm-serve` base — if set, proxies instead of native |
| `ZEROLLAMA_LAYA_URL` | Optional external Laya Decider |

Routing:

1. `modality_backends.decisions=clm` or name `clm` / `clm-*` / `contrastive-lm/*`
2. If `ZEROLLAMA_CLM_URL` set → proxy `/v1/systemone`
3. Else if heads+emb set → **native Go**
4. Else → **503** with setup hint

## Lab recipe (non-production ports)

Never bind `:11434` / `:8081` from agent lab smokes.

```bash
# 1) Embedder
./build/bin/llama-server -m /path/to/qwen3-8b.gguf --embeddings --port 18090

# 2) Zerollama lab — no Python
ZEROLLAMA_CLM_HEADS=$HOME/.cache/clm/CLM_v0.1-8B.gguf \
ZEROLLAMA_CLM_EMB_URL=http://127.0.0.1:18090 \
OLLAMA_HOST=127.0.0.1:11435 ./zerollama serve

curl -s http://127.0.0.1:11435/v1/systemone -H 'content-type: application/json' -d '{
  "model": "clm",
  "state": "Customer: charged twice, no one answers the phone",
  "questions": {
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "dept": {
      "type": "choice",
      "instructions": "Which team?",
      "criteria": {"billing": "Charges and refunds", "technical": "Bugs"}
    }
  }
}'
```

## Follow-ups

| ID | Goal |
|----|------|
| **CLM3** | `/v1/rank` passthrough / native rank |
| **CLM4** | Managed spawn of embedder (CUDA boxes) |

## WHYs vs Laya

| | Laya | CLM |
|---|------|-----|
| Encoder | ModernBERT in GGUF | Qwen3-8B embeddings (separate process) |
| Heads | In same GGUF | Heads-only GGUF + Go MLP |
| Act/escalate | Yes | No |
| Serve Python | No | **No** (convert-only) |
