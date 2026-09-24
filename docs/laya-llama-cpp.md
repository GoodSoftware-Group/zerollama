# Laya (llama.cpp) — operator notes

**ROADMAP:** Typed decisions (Laya) LAYA1–LAYA2 active; MLX sidecar parked (LAYA4).  
**Findings / WHYs:** [laya-llama-cpp-findings.md](./laya-llama-cpp-findings.md)  
**OpenAPI:** `POST /v1/decisions`, alias `/v1/systemone` · skill `agentskills/typed-decisions/`

## Why this exists

Laya ([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)) is a **non-autoregressive System-1** model: ModernBERT-class encoder + small decision Transformer + scorer + act/escalate head. It answers typed questions (`choice` / `score` / `noul`) in **one forward pass** with calibrated probabilities.

**Why not chat:** sampling free text loses the calibrated distribution and act head.  
**Why not `/v1/rerank`:** RANK pooling returns a single scalar (`embd[0]`); Laya needs per-MASK logits + a 4-feature act head.  
**Why not overload `LLM_ARCH_MODERN_BERT`:** same encoder family, different product graph and GGUF tensors — a separate `LLM_ARCH_LAYA` keeps convert/load/serve honest.  
**Why CPU decision head:** encoder runs on GPU like embed; head (~26 MB) applies on CPU after `t_embd` — avoids new GPU kernels while keeping GPU encode.  
**Why Go packs:** LA16 pattern — Go owns `{state,questions}` → tokenized batch; llama-server owns encode+head. Same split as `/v1/rerank`.

Patches: **0127** (arch + `laya.cpp`) · **0128** (`--decisions`, `POST /v1/decisions`). Pin: [runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md).

## Convert

```bash
# From zerollama root; needs sibling ../llama.cpp/gguf-py + safetensors/torch/transformers
PYTHONPATH=../llama.cpp/gguf-py python3 scripts/convert_laya_to_gguf.py \
  --model /path/to/convaiinnovations-laya \
  --outfile /tmp/laya-f32.gguf
```

Checkpoint must include `model.safetensors`, `rl_agent_config.json`, `encoder/config.json`, `tokenizer/`.

**WHYs from lab smoke:**
- Vocab is required (`tokenizer.ggml.tokens` + merges) — without it llama.cpp refuses load.
- Prefer **F32** GGUF for CPU smoke; F16 norms hit `binary_op: unsupported types f32/f16` on ggml-cpu warmup.
- Leave HF `type_emb` as `[3, d]` — gguf-py reverses dims into ggml `ne=[d,3]`; do not transpose.
- Lab: `OLLAMA_HOST=127.0.0.1:11435`, `ZEROLLAMA_LLAMA_SERVER=1`, `LLAMA_SERVER_BIN=…/llama-server` (Darwin default does not route to llama-server).

## llama-server (lab ports only)

```bash
./scripts/build/build_llama_server.sh   # after syncing patches 0127–0128
./build/bin/llama-server -m /tmp/laya-f16.gguf --decisions --port 18082 \
  --embeddings --pooling none

curl -s http://127.0.0.1:18082/v1/decisions \
  -H 'content-type: application/json' \
  -d '{"inputs":[{"tokens":[...],"marker_pos":[...],"qtype":0,"question_id":"dept"}]}'
```

C++ v1 is **tokenized-only**. High-level `{state, questions}` packing is owned by Go (LAYA2). Results echo `question_id` when provided so clients can verify alignment.

Never bind `:11434` / `:8081` from agent lab smokes.

## zerollama

```bash
# create from GGUF (arch laya), then:
OLLAMA_HOST=127.0.0.1:11435 ./zerollama serve   # lab only
curl -s http://127.0.0.1:11435/v1/decisions -H 'content-type: application/json' -d '{
  "model": "laya",
  "state": {"body": "billed twice, refund please"},
  "questions": {
    "dept": {
      "type": "choice",
      "instructions": "Which team?",
      "criteria": {"billing": "refunds", "tech": "bugs"}
    }
  }
}'
```

Go packs via `llm.PackQuestions` + llama-server `/tokenize`; subprocess gets `--decisions` when GGUF `general.architecture=laya`. Alias: `POST /v1/systemone`.

**WHY sort question ids:** map iteration is non-deterministic; both pack and result alignment sort by `question_id` so logits line up with labels.

## Parity

Unit coverage (no weights):

```bash
go test ./llm/ -run 'Laya|Pack|Confidence|Render|GgufIsLaya'
go test ./server/ -run Decisions
go test ./server/openapi -run OpenAPI
```

Full CPU/CUDA parity vs Python `laya` Agent requires a converted GGUF + frozen fixtures (argmax + calibrated probs). Lab: CPU first, then CUDA `CMAKE_CUDA_ARCHITECTURES=120-real` on CT 1564 (not the PVE hypervisor).

## Parked / later

| Track | Status | Why |
|-------|--------|-----|
| LAYA3 HF pull + mmBERT | Later | English convert path first |
| LAYA4 laya-mlx sidecar | Parked | Prefer external `ZEROLLAMA_LAYA_URL` when unparked; no Darwin-managed spawn |

→ [mlx-serve-borrowings.md](./mlx-serve-borrowings.md) · ROADMAP **Typed decisions (Laya)**
