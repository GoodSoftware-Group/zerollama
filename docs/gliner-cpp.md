# GLiNER.cpp — operator notes

**ROADMAP:** Entity extract — **GL0–GL5a** (C++/ONNX sibling; dual wire; CUDA; token multitask). RelEx/bi-encoder **Parked** (Finding 9).  
**Findings / WHYs:** [gliner-cpp-findings.md](./gliner-cpp-findings.md)  
**Upstream:** [Knowledgator/GLiNER.cpp](https://github.com/Knowledgator/GLiNER.cpp) — pin `GLINER_CPP_COMMIT`  
**In-repo server:** `gliner/server/` (links vendor lib + ORT)

## Why this exists

Zero-shot **span NER** for agents: given text + label strings, return spans with scores. Not chat, not `/v1/decisions` (those are System-1 triage).

**Why C++ + ONNX Runtime:** no Torch/Ray at serve; portable ONNX models; CPU (GL0) or CUDA ORT (GL4).  
**Why sibling binary:** keep ORT/tokenizers out of prod `b10615` `llama-server`.  
**Why dual wire:** portable agents use `/v1/extract`; engine tuners use `/v1/gliner` with full GLiNER.cpp knobs.

## Dual wire

| Path | Role | Body |
|------|------|------|
| `POST /v1/extract` | Abstract NER | `{text\|texts, labels, threshold?}` → `{entities:[…]}` or `{results:[…]}` for batch |
| `POST /v1/gliner` | Mechanical | Same core fields **plus** `max_width`, `max_length`, `model_type` (`span`\|`token`), `device_id`, `flat_ner`, `multi_label`, … |

Go proxies both when `ZEROLLAMA_GLINER_URL` is set. Abstract path strips unknown engine keys before forward; `/v1/gliner` forwards the engine bag (400 on unknown keys at the C++ server).

## Ensure + build

```bash
./scripts/vendor/ensure_gliner_model.sh small       # onnx-community/gliner_small-v2.1
./scripts/vendor/ensure_gliner_model.sh multitask   # token-level large (GL5a)

./scripts/vendor/ensure_gliner_cpp.sh
./scripts/vendor/ensure_onnxruntime.sh
./scripts/build/build_gliner_server.sh
# GL4: GLINER_ORT=cuda ./scripts/build/build_gliner_server.sh
```

| Var | Role |
|-----|------|
| `ZEROLLAMA_GLINER_CPP_ROOT` | Override vendor tree |
| `ZEROLLAMA_GLINER_SERVER_BIN` | Server after build |
| `ZEROLLAMA_GLINER_URL` | Go → C++ base (e.g. `http://127.0.0.1:18094`) |
| `ONNXRUNTIME_ROOTDIR` | ORT unpack root (`include/` + `lib/`) |
| `ONNXRUNTIME_VARIANT` | `cpu` (default) \| `gpu` |
| `GLINER_ORT` | `cpu` \| `cuda` (selects ORT variant + build dir) |
| `GLINER_MODEL_DIR` | Dir with `model.onnx` + `tokenizer.json` |

**VRAM (5080 16 GB):** CUDA `gliner_small` adds ~1 GiB. Do **not** co-reside with DiffusionGemma `-ngl 25` / heavy emb — unload those first, or keep GLiNER on CPU ORT (Finding 7). Multitask-large prefers CPU unless VRAM is free.

**Token multitask (GL5a):** `ensure_gliner_model.sh multitask` + `--model-type token`. onnx-community logits are channel-first — patch `gliner/patches/0001-…` (Finding 10). RelEx/bi-encoder remain Parked (Finding 9).

## Smoke (lab only)

Never `:11434` / `:8081` / `:8080`.

```bash
./scripts/smoke/gliner_cpp_smoke.sh
./scripts/smoke/gliner_server_smoke.sh
./scripts/smoke/gliner_go_e2e.sh
./scripts/smoke/gliner_cuda_smoke.sh           # GL4
./scripts/smoke/gliner_token_smoke.sh          # GL5a token multitask
./scripts/smoke/gliner_token_go_e2e.sh         # GL5a Go proxy → token

# CI (no ORT download)
./scripts/check_gliner_scripts.sh
```

## Server (lab port — never 11434/8081)

```bash
# One-shot sibling + Go (preferred)
./scripts/serve/serve_gliner_lab.sh
./scripts/serve/serve_gliner_lab.sh --cuda
./scripts/serve/serve_gliner_lab.sh --multitask

# Manual
"$ZEROLLAMA_GLINER_SERVER_BIN" \
  --model /path/to/model.onnx --tokenizer /path/to/tokenizer.json \
  --host 127.0.0.1 --port 18094 --model-type span   # or token
```

## zerollama Go (GL2)

```bash
ZEROLLAMA_GLINER_URL=http://127.0.0.1:18094 OLLAMA_HOST=127.0.0.1:11437 ./zerollama serve

# Abstract
curl -s http://127.0.0.1:11437/v1/extract -H 'content-type: application/json' -d '{
  "model": "gliner",
  "text": "Kyiv is the capital of Ukraine.",
  "labels": ["city", "country"]
}'

# Mechanical
curl -s http://127.0.0.1:11437/v1/gliner -H 'content-type: application/json' -d '{
  "model": "gliner",
  "text": "Kyiv is the capital of Ukraine.",
  "labels": ["city", "country"],
  "threshold": 0.4,
  "flat_ner": true,
  "max_width": 12,
  "max_length": 512,
  "model_type": "span",
  "device_id": 0
}'
```

Routing: `modality_backends.extract=gliner` or name `gliner` / `gliner:*`.

**PVE:** run inside CT 1564 only; do not bind on the hypervisor.
