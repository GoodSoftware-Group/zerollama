# GLiNER2.5-Decide — operator notes

**ROADMAP:** Typed decisions — **GD0–GD4** (Fastino encoder Decider; self-host).  
**Findings / WHYs:** [gliner-decide-findings.md](./gliner-decide-findings.md)  
**Upstream:** [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) via [`gliner2`](https://github.com/fastino-ai/gliner2)  
**In-repo sibling:** `gliner/decide_server/` (FastAPI + AutoExtractor)

## Why this exists

Zero-shot **typed triage** (intent / route / urgency / yes-no) with a 340M encoder — same *job family* as Laya/CLM/OpenJev, **not** span NER.

**Why not `ZEROLLAMA_GLINER_URL` / C++ `gliner-server`:** that path is ONNX span/token NER (`/v1/extract`). Decide uses `classify_text` schemas.  
**Why Python sibling:** upstream ships as `gliner2[local]` Torch; no C++ Decide path yet.  
**Why dual wire:** portable agents use Jev `/v1/decisions`; engine tuners use `/v1/gliner-decide` with Fastino schemas.

## Dual wire

| Path | Role | Body |
|------|------|------|
| `POST /v1/decisions` / `/v1/systemone` | Abstract Jev | `{model, state, questions}` → `{answers}` (`calibrated:false` until a fixture gate) |
| `POST /v1/gliner-decide` | Mechanical | `{text, schema, include_confidence?}` → Fastino-shaped classify result |

Go proxies both when `ZEROLLAMA_GLINER_DECIDE_URL` is set.

## Ensure + serve (lab)

```bash
./scripts/vendor/ensure_gliner_decide_model.sh
# venv + deps once
./scripts/serve/serve_gliner_decide_lab.sh
```

| Var | Role |
|-----|------|
| `ZEROLLAMA_GLINER_DECIDE_URL` | Go → Python base (e.g. `http://127.0.0.1:18098`) |
| `GLINER_DECIDE_MODEL` | HF id or local dir (default `fastino/GLiNER2.5-Decide`) |
| `GLINER_DECIDE_DEVICE` | `cpu` (default) or `cuda:0` |

**VRAM (5080 16 GB):** ~0.7–1.5 GiB class for 340M FP16 — unload OpenJev `-ngl 25` / heavy NER CUDA ORT first (Finding 2).

## Smoke (lab only)

Never `:11434` / `:8081` / `:8080`.

```bash
./scripts/smoke/gliner_decide_smoke.sh
./scripts/smoke/gliner_decide_go_e2e.sh
```

## Example

```bash
ZEROLLAMA_GLINER_DECIDE_URL=http://127.0.0.1:18098 OLLAMA_HOST=127.0.0.1:11439 ./zerollama serve

curl -s http://127.0.0.1:11439/v1/decisions -H 'content-type: application/json' -d '{
  "model": "gliner-decide",
  "state": "My subscription renewed after the service was already down. Can I get a refund?",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Customer intent",
      "criteria": {
        "refund_request": "wants money back",
        "cancel_subscription": "wants to cancel",
        "order_status": "tracking an order",
        "other": "none of the above"
      }
    }
  }
}'

# Mechanical
curl -s http://127.0.0.1:11439/v1/gliner-decide -H 'content-type: application/json' -d '{
  "model": "gliner-decide",
  "text": "My subscription renewed after the service was already down. Can I get a refund?",
  "schema": {"intent": ["refund_request", "cancel_subscription", "order_status", "other"]},
  "include_confidence": true
}'
```

Routing: `modality_backends.decisions=gliner-decide` or name `gliner-decide` / `gliner2.5-decide*`.

**PVE:** run inside CT 1564 only; do not bind on the hypervisor.
