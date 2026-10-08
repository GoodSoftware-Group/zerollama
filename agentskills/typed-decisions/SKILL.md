---
name: typed-decisions
description: "Answer typed System-1 questions (choice / score / noul) via zerollama POST /v1/decisions — Laya GGUF (calibrated), Contrastive-LM (calibrated), DiffusionGemma/OpenJev (DG8/DG9 opt-in), or GLiNER2.5-Decide (uncalibrated self-host); not chat, not score endpoint, not rerank, not NER extract."
version: 1.3.1
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, laya, clm, openjev, diffusiongemma, gliner-decide, decisions, systemone, classification, routing, calibrated]
    category: mlops
    related_skills: [zerollama-integration, entity-extract, rerank-candidates, download-model]
---

# Typed Decisions Skill

Run **System-1** typed decisions on
[zerollama](https://github.com/GoodSoftware-Group/zerollama) via
`POST /v1/decisions` (alias `POST /v1/systemone`):

| Backend | How | Calibrated? |
|---------|-----|-------------|
| **Laya** GGUF | llama-server `--decisions` + act/escalate | **Yes** |
| **Score System One** (tev1 / nimble / Clef text) | `decision.Compile` → llama-server **or MLX** score (`CapabilityDecision`) | **Yes** (softmax over candidates) |
| **Strands** (PointerRows) | MLX only (`x/mlxrunner` + `x/models/strands`) — CUDA/llama-server returns 400 | **Yes** (pointer head) |
| **CLM** | `ZEROLLAMA_CLM_HEADS` + emb URL (or `ZEROLLAMA_CLM_URL`) | **Yes** |
| **OpenJev / DiffusionGemma** | `ZEROLLAMA_OPENJEV_URL` → sibling `llama-diffusion-gemma-server` | **Choice/score (DG8)** when `CALIBRATED=1`; **noul (DG9)** when + `NOUL_CALIBRATED=1` |
| **GLiNER2.5-Decide** | `ZEROLLAMA_GLINER_DECIDE_URL` → Python `gliner2` sibling | **No** (`calibrated:false` until fixture gate) |

**Why not chat:** calibrated (or honest uncalibrated) labels need a decisions modality — not free-text sampling, not `/api/score`, not `/v1/rerank`.

## Compatibility check

```bash
zerollama --version
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/decisions -d '{}'
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:11434/v1/systemone -d '{}'
```

404 → build predates the feature; check [`CHANGELOG.md`](../../CHANGELOG.md).

## When to Use

- Discrete **routing / triage** with known labels (`choice`), ordinal (`score`), or boolean (`noul`)
- **Laya/CLM:** you need **calibrated probs + confidence** (+ Laya act)
- **OpenJev wire:** choice/score after DG7 gate + `ZEROLLAMA_OPENJEV_CALIBRATED=1`; noul needs `--noul-calibrated` / `NOUL_CALIBRATED=1` (Finding 19; English true/false prior was Finding 17)

## When NOT to Use

- Open-ended generation → `/api/chat`
- Rank documents → `/v1/rerank`
- Score chat continuations → `/api/score`
- Production OpenJev without opt-in flags when you need `calibrated:true` (`CALIBRATED`; noul also `NOUL_CALIBRATED`)
- Treating OpenJev marker-logit noul as identical to Laya’s trained noul head

## Prerequisites

- **Laya:** patches **0125–0126** + Laya GGUF — [docs/laya-llama-cpp.md](../../docs/laya-llama-cpp.md)
- **Score System One:** [docs/system-one-score.md](../../docs/system-one-score.md) — tev1/nimble Rows on CUDA **or** Mac MLX; Strands **MLX only**; Clef **text** on MLX now, Clef **GGUF/images** need pin **L6** / vision merge
- **CLM:** heads GGUF + emb URL — [docs/clm.md](../../docs/clm.md)
- **OpenJev:** sibling build + `ZEROLLAMA_OPENJEV_URL` — [docs/diffusion-gemma-llama-cpp.md](../../docs/diffusion-gemma-llama-cpp.md) · [findings](../../docs/diffusion-gemma-llama-cpp-findings.md)
- **GLiNER2.5-Decide:** `serve_gliner_decide_lab.sh` + `ZEROLLAMA_GLINER_DECIDE_URL` — [docs/gliner-decide.md](../../docs/gliner-decide.md) (not NER `/v1/extract`)
- Lab ports only (`11435`, `11439`, `18082`, `18093`, `18098`) — never bind `:11434` / `:8081` from agent work

## API Contract

`POST /v1/decisions` (same body as `POST /v1/systemone` — Jev / Unsloth Desktop Decision wire)

| Field | Required | Notes |
|---|---|---|
| `model` | yes | Laya tag, **decision-capable** safetensors (renderer `tev1`/`clef`/`strands` or capability `decision`), `clm`, `openjev`, or alias |
| `state` | yes | string \| object \| array — shared context |
| `images` | no | Base64 image blobs — **Clef** multimodal only; rejected for tev1/nimble/strands score encodings |
| `questions` | yes | map of `question_id` → `{type, instructions, criteria?, labels?}` (**≤64**); **insertion order** matters for score models |

`questions[id].type` ∈ `choice` | `score` | `noul`. Caps: **≤255** choice options, **≤10** score levels.

**Policy:** gate on `probabilities` / `noul`, not `confidence` (Laya entropy confidence ≠ cloud Jev thresholds).

OpenJev answers: default `"calibrated": false`. With `ZEROLLAMA_OPENJEV_CALIBRATED=1`, **choice** and **score** become `calibrated:true` (DG8). With both that and `ZEROLLAMA_OPENJEV_NOUL_CALIBRATED=1`, **noul** is calibrated too (DG9; slots `0|1`, bias default 0). Temperature default 0.5 via `ZEROLLAMA_OPENJEV_TEMP`.

## How to Run

```bash
# Calibrated (Laya)
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

# Uncalibrated wire (DiffusionGemma sibling — lab)
# ZEROLLAMA_OPENJEV_URL=http://127.0.0.1:18093
curl -s http://127.0.0.1:11435/v1/systemone -H 'content-type: application/json' -d '{
  "model": "openjev",
  "state": {"body": "billed twice, refund please"},
  "questions": {
    "dept": {
      "type": "choice",
      "instructions": "Which team?",
      "criteria": {"billing": "refunds", "tech": "bugs"}
    }
  }
}'

# GLiNER2.5-Decide (self-host — lab :11439)
# ./scripts/serve/serve_gliner_decide_lab.sh
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
        "other": "none of the above"
      }
    }
  }
}'
```

## Pitfalls

- **Wrong model arch → 501** — chat GGUFs cannot decisions (Laya needs `arch=laya`).
- **Strands on CUDA → 400** — `pointer_rows` need an MLX decision runner; do not expect llama-server to score Strands.
- **Clef GGUF on b10615 → runner error** — create/import OK; live joint-head score needs pin **b11232+** (L6). Prefer MLX text Clef or Laya until then.
- **Clef images on MLX** — text schema works; multimodal images still blocked until Qwen3.5 MLX vision lands.
- **Oversize batch → 400** — >64 questions, >255 choice options, or >10 score levels (Jev/Unsloth caps).
- **Confidence ≠ Jev** — do not reuse cloud Jev confidence thresholds; use `probabilities` / `noul`.
- **OpenJev default uncalibrated** — without opt-in envs, `calibrated:false` even with canvas gather; do not escalate policy on `probabilities` as if Laya unless flags are on.
- **OpenJev noul** — needs `0|1` markers + both calibrated flags; prefer Laya/CLM if you need a trained boolean head.
- **GLiNER2.5-Decide** — always `calibrated:false` today; mechanical schema is `/v1/gliner-decide` (not NER `/v1/gliner`).
- **Choice key order** — criteria object key order is load-bearing for Laya; OpenJev pack sorts question ids.
- **VRAM** — DiffusionGemma Q4 on 16 GB needs `-ngl ~25` and usually exclusive GPU ([findings](../../docs/diffusion-gemma-llama-cpp-findings.md)). Gather needs host logits (GPU sampling off during score — Finding 7). Decide GPU: unload OpenJev/NER CUDA first.
- **Agent labs** — never kill production `:11434` / `:8081`.

## Related

- Docs: [laya](../../docs/laya-llama-cpp.md) · [system-one-score](../../docs/system-one-score.md) · [clm](../../docs/clm.md) · [diffusion](../../docs/diffusion-gemma-llama-cpp.md) · [readout design](../../docs/diffusion-gemma-readout-design.md) · [gliner-decide](../../docs/gliner-decide.md)
- ROADMAP typed-decisions **LAYA\*** / **CLM\*** / **DG\*** / **GD\***
