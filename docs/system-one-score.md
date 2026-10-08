# System One score path (tev1 / nimble / Clef / Strands)

**ROADMAP:** L2 product rematch from upstream `decision/` + `SystemOneHandler` patterns.  
**Wire:** same `POST /v1/decisions` and `POST /v1/systemone` as Laya/CLM/OpenJev — zerollama keeps **one handler** (`DecisionsHandler`) with multi-backend dispatch.

## What landed (L2)

| Piece | Location |
|-------|----------|
| Compile tev1 / nimble / clef / strands prompts → logits | `decision/` (upstream port) |
| Score via llama-server (Rows + shared-prefix primer) | `llm/llama_server_score_systemone.go` |
| Clef segment + `score_fields` request shape | `decision/clef.go` + `scoreSystemOne` `/embedding` path |
| Handler branch + `CapabilityDecision` scheduling | `server/decisions_systemone.go`, `decisions_handler.go` |
| MLX subprocess scoring (Mac) | `x/mlxrunner/score.go`, `client_score.go`, `POST /v1/score` |
| Create: Laya HF layout, Clef joint head, Qwen3.5 decision quant policy | `x/create/laya.go`, `clef.go`, `qwen35_decision.go` |
| Request `images` (Clef multimodal) | `api.DecisionsRequest`, OpenAPI |

**WHY not replace DecisionsHandler:** zerollama ships Laya GGUF (`Decider`), native CLM heads, OpenJev/DG, GLiNER-Decide, and external URLs — upstream’s standalone `/v1/systemone` only covers local score models.

## Deferred to L6 (llama.cpp pin ladder → **b11232+** / tip **b11351**)

| Item | WHY blocked on **b10615** |
|------|---------------------------|
| Clef **runtime** joint head on GGUF / llama-server | **Wired on tip b11351**; synth + **Cloudflare Clef-Flash product** text + **mmproj image** `/v1/decisions` PASS (`l6_clef_product_convert.sh` → Q8_0 + `mmproj-*-f16.gguf`; tag `clef-flash-vl`). Do not use `ggml-org/Clef-Flash-GGUF` with `llama/clef` ([pin ladder](./llama-cpp-pin-ladder.md)) |
| Strands **PointerRows** on CUDA llama-server | **Deferred:** `llm/llama_server_score_systemone.go` rejects `pointer_rows` (no head in tip llama-server). MLX Strands on Mac works via `x/models/strands/`. CUDA needs a new llama-server score wire (not a pin-ladder patch). |)

### MLX path (L2, Mac)

| Piece | Location |
|-------|----------|
| Row scoring (tev1 / nimble Qwen3.5) | `x/mlxrunner/score.go` + `x/models/qwen3_5/score.go` (`UnembedCandidates`) |
| Strands `PointerRows` | `x/models/strands/` (`base.CachedScorer`) |
| Clef joint head (text schema) | `x/models/clef/` |
| Hidden row + score prefix cache | `x/mlxrunner/cache/hidden.go`, `cache_score.go`, `media_score.go` |
| Wire | `x/mlxrunner/server.go` `POST /v1/score`; schedule via existing `DecisionsHandler` → `llm.Scorer` |

**Linux CI:** `go build ./x/mlxrunner/...` and `./x/models/{clef,strands}/...` compile; MLX score **runtime** tests are `//go:build darwin` (Metal). **Mac soak:** import a decision safetensors tag (renderer `tev1`/`clef`/`strands`, capability `decision`), ensure MLX routing, then `curl` lab `:11435/v1/decisions` as below.

**Still deferred on b10615 / partial MLX:**

| Item | Gap |
|------|-----|
| Clef **images** (MLX) | Qwen3.5 MLX `vision.go` not merged; Mac MLX Clef stays text-only |
| Clef **images** (GGUF / tip) | **Done on CT:** `clef-flash-vl` (text Q8 + mmproj) + `l6_clef_vl_decisions_smoke.sh` |
| Clef **GGUF** / llama-server | Tip wire + product text/VL e2e; CT serve on tip b11351 |
| Upstream MLX score integration tests | Port `score_test.go` / `score_hidden_test.go` on Mac lab |

**L6 enabler (rung 5 / b11351 tip):** `llama/clef/` linked into `llama-server` via `ZEROLLAMA_CLEF_DIR`; server wire is patch **0135** (tip `common_batch` / `batch.view`). Follow-ups **0136** (ANE shim) + **0137** (Laya `llama_process`).

## Lab smoke (non-production ports)

```bash
# After create/import a decision-capable safetensors tag (renderer tev1|clef|strands or capability decision)
curl -sS http://127.0.0.1:11435/v1/decisions -H 'content-type: application/json' -d '{
  "model": "my-decision",
  "state": "Customer charged twice.",
  "questions": {
    "refund": {"type":"noul","instructions":"Refund requested?","criteria":{"false":"no","true":"yes"}}
  }
}'
```

Clef with images (when runner supports vision + score_fields):

```bash
curl -sS http://127.0.0.1:11435/v1/decisions -H 'content-type: application/json' -d '{
  "model": "my-clef",
  "state": {"text":"see image"},
  "images": ["'"$(base64 -w0 screenshot.png)"'"],
  "questions": { "label": {"type":"choice","instructions":"topic","criteria":{"a":"A","b":"B"}} }
}'
```

## Related

- [laya-llama-cpp.md](./laya-llama-cpp.md) — Laya **Decider** path (not score rows)  
- [upstream-ollama-diff.md](./upstream-ollama-diff.md) — L2 status  
- Skill `agentskills/typed-decisions/SKILL.md`
