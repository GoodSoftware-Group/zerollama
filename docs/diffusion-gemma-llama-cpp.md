# DiffusionGemma (llama.cpp) — operator notes

**ROADMAP:** Typed decisions — **DG0–DG9** (`ZEROLLAMA_OPENJEV_CALIBRATED` for choice/score; + `NOUL_CALIBRATED` for noul).  
**Findings / WHYs:** [diffusion-gemma-llama-cpp-findings.md](./diffusion-gemma-llama-cpp-findings.md)  
**Readout design:** [diffusion-gemma-readout-design.md](./diffusion-gemma-readout-design.md)  
**Upstream:** [ggml-org/llama.cpp#24427](https://github.com/ggml-org/llama.cpp/pull/24427) — pin file `DIFFUSION_LLAMA_COMMIT`  
**Patch series:** `llama/patches/diffusion/` (`0001` prompt-only, `0002` tokenize+gather; applied by ensure; **not** on prod `b10615`)

## Why this exists

Self-host OpenJev-class **structured read** needs DiffusionGemma **block diffusion** in CUDA — not a Go reimplementation, not MLX, not cloud. Prod chat already runs a patched `b10615` `llama-server` (Laya `--decisions`, CLM emb). Diffusion lands as a **sibling** binary so draft-PR risk cannot take down chat/embed.

**Why not chat `/v1/completions` alone:** Agents need the same public `POST /v1/systemone` shape as Laya/CLM. Go owns packing so clients do not learn a second request schema.

## Why sibling, not prod llama-server

| Concern | Why sibling wins |
|---------|------------------|
| Draft PR `#24427` | Unstable tip; must not break Laya/CLM on `b10615` |
| Different binary | `llama-diffusion-gemma-*`, not stock `llama-server` |
| VRAM | Q4_K_M needs `-ngl ~25` on 16 GB; conflicts with concurrent emb |

Do **not** point `LLAMA_SERVER_BIN` at the diffusion binary.

## What DG2 / DG2b is (and is not)

| Is | Is not | Why |
|----|--------|-----|
| Go packs `{state,questions}` → prompt; C++ denoise; Go parses JSON | Calibrated System-1 | Text confidence would lie to agents (same failure as “JSON in chat”) |
| DG3a: Go packs `<<qid>>` markers → C++ post-marker option logits → Go softmax | Calibrated OpenJev / Laya | Raw vocab logits at a found marker; no decision-head temperature |
| `answers.*.calibrated=false`; `score_source=answer_marker_logit` (or gather fallback) | Trustworthy `confidence` | Model-written `1.0` is not a score |
| Lab Decider + optional prod choice/score/noul (`CALIBRATED` / `NOUL_CALIBRATED`) | Drop-in Laya decision head | Still vocab-marker logits, not a trained noul head |

## Model artifact (DG0/DG1 smoke)

```text
unsloth/diffusiongemma-26B-A4B-it-GGUF
→ /root/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf   # ~16.8 GB
```

**Why this quant:** Named smoke for DG0 kill criteria. Full `-ngl 99` OOMs on 5080 16 GB; `-ngl 25` is the measured default ([findings](./diffusion-gemma-llama-cpp-findings.md)).

## Ensure + build (5080 / CUDA 120-real)

```bash
./scripts/vendor/ensure_diffusion_llama.sh   # real vendor checkout + apply diffusion patch
./scripts/build/build_llama_diffusion.sh     # llama-diffusion-gemma-cli + server
```

**Why ensure applies patches:** C++ `/v1/systemone` prompt-only endpoint lives in-repo as a patch so fresh clones reproduce without a host-only spike tree. Ensure **removes** spike symlinks unless `DIFFUSION_ALLOW_SPIKE_LINK=1`.

| Var | Role | Why |
|-----|------|-----|
| `ZEROLLAMA_DIFFUSION_ROOT` | Override vendor tree | Lab layouts / alternate pins |
| `ZEROLLAMA_DIFFUSION_BIN` | CLI after build | Smoke without recalling PR checkout |
| `ZEROLLAMA_DIFFUSION_SERVER_BIN` | Server after build | Go proxy target |
| `ZEROLLAMA_OPENJEV_URL` | Go → C++ base URL | Same modality pattern as `ZEROLLAMA_CLM_URL` |
| `LD_LIBRARY_PATH` | `/root/nvidia-host` on CT 1564 | CT CUDA driver libs |

## Smoke (lab only)

Free heavy GPU jobs first (e.g. stop CLM emb on `:18092`). **Do not** touch prod `:8080` / `:11434`.

```bash
export LD_LIBRARY_PATH=/root/nvidia-host:${LD_LIBRARY_PATH:-}
./scripts/smoke/diffusion_gemma_smoke.sh          # CLI denoise
DIFFUSION_NGL=8 ./scripts/smoke/diffusion_gemma_gather_smoke.sh   # tokenize + slots/gather
DIFFUSION_NGL=8 ./scripts/smoke/diffusion_gemma_openjev_e2e.sh    # Go :11435 → sibling :18093
DIFFUSION_NGL=8 ./scripts/smoke/diffusion_gemma_oracle_smoke.sh   # DG4a seed ranking oracle
DIFFUSION_NGL=8 ./scripts/smoke/diffusion_gemma_fixture_accuracy.sh
DIFFUSION_NGL=8 ./scripts/smoke/diffusion_gemma_fit_temperature.sh
DIFFUSION_NGL=8 ./scripts/smoke/diffusion_gemma_fit_noul_bias.sh   # DG9 → noul_bias_v0.json
```

## Server (lab port — never 11434/8081)

```bash
"$ZEROLLAMA_DIFFUSION_SERVER_BIN" \
  -m /root/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  -ngl 25 -c 2048 --host 127.0.0.1 --port 18093
```

| Endpoint | Body | Response | Why |
|----------|------|----------|-----|
| `POST /v1/chat/completions` | OpenAI chat | chat completion | Upstream PR surface |
| `POST /tokenize` | `{content}` | `{tokens:[…]}` | Go builds DG2b gather specs with engine vocab |
| `POST /v1/systemone` | `{model, prompt\|messages, readout.gather?}` | `{model, answer, usage, gather?}` | Denoise + optional canvas gather (LA16) |

C++ **rejects** high-level `{state, questions}` alone (400) — packing belongs in Go.

**DG2b gather smoke** (free emb VRAM first; lab port only):

```bash
# after server is up on :18093
curl -s http://127.0.0.1:18093/tokenize -H 'content-type: application/json' -d '{"content":"billing"}'
# Go /v1/systemone with ZEROLLAMA_OPENJEV_URL set will tokenize + attach readout.gather automatically
```

**Why disable GPU sampling during gather:** with diffusion GPU sampling on, `llama_decode` keeps dense logits on device and host `llama_get_logits*` reads zeros (Finding 7).

## zerollama Go (DG2 / DG8 / DG9)

```bash
# One-shot sibling + Go proxy (never :11434/:8081/:8080)
./scripts/serve/serve_openjev_lab.sh                              # calibrated:false
./scripts/serve/serve_openjev_lab.sh --calibrated                 # choice/score calibrated:true
./scripts/serve/serve_openjev_lab.sh --calibrated --noul-calibrated  # + noul (DG9)

# or manual
ZEROLLAMA_OPENJEV_URL=http://127.0.0.1:18093 OLLAMA_HOST=127.0.0.1:11436 ./zerollama serve
curl -s http://127.0.0.1:11436/v1/systemone -H 'content-type: application/json' -d '{
  "model": "openjev",
  "state": {"body": "customer asking for refund"},
  "questions": {
    "dept": {
      "type": "choice",
      "instructions": "Which team?",
      "criteria": {"billing": "refunds", "tech": "bugs"}
    }
  }
}'
```

| Env | Role |
|-----|------|
| `ZEROLLAMA_OPENJEV_URL` | Sibling diffusion base |
| `ZEROLLAMA_OPENJEV_TEMP` | Softmax T (default 0.5) |
| `ZEROLLAMA_OPENJEV_CALIBRATED` | `1` → choice/score `calibrated:true` |
| `ZEROLLAMA_OPENJEV_NOUL_BIAS` | Subtract from positive (`"1"`) logit before softmax (default **0**) |
| `ZEROLLAMA_OPENJEV_NOUL_CALIBRATED` | `1` + `CALIBRATED` → noul `calibrated:true` (Finding 19) |

**VRAM:** Free emb/CLM on the 5080 before `-ngl 25`; default script uses `-ngl 8`. Do not compete with prod `:8080` chat without unloading heavy jobs.  
**PVE:** script sources `refuse_pve_host.sh` — run inside CT 1564 only.

Routing: `modality_backends.decisions=openjev` or name `openjev` / `openjev:*` / `diffusiongemma*`.  
Proxy timeout **10 minutes** — why: cold Q4 load + denoise under partial `-ngl` exceeds the 120 s CLM/Laya client.
