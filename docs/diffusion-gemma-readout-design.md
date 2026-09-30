# DiffusionGemma structured readout — design (DG2–DG9)

**Status:** Design through **DG9**: `ZEROLLAMA_OPENJEV_CALIBRATED=1` marks **choice/score** `calibrated:true`; + `ZEROLLAMA_OPENJEV_NOUL_CALIBRATED=1` for **noul** (Finding 19). Default remain `calibrated:false`.  
**Ops:** [diffusion-gemma-llama-cpp.md](./diffusion-gemma-llama-cpp.md) · **Findings:** [findings](./diffusion-gemma-llama-cpp-findings.md)

## Why a design note before more C++

Full OpenJev readout is months-class (canvas markers, logit API, parity oracle). Without a one-page contract, it is easy to (a) pack in C++, (b) return fake `confidence`, or (c) bolt diffusion into prod chat `llama-server`. This note locks the split and names the interim.

## Goal

Map OpenJev / Jev **structured read** (`POST /v1/systemone`) onto DiffusionGemma **block diffusion** without pretending denoise text or raw vocab logits are calibrated System-1 — unless the operator explicitly opts in after the DG7/DG9 gates.

## Split (LA16) — locked

| Layer | Owns | Why |
|-------|------|-----|
| **Go** | Pack `{state,questions}` → prompt (+ `<<qid>>` markers); tokenize markers/options; parse JSON; merge gather → softmax; calibrated policy | One packing/calibration policy (same WHY as Laya / `/v1/rerank`) |
| **C++** | Denoise + slot/gather logit readout | Engine stays dumb; draft PR stays out of prod pin |
| **Not Go/MLX** | Weights / CUDA | Where kernels belong |

## Shipped ladder

| Stage | Mechanism | `score_source` / `readout_mode` |
|-------|-----------|----------------------------------|
| DG2 | Parse denoise JSON | (none) |
| DG2c | Max option logit over **answer** span | `final_answer_logit_gather` |
| DG3a | Find `<<qid>>` in answer; logits at **next** position(s) | `answer_marker_logit` |
| DG3b | Per-question: slots first, gather fill for missing markers | `hybrid_marker_gather` when both |
| DG4 (partial) | Temperature fit on labeled choice fixtures | still `calibrated:false` by default |
| DG5a | noul marker options; `noul = P(positive)` | same `answer_marker_logit` / hybrid |
| DG5b | score level indices `"0".."k-1"`; `score = E[level]` | same |
| DG6 | Markers **before** JSON; larger fixtures; harness `marker_rate` | prefer `answer_marker_logit` |
| DG7 | Explicit `calib_gate_v0.json` floors + `EvalOpenJevCalibGate` | `gate_ready` without auto-flip |
| DG8 | `ZEROLLAMA_OPENJEV_CALIBRATED=1` | choice/score `calibrated:true` |
| DG9 | noul slots `"0"`/`"1"` + bias fit; `NOUL_CALIBRATED` | noul `calibrated:true` when both flags on |

## DG4 / DG8 / DG9 — what “calibrated” means now

Do **not** set `ZEROLLAMA_OPENJEV_CALIBRATED=1` until:

1. **Frozen logit / ranking oracle** — (**DG4a shipped**).
2. **Labeled fixture set** — (**DG7:** 18/18 gating fixtures, `marker_rate=1.0`).
3. **Explicit score transform** — T=**0.5**, choice holdout **6/6**, `gate_ready=true` in `testdata/openjev/calib_gate_v0.json`.
4. **Operator opt-in** — env flag (or `serve_openjev_lab.sh --calibrated`). Lab gate JSON keeps `operator_signoff:false`; serve opt-in is separate.

**Noul** additionally needs DG9: `0|1` slots (not English true/false — Finding 17), optional `ZEROLLAMA_OPENJEV_NOUL_BIAS` (fit default **0**), and `ZEROLLAMA_OPENJEV_NOUL_CALIBRATED=1`.

Lab:

```bash
DIFFUSION_NGL=8 ./scripts/smoke/diffusion_gemma_oracle_smoke.sh
DIFFUSION_NGL=8 ./scripts/smoke/diffusion_gemma_fit_temperature.sh  # write temperature_v0.json + calib_gate
DIFFUSION_NGL=8 ./scripts/smoke/diffusion_gemma_fit_noul_bias.sh    # write noul_bias_v0.json
./scripts/serve/serve_openjev_lab.sh --calibrated --noul-calibrated
```

Go applies `ZEROLLAMA_OPENJEV_TEMP` (default **0.5**) and `NOUL_BIAS` (default **0**) in `MergeOpenJevGather`; calibrated flags control `calibrated` per type.

## DG3a/b wire

**Prompt (Go):** Emit filled marker lines **first**, then JSON — choice: `<<qid>> <criteria_key>`; noul: `<<qid>> 0|1` (1 = statement holds / immediate action); score: `<<qid>> <level_index>` (DG6/DG9).

**Request extras:** `readout.slots` + `readout.gather` (fallback / hybrid fill).

**Why post-marker position:** Laya reads at MASK.  
**Why hybrid:** Marker emission is model-dependent (Finding 9); multi-question requests should not drop unscored ids.

## HTTP

| Path | Role |
|------|------|
| C++ `POST /tokenize` | Engine vocab for markers + options |
| C++ `/v1/systemone` | Denoise + slots (prefer) / gather (fill) |
| Go `/v1/systemone` | Public Decider wire |

Lab smokes:

```bash
DIFFUSION_NGL=8 ./scripts/smoke/diffusion_gemma_gather_smoke.sh
DIFFUSION_NGL=8 ./scripts/smoke/diffusion_gemma_openjev_e2e.sh
```

## Non-goals

- Cloud OpenJev  
- Entity NER (see GLiNER track / [gliner-cpp.md](./gliner-cpp.md))  
- Patching diffusion into prod chat `llama-server`  
- Claiming noul marker logits ≡ Laya noul (different head)  
- Auto-flipping calibrated from `gate_ready` alone
