# DiffusionGemma llama.cpp — findings & learnings

**Audience:** anyone extending the sibling diffusion tree, OpenJev Go pack, or deciding go/no-go vs vLLM OpenJev.  
**Related:** [ops](./diffusion-gemma-llama-cpp.md) · [readout design](./diffusion-gemma-readout-design.md) · ROADMAP typed-decisions **DG0–DG2**.  
**Pin:** `DIFFUSION_LLAMA_COMMIT` → ggml-org PR [`#24427`](https://github.com/ggml-org/llama.cpp/pull/24427) tip.  
**Vendor:** `vendor/llama-cpp-diffusion-<shortsha>/` (real checkout; **not** a spike symlink). Lab spike may still exist at `/root/llama.cpp-diffusion-spike` for one-off experiments.

---

## Why this doc exists

DiffusionGemma looked “just another GGUF chat” until VRAM, draft-PR binaries, and OpenJev product semantics collided. These findings record **why** we sibling-vendor, keep LA16 packing in Go, refuse model-written `confidence`, and treat DG2 v0 as a wire — so the next change does not merge diffusion into prod `llama-server` or ship fake calibration.

---

## DG0 spike results (CT 1564 / RTX 5080 16 GB)

| Item | Value |
|------|--------|
| Build | CUDA 13.3 + `CMAKE_CUDA_ARCHITECTURES=120-real` → `120a-real`; `llama-diffusion-gemma-cli` **OK** |
| Model | Unsloth `diffusiongemma-26B-A4B-it-Q4_K_M.gguf` (~16.8 GB on disk) |
| Load | `-ngl 99/60/45/35` → **cudaMalloc OOM** (~16013 MiB) |
| Load that works | **`-ngl 25`** |
| Peak VRAM | **15318 MiB** |
| Wall | **~24 s** cold (load + denoise) |
| Sample | `Hello!` (non-empty) |

**Kill criteria → GO** (build, load with some `-ngl`, generate, upstream alive). Proceed DG1; do **not** merge into `b10615` yet. Q4 does **not** full-GPU on 16 GB.

---

## Finding 1 — Weights ≈ GPU size; `-ngl` is mandatory on 16 GB

**What:** Full offload allocates ~16.0 GiB CUDA buffer and OOMs on ~15.9 GiB visible VRAM.

**Why that matters:** Published guides often assume **24 GB**. On a 5080 16 GB box, “it fits on disk” ≠ “it fits with KV + activations + concurrent CLM emb”.

**Learning:** Default smoke `-ngl 25`. Free competing loads (e.g. CLM `:18092`) before smoke. Do not promise concurrent Laya/CLM emb + DiffusionGemma on one 16 GB card.

---

## Finding 2 — Dedicated binary, not stock `llama-cli` / prod `llama-server`

**What:** PR `#24427` adds `examples/diffusion-gemma/` → `llama-diffusion-gemma-cli` + `llama-diffusion-gemma-server`.

**Why sibling:** Block-diffusion kernels and server loop are incompatible with the chat/Laya/CLM-emb `llama-server` we pin at `b10615`. Pointing `LLAMA_SERVER_BIN` at the diffusion binary would break embedder/decisions paths already in production.

**Learning:** `ZEROLLAMA_DIFFUSION_BIN` / `ZEROLLAMA_DIFFUSION_SERVER_BIN` only. Never overwrite prod `LLAMA_SERVER_BIN`.

---

## Finding 3 — Generation ≠ calibrated OpenJev readout

**What:** Denoise produces answer **text**. Asking the model for JSON + parse is still chat-shaped. Real OpenJev needs logits at structured positions (canvas markers), like Laya MASK.

**Why we still ship a v0 wire:** Agents and Go need one public path (`/v1/systemone`) to exercise the sibling server without inventing a second product API. Honesty requires `calibrated:false` and **stripping** model-written `confidence` (a `1.0` in JSON is not a score).

**Learning:** Do not route production triage through `ZEROLLAMA_OPENJEV_URL` until marker/logit readout lands. Use Laya/CLM for calibrated System-1 today.

---

## Finding 4 — Vendor must be a real tree + in-repo patch

**What:** Symlinking `vendor/…` → `/root/llama.cpp-diffusion-spike` left C++ edits outside git and non-portable.

**Why in-repo patch:** Prod llama uses numbered `llama/patches/0127+`. Diffusion must **not** land on that pin until stable — a separate `llama/patches/diffusion/` series applied only by `ensure_diffusion_llama.sh` keeps the blast radius clear.

**Learning:** Ensure materializes a real checkout, applies `0001` then `0002` (`tokenize` + gather). Spike symlink only with `DIFFUSION_ALLOW_SPIKE_LINK=1` (discouraged).

---

## Finding 5 — LA16: Go packs; C++ denoise only

**What:** An early draft packed `{state,questions}` inside C++ and returned fake-confident `answers`. That inverted the Laya split and duplicated packing policy.

**Why Go packs:** Same control plane as `/v1/rerank` and Laya — tokenizer/prompt policy, question-id sort order, and answer schema live in one place (Go). C++ stays a denoise engine: `{prompt}` → `{answer}`.

**Learning:** Reject C++ bodies with only `state`/`questions` (HTTP 400). Go `PackOpenJevPrompt` / `ParseOpenJevAnswers`; proxy timeout **10 m** (cold load + denoise on partial offload).

---

## Finding 6 — Name `openjev` vs engine DiffusionGemma

**What:** Backend id and env are `openjev` / `ZEROLLAMA_OPENJEV_URL`; weights are DiffusionGemma GGUF.

**Why:** Public wire is Jev/OpenJev-shaped (`/v1/systemone`). The engine underneath is DiffusionGemma in llama.cpp. Agents already say `model=openjev`; renaming the modality would fork clients. Docs must say **engine ≠ calibrated OpenJev product**.

**Learning:** Keep the name; document the honesty gap; never imply v0 ≡ vLLM OpenJev oracle.

---

## Finding 7 — GPU sampling leaves host gather logits at zero

**What:** First DG2b gather responses returned `logit_max=0.0` / `logit_mean=0.0` for every option (including token id 0).

**Why:** With `llama_set_diffusion_gpu_sampling(true)`, `llama_decode` keeps dense diffusion logits on the backend and **does not** copy them to the host output buffer (see `llama.h`). Denoise uses `llama_diffusion_sample_topk` on device; gather incorrectly called `llama_get_logits` afterward and read zeros.

**Learning:** `score_gathers` must `llama_set_diffusion_gpu_sampling(ctx, false)` for the gather forward (then restore). Prefer `llama_get_logits_ith`. Smoke must assert gather logits are finite and not all equal zeros.

---

## Finding 8 — Full-canvas max lets thought dominate ranking

**What:** DG2b took `max` over the entire denoised canvas (thought + answer). Option tokens that appear in scratchpad/thought could outrank the answer region.

**Why that matters:** Agents consume the answer JSON; ranking should track what the model wrote as the response, not internal monologue tokens.

**Learning:** DG2c restricts gather to the answer span after the last `<channel|>` (same cut used for `answer` text). Fallback to full canvas if the span is empty. `score_source` / `readout_mode` → `final_answer_logit_gather`. Still not marker calibration.

---

## Finding 9 — Marker slots need the model to emit `<<qid>>`

**What:** DG3a finds Go-tokenized `<<question_id>>` in the answer span and scores option logits at the next position(s).

**Why that matters:** Without the marker in the canvas, slot scoring is a no-op and C++ falls back to DG2c answer-span max. Short denoise / weak instruction following → gather mode.

**Learning:** Prompt must require marker lines; smoke accepts either `answer_marker_logit` or `final_answer_logit_gather`. Lab saw `<<dept>>` → three tokens `[6143,79817,6985]` on this GGUF.

---

## Finding 10 — Go e2e needs exclusive-ish VRAM + lab ports

**What:** `diffusion_gemma_openjev_e2e.sh` starts sibling `:18093` + `zerollama serve` on `:11435` with `ZEROLLAMA_OPENJEV_URL`. Live result: `choice=billing`, `score_source=answer_marker_logit`, `calibrated=false`.

**Why lab ports:** Must not touch prod `:8080` / `:11434` / `:8081`.

**Learning:** Use `-ngl 8` when CLM emb already holds ~6 GiB; full `-ngl 25` needs ~15 GiB free. Always tear down the lab pair after smoke.

---

## Finding 11 — Seeded denoise can be ranking- and bit-stable on 5080

**What:** DG4a oracle (`seed=42`, `n_steps=8`) ran twice; gather logits matched exactly (`max_abs_delta=0`) and argmax stayed `billing`. Fixture harness scored **3/3** on `choice_fixtures_v0.json` with `calibrated:false`.

**Why that matters:** Calibration (DG4) needs a reproducible engine before fitting temperature/heads. Bit-exactness is a bonus; ranking stability is the product gate.

**Learning:** Forward `options.seed` + `options.n_steps` from Go; C++ must accept `n_steps` (not only `diffusion_steps`). Echo `diffusion.seed` / `steps_requested` for oracle asserts. Still do not set `calibrated:true`.

---

## Finding 12 — Fixture temperature fit hits Laya’s T floor (0.5)

**What:** DG4b NLL grid on three seeded fixture logit vectors chose **T=0.5** (Laya `ClampTemperature` minimum). Margins were large (~2–3 logit), so sharpening always reduced train NLL.

**Why not `calibrated:true`:** Train=test (n=3), no holdout, no decision head — temperature only reshapes already-correct argmax probs.

**Learning:** Ship `ZEROLLAMA_OPENJEV_TEMP` (default 0.5) + `"temperature"` on answers; keep `calibrated:false` until a held-out set + operator sign-off. Refit with `diffusion_gemma_fit_temperature.sh` when fixtures grow.

---

## Finding 13 — Holdout ranking can pass without justifying calibrated:true

**What:** DG4c expanded fixtures to 6 cases (train 4 / holdout 2). Fit T=0.5 on train; holdout ranking **2/2** (`holdout_gate_ok=true`).

**Why still uncalibrated:** n_holdout=2 is a smoke gate, not a calibration study. Large logit margins make temperature mostly cosmetic.

**Learning:** Persist `testdata/openjev/holdout_v0.json`; exit 2 on holdout miss; never auto-flip `calibrated:true` from `holdout_gate_ok`.

---

## Finding 14 — OpenJev noul is P(positive) over marker logits (DG5a / DG9)

**What:** Same slot/gather path as choice. Options are `"0"`/`"1"` (DG9; historically `false`/`true`). `MergeOpenJevGather` writes `"noul": probs["1"]` (or legacy `probs["true"]`) and strips `choice`. Prompt marker line: `<<qid>> 0|1`.

**Why not a separate C++ path:** Vocab logits at the post-marker position already discriminate boolean tokens; Go owns the typed merge (LA16).

**Learning:** Multi-q fixture `multi_refund_urgent` checks choice + `noul_min`. Urgent noul≈0.99, low ≪0.5 with `0|1` slots (Finding 19). Score-type (ordinal) is **DG5b**. Calibrated noul requires DG9 opt-in.

---

## Finding 15 — OpenJev score is E[level] over index marker logits (DG5b)

**What:** Score criteria is a JSON string array (Laya). Slot options are `"0".."k-1"`; merge writes `"score": Σ i·P(i)` plus `legend`. Prompt marker: `<<qid>> <level_index>`.

**Why indices not level text:** Tokenizing free-form level phrases is unstable; digit ids match Laya’s head layout and stay short for diffusion markers.

**Learning:** Still `calibrated:false`. Fixture `outage_severity_high` uses `score_min`. Lab: 8/8 fixtures; sev score≈2.0 (`final_answer_logit_gather` when markers miss).

---

## Finding 16 — Marker-first prompt recovers slot readout (DG6)

**What:** With `max_tokens=64` and “JSON answers object:” as the final cue, denoise often stopped after JSON → C++ never found `<<qid>>` → all `final_answer_logit_gather`. Asking for filled marker lines **before** JSON + `max_tokens=128` yielded `marker_rate=1.0` (10/10 `answer_marker_logit`) on the expanded fixture set.

**Why:** Diffusion has a tight canvas budget; the last prompt line dominates emission order. Slot scoring needs the marker tokens present in the answer span (Finding 9).

**Learning:** Keep markers-first in `PackOpenJevPrompt`. Harness reports `marker_rate`. Still do not flip `calibrated:true` (n_holdout choice=3 is a gate, not a calib study).

---

## Finding 17 — Noul marker logits skew toward true (DG7; fixed DG9)

**What:** With English `true`/`false` option ids, fixtures expecting `noul_max=0.5` on polite non-urgent asks still scored `noul≈0.99` with `answer_marker_logit`.

**Why:** Post-marker English `true` vocab has a strong instruction prior; temperature sharpening does not fix ranking bias toward true.

**Learning (historical):** DG8 calibrated **choice/score** only. **DG9** switched slots to `"0"`/`"1"` (noul=`P("1")`) and optional logit bias — see Finding 19.

---

## Finding 18 — DG8 calibrated opt-in is choice/score only

**What:** `ZEROLLAMA_OPENJEV_CALIBRATED=1` (or `serve_openjev_lab.sh --calibrated`) sets `calibrated:true` on choice and score answers after marker/gather merge. Noul remains `calibrated:false` unless DG9 `NOUL_CALIBRATED` is also on.

**Why not auto-flip from gate_ready:** Lab JSON (`calib_gate_v0.json`) records floors; serve must require an explicit operator env so agents never inherit calibrated by accident.

**Learning:** Free VRAM before sibling `-ngl` (prod emb/CLM on 5080 can leave &lt;200 MiB). Never bind OpenJev on `:11434`/`:8081`/`:8080`.

---

## Finding 19 — DG9 noul 0|1 slots remove English true prior; bias=0

**What:** Prompt markers `<<qid>> 0|1` with option ids `"0"`/`"1"`; `MergeOpenJevGather` sets `noul=P("1")` (legacy `"true"` still accepted). Fit on train noul (`fit_noul_bias.sh`) yields **`noul_bias=0`**, holdout **2/2**, `holdout_gate_ok`. With `ZEROLLAMA_OPENJEV_CALIBRATED=1` **and** `ZEROLLAMA_OPENJEV_NOUL_CALIBRATED=1`, noul answers get `calibrated:true`. Lab: 20/20 fixtures, low-urgency noul ≪ 0.5.

**Why 0|1 works:** Digit tokens lack the English “true” instruction prior that skewed Finding 17; margins on low cases are large negative (~−9) so bias fit collapses to 0.

**Learning:** Default `DefaultOpenJevNoulBias=0`. Opt-in noul calibrated separately from choice/score (Finding 17 could return under new prompts — keep `NOUL_CALIBRATED` gated). `serve_openjev_lab.sh --calibrated --noul-calibrated`.
