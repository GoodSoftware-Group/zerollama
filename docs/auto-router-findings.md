# Auto-router — findings (WHYs)

Lab notes for the **AR\*** track ([auto-router.md](./auto-router.md)). Keep the non-obvious lessons here so the router is not re-derived, and so AR5 is not promoted past opt-in without the AR7 gate.

## Finding 1 — Most of the value needs no Jev

**What:** Capability + availability dispatch (AR1–AR3) fixes wrong-class picks (chat model on `/v1/images/*`, tools request to a model without tools) and avoids model swaps — deterministically, with no decider call.

**Why:** The request already states images / video / audio / tools / context size; on one GPU a swap costs seconds to minutes, so "prefer warm" dominates any quality gain.

**Learning:** Jev is reserved for what the request does not say: difficulty / escalation (AR4–AR5) and cross-modal intent (AR8). Media surfaces never call a decider.

## Finding 2 — Proxies pick a backend before handlers run

**What:** `runtimeChatProxy`, `runtimeGenerateProxy`, `runtimeV1ChatCompletionsProxy` and the SGLang proxy decide on `req.Model` before `ChatHandler` / `GenerateHandler`. In-handler rewriting (LA11, AR1) routed a runtime-sidecar model onto the ggml path → `ErrRuntimeInferenceModel` 400.

**Why:** Middleware order in `routes.go`; the router name is not a model, so the proxies fall through.

**Learning:** `routeAheadMiddleware` (AR3) rewrites the body's `model` on `/api/chat`, `/api/generate`, `/v1/chat/completions` before the proxies. In-handler routing keeps excluding runtime models on the routes that have no proxy.

## Finding 3 — `effectiveMaxLoadedModels` can trigger 30 s of GPU discovery

**What:** With `OLLAMA_MAX_LOADED_MODELS` unset, the max-loaded lookup calls `cachedGPUCount`, which runs discovery on a cold cache. The first routed request in a test took 30 s.

**Why:** Discovery has a 30 s timeout when no runner library is present.

**Learning:** Only consult it when something is resident (`sync.OnceValue`, AR1). With nothing loaded no load can evict, and on a real node discovery has already run before any model loads.

## Finding 4 — Embedders and rerankers must be pinned

**What:** Two embed calls routed to different models return vectors that silently do not compare.

**Why:** Embedding spaces are model-specific; rerank scores likewise.

**Learning:** Embed / rerank resolve to router.yaml `embedder` / `reranker`, or implicit `auto` only when exactly one model qualifies — otherwise refuse (`ErrAmbiguousPinned`). Never pick per request.

## Finding 5 — `/api/tags` is an API for tools, not just a list

**What:** `zerollama bench`, `launch` and `ls` enumerate `/api/tags`. Listing virtual `auto` there would make bench try to benchmark a router.

**Learning:** Virtual router entries go on `/v1/models` only (OpenAI pickers), after the real models, and only when ≥1 local model exists. `TestRoutes/openai_list_models_with_tags` was updated for the trailing entry — a visible default change for OpenAI clients.

## Finding 6 — A decider timeout cancels the decider's own cold load

**What:** With a 150 ms `decider_timeout`, the first request's context is cancelled during the Laya load, so the scheduler drops the pending load and every request times out.

**Why:** `runDecisions` uses the request context for `scheduleRunner`.

**Learning:** Warm deciders at serve start (`prewarmJevDeciders`), send `keep_alive: -1` on decider calls (decodes to "forever", not a negative duration), and on a deadline start one untimed background warm-up (singleflight). Fail open meanwhile.

## Finding 7 — Calibration is reported only by the uncalibrated

**What:** Laya, CLM and System One answers omit `calibrated`; OpenJev and GLiNER-Decide set it (`false` by default).

**Learning:** Missing ⇒ calibrated. Thresholds (activation, escalation, act head) apply only when calibrated; otherwise argmax only and never escalate. Laya's `action.act_probability` is used only when present — other backends do not emit it, and a zero would otherwise escalate everything.

## Finding 8 — Context fit must be soft for picking, strict for keeping

**What:** Decide's context filter never rejects every candidate (the runner may still truncate / LA22-compress). Re-validating a single remembered model through Decide therefore kept a model the conversation had outgrown.

**Learning:** `autoroute.Validate` applies every hard filter strictly for sticky / cached hints (AR6); Decide keeps the soft behaviour for fresh picks. Voice ownership follows the same split.

## Finding 9 — The in-tree CGO build broke at the b11351 pin (fixed)

**What:** On a Linux checkout without a sibling `../llama.cpp`, `go test ./server/` failed: gitignored `vendor/cpp-httplib` / `vendor/hash` / `vendor/sheredom`, CMake-generated `ggml-version.h` / `llama-version.h`, a stale `build-info.cpp.in` (`llama_print_build_info` lost its `FILE *` parameter upstream), unwrapped `ggml-cpu/tiled/*.cpp` and `common/parsers/*.cpp`, and an mtmd helper call missing its new `opt` argument.

**Why:** Each was a gap in the sync tooling, not a one-off: `.gitignore`'s blanket `vendor/` silently dropped new vendor dirs on `git add`, and the sync had no step for configure-time outputs or new subdirectories.

**Fix:** upstream `61634007` covers the generated parts — `scripts/phase/ensure_ggml_version_h.sh` writes both version headers (gitignored), Go cgo packages `common/parsers` and `ggml-cpu/tiled` compile the new subdirectories, and `build-info.cpp.in` matches the new signature. Still needed on top: `.gitignore` re-includes `llama/llama.cpp/vendor/` so `cpp-httplib` / `hash` / `sheredom` are tracked (`mtmd-helper.cpp` and `http.h` fail without them). (An earlier local `gen_llama_cgo_glue.sh` with unity wrappers was dropped in favour of upstream's approach on rebase — keeping both would compile the sources twice.)

**Newly visible:** `./llama` links for the first time and shows two pre-existing expectation mismatches (`TestSchemaToGrammar/{"type":"object"}`, `TestIssue7978`) — the tests expect pre-b11351 `json-schema-to-grammar` output. Not touched here.

## Finding 11 — The keyword fallback is scored on a set written by the same hand

**What:** `GuessOutputModality` reaches 13/13 recall and 0/96 false positives on `testdata/router_eval/prompts.jsonl`. The first version flagged "you may **draw** one fruit" as an image request; bare drawing verbs now count only at a clause start or after "can you / please".

**Why it matters:** the rules and the dataset were written in the same session, so the score measures fit to known phrasings, not real traffic. Real users write "give me an image of…", "visualise…", "voice this" — some will miss.

**Learning:** keep the fallback advisory-only (a header never changes behaviour), prefer a jev `output_modality` question where a decider exists, and judge the fallback on logged real turns (`zerollama_router_suggest_total` + client follow-through) before relying on it for AR9 handoff.

## Finding 12 — Warm-first collapses implicit `auto` onto the router's own classifier (fixed)

**What:** On the first real run, implicit `auto` sent all 104 servable prompts to `llama3.2:1b` (`reason: warm`) — the LA11 score classifier, resident only because a router used it.

**Why:** The warm bonus (weight 4) dominates the deterministic score by design (Finding 1); a helper model is the one most likely to be warm.

**Fix:** implicit `auto` drops router helpers (score classifier, jev decider, embedder, reranker) from text candidates unless a router also lists them as a candidate; soft (a helper-only node still answers). Re-run: `auto` picks spread to the real models.

## Finding 13 — Laya GGUFs look like chat models to `Model.Capabilities` (fixed in the router view)

**What:** `laya:latest` reported `["completion"]`, not `decision` — the decisions handler detects Laya by family (`isLayaDecisionModel`) instead.

**Fix:** the router inventory maps Laya models to `[decision]` only, so chat never lands on the encoder and `/v1/decisions` `auto` finds it. `Model.Capabilities` itself is unchanged (scheduler / `/api/tags` depend on it) — worth fixing there separately.

## Finding 14 — A system Ollama install crashes zerollama runners (not fixed)

**What:** With vanilla Ollama installed, `ml.LibOllamaPath` picked `/usr/local/lib/ollama` (it has `rocm/` / `cuda_*` plugin dirs and the search prefers dirs with plugins). Loading Ollama's `libggml-*` into zerollama's patched ggml aborts (`SIGABRT` in `ggml_backend_load_all_from_path`); every runner start failed and requests hung on load cooldown.

**Workaround:** `OLLAMA_LIBRARY_PATH` pointing at a zerollama-owned dir that contains a plugin subdir (an empty `rocm/` suffices). **Fix needed:** skip system paths whose `libggml-base` version / build does not match the binary, or an explicit opt-out.

## Finding 15 — `-Inf` log-probs broke router JSON (fixed)

**What:** Laya rounds probabilities to 4 decimals, so jev decisions carried `log(0) = -Inf`; `encoding/json` refused it and `/v1/route` answered 200 with an empty body. The LA11 score tier can produce `-Inf` the same way for unscored labels.

**Fix:** `RouterLabelScore` encodes non-finite `log_prob` as `null` (field order unchanged).

## Finding 16 — Multi-question latency excluded the decider (fixed)

**What:** `smart` reported 0 ms decisions; `decideAuto` only timed its deterministic pick, not the decider batch before it (wall clock 4.8 s on CPU).

**Fix:** `decideRouterJevMulti` times the whole decision.

## Finding 17 — The LA11 score classifier decodes token by token

**What:** One `/api/router/decide` on a warm 1B scorer issued ~530 decode batches (≈ prompt tokens × labels, each forcing full-output logits) — 15 s per decision on CPU, ~60 s with chat-formatted prompts.

**Why it matters:** the score tier cannot meet a 50 ms gate on CPU and will stay slow on GPU. The LA9 `/api/score` path should batch candidates and share the prompt prefix. Not changed here.

## Finding 18 — Zero-shot Laya is noisier than the keyword fallback for output modality

**What:** `smart`'s `output_modality` question flagged 6/84 text prompts ("hey! how's it going?" → speech, limerick / sonnet → music) where the keyword fallback flagged 0/84 on the same set.

**Learning:** keep `ZEROLLAMA_ROUTER_SUGGEST=auto` but consider raising `suggest_threshold` for jev or preferring the keyword rule until a decider is trained / calibrated for this question.

## Finding 19 — The vendor workspace switches Go into vendor mode

**What:** `rebase_vendor_unified.sh` creates the gitignored top-level `vendor/llama-cpp-*`; any top-level `vendor/` makes `go build` / `go test` default to `-mod=vendor` and fail on `vendor/modules.txt`.

**Learning:** build with `GOFLAGS=-mod=mod` (as `build_mlx_dylibs_mac.sh` already does) whenever the vendor workspace exists.

## Finding 20 — Laya decisions cost ~0.4 s + ~0.1 s per option, per question — even on GPU (fixed)

**What:** With Laya fully offloaded to the RX 7900 XTX (29/29 layers, 1.4 GB VRAM), warm `/v1/decisions` medians were 491 ms (1 option), 899 ms (5 options), 1.42 s (10 options); 5 options × 2 questions 1.72 s, × 4 questions 3.41 s — the same order as CPU F32 (~1.1 s for 5 options).

**Why it matters:** the encoder is not the bottleneck; the serving path is (fixed ~0.4 s per question plus a per-option cost, linear in questions — consistent with per-question encodes and heavy padding, or the CPU decision head / Go-side packing). Until that is fixed no jev router can meet a 50 ms gate, and `smart` (4 questions) costs seconds.

**Learning:** profile the Laya runner (`llama_server_decisions.go` pack → `--decisions` encode → CPU head) before tuning router timeouts; batch all questions into one encode.

**Fix (2026-10-15):** `LLAMA_LAYA_TIMING=1` showed encode ~25 ms on GPU and the scalar, single-threaded CPU decision head at 400–1,360 ms. Patch **0142** batches every question into one encode and makes the head vectorized and threaded (`LLAMA_LAYA_THREADS`, default min(16, cores/2)); outputs match the Python reference (max |Δ| 0.0004, no flipped choices). The Go side then cost ~50 ms on top, which was **not** tokenization (cached anyway: `layaTokenCache`, ~28 `/tokenize` round trips → 0–1) but the scheduler (Finding 22). End to end at production timeouts: `agent-jev` p50 ~70 ms, `smart` (4 questions) ~115 ms, `/v1/decisions` 4 questions × 5 options ~115 ms — of which Laya itself is ~30 ms encode + ~45 ms head.

## Finding 21 — No AMD/HIP build of this tree worked at b11351 (fixed)

**What:** Building ggml-hip for gfx1100 hit three independent gaps, each of which would also block any ROCm user:

1. `GGML_CUDA_FA_QUANTS` unset — upstream defines it in `ggml/CMakeLists.txt`, which zerollama's root CMake skips → CUDA / HIP configure fails. **Fix:** mirror the upstream default in root `CMakeLists.txt`.
2. `cudaDriverGetVersion` (patch 0015, Ollama GPU discovery) had no HIP mapping in `ggml-cuda/vendors/hip.h`. **Fix:** patch **0140**.
3. `ggml-hip` did not compile the zerollama TBQ `fattn-vec*tbq*.cu` instances that `ggml-cuda` adds → `libggml-hip.so` failed to load (undefined `ggml_cuda_flash_attn_ext_vec_case<64, TBQ3_0, TBQ3_0>`). **Fix:** patch **0141**.

**Also:** installing only the CMake `HIP` component (with the ROCm preset) leaves `libggml-base` / `libggml.so` / CPU variants out of the `lib/ollama` root the native GPU probe dlopens; build target `ggml` and install `CPU` too, laid out as `lib/ollama/{libggml*.so}` + `lib/ollama/rocm/`. Under WSL, the KFD sysfs warning (`/sys/class/kfd/...`) is harmless — HIP enumerates the card through `/dev/dxg`. With all fixes, discovery reports `library=ROCm compute=gfx1100 "AMD Radeon RX 7900 XTX"` and `qwen3:8b` decodes ~100 tok/s.

## Finding 22 — Every scheduled request probed for `nvidia-smi` (fixed)

**What:** `applyGgmlNumCtxClamp` runs on every `scheduleRunner` call, warm runners included, and builds its load profile with `ggmlLoadProfileFor(model, opts)` → `SelectGpuProfile(nil)`. With no devices, `primaryGPUIdentity` falls back to `nvidia-smi` — so an NVIDIA node spawned `nvidia-smi` per request, and on this AMD/WSL lab box the failed PATH lookup alone (23 `/mnt/c` entries) cost **~35 ms per request**: a third of a Laya decision, and the same tax on every chat / generate request.

**Fix:** cache the `nvidia-smi` identity in `llm/gpu_profile.go` (a hit for the process; a miss retried after a minute so a slow startup probe does not disable profile matching). Found with temporary phase timing: `scheduleRunner` 53 ms of which `GetRunner` was 6.5 ms; `suggestMaxGgmlNumCtx`, metadata parse and the free-VRAM probe were all < 3 ms.

**Then the rest of the scheduler bookkeeping (fixed):** two more per-request file reads, both re-reading immutable GGUF headers:

| Call | Per call before (Laya / `qwen3:8b`) | Fix |
|------|-------------------------------------|-----|
| `Model.Capabilities()` — `gguf.Open` reads the whole KV section (150k vocab strings) to test 3 keys; runs in `CheckCapabilities` and again in `GetRunner` | 6.5 ms / **17.5 ms** | `ggufCapabilityFlagsFor` cache in `server/images.go` |
| `llm.LoadModelMetadata` — skips the vocab string by string; called by the n_ctx_train cap and the VRAM clamp | 2.5 ms / 7.5 ms | per-file cache in `llm/server_shared.go` (shared, read-only result) |

Both caches key on path + size + mtime, so a rewritten file (LM Studio repair, an external `--link` GGUF) is read again (`TestModelFileCachesRevalidate`). Result: everything before the runner call in `/api/chat` takes **~2.5 ms**; `scheduleRunner` went from 25–50 ms to 1–2 ms. Router: `agent-jev` p50 65 → **~44 ms**, `smart` ~115 → **~83 ms**, 4-question `/v1/decisions` ~115 → **~80 ms** (Laya's own encode + head).

## Finding 24 — llama-server's RAM prompt cache costs ~70 ms per cold request on this box

**What:** after Finding 22, a one-token `qwen3:8b` chat still spent ~100 ms in `r.Completion` against ~30 ms of llama-server prompt + eval time. llama-server logged `prompt cache update took ~70 ms` on every request: when no slot holds a similar prompt it picks one by LRU, then saves that slot's state to the `--cache-ram` host cache, tries a load, and updates the cache. The cost was flat (13 → 20 entries, ~70 ms), so it is not the lookup; each saved prompt is only 4.8 MiB, so likely many small synchronous device↔host transfers, which are slow through WSL's `/dev/dxg`. Not measured on native Linux.

**Evidence:** `LLAMA_ARG_CACHE_RAM=0` → server overhead per chat 74 ms → **2 ms**, one-shot request ~104 ms → 25–31 ms.

**Not changed:** the RAM cache is what lets a returning conversation skip prefill once there are more live conversations than slots (`-np 8`). Disabling it trades ~70 ms on cold or one-shot requests for full re-prefill of long returning chats. Options: leave the default; disable it per node (`LLAMA_ARG_CACHE_RAM=0`) where traffic is mostly one-shot (routing helpers, eval, short API calls); or patch llama-server to skip saving prompts too short to be worth caching (re-prefilling 34 tokens costs ~37 ms here; saving costs ~70 ms). Measure on a native Linux / CUDA node before choosing — the save cost there is probably a few ms.

## Finding 23 — One judge, one order, 16 pairs is noise

**What:** `agent-jev` went from 8 / 2 / 6 (lab run 2, "PASS") to 5 / 2 / 9 (lab run 3) on the same 16-pair budget. A sign test puts the first at p = 0.79 — indistinguishable from a coin. Single-order LLM judging also bakes in position bias.

**Fix (harness):** judge every pair in **both orders** (a preference that flips with position is a tie), allow a **panel** (`--judge` repeatable; a win / loss needs every judge to agree), print a **sign-test p**, break results down **by routed model**, and grow the text-chat pool from 78 to 198 prompts (`prompts.jsonl` r130–r249, balanced 8 per task × tier). Also: unload other models before each generation / judge phase (a resident baseline pushed a 27B judge half onto CPU — 2.9 tok/s prefill, 300 s timeouts), treat timeouts as failed calls instead of crashing, warm the decider before each routing pass, and `--cache` answers / verdicts so a rerun resumes.

## Finding 10 — AR7 gate (definition and lab runs)

**What:** `scripts/eval/router_eval.py` replays `testdata/router_eval/prompts.jsonl` (129 model-agnostic prompts: 5 tasks, 4 surfaces, tools / image / video / audio attachments, expected tier, expected output modality) through `/v1/route` with `route: "fresh"`.

**Gate (per router):**

| Check | Pass condition |
|-------|----------------|
| Capability correctness | 100% of servable rows (rows no local model can serve count as `unservable`, not failures) |
| Decision latency | p50 ≤ 50 ms (`--gate-ms`) for deterministic routing; ≤ 150 ms (`--gate-ms-model`) for a router whose classifier calls a model (jev, score, knn, rerank) — a forward pass cannot fit in 50 ms |
| Quality (`--judge` + `--baseline`) | routed wins + ties ≥ 50% of judged pairs **and** wins ≥ losses vs a fixed baseline model |

No quality run ⇒ **INCOMPLETE** — tier / strength accuracy shows the router follows its policy, not that the policy is good.

**Status:** **AR4 `agent-jev` PASS on lab run 4 (2026-10-15)** — quality parity with a fixed 8B at ~9% more average compute (8.75 B vs 8 B params per prompt), p50 ~44 ms after Finding 22. AR5 `smart` FAIL (dominated) — stays opt-in.

**Lab run 1** — RX 7900 XTX box, but **CPU inference** (ROCm runtime present, HIP / hipBLAS dev headers missing, so no ggml-hip build). Models: `llama3.2:1b` (score classifier), `llama3.2:3b` (low), `qwen3:8b` (mid / fallback), `qwen3:14b` (high), Laya F32 GGUF (`convaiinnovations/laya`) via patched `llama-server --decisions`. Lab copy of `router_lab.example.yaml` with `qwen3:14b` as high tier and decider timeouts raised (150 ms → 5 s / 8 s) so CPU latency did not force every jev decision to fail open. 25 rows unservable (no local vision / audio / image model) — correctly 503 with reasons.

| Router | Errors | Caps | Picks 3b / 8b / 14b | Tier exact / under / over | Strength hit | Decision p50 | Cross-modal recall / FP |
|--------|--------|------|---------------------|---------------------------|--------------|--------------|-------------------------|
| `auto` (implicit) | 0 | 6/6 | 0 / 0 / 104 | — (no tiers) | — | < 1 ms | 10/10, 0/84 (keyword) |
| `agent-score` (LA11) | 0 | — | sample of 8 only | 25% accuracy, 4/8 fallback | — | **15–60 s** | — |
| `agent-jev` (AR4) | 0 | 6/6 | 13 / 62 / 29 | 33% / 26% / 41% | 88% | 1.6 s | 10/10, 0/84 (keyword) |
| `smart` (AR5) | 0 | 6/6 | 1 / 33 / 70 | 36% / 6% / 57% | 69% | ~4.8 s † | 9/10, **6/84** (jev) |

† `smart` reported 0 ms until Finding 16 was fixed; wall-clock p50 was 4.8 s.

**Read-out:**

- **Capability routing is right** on every router (tools rows 6/6, 25 unservable rows explained, zero routing errors) — the AR1–AR3 core holds on real models.
- **AR4 beats LA11 score routing** on the A/B that could be run: jev 88% strength hit vs score 25% sample accuracy with half its decisions falling back (multi-label activation at 0.15 makes "code 0.69 + extract 0.21" match no single candidate). Laya's own answers were sharp (code 0.998, math 0.988, extract 0.98).
- **Latency fails the 50 ms gate on CPU for every classifier** — Laya ~1.1–1.2 s warm per decision (F32, 421M), the score classifier 15–60 s (Finding 17). GPU numbers are still needed before judging the gate.
- **AR5 over-escalates** with the example policy: 70/104 to the 14B model, 57% over-tier, 24 escalations. Tune `escalate_at` / `escalate_below` / strengths before considering it outside opt-in.
- **Tier labels are the author's guesses** (Finding 11 applies); treat tier accuracy as a policy-consistency signal, not quality.
- **Quality (judged) not run:** needs the 8B / 14B models generating plus a judge larger than the baseline; ~7 GB free RAM and no GPU backend made that infeasible.

**Lab run 2 — GPU, judged (2026-10-15).** Same box after installing `amdrocm-runtime-dev10.1` / `amdrocm-blas-dev10.1`: ggml-hip + patched `llama-server` (HIP) on the RX 7900 XTX (gfx1100; Finding 21 fixes). Judge `gemma3:27b`, baseline `qwen3:8b`, `think: false` on every answer, decider timeouts 5 s / 8 s (production 150 / 250 ms would make every jev decision fail open on this Laya path — Finding 20). Harness fixes from this run: servability per router candidate set, batched + cached generation (no judge VRAM thrash), gate counts only real (non-same-model) pairs with an absolute `--min-pairs` (10) and a 25% failure cap. `ZEROLLAMA_HOST_MEM_GUARD=false` in the lab only (host RAM held by an unrelated process).

| Router | Sample | Real pairs | Routed W / T / L vs `qwen3:8b` | Decision p50 | Quality | Gate |
|--------|--------|------------|-------------------------------|--------------|---------|------|
| `agent-jev` (AR4) | 50 | 16 (34 picked the baseline) | **8 / 2 / 6** | 1.47 s | **PASS** | FAIL (latency) |
| `smart` (AR5) | 20 | 16 | 5 / 4 / 7 | 4.23 s | FAIL | FAIL |
| `auto` | — | — | not judged: picks the largest local model, which is the judge itself (`gemma3:27b`) — self-preference | < 1 ms | — | — |

**Read-out:** AR4 jev routing beats a fixed 8B on the pairs where it routes away (8 W vs 6 L) — the quality case for jev holds, on a small sample. AR5 as configured does not (more losses than wins; it escalates to the 14B too often). **Neither meets 50 ms**, and the cause is the Laya serving path (Finding 20), not the router. AR5 stays opt-in; AR4 is the candidate for wider use once Laya decisions are batched into one encode. Small samples, one judge, author-labelled tiers (Finding 11) — treat as directional.

**Next:** fix Laya batching (Finding 20) and re-measure latency at production timeouts; tune `smart` (`escalate_at`, `escalate_below`, strengths) and re-judge; add a second judge family to check `agent-jev`'s margin.

**Lab run 3 — production decider timeouts (150 / 250 ms), patch 0142 (2026-10-15).** Same models and judge as run 2. Laya now decides inside the timeouts, so jev routing is real rather than falling back.

| Router | Real pairs | W / T / L vs `qwen3:8b` | sign p | Decision p50 / p95 | Cross-modal recall / FP |
|--------|------------|-------------------------|--------|--------------------|-------------------------|
| `agent-jev` | 16 | 5 / 2 / 9 | 0.42 | 99 / 111 ms | 10/10, 0/84 |
| `smart` | 39 | 9 / 8 / 22 | **0.03** | 156 / 179 ms | 9/10, 6/84 |

**Read-out:** `smart` looked measurably worse than a fixed 8B. `agent-jev` flipped from run 2 — evidence that 16 single-judge pairs cannot decide it (Finding 23). Latency was then cut further (Finding 22).

**Lab run 4 — panel judged, 249 prompts, Findings 22–23 fixes (2026-10-15).** Production timeouts, 198 text-chat prompts, every one judged (`--quality-sample 250`), both orders, panel `gemma3:27b` + `mistral-small3.2:24b` (a win / loss needs both judges in both orders), baseline `qwen3:8b`, `--min-pairs 30`.

| Router | Picks 3b / 8b / 14b | Real pairs | Panel W / T / L | sign p | By routed model (W / T / L) | Decision p50 / p95 | Cross-modal FP | Gate |
|--------|---------------------|------------|-----------------|--------|-----------------------------|--------------------|----------------|------|
| `agent-jev` | 28 / 122 / 48 | 76 | 11 / 57 / 8 | 0.65 | 3b 4 / 19 / 5 · 14b 7 / 38 / 3 | **65 / 78 ms** | 0/204 | **PASS** |
| `smart` | 1 / 62 / 135 | 136 | 16 / 101 / 19 | 0.74 | 14b 16 / 100 / 19 | 148 / 176 ms | 12/204 | FAIL (W < L) |

Single judges agree on the shape: `agent-jev` gemma 17 / 40 / 19, mistral 13 / 50 / 13; `smart` gemma 26 / 75 / 35, mistral 20 / 94 / 22.

**Read-out:**

- **Neither router changes answer quality measurably** versus a fixed `qwen3:8b` (p 0.65 / 0.74). Run 3's "`smart` is worse" (p 0.03) did not survive both orders and a second judge — most single-order verdicts were position noise; 75–85% of real pairs are panel ties.
- **`agent-jev` neither gains quality nor saves compute on this ladder:** it sends 28 prompts to the 3B with no measurable loss (4 / 19 / 5) but 48 to the 14B for a small, non-significant gain (7 / 38 / 3) — an average of 8.75 B parameters per prompt vs 8 B for the fixed baseline (`smart`: 12.1 B). It passes the gate (`--gate-ms-model` 150) because the gate checks "no worse", not "cheaper". The 3B result is the promising part: Laya found prompts a 3B handles as well as an 8B.
- **`smart` is dominated:** 68% of prompts to the 14B (46 escalations) for the same quality as the 8B — strictly more compute than `agent-jev` for no gain, plus 12 cross-modal false positives. It stays opt-in; tuning its thresholds against this set would overfit it (Finding 11), and loosening escalation only converges on `agent-jev`.
- **The candidate ladder limits what any router can show here:** with `think: false`, the local 14B barely beats the 8B on these prompts. A router earns quality only when its candidates differ in quality; on this node it earns savings.

**Status:** AR4 (`agent-jev`) meets the AR7 gate on lab run 4. AR5 (`smart`) stays opt-in.

**Run** (copy [`scripts/eval/router_lab.example.yaml`](../scripts/eval/router_lab.example.yaml) to the node's `router.yaml` and set its model tags first):

```bash
./scripts/eval/router_eval.py --host http://127.0.0.1:11435 \
  --router agent-score --router agent-jev --router smart \
  --judge <judge-model> --judge <second-judge-family> --baseline <fixed-model> \
  --quality-sample 250 --min-pairs 30 --cache /tmp/router_eval_cache.json \
  --json /tmp/router_eval.json testdata/router_eval/prompts.jsonl
```
