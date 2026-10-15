# Auto-router (`model: "auto"`) — capability dispatch + Jev decisions, all modalities

**Status:** In progress — **AR0** landed (`c5f02942`), **AR1** landed (`5ed9f886`), **AR2** landed (`0fadfe62`), **AR3** landed (`7228ffc7`), **AR4** landed (`40d49d92`), **AR5** landed (`ecb71df2`), **AR6** landed (`21c77d4c`), **AR7** harness landed, gate run (lab run 4: AR4 PASS, AR5 FAIL — [findings §10](auto-router-findings.md#finding-10--ar7-gate-definition-and-lab-runs)), **AR8** landed (`8ae6fffc`); **AR9 deferred**, **AR10 parked** (see §15). Milestones **AR0–AR10**.  
**Builds on:** LA11 router ([router_policy.go](../server/router_policy.go), [router_handler.go](../server/router_handler.go)) · typed decisions ([laya-llama-cpp.md](./laya-llama-cpp.md), [system-one-score.md](./system-one-score.md), [gliner-decide.md](./gliner-decide.md)) · can-load ([can_load.go](../server/can_load.go)) · bench cache ([bench-cache.md](./bench-cache.md)) · QoS ([agent-qos-and-project-tracking.md](./agent-qos-and-project-tracking.md)) · fleet ([fleet-scheduling.md](./fleet-scheduling.md))  
**Audience:** zerollama contributors deciding whether and how to build this; operators who will write `router.yaml`.

---

## 1. Why this exists

Three gaps show up when one daemon serves chat, vision, images, video, speech and decisions:

1. **Clients hard-code one model name.** Hermes, OpenAI SDK clients and simple agents send the same `model` on every turn. An easy turn pays for the 30B; a hard turn gets the 3B.
2. **The request often already says which model class it needs — and nothing uses that.** Images in the payload need a vision model; `tools` needs tool support; a 40k-token prompt needs a long-context model; `/v1/audio/speech` with `voice: "nova"` needs whichever TTS tag owns that voice.
3. **Cross-modal intent arrives as chat text.** "Draw a diagram of the scheduler" or "read that back to me" lands on `/v1/chat/completions`; nothing knows it wants image gen or TTS. This is the only place a *semantic* classifier is required — and it is what Jev / `/v1/decisions` is good at.

**What LA11 already does:** `model: <router-name>` on `/api/chat` and `/api/generate` (and `/v1/images/*`, which the middleware turns into a generate request) runs one classifier — logprob score, KNN or rerank — over the last user message, then picks the first candidate whose labels cover the active set. Rewrite is in-band; headers `X-Zerollama-Router*` report the choice.

**What LA11 does not do:**

| Gap | Consequence |
|-----|-------------|
| Single label question | Cannot ask "task + difficulty + needs reasoning" in one call |
| No capability / surface filter | A router name sent to `/v1/images/generations` can pick a **chat** model today (classifier runs on the image prompt, candidate list is surface-blind) |
| No availability | Ignores warm residency, busy runners, load cooldown, context fit, bench speed |
| Chat/generate only | Video, speech, transcription, music, embed, rerank, extract, decisions never consult the router (video / speech / transcription / music / extract do not even apply aliases) |
| No stickiness | Every turn re-classifies; a model flip mid-thread loses the KV prefix and the persona |

## 2. Is it worth it? (scope verdict)

**Yes, narrowly.** The value is front-loaded in the cheap part:

| Layer | Value | Cost | Needs Jev? |
|-------|-------|------|------------|
| Deterministic capability + availability dispatch | High — fixes wrong-class picks, avoids model swaps, works on every surface | Low | **No** |
| Small/large escalation by difficulty | Medium — real on multi-GPU / 48 GB+ hosts; marginal on one 16 GB GPU where swap cost dominates | Medium | Yes (calibrated) |
| Cross-modal intent from chat | High *product* value (README "same place as chat") | Medium + contract decision | Yes |
| Jev-scored choice among media backends | Low — most hosts have 1 TTS, 1 image pipeline, 1–2 video families | — | **Do not build** |

**Single-GPU caveat:** a load costs seconds to minutes. On one GPU the router should almost always prefer the warm model; quality routing only wins when the alternative is already resident or the host has headroom. The scorer encodes this via `needs_eviction` / `already_loaded` from can-load (§7).

**Evidence gate:** hand-written `strengths:` are guesses. AR7 (eval harness) gates AR5+ — Jev escalation only expands if it beats a fixed baseline on a labeled set.

## 3. Non-goals

- **Not cross-node placement.** Fleet (F7 `ScoreCandidates`) picks a *node* for a fixed model. The auto-router picks the *model* on this node. They compose: route → assign (§12).
- **Not dynamic embedder / reranker switching.** Embedding spaces are incompatible across models; `auto` on embed resolves to **one pinned model per router**, forever (§6.8).
- **Not a load trigger.** Routing uses dry-run signals (`evaluateCanLoad`, cooldown) only; the chosen handler loads as it does today.
- **Not mid-thread switching by default.** Sticky per session; re-route only when a hard filter fails (§8).
- **No silent response-shape changes.** Chat never returns an image unless the client opts in (§9).
- **Not replacing aliases.** Aliases resolve first (one hop); router names are a separate namespace.
- **No Jev on the decider itself.** `/v1/decisions` with `model: auto` uses deterministic rules (§6.9) — no recursion.

---

## 4. Architecture

```text
 HTTP surface (chat, generate, images, videos, speech, transcribe, music,
               embed, rerank, extract, decisions)
        │
        ▼
 ① surface adapter ──► RouteSignals{surface, text, attachments, tools, tokens,
        │                            voice, size, keyframes, qos, session_key}
        ▼
 ② applyModelAlias  (now on every surface)
        ▼
 ③ lookupRouter(model)? ── no ──► unchanged path (zero overhead)
        │ yes
        ▼
 ④ sticky cache hit? ── yes ──► reuse decision (re-check hard filters only)
        │ no
        ▼
 ⑤ inventory (cached per digest) ─► hard filters ─► availability (can-load dry run)
        │
        ├── 0 left  ─► fallback / 503 with reasons
        ├── 1 left  ─► done (skip Jev)
        ▼
 ⑥ Jev tier (optional, per router): one /v1/decisions batch on the pinned decider
        ▼
 ⑦ score + escalate ─► RouteDecision ─► cache + headers + metrics + log
        ▼
 ⑧ dispatch: in-band rewrite of req.Model  │  or advisory / handoff for cross-modal (§9)
```

**Why one function, many adapters:** every handler already does `alias → parse → getExistingName → GetModel → CheckCapabilities`. The router slots in after alias and before `GetModel`, so each handler gains ~10 lines, and the policy lives in one place (`server/autoroute/`).

**Why route before video expansion:** chat calls `ExpandVideosInChatRequest` (ffmpeg → frames) after capability checks ([routes.go:3593–3667](../server/routes.go)). The router must see `Videos[]` pre-expansion so it requires `CapabilityVideo`, and must not pay ffmpeg for a candidate that will be rejected.

---

## 5. Core types (sketch)

```go
// server/autoroute/signals.go
type Surface string // chat | generate | image | image_edit | video_gen | speech | transcribe | music | embed | rerank | extract | decisions

type RouteSignals struct {
	Surface      Surface
	Text         string   // last user message / prompt / input — what Jev sees
	System       string   // truncated system prompt (stickiness key + Jev state)
	HasImages    bool
	HasVideo     bool     // pre-expansion Videos[] / video_url parts
	HasAudio     bool
	HasTools     bool
	WantsThink   bool
	PromptTokens int      // cheap estimate (bytes/4) unless tokenizer is warm
	Width, Height int     // image / video
	DurationSec  float64  // video / music / speech max
	Keyframes    int      // options.keyframes count (video)
	ControlImage bool     // comfy control_image / LoRA hints
	Voice, Language string
	QoSClass     string   // options.zerollama.qos_class (resolved default)
	SessionKey   string   // prompt_cache_key | options.zerollama session | hash(system+first user)
}

// server/autoroute/inventory.go — one row per local manifest, rebuilt when digest changes.
type Candidate struct {
	Name, Digest   string
	Caps           []model.Capability // memoised: m.Capabilities() reopens the GGUF each call
	Backends       map[string]string  // ModalityBackends
	ContextLength  int
	SizeBytes      int64
	VideoProfile   string             // resolveVideoGenerationConfig result
	Voices         []string           // modality.ListSpeechVoices
	PerfTokS       float64            // bench.json (completion)
	PerfSec        float64            // bench.json (image / video_gen)
	Platform       []string           // e.g. ltx-mlx → darwin only
}

type RouteDecision struct {
	Router, Surface, Model string
	Fallback, Sticky, Escalated bool
	Signals   map[string]any        // Jev answers (probabilities, noul) + hard signals
	Filtered  []Rejection           // {model, reason} — honest "why not"
	Ranked    []Ranked              // {model, score, warm, needs_eviction, reasons}
	Decider   string
	Calibrated bool
	LatencyMs int64
}
```

**Why memoise capabilities:** `GetModel` is digest-cached but `Capabilities()` reopens the GGUF on every call. Building a candidate list on every request would open every GGUF on disk. The inventory recomputes a row only when its manifest digest changes (same invalidation rule as `inferenceModelCacheDefault`).

**Why bench is read server-side:** `cmd/benchcache` is client-only today. AR0 moves the reader to a shared package (read-only; `zerollama bench` stays the only writer). Missing perf ⇒ neutral speed score, not exclusion.

---

## 6. Per-modality routing

Every row: what the handler does today → hard filters the router applies → whether Jev participates → soft score inputs.

| # | Surface | Entry today | Hard filters | Jev? | Soft score |
|---|---------|-------------|--------------|------|-----------|
| 6.1 | Text chat / generate | `ChatHandler` [routes.go:3518](../server/routes.go), `GenerateHandler` [:601](../server/routes.go) — LA11 already hooked | `completion`; `tools` if tools; `thinking` if think requested; context ≥ tokens + reply budget | **Yes** — task, difficulty, reasoning | fit, warm, tok/s, size |
| 6.2 | Vision / video / audio understanding (chat) | Cap checks [routes.go:3593–3616](../server/routes.go) | `vision`; `vision`+`video` if videos; `audio` if audio; plus 6.1 filters | Same as 6.1 (optional) | warm dominates (vision GGUFs are large) |
| 6.3 | Image generate | Middleware → `GenerateHandler` → `handleImageGenerate` (MLX / ComfyUI / external / OpenVINO) | `image`; ≤4096 px; platform | Only if ≥2 image models: `style` choice (photo / illustration / diagram / text-heavy) | sec/image, warm, background QoS |
| 6.4 | Image edit | Same, `ImageEditsMiddleware` | `image` **and** backend accepts input image (comfyui edit workflows); `control_image` / LoRA ⇒ comfyui | No | sec/image |
| 6.5 | Video generate | `VideoCreateHandler` [video_generate.go:106](../server/video_generate.go) — **not routed** | `video_gen`; backend ∈ wan/ltx/h3; keyframes supported by backend; platform (ltx-mlx ⇒ darwin); **host-RAM admission as a predicate** (`admitLtxHostRAM` / `admitH3HostRAM` / `admitWanHostRAM`) | No (optional `quality` vs `draft` from `options`) | gen_sec, profile size |
| 6.6 | Speech (TTS) | [speech.go:25](../server/speech.go) — **not routed, no alias** | `speech`; backend piper/remote-tts (not music3); **voice ∈ model voices**; language | No | warm, latency |
| 6.7 | Transcription | [transcription.go:25](../server/transcription.go) — multipart `model` | whisper backend, or `audio`-capable chat model; language | No | whisper preferred (cheaper) |
| 6.8 | Embed | `EmbedHandler` [routes.go:1295](../server/routes.go) — no cap gate today | `embedding` | **Never** | **Pinned**: router's `embedder:` only |
| 6.9 | Rerank | [rerank_handler.go:38](../server/rerank_handler.go) | reranker-capable | **Never** | **Pinned** |
| 6.10 | Extract (NER) | [extract_handler.go](../server/extract_handler.go) — proxy only | `ZEROLLAMA_GLINER_URL` reachable | No | single backend |
| 6.11 | Decisions | `DecisionsHandler` | by question shape (below) | **No (recursion)** | calibrated first |
| 6.12 | Music | [music_generate.go:46](../server/music_generate.go) | `speech` + `music3`; darwin | No | single backend |

### Notes that change implementation

- **6.3 bug fix falls out of AR1:** today a router name on `/v1/images/*` runs a text classifier and may return a chat model. Surface filtering makes that impossible.
- **6.5 admission must become pure.** `admitLtxHostRAM` and friends return 503 + alternatives inside the handler. AR0 splits each into `predict(cfg, req) (ok bool, reason string, alts []string)` (pure) and the existing caller. GPU best-fit placement **stays in Python** (`cuda_device.py pick_cuda_device`) — the router only decides *which tag*, never *which GPU*.
- **6.6 voice is the real selector.** For TTS, `auto` + `voice: "nova"` resolves to the tag listing that voice (`/v1/audio/voices` data). Ambiguous voice ⇒ prefer warm, then router order.
- **6.8/6.9 pinned on purpose.** Two calls in one RAG pipeline routed to different embedders produce vectors that silently do not compare. `auto` here is an indirection for config, not a decision.
- **6.11 decider selection (deterministic):** `noul` questions ⇒ prefer Laya / CLM (trained boolean heads); `images` ⇒ Clef / tev1 score models; caller needs `calibrated:true` ⇒ exclude GLiNER-Decide and OpenJev unless `ZEROLLAMA_OPENJEV_CALIBRATED=1` (and `NOUL_CALIBRATED=1` for noul); `pointer_rows` ⇒ Strands on MLX only.
- **QoS is preserved, not decided.** `ensureQoSDefaults` (background for image/video) runs after routing exactly as today. The router reads `QoSClass` only to prefer warm models for `interactive`.

---

## 7. Selection: filter → (Jev) → score

### Hard filters (always, cheap, deterministic)

In order, each rejection recorded with a reason (`Filtered[]`):

1. Surface capability (table §6)
2. Request-derived capabilities (images / video / audio / tools / think)
3. Context fit: `ContextLength ≥ PromptTokens + max(num_predict, 1024)`
4. Platform / backend constraints (ltx-mlx darwin, Strands MLX, music3 darwin)
5. Surface specifics: voice ownership, keyframe support, edit-capable backend, host-RAM predicate
6. Availability: `cooldownErr(key) == nil`; `evaluateCanLoad` ⇒ drop if `!can_load && !already_loaded`, or `loads_paused` and not warm

**Why can-load and not the scheduler:** `evaluateCanLoad` is explicitly a dry run that never enqueues ([can_load.go:24–28](../server/can_load.go)). Calling `GetRunner` would start loads for every candidate.

### Jev tier (only when the router declares `questions:` and ≥2 candidates survive)

- **One batch call** to the router's `decider` through an extracted `s.runDecisions(ctx, llm.DecisionsRequest)` (AR0 refactor of `DecisionsHandler`; HTTP handler becomes a thin wrapper).
- **State packing:** `{"request": Text (≤ ~2k tokens, head+tail), "system": System (≤256 tokens), "attachments": ["image"], "has_tools": true}`. Laya is an encoder; long state truncates silently — pack deliberately.
- **Caps enforced by existing validation:** ≤64 questions, ≤255 choice options, ≤10 score levels. Question **insertion order** is load-bearing for score models — router YAML is parsed into an ordered list, not a map.
- **Calibration gate:** `calibrated:false` (OpenJev default, GLiNER-Decide) ⇒ argmax only; no threshold activation, no escalation.
- **Budget:** decider timeout (default 150 ms); on timeout or error ⇒ continue with deterministic score (`fallback_reason: decider_timeout`). **Fail open** unless `on_decider_error: fail`.

### Score

```text
score(c) =  w_fit   · fit(c, answers)          // strengths ∩ task answer, difficulty band match, reasoning ⇒ thinking cap
          + w_warm  · already_loaded(c)        // 1 / 0
          − w_evict · needs_eviction(c)        // swap cost; heavier on single GPU
          + w_speed · norm(perf(c))            // tok/s or 1/sec from bench; neutral if unknown
          − w_size  · norm(size(c))            // prefer small when fit ties
          + w_stick · sticky_prev(c)           // session affinity
```

Defaults: `w_fit 1.0, w_warm 0.6, w_evict 0.8, w_speed 0.2, w_size 0.1, w_stick 1.5`. On a single GPU (`device_count ≤ 1` from can-load topology) `w_evict` doubles. Ties ⇒ router candidate order.

**Escalation** (calibrated decider only): if top task probability `< escalate_below` (default 0.55) **or** `difficulty ≥ escalate_at` **or** reasoning noul true ⇒ restrict to candidates with `tier: high` before scoring; record `escalated: true`. Laya's act/escalate head, when present, is used directly instead of the heuristics.

---

## 8. Stickiness and caching

- **Session key:** `prompt_cache_key` → `options.zerollama.session_parent`/`session_group` → `hash(router, system, first user message)`.
- **Sticky table:** `session_key → (model, digest, surface, expires)`; TTL default 30 min, refreshed on use; LRU-bounded.
- **Re-route only when** the sticky model fails a hard filter for *this* request (images appear, context outgrown, model deleted / cooled down). Recorded as `sticky_break: <reason>`.
- **Decision cache:** `(router, surface, hash(Text), inventory_epoch)` ⇒ decision, short TTL (60 s) — dedupes agent retries and Hermes batch fan-out.

**Why sticky by default:** a flip mid-thread discards the Radix / prompt-cache prefix and changes voice; for agent loops that is worse than a slightly sub-optimal model.

---

## 9. Cross-modal intent (chat → image / speech / video / music)

Jev question (optional per router):

```yaml
output_modality:
  type: choice
  instructions: "What should the reply be?"
  criteria:
    text:   "a written answer"
    image:  "draw, render, illustrate or edit a picture"
    speech: "read aloud / say it"
    video:  "make a clip or animation"
    music:  "compose a song or melody"
```

Three contracts, in shipping order:

| Mode | Behaviour | Risk |
|------|-----------|------|
| **A. Advisory** (AR8, default) | Chat answers as text; response carries `X-Zerollama-Route-Suggest: image` + `/v1/route` returns the suggested surface, model and a ready request body | None — clients opt in by acting on it |
| **B. Handoff** (AR9, opt-in `options.zerollama.modality_handoff: true`) | Server runs the target surface and returns it in the chat response: image ⇒ content part (data URL / media id); speech ⇒ audio content part (OpenAI `audio` output shape); video / music ⇒ async job reference (`/v1/videos/{id}`) because they are minutes-class | Response shape differs per client wire (OpenAI vs Ollama vs Anthropic) — needs per-wire encoders and tests |
| **C. Tool-shaped** (directional) | Expose `generate_image` / `speak` / `make_video` as server-side tools; let the chat model call them | Depends on model tool quality; overlaps with agent harnesses |

**Why advisory first:** it is useful immediately for agents, has zero compatibility risk, and produces the labeled data (suggestion vs what the client actually did) that AR10 needs.

---

## 10. API surface

### In-band (every surface)

`model: "<router-name>"` — `auto` by convention — on `/api/chat`, `/api/generate`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`, `/v1/images/*`, `/v1/videos`, `/v1/audio/speech`, `/v1/audio/transcriptions` (multipart field), `/v1/audio/generations`, `/api/embed`, `/v1/embeddings`, `/v1/rerank`, `/v1/extract`, `/v1/decisions`.

Response headers (extends LA11): `X-Zerollama-Router`, `X-Zerollama-Router-Chosen`, `X-Zerollama-Router-Fallback`, **new** `X-Zerollama-Router-Reason` (short: `warm`, `escalated`, `sticky`, `only-candidate`), `X-Zerollama-Route-Suggest` (§9).

Per-request opt-out: `options.zerollama.route: "off"` (also accepted flat, per the M15c alias rules).

### Dry run

`POST /v1/route` — body is the target surface's normal request plus `"surface"` (inferred when obvious). Returns `RouteDecision` (§5) without inference. `/api/router/decide` stays as the LA11-compatible alias (`{router, prompt}` ⇒ chat surface).

```json
{
  "router": "auto", "surface": "chat", "model": "qwen3-coder:30b",
  "sticky": false, "escalated": true, "fallback": false,
  "signals": {
    "task": {"answer": "code", "probabilities": {"code": 0.91, "chat": 0.06, "math": 0.03}},
    "difficulty": {"answer": 4},
    "reasoning": {"noul": false},
    "has_tools": true, "prompt_tokens": 812
  },
  "filtered": [{"model": "llama3.2:3b", "reason": "missing capability: tools"}],
  "ranked": [
    {"model": "qwen3-coder:30b", "score": 2.31, "warm": true,  "needs_eviction": false},
    {"model": "qwen3:8b",        "score": 1.12, "warm": false, "needs_eviction": true}
  ],
  "decider": "laya", "calibrated": true, "latency_ms": 41
}
```

### Discovery

- `GET /v1/models` / `/api/tags`: router names appear as virtual models (`owned_by: "zerollama-router"`, `capabilities` = union of candidates) so OpenAI-style pickers can select `auto`.
- `GET /api/router/inventory` (debug): candidate rows with memoised caps, perf, warm state.
- `GET /api/version` → `zerollama.capabilities.auto_router: true` + supported surfaces (progressive client ladder).

---

## 11. Configuration (`router.yaml`, backward compatible)

Existing LA11 routers (`classifier: score|knn|rerank|colbert`) are untouched. New keys are additive; a router without `surfaces:` keeps LA11 semantics (chat/generate).

```yaml
routers:
  auto:
    surfaces: [chat, generate, image, image_edit, video_gen, speech, transcribe, music, embed, rerank, extract, decisions]
    classifier: jev                # jev | score | knn | rerank | none (deterministic only)
    decider: laya                  # any /v1/decisions model; pinned resident
    decider_timeout: 150ms
    on_decider_error: fallback     # fallback | fail
    sticky_ttl: 30m
    escalate_below: 0.55
    escalate_at: 4
    weights: {fit: 1.0, warm: 0.6, evict: 0.8, speed: 0.2, size: 0.1, stick: 1.5}

    questions:                     # ordered; only consulted on chat/generate (+ image style)
      - id: task
        type: choice
        instructions: "What kind of work is this?"
        criteria: {code: "write/fix/explain code", math: "calculation or proof", chat: "conversation or advice", extract: "pull facts from text", creative: "fiction, copy, poems"}
      - id: difficulty
        type: score
        instructions: "How hard is this for a small model?"
        labels: ["1", "2", "3", "4", "5"]
      - id: reasoning
        type: noul
        instructions: "Does this need multi-step reasoning?"
      - id: output_modality        # §9; omit to disable cross-modal suggestions
        type: choice
        instructions: "What should the reply be?"
        criteria: {text: "...", image: "...", speech: "...", video: "...", music: "..."}

    candidates:
      # text
      - {model: llama3.2:3b,      surfaces: [chat, generate], tier: low,  strengths: {task: [chat, extract]}, max_difficulty: 2}
      - {model: qwen3:8b,         surfaces: [chat, generate], tier: mid}
      - {model: qwen3-coder:30b,  surfaces: [chat, generate], tier: high, strengths: {task: [code, math]}}
      - {model: gemma3:12b,       surfaces: [chat, generate]}             # vision inferred from caps
      # media — usually one each; order is the tiebreak
      - {model: z-image-turbo,    surfaces: [image]}
      - {model: qwen-image-edit,  surfaces: [image, image_edit]}
      - {model: ltxv-2b-distilled:lab, surfaces: [video_gen]}
      - {model: kokoro,           surfaces: [speech]}
      - {model: whisper-large-v3, surfaces: [transcribe]}
      # pinned
    embedder: nomic-embed-text     # embed surface → always this
    reranker: bge-reranker-v2      # rerank surface → always this
    fallback: qwen3:8b             # chat/generate only; media surfaces 503 with reasons instead
```

**Why candidates can omit capabilities:** capabilities come from the inventory (manifest + GGUF). YAML declares *policy* (surfaces, tier, strengths), never facts the model file already states — avoids drift.

**Why no media fallback:** silently generating an image on a different pipeline than requested is worse than a 503 that lists `filtered[]` reasons.

**Built-in default (AR1):** if `router.yaml` has no `auto` router, `model: "auto"` uses an implicit deterministic router over all local models (all surfaces, no Jev). Opt-out: `ZEROLLAMA_AUTO_ROUTER=0`.

---

## 12. Interactions

| System | Interaction |
|--------|-------------|
| **Aliases (LA17)** | Resolve first; an alias may point at a router name; a router candidate may not be a router (existing check) |
| **Fleet (F7)** | Management node: route model first (needs inventory from `loaded_model_details`), then `ScoreCandidates` picks the node. Directional — local-only in AR1–AR9 |
| **QoS / session gate** | Unchanged; `ensureQoSDefaults` runs after routing; router never alters `qos_class` |
| **Concurrency groups (LA4)** | Decider should sit in a group that co-resides with chat models, or be CPU-only (GLiNER-Decide) — otherwise the decider load evicts the target |
| **Load cooldown (LA18)** | Hard filter; cooled-down models are skipped, not retried |
| **Pins / prewarm (LA21)** | Router decider is pinned at serve start when `classifier: jev` (`keep_alive: -1`) |
| **Cloud `:cloud` models** | Excluded from candidates in v1 (decisions and can-load are local-only) |
| **Python runtime proxy** | Routing happens in Go before `runtimeChatProxy`; the proxied request carries the rewritten model |

**Decider residency is the main operational risk.** On a single 16 GB GPU, use Laya (small encoder) or GLiNER-Decide on CPU. DiffusionGemma / OpenJev wants the whole GPU and is **not** a viable router decider there.

---

## 13. Observability and evaluation

- **Per-request log line** (existing `withInferenceRequestLogging`): router, surface, chosen, reason, sticky, escalated, fallback, decider latency, filtered count.
- **`/api/metrics`:** `router_decisions_total{router,surface,reason}`, `router_fallback_total`, `router_decider_latency_ms` (histogram), `router_model_switches_total` (sticky breaks), `router_suggest_total{modality}`.
- **Eval harness (AR7):** `scripts/eval/router_eval.sh` + `testdata/router_eval/*.jsonl` (`{prompt, surface, attachments, expected_model_class, expected_output_modality}`). Reports accuracy vs labels, p50/p95 added latency, and a pairwise-quality spot check (routed vs fixed baseline judged by a held-out model). **Gate:** AR5 escalation stays opt-in unless it beats baseline on quality within the latency budget (≤ 50 ms p50 deterministic, ≤ 150 ms when the classifier calls a model — findings §10).
- **Feedback (AR10):** decisions + client outcome (regenerate, thumbs, suggestion followed) ⇒ KNN corpus (`/api/router/corpus`) and Laya calibration fixtures.

---

## 14. Failure modes

| Failure | Behaviour |
|---------|-----------|
| No candidate survives filters | Chat/generate ⇒ `fallback` if it passes filters, else 503; media ⇒ 503 with `filtered[]` reasons |
| Decider timeout / error | Deterministic score, `fallback_reason: decider_*`; `on_decider_error: fail` ⇒ 503 |
| Decider uncalibrated | Argmax only; no escalation; `calibrated:false` in decision |
| Router YAML invalid | Router disabled, error logged once per mtime; `/api/router/inventory` shows the parse error; `auto` falls back to built-in deterministic |
| Model deleted mid-session | Sticky break ⇒ re-route |
| Candidate is a router | Rejected at load (existing LA11 check) |
| Text is empty (image edit w/o prompt, transcription) | Skip Jev; deterministic only |
| Surface not in router `surfaces:` | 400 `router "x" does not serve surface "speech"` |

---

## 15. Milestones

| Milestone | Goal | Owner | Exit criteria |
|-----------|------|-------|---------------|
| **AR0** | **Refactors (no behaviour change)** | Go | **Done (Oct 2026)** — `runDecisions` extracted from `DecisionsHandler`; inventory package with memoised caps + digest invalidation; bench reader shared from `cmd/benchcache`; video admission split into pure predicates; all existing tests green |
| **AR1** | **Deterministic `auto` — chat/generate/vision/video-understanding** | Go | **Done (Oct 2026)** — Surface + request-cap filters, context fit, can-load/cooldown availability, warm preference; fixes image-surface router bug; built-in implicit `auto`; unit tests per filter |
| **AR2** | **`/v1/route` + discovery + metrics** | Go | **Done (Oct 2026)** — Dry-run endpoint, virtual models in `/v1/models`, `auto_router` in `/api/version`, `/api/router/inventory`, reason header, metrics; OpenAPI updated |
| **AR3** | **All surfaces** | Go | **Done (Oct 2026)** — Alias + router hook in video, speech (voice ownership), transcription (multipart), music, embed/rerank (pinned), extract, decisions (deterministic decider pick); per-surface handler tests |
| **AR4** | **`classifier: jev` single question** | Go | **Done (Oct 2026; AR7 PASS on lab run 4 — quality parity with a fixed 8B, p50 65 ms)** — One `choice` question mapped to LA11 labels → existing `finish()`; decider pin at startup; timeout fail-open; A/B vs `classifier: score` on eval set |
| **AR5** | **Multi-question + scoring + escalation** | Go | **Done (Oct 2026; opt-in — AR7 FAIL on lab run 4: same quality as the 8B while sending 68% to the 14B)** — Ordered questions, weights, tiers, calibration gate, Laya act head; opt-in until AR7 gate passes |
| **AR6** | **Stickiness + decision cache** | Go | **Done (Oct 2026)** — Session key derivation, sticky table, re-route only on hard-filter break; Hermes batch dedupe verified |
| **AR7** | **Eval harness + gate** | Go + scripts | **Done (Oct 2026)** — panel-judged gate run recorded (lab runs 1–4) — ≥100 labeled prompts across surfaces; report script; CI-optional; documented gate result in findings doc |
| **AR8** | **Cross-modal advisory** | Go | **Done (Oct 2026)** — `output_modality` question; suggest header + `/v1/route` ready body; no response-shape change |
| **AR9** | **Cross-modal handoff (opt-in)** | Go | **Deferred (Oct 2026)** — stays advisory until the AR7 lab run measures suggestion accuracy on real traffic (findings Finding 11). When resumed: per-wire encoders (OpenAI / Ollama / Anthropic); image + speech inline, video/music job refs; integration tests per wire |
| **AR10** | **Feedback loop** *(parked)* | Go | Outcome capture → KNN corpus + Laya calibration fixtures |

**AR1 as built (gaps carried forward):**

- `model: "auto"` (or `auto:latest`, `library/auto`) routes on `/api/chat`, `/api/generate`, and every wire that funnels into them (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`, `/v1/images/*`). A `router.yaml` router named `auto` wins; so does a real local tag named `auto`. `ZEROLLAMA_AUTO_ROUTER=0` turns the implicit router off.
- LA11 routers now pass through the same surface/capability gate (`enforceRouterSurface`): a pick that cannot serve the request falls back to `fallback:` when that qualifies, otherwise 503 with the reason. Closes the image-surface gap in §6.3.
- Errors are **503** with `filtered` reasons in the message (same status LA11 already used for router failures).
- **Runtime-sidecar models are excluded** (`modelUsesRuntimeInference`) while a runtime URL is active: the runtime proxy middlewares run before the handler rewrites `model`, so a routed runtime model would 400 on the ggml path. Fix = route ahead of the proxies (AR3 follow-up).
- Availability is one scheduler-lock snapshot per request + per-candidate cooldown / concurrency-group / max-loaded checks — the `evaluateCanLoadGgml` rules without its per-call health probe and GPU discovery. `effectiveMaxLoadedModels` is only consulted when something is resident (it can trigger GPU discovery).
- Prompt tokens are a bytes/4 estimate; context fit is soft (never rejects every candidate).
- Cold-start preference among equals is **largest model** (text) / **fastest bench** (image) — a crude quality proxy until AR5 tiers and weights.

**AR2 as built:**

- `POST /v1/route` accepts the request you are about to send — native `/api/chat`, OpenAI chat-completions (content parts), or `/api/generate` — plus `router` / `model` / `surface`; aliases resolve first. Response = `RouterDecision` + `surface` + `model`; 503 bodies carry `filtered`. One `resolveRouter` backs the in-band rewrite, `/v1/route` and `/api/router/decide`, so dry runs cannot drift from real routing.
- `/api/router/decide` now also answers for implicit `auto` (prompt treated as a chat turn) and applies the surface gate to router.yaml picks.
- `GET /api/router/inventory` — candidates with capabilities, context, bench perf, live warm / can_load / needs_eviction / excluded, plus unreadable models. Scheduler keys are not serialised.
- `/v1/models` appends router names (router.yaml routers + implicit `auto` when ≥1 local model) after real models, `family: router`, capabilities = union of candidates. **Not** `/api/tags` (bench / launch / ls enumerate it). `GET /v1/models/{router}` answers with the virtual entry (`virtualRouterModelGet`, added after AR8) so clients that validate an id before use accept `auto`.
- `GET /api/version` → `zerollama.capabilities.auto_router` `{enabled, name, surfaces, dry_run, inventory}`.
- `/api/metrics`: `zerollama_router_decisions_total{router,surface,reason}`, `zerollama_router_errors_total{router,surface}`, `zerollama_router_fallback_total`, `zerollama_router_decision_latency_ms_{sum,count}` — in-band rewrites only.
- OpenAPI: `/v1/route`, `/api/router/inventory`, `AutoRouteDecision`, `RouteRequest/Response/ErrorResponse`, `RouterInventoryResponse`, `X-Zerollama-Router-Reason`.

**AR3 as built:**

- **Routing ahead of the proxies.** `routeAheadMiddleware` runs on `/api/chat`, `/api/generate` and `/v1/chat/completions` after the `:cloud` passthrough and before the runtime / SGLang proxies; it rewrites the body's `model`, so runtime-sidecar models are routable there. In-handler routing (`/v1/responses`, `/v1/messages`, `/v1/completions`, images) still excludes them — the AR1 gap is closed only where a proxy exists.
- **Surfaces:** `video_gen` (`/v1/videos`), `speech`, `music`, `transcribe`, `embed` (`/api/embed`, `/api/embeddings`, `/v1/embeddings`), `rerank`, `decisions`. Each handler now applies aliases (new for video / speech / music / transcription) then `routeForSurface`; router errors are 503 + `filtered` on every surface.
- **Per-surface rules:** video candidates must pass `videoBackendErr` + `resolveVideoGenerationConfig` + `validateVideoPlan` + `admitVideoHostRAM` (via `Signals.Check`); speech excludes music3, music requires it; voice ownership is a soft filter (Piper voices, or remote-TTS with `tts_voices_file`); transcription accepts whisper or an audio-capable chat model and rewrites the body for the ChatHandler fall-through; decisions require `decision`, images require a clef score model, and calibrated deciders rank first (Laya +noul bonus) via `Signals.Bonus`.
- **Pinned embed / rerank:** router.yaml `embedder` / `reranker` pin; implicit `auto` resolves only when exactly one model qualifies, otherwise refuses (`ErrAmbiguousPinned`) — never silently switches embedders.
- **router.yaml routers on non-text surfaces** skip the LA11 classifier and pick deterministically among their own candidates + fallback.
- **Not routed:** `/v1/extract` (already proxies any model name to `ZEROLLAMA_GLINER_URL`), `/v1/chat/completions/batch` (runtime-only, single model per batch). `/v1/route` now dry-runs every routed surface: name `surface` (`video_gen`, `speech`, `music`, `transcribe`, `embed`, `rerank`, `decisions`) and send that endpoint's normal body; chat / generate / image are still inferred.

**AR4 as built:**

```yaml
routers:
  agent:
    classifier: jev
    decider: laya              # any /v1/decisions model (Laya, CLM, Clef/tev1, openjev, gliner-decide)
    decider_timeout: 150ms     # default 150ms
    on_decider_error: fallback # default; "fail" ⇒ 503 instead of falling back
    activation_threshold: 0.15 # calibrated deciders only; uncalibrated ⇒ argmax
    policies:
      - {label: code, description: write, fix or explain code}
      - {label: chat, description: conversation or advice}
    candidates:
      - {model: qwen3-coder:30b, labels: [code]}
      - {model: llama3.2:3b, labels: [chat]}
    fallback: qwen3:8b
```

- One `choice` question (`route`) over the policy labels, sent in-process through `runDecisions` (no HTTP hop). The body is hand-built so criteria keep **policy order** (Laya logit ↔ label alignment) and carries `keep_alive: -1`.
- **Calibration:** answers without a `calibrated` field (Laya / CLM / System One) count as calibrated ⇒ every label ≥ `activation_threshold` is active (same multi-label semantics as the score tier). `calibrated:false` (OpenJev default, GLiNER-Decide) ⇒ argmax only. `RouterDecision` gains `decider` and `calibrated`.
- **Fail-open:** timeout / error ⇒ router `fallback`; with no fallback ⇒ deterministic pick among the router's candidates (`reason: decider-fallback`); `on_decider_error: fail` ⇒ 503. A router used as a decider is refused (no recursion).
- **Residency:** deciders are warmed at serve start (`prewarmJevDeciders`), and a decider timeout triggers a background, untimed warm-up (singleflight) — otherwise the per-call deadline would cancel every cold load and the decider would never become resident.
- **A/B:** `scripts/eval/router_ab.py --routers agent-score agent-jev scripts/eval/router_ab_sample.jsonl` (dry runs via `/api/router/decide`, lab port default). Not yet run against real models — needs a lab node with a Laya GGUF.

**AR5 as built (opt-in: `classifier: jev` + non-empty `questions:`):**

```yaml
routers:
  smart:
    classifier: jev
    decider: laya
    escalate_below: 0.55       # top task probability below ⇒ escalate (calibrated only)
    escalate_at: 4             # difficulty level (1-based) at/above ⇒ escalate
    act_escalate_below: 0.5    # Laya act head (only when the decider emits action)
    weights: {fit: 1.0, warm: 0.6, evict: 0.8, size: 0.1, speed: 0.2}   # defaults shown
    questions:                 # ordered; choice criteria keep YAML key order
      - id: task
        type: choice
        instructions: What kind of work is this?
        criteria: {code: write or fix code, chat: conversation, math: calculation or proof}
      - id: difficulty
        type: score
        instructions: How hard is this for a small model?
        labels: ["1", "2", "3", "4", "5"]
      - id: reasoning
        type: noul
        instructions: Does this need multi-step reasoning?
    candidates:
      - {model: llama3.2:3b,     tier: low,  strengths: {task: [chat]}, max_difficulty: 2}
      - {model: qwen3:8b,        tier: mid,  strengths: {task: [chat, math]}}
      - {model: qwen3-coder:30b, tier: high, strengths: {task: [code, math]}}
    fallback: qwen3:8b
```

- One batch call answers every question. **Fit** per candidate = Σ probability of its `strengths` labels per choice question (argmax-only when uncalibrated) − 1 per difficulty-band miss + 0.5 for thinking-capable models when the first noul ≥ 0.5; multiplied by `weights.fit` and added to the deterministic score (filters, residency, eviction, bench) with the router's weights.
- **Escalation** (calibrated deciders only) restricts the pick to `tier: high` when the first choice is unsure, difficulty ≥ `escalate_at`, the first noul ≥ 0.5, or Laya's act head < `act_escalate_below`. No high tier able to serve ⇒ retry without the gate (`signals.escalation_unavailable`) rather than 503.
- `RouterDecision` gains `escalated` and `signals` (per-question answers + `calibrated`, `escalate_reasons`). Config errors (bad `questions`) are 503s; decider errors fail open exactly as AR4.
- Score answers are expected **0-based** level indices from the decider; the router reports and compares **1-based** levels.
- **Evaluated (lab run 4):** fails AR7 — no quality gain over a fixed baseline for more compute; stays opt-in.

**AR6 as built:**

- **Session key:** the client's `prompt_cache_key` / `session_id` / `conversation_id` / Eliza `conversationId` (OpenAI top-level `prompt_cache_key` maps into options), else `sha256(system prompt ∥ first user message)` — identical on every turn of one conversation. Generate requests are sticky only with an explicit key.
- **Sticky table:** router-scoped, 30 min sliding TTL (`ZEROLLAMA_ROUTER_STICKY_TTL`, `0` = off), 4096 conversations LRU. A remembered model is reused only if `autoroute.Validate` passes — every hard filter **strictly** (surface, capabilities, `Check`, context fit, voice, availability) — and the model is still one of that router's candidates. Otherwise the turn re-routes, the decision carries `sticky_break` (also `X-Zerollama-Router-Sticky-Break`), and the conversation sticks to the new pick.
- **Decision cache:** identical text requests (router, surface, capability flags, token bucket, exact text) reuse the last pick for 60 s (1024 entries), re-validated the same way. Saves the classifier / Jev decider call on agent retries. The batch endpoint is not routed (AR3), so "Hermes batch dedupe" reduces to retries hitting this cache.
- **Opt-out per turn:** `options.zerollama.route: "fresh"` skips remembered picks (the new pick is still remembered).
- **Dry runs** (`/v1/route`, `/api/router/decide`) read memory so they show what the next request would do, but never write it. New reasons: `sticky`, `cached`; fields `sticky`, `cached`, `sticky_break`.
- **Trade-off, by design:** a jev router that routed turn 1 ("hi") to a small model keeps it on later turns unless a hard filter breaks — escalation does not re-run mid-conversation. Clients that want per-turn routing send `route: "fresh"` or disable stickiness.

**AR7 as built:**

- **Dataset:** [`testdata/router_eval/prompts.jsonl`](../testdata/router_eval/prompts.jsonl) — 129 prompts, model-agnostic labels (`task`, `difficulty` 1–5, expected minimum `tier`, expected `output` modality, `surface`, `attach`). Covers code / math / chat / extract / creative, chat / generate / image / image_edit, ≥3 rows each for tools / image / video / audio, and 21 cross-modal intent rows (labels for AR8).
- **CI-optional check:** `TestEvalDatasetSchema` (pure Go, `server/autoroute`) keeps the set loadable and representative — ≥100 rows, valid enums, every label value present.
- **Harness:** [`scripts/eval/router_eval.py`](../scripts/eval/router_eval.py) (stdlib) — dry runs through `/v1/route` with `route: "fresh"`; capability correctness, tier exact / under / over, strength hit rate, escalation count, p50 / p95 decision latency; optional judged quality vs a fixed baseline. Exit 1 on gate FAIL.
- `/api/router/inventory` now includes `routers` (classifier, decider, fallback, question count, candidates with tiers / strengths) so the harness needs no YAML parser.
- **Invalid router.yaml is now visible:** a parse error logs once per file version and appears as `router_config_error` in `/api/router/inventory` (it previously made every router silently vanish — §14 said it should be reported).
- **Lab config:** [`scripts/eval/router_lab.example.yaml`](../scripts/eval/router_lab.example.yaml) — `agent-score` / `agent-jev` (identical policies + candidates, different classifier — the AR4 A/B) and `smart` (AR5 multi-question + AR8 `output_modality`); `TestLabExampleRouterYAML` keeps it loadable and the A/B fair.
- **Gate + result:** [auto-router-findings.md](./auto-router-findings.md) Finding 10. **Pending** — needs a lab node with real models.

**AR8 as built (advisory only — the chat request is still served as text):**

- **Detection, best first:** (1) a jev router's `output_modality` choice question (`modality_question:` to rename) answered in this turn's batch — calibrated ⇒ P(choice) ≥ `suggest_threshold` (0.5), else argmax; (2) on sticky / cached turns the decider batch is skipped, so that one question is asked alone (intent changes per turn) — not after a decider failure; (3) the keyword fallback `autoroute.GuessOutputModality` (generate-verb near an artefact noun; bare "draw / sketch" only as a request).
- **Suggestion:** `modality`, `source` (jev | heuristic), `probability`, `endpoint` (`/v1/images/generations`, `/v1/audio/speech`, `/v1/videos`, `/v1/audio/generations`), `model` (what the target surface would route to, or `unavailable` with the reason) and a ready `body` (speech reads back the previous assistant reply; music sends the turn as the Music 3 caption).
- **Delivery:** `X-Zerollama-Route-Suggest` / `-Endpoint` / `-Model` headers on routed chat responses, `suggestion` on `/v1/route` and `/api/router/decide`, `zerollama_router_suggest_total{modality,source}`.
- **Modes:** `ZEROLLAMA_ROUTER_SUGGEST` = `auto` (default: jev, else heuristic) | `jev` (no keyword fallback) | `off`.
- **Measured here:** keyword fallback 13/13 cross-modal chat rows, 0/96 false positives on the AR7 set (`TestGuessOutputModalityOnEvalSet`) — but see findings Finding 11: the rules and the set share an author. `router_eval.py` now reports suggestion recall / false positives per router for the lab run.

**Suggested cut:** AR0–AR3 is the useful, low-risk core (no Jev). AR4–AR7 is the Jev experiment with an evidence gate. AR8–AR9 is the product feature; AR9 needs an owner decision on response contracts first.

### Files likely touched

| Area | Files |
|------|-------|
| New | `server/autoroute/{signals,inventory,filter,score,sticky,decide}.go` + tests; `server/route_handler.go` (`/v1/route`); `scripts/eval/router_eval.sh`; `docs/auto-router-findings.md` |
| Router | `server/router_policy.go` (spec additions, ordered questions), `server/router_handler.go` (delegate to autoroute) |
| Handlers | `server/routes.go` (chat/generate/embed, image path, list/version), `server/video_generate.go`, `server/video_admit.go`, `server/speech.go`, `server/transcription.go`, `server/music_generate.go`, `server/rerank_handler.go`, `server/extract_handler.go`, `server/decisions_handler.go` |
| Shared | `cmd/benchcache` → shared package; `envconfig/config.go` (`ZEROLLAMA_AUTO_ROUTER`, decider timeout); `server/openapi/openapi.yaml`; `api/types.go` (`RouteDecision`) |
| Docs | this file, ROADMAP row, `docs/README.md` index, skill `agentskills/typed-decisions/` (router section) |

---

## 16. Owner decisions

| # | Question | Decision |
|---|----------|----------|
| 1 | Reserve `auto`? | **Yes** — implicit deterministic router; a router.yaml `auto` or a real local tag named `auto` wins; `ZEROLLAMA_AUTO_ROUTER=0` opts out (AR1). |
| 2 | Fail-open vs fail-closed on decider error | **Open by default** (router fallback, else deterministic among its candidates); `on_decider_error: fail` per router (AR4). |
| 3 | Handoff contract (AR9) | **Advisory for now** — suggestions only, no response-shape change, until the AR7 lab run measures suggestion accuracy on real traffic. |
| 4 | Sticky TTL / per-turn re-route | **30 min sliding** (`ZEROLLAMA_ROUTER_STICKY_TTL`, `0` = off); per-turn opt-out `options.zerollama.route: "fresh"` (AR6). |
| 5 | Default decider | **Open** — Laya GGUF (calibrated, GPU encoder) vs GLiNER-Decide (CPU, uncalibrated, argmax-only). Decide from the AR7 lab run. |
