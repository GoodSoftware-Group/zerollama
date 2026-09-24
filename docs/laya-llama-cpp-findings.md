# Laya llama.cpp — findings & learnings

**Audience:** anyone extending `LLM_ARCH_LAYA`, the convert script, or `/v1/decisions`.  
**Related:** [laya-llama-cpp.md](./laya-llama-cpp.md), ROADMAP **Typed decisions (Laya)**, patches **0127–0128**, OpenAPI `/v1/decisions`, skill `typed-decisions`.  
**Status:** LAYA1–LAYA2 active (Sep 2026); LAYA4 MLX parked.

---

## Why this doc exists

Laya looked “just another ModernBERT” until product requirements collided with llama.cpp’s embed/rerank paths. These findings record **why** we chose a new arch, CPU head, Go packing, and specific GGUF keys — so the next change does not collapse decisions back onto chat or RANK pooling.

---

## Finding 1 — Cannot overload ModernBERT / RANK

**What:** `LLM_ARCH_MODERN_BERT` + `POOLING_TYPE_RANK` / `/v1/rerank` returns a single pooled scalar (`embd[0]`).

**Why that fails for Laya:** the decision head needs **per-MASK token states**, then scorer logits per option, then a separate act head. One scalar cannot carry choice probs + act.

**Learning:** Register `LLM_ARCH_LAYA` with its own tensor names (`laya.type_embd`, `laya.head.%d.*`, `laya.scorer.*`, `laya.act.*`). Reuse ModernBERT **encoder graph shape**, not the RANK product path.

**Non-goal:** Bolting Laya onto `/api/chat` or treating decisions as “chat with a grammar.”

---

## Finding 2 — Decision head on CPU after GPU encode

**What:** Encoder runs like embed (GPU/Metal/CUDA via ggml). Head weights (~26 MB) load to host floats; `laya_head_apply` runs in `server-laya.cpp`.

**Why:** Avoid new Metal/CUDA kernels for a tiny Transformer+MLP while still accelerating the expensive encoder. `res->t_embd` already exposes per-token states.

**Learning:** Keep head load lazy (`laya_weights` on first `/v1/decisions`). Eps / RoPE metadata must match convert keys or numerics silently drift.

---

## Finding 3 — Go packs; llama-server scores (LA16)

**What:** Public API is Jev-shaped `{model, state, questions}` → typed `answers`. llama-server accepts **tokenized** `{inputs:[{tokens, marker_pos, qtype, question_id?}]}` only.

**Why:** Same control-plane split as `/v1/rerank` — Go owns wire + packing/calibration; C++ owns the graph. High-level packing in C++ would duplicate tokenizer policy and make Go/MLX clients diverge.

**Learning:** `layaServerTokenizer` uses `/tokenize` + GGUF special IDs (`cls`/`sep`/`mask`). Sort `question_id` before pack **and** when aligning results; echo `question_id` in server results as belt-and-suspenders.

---

## Finding 4 — GGUF key footguns (audit Sep 2026)

| Bug | Wrong | Right | Effect |
|-----|-------|-------|--------|
| SWA RoPE freq | `laya.attention.rope.freq_base_swa` | `laya.rope.freq_base_swa` | `LLM_KV_ROPE_FREQ_BASE_SWA` miss → default freq → **wrong RoPE on all SWA layers** |
| Head LN eps | `laya.norm_eps` / `attention.layernorm_eps` | Prefer `laya.attention.layer_norm_epsilon` | Silent default eps |
| `n_act` | Always `len(act_costs)+1` | `cfg.get("n_act", …)` | Wrong act softmax width when config sets `n_act` explicitly |

**Learning:** Convert script and `laya_head_load` / hparams must share one key dictionary. Treat silent `get_key(..., false)` misses as **correctness bugs**, not cosmetics.

---

## Finding 5 — Map order is load-bearing

**What:** Choice `criteria` is a JSON object; logits align to key order. Go `PackQuestions` sorts question ids for stable batch order.

**Why:** Go map iteration is random; server results without echoed ids were position-aligned only. Unstable order → wrong label on the winning logit.

**Learning:** Preserve criteria key order (`json.RawMessage`); sort question ids; prefer echoed `question_id` when present.

---

## Finding 6 — Park MLX sidecar

**What:** `laya-mlx` / mlxrunner graph is ROADMAP **LAYA4 parked**.

**Why:** CPU/CUDA llama.cpp covers the primary GPU-box path first; Darwin-managed sidecars add spawn/VRAM policy surface (same class of pain as remote-tts/Comfy). When unparked, prefer external `ZEROLLAMA_LAYA_URL` proxy with the **same** public `/v1/decisions`.

**Learning:** Do not invent a second public API for Metal.

---

## Finding 7 — Lab ports only

**What:** Smokes use `:18082` (llama-server) and `:11435` (zerollama).

**Why:** `:11434` / `:8081` are production on the Mac lab. Agents must not bind or kill them.

**Learning:** Document lab ports in every operator curl block; refuse “free the port” by killing production.

---

## Open questions (not blockers)

- Full fixture parity vs Python Agent (argmax + temp-calibrated probs) on converted GGUF — unit tests cover pack/decode only.
- mmBERT multilingual: same `LLM_ARCH_LAYA`, different vocab — LAYA3.
- Whether Go should refuse misaligned `question_id` echoes instead of trusting sort order alone.
