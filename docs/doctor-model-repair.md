# Doctor model repair (`--repair-models`)

**Why this exists:** Benchmarks and harnesses often score a GGUF as “broken” (0/N traps, empty `response`, slash loops) when the weights are fine and the **assembled prompt / parser / default think routing** is wrong. Operators then delete or re-quantize models that only needed a Modelfile overlay. Chat-template hygiene (stops, empty TEMPLATE, `{{ .Response }}`) is the same class of fault—borrowed from Unsloth/Ollama-style “pair the template with the stops the model was trained on.”

This is separate from:

| Command | Job |
|---------|-----|
| `zerollama doctor` | Host + minefield live probes (warn only) |
| `zerollama doctor --fix` | Bootstrap toolchain (uv, Metal llama.cpp) |
| `zerollama doctor --models --fix` | Delete **orphaned** manifests only |
| `zerollama repair` | Rewrite params/config from GGUF headers (no live probes) |
| **`zerollama doctor --repair-models`** | Diagnose + optional **template/parser/stop** recreate for known serving traps |

Canonical trap map: [model-serving-minefield.md](./model-serving-minefield.md) §3.1.

---

## Quick start

```bash
# Dry-run: warm models from /api/ps (no surprise cold loads)
OLLAMA_HOST=127.0.0.1:8080 ./zerollama doctor --repair-models

# Named tags (may load GGUF — expect minutes on 30B+)
./zerollama doctor --repair-models milkey/Kalomaze-Qwen3-16B-A3B:latest

# Every local tag (explicit opt-in — may cold-load the library)
./zerollama doctor --repair-models --all-local

# Apply: recreate the same tag FROM itself with patched TEMPLATE/PARSER/stop
./zerollama doctor --repair-models --apply milkey/Kalomaze-Qwen3-16B-A3B:latest
```

Progress lines go to **stderr**; the report (and proposed Modelfile on dry-run) goes to **stdout**. Use `--json` for machine-readable reports.

---

## Recipes (what we auto-fix)

### Family-safe hygiene (any ChatML tag)

| Recipe ID | Symptom | Patch | Why |
|-----------|---------|-------|-----|
| `chatml_missing_stops` | ChatML TEMPLATE without `PARAMETER stop <|im_end|>` and/or `<\|im_start\|>` | Add those stops only (TEMPLATE/PARSER inherited) | Without stops the model emits role markers into the next turn |
| `missing_response_placeholder` | Go TEMPLATE has `{{` but no `{{ .Response }}` | Append Response + `<\|im_end\|>` suffix; keep Messages layout | `/api/generate` continuation needs the assistant-so-far placeholder |

### Invasive TEMPLATE rewrites (Qwen3 family only)

Auto-patch that **replaces** TEMPLATE runs **only** when the tag is in the **Qwen3 family** (`PARSER` / GGUF `general.architecture` / Modelfile `PARSER` contains `qwen3`).

**Why the family gate:** Patches replace the chat template with Qwen3 ChatML + `/think`/`/no_think` (or drop-system ChatML). Applying that to Llama/Hermes/DeepSeek that merely *look* ChatML-ish would make serving worse.

| Recipe ID | Symptom | Patch | Why |
|-----------|---------|-------|-----|
| `empty_template` | TEMPLATE layer is empty | Stock ChatML (thinking → `/think` template; else `templateChatMLStock`) | Empty TEMPLATE skips chat assembly |
| `think_generate_empty` | Default `/api/generate` (omit `think`) → empty `response`, answer in `thinking`; or ChatML thinking template lacks `/no_think` | Inject `/think`\|`/no_think`, closed empty `<think>` when off; `PARSER qwen3` | Stock `qwen3-thinking` + ChatML without toggles + older serve Init-order parks answers in the thinking channel |
| `think_parser_mismatch` | `PARSER` contains `thinking` but TEMPLATE has no `/think`\|`/no_think` | Same as think_generate_empty | Explicit hygiene signal for operators comparing PARSER vs TEMPLATE |
| `slash_system_collapse` | User-only chat OK; `system`+user or harness `System:`/`User:`/`Assistant:` lines → `/` loops or empty `eval≤4` | Drop system role; `stripRolePrefixes` on user/prompt; one-line “Reply with useful content only.” steer; ChatML stops (not `///`) | Some Qwen3-Coder GGUFs treat roleplay labels as code-comment completion. **Needs serve built with template func `stripRolePrefixes`.** |

Non-qwen3 tags with invasive symptoms → **`manual_review`** (reported, never applied). Hygiene recipes (`chatml_missing_stops`, `missing_response_placeholder`) still apply.

### Intentionally unfixable

- Image-only models, weight loops, VRAM spill, unrelated 0/N failures.
- Weak instruction-following checkpoints (e.g. `llama2-uncensored`) that strip XML tags / paraphrase — Modelfile cannot add capability the weights lack.
- Roleplay prompts that embed `System:` / `Assistant:` in the user text — template cannot strip that.

---

## Server half (think default before parser Init)

**Why:** Chat already defaulted `think=false` for thinking models *before* parser `Init`. Generate used to call `builtinParser.Init(..., req.Think)` while `Think` was still `nil`, then set `Think=false` afterward. For `PARSER qwen3-thinking`, `nil` means `defaultThinking=true` → CollectingThinking → answer lands in `thinking`.

Fixed in [`server/routes.go`](../server/routes.go) GenerateHandler: default `Think=false` **before** `Init`.

**Why `PARSER qwen3` in the Modelfile patch anyway:** Production may still run a binary without that fix. Content-mode parser keeps default generate usable until serve is rebuilt/restarted. After restart, you can prefer `qwen3-thinking` again if you want a real thinking channel when `think:true`.

---

## Safety rules

1. **Dry-run by default** — `--apply` is required to write.
2. **Warm-only by default** — without MODEL args, only `/api/ps` warm models. Pass **`--all-local`** to scan `/api/tags` (may cold-load).
3. **Unload between probes** — prefix KV can poison later turns (especially after a slash generation). Repair probes and the trap-12 generate arm unload first.
4. **Do not overload `doctor --fix`** — that flag is host bootstrap; mutating Modelfiles under `--fix` would surprise operators who only meant “install Metal llama.cpp.”

---

## Code map

| Path | Role |
|------|------|
| [`internal/modelrepair`](../internal/modelrepair) | Recipes, probes, patch builders, HTTP `/api/create` apply |
| [`cmd/doctor_repair_models.go`](../cmd/doctor_repair_models.go) | CLI wiring for `--repair-models` / `--apply` / `--all-local` |
| [`cmd/doctor_serving_traps.go`](../cmd/doctor_serving_traps.go) | Live trap-12/64 + **default generate** arm (warn → FixHint points here) |
| [`server/routes.go`](../server/routes.go) | Generate think default before parser Init |

---

## Related

- [model-serving-minefield.md](./model-serving-minefield.md) — trap IDs 12/64/65/66/29
- [ROADMAP § GPU training T8](./ROADMAP.md#gpu-training-fine-tuning) — train-time chat-template alignment (separate from serve doctor)
- `zerollama repair` — GGUF metadata hygiene (different tool)
- Skills: [`skills/doctor-model/SKILL.md`](../skills/doctor-model/SKILL.md)
