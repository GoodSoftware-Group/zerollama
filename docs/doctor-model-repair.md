# Doctor model repair (`--repair-models`)

**Why this exists:** Benchmarks and harnesses often score a GGUF as “broken” (0/N traps, empty `response`, slash loops) when the weights are fine and the **assembled prompt / parser / default think routing** is wrong. Operators then delete or re-quantize models that only needed a Modelfile overlay.

This is separate from:

| Command | Job |
|---------|-----|
| `zerollama doctor` | Host + minefield live probes (warn only) |
| `zerollama doctor --fix` | Bootstrap toolchain (uv, Metal llama.cpp) |
| `zerollama doctor --models --fix` | Delete **orphaned** manifests only |
| `zerollama repair` | Rewrite params/config from GGUF headers (no live probes) |
| **`zerollama doctor --repair-models`** | Diagnose + optional **template/parser** recreate for known serving traps |

Canonical trap map: [model-serving-minefield.md](./model-serving-minefield.md) §3.1.

---

## Quick start

```bash
# Dry-run: warm models from /api/ps (no surprise cold loads)
OLLAMA_HOST=127.0.0.1:8080 ./zerollama doctor --repair-models

# Named tags (may load GGUF — expect minutes on 30B+)
./zerollama doctor --repair-models milkey/Kalomaze-Qwen3-16B-A3B:latest

# Apply: recreate the same tag FROM itself with patched TEMPLATE/PARSER/stop
./zerollama doctor --repair-models --apply milkey/Kalomaze-Qwen3-16B-A3B:latest
```

Progress lines go to **stderr**; the report (and proposed Modelfile on dry-run) goes to **stdout**. Use `--json` for machine-readable reports.

---

## Recipes (what we auto-fix)

Auto-patch runs **only** when the tag is in the **Qwen3 family** (`PARSER` / GGUF `general.architecture` / Modelfile `PARSER` contains `qwen3`).

**Why the family gate:** Patches replace the chat template with Qwen3 ChatML + `/think`/`/no_think` (or drop-system ChatML). Applying that to Llama/Hermes/DeepSeek that merely *look* ChatML-ish would make serving worse.

| Recipe ID | Symptom | Patch | Why |
|-----------|---------|-------|-----|
| `think_generate_empty` | Default `/api/generate` (omit `think`) → empty `response`, answer in `thinking`; or ChatML thinking template lacks `/no_think` | Inject `/think`\|`/no_think`, closed empty `<think>` when off; `PARSER qwen3` | Stock `qwen3-thinking` + ChatML without toggles + older serve Init-order (`Think=nil` → parser defaultThinking) parks answers in the thinking channel. Harnesses that only read `response` score 0. |
| `slash_system_collapse` | User-only chat OK; `system`+user or harness `System:`/`User:`/`Assistant:` lines → `/` loops or empty `eval≤4` | Drop system role; `stripRolePrefixes` on user/prompt; one-line “Reply with useful content only.” steer; ChatML stops (not `///`) | Some Qwen3-Coder GGUFs treat roleplay labels / “You output …” as code-comment completion and emit `///…`. That also poisons the runner until unload. `stop ///` fails the one request but leaves the slot bad — prevention beats stop. **Needs serve built with template func `stripRolePrefixes`.** |

Non-qwen3 tags with the same symptoms → **`manual_review`** (reported, never applied).

### Intentionally unfixable

- Image-only models, weight loops, VRAM spill, unrelated 0/N failures.
- Weak instruction-following checkpoints (e.g. `llama2-uncensored`) that strip XML tags / paraphrase — Modelfile cannot add capability the weights lack.

---

## Server half (think default before parser Init)

**Why:** Chat already defaulted `think=false` for thinking models *before* parser `Init`. Generate used to call `builtinParser.Init(..., req.Think)` while `Think` was still `nil`, then set `Think=false` afterward. For `PARSER qwen3-thinking`, `nil` means `defaultThinking=true` → CollectingThinking → answer lands in `thinking`.

Fixed in [`server/routes.go`](../server/routes.go) GenerateHandler: default `Think=false` **before** `Init`.

**Why `PARSER qwen3` in the Modelfile patch anyway:** Production may still run a binary without that fix. Content-mode parser keeps default generate usable until serve is rebuilt/restarted. After restart, you can prefer `qwen3-thinking` again if you want a real thinking channel when `think:true`.

---

## Safety rules

1. **Dry-run by default** — `--apply` is required to write.
2. **No `--all-local` yet** — without MODEL args, only `/api/ps` warm models (same rule as live doctor: don’t surprise-load the library).
3. **Unload between probes** — prefix KV can poison later turns (especially after a slash generation). Repair probes and the trap-12 generate arm unload first.
4. **Do not overload `doctor --fix`** — that flag is host bootstrap; mutating Modelfiles under `--fix` would surprise operators who only meant “install Metal llama.cpp.”

---

## Code map

| Path | Role |
|------|------|
| [`internal/modelrepair`](../internal/modelrepair) | Recipes, probes, patch builders, HTTP `/api/create` apply |
| [`cmd/doctor_repair_models.go`](../cmd/doctor_repair_models.go) | CLI wiring for `--repair-models` / `--apply` |
| [`cmd/doctor_serving_traps.go`](../cmd/doctor_serving_traps.go) | Live trap-12/64 + **default generate** arm (warn → FixHint points here) |
| [`server/routes.go`](../server/routes.go) | Generate think default before parser Init |

---

## Related

- [model-serving-minefield.md](./model-serving-minefield.md) — trap IDs 12/64/65/66/29
- `zerollama repair` — GGUF metadata hygiene (different tool)
- Skills: [`skills/doctor-model/SKILL.md`](../skills/doctor-model/SKILL.md)
