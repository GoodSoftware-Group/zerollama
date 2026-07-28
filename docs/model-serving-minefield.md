# Model serving minefield ↔ zerollama

Cross-reference between the community [model-serving-minefield](https://github.com/Blackwellboy/model-serving-minefield) registry and zerollama.

This is a living map, not a copy of the registry. Trap IDs match the upstream entries.

Status vocabulary used below:

| Status | Meaning |
|--------|---------|
| `safe` | Owned path does not share the trap's failure mode |
| `at-risk` | Owned code trusts a signal the trap says can lie |
| `gap` | Footgun not checked / not surfaced to operators |
| `fixed` | Regression locked after confirmation |
| `covered via doctor` | `zerollama doctor` exercises a check |
| `covered via test` | Unit/integration test locks behavior |
| `partial` / `n/a` | Related code or architecture differs |

Upstream premise that still applies: **inspect the assembled prompt, not the request**; state build + revision next to numbers; diff the kwarg surface in both directions.

---

## 1. Broker / admission / llama-server substrate (primary)

These traps matter because zerollama **spawns llama-server** and owns VRAM admission/scheduling. A wrong free-memory or context signal is a correctness bug in *our* load decisions, not just a measurement footnote.

| Trap | Topic | Verdict | Evidence |
|------|-------|---------|----------|
| **96** | `--list-devices` free may be host memory, not device VRAM | **mitigated** | Clamp free ≤ total; for **CUDA/ROCm** prefer native probe (NVML/HIP) free when present ([`preferNativeDeviceFree`](../discover/llama_server.go)). Metal host-as-free remains intentional. Python admission still uses NVML/smi independently. |
| **97** | Partial GPU offload invisible in health output | **fixed** | `/api/ps` `loaded_metadata.gpu_layers_offloaded` / `gpu_layers_total` from llama-server load logs; `zerollama ps` PROCESSOR column prefers layer counts over VRAM ratio. |
| **87** | `/props` reports per-slot context, not trained | **safe** | No serving path reads llama-server `/props` for context. Served = load `options.NumCtx` ([`llm/llama_server.go`](../llm/llama_server.go) `ContextLength()`). `/api/ps` `loaded_metadata` separates `num_ctx` vs `train_context_length` ([`server/runner_metadata.go`](../server/runner_metadata.go)). |
| **71** | MTP layer count ≠ draft token count | **safe** | `nextn_predict_layers` / tensor presence enables MTP; draft depth is `draft_num_predict` → `--spec-draft-n-max` ([`llm/llama_server.go`](../llm/llama_server.go) `appendMTPDraftArgs`). Guessing only sets `spec_type=draft-mtp`, not draft count from layer count ([`server/gguf_guess.go`](../server/gguf_guess.go)). |
| **91/92** | Temp-0 determinism has prompt-length floor; prefix cache is a second divergence source | **documented** | Caveats in [`docs/testing-smoke.md`](./testing-smoke.md), [`scripts/mlx/mlx_prefix_cache_smoke.sh`](../scripts/mlx/mlx_prefix_cache_smoke.sh), phase15 batch/auto-batch smokes, [`scripts/e2e/e2e_runtime_smoke.sh`](../scripts/e2e/e2e_runtime_smoke.sh). |

**Operator takeaway:** On discrete CUDA, do not treat Go discovery `available=` from list-devices as NVML ground truth without cross-check. On Metal, free-as-host is by design. Prefer `/api/ps` `loaded_metadata` over raw llama-server `/props` for context. Do not infer full GPU residency from VRAM alone — require layer offload counts.

---

## 2. Lab minefield-doctor run

### Re-verify after trap 78 fix (2026-07-28, same lab layout)

```text
OLLAMA_HOST=127.0.0.1:11435  model=qwen2.5:0.5b  build=0.30.11
```

| Bucket | Result |
|--------|--------|
| **PROBLEMS** | **none** (lab re-run Jul 28 after traps **77** + **78**) |
| **CLEAN** | **77** (unknown field → 400), **78** (`tool_choice none`), **19**, **07** (dead `chat_template_kwargs` loud), **26**, mm-* |
| Coverage | `problems 0` · executed CLEAN includes **77** and **78** |

### Earlier baseline (pre-fix)

Empirical baseline against a **lab** stack only (never production `:11434` / `:8081`):

```text
OLLAMA_HOST=127.0.0.1:11435
ZEROLLAMA_RUNTIME=0
ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0
model=qwen2.5:0.5b
build=0.30.11
tool=minefield_doctor.py (upstream registry)
base-url=http://127.0.0.1:11435/v1
```

#### PROBLEMS (2) — pre-fix

| Trap | Finding |
|------|---------|
| **77** | Invented top-level field `__minefield_unvalidated_field_probe__` accepted with HTTP 200 — request surface is largely unvalidated; thinking-off / typo arms can measure the wrong configuration. **Fixed:** `/v1/chat/completions` rejects unknown top-level keys with HTTP 400 ([`openai/chat_unknown_fields.go`](../openai/chat_unknown_fields.go), [`BindChatCompletionRequest`](../openai/chat_extras.go), runtime v1 proxy). |
| **78** | `tool_choice: "none"` was accepted and ignored (fails open). **Fixed:** `tool_choice: "none"` now omits tools from `/v1/chat/completions` and `/v1/responses` conversion ([`openai/openai.go`](../openai/openai.go), [`openai/responses.go`](../openai/responses.go)). |

### CHECKED AND CLEAN (selected)

| Trap / advisory | Result |
|-----------------|--------|
| **19** | Structured `tool_calls` on forced tool probe (`get_time`) |
| **12** | Cap reached (`finish=length`) but content still returned at `max_tokens=512` (this budget / sample only) |
| **23** | Streamed answer in `content` deltas |
| **26** | Tool markup parsed into `tool_calls`, not left in reasoning/content text |
| mm-surface / mm-usage / mm-errors | Inline image accepted; usage attributes media tokens; bad media path → HTTP 400 |

### INCONCLUSIVE / COULD NOT CHECK (highlights)

- **02** orphaned `</think>` — partial arm failures
- **03/29** thinking-toggle map — not all arms returned
- **04/20/25** assembled-prompt inspection — no render/`/apply-template` path on this Ollama-shaped stack for the upstream doctor
- **07/10/17/21** need `--hf-repo` or readable chat template

Coverage line from the tool: `implemented 19/103 | executed 6 | clean 4 | problems 2 | …` — a clean subsection is **not** a bill of health for the full registry.

---

## 3. `zerollama doctor` coverage (native)

`zerollama doctor` includes minefield-style checks in-tree:

1. **Model config traps** ([`internal/modelhealth/traps.go`](../internal/modelhealth/traps.go)): **21**, **10**, **56**, **55/61**
2. **Live serving traps** ([`cmd/doctor_serving_traps.go`](../cmd/doctor_serving_traps.go)): **01/03**, **12/64/65**, **19** — warm `/api/ps` only; warn (not fail) when inconclusive

```bash
./zerollama doctor
./zerollama doctor --models
./zerollama run <model> && ./zerollama doctor   # enable live serving probes
```

External upstream doctor (lab ports only):

```bash
curl -sO https://raw.githubusercontent.com/Blackwellboy/model-serving-minefield/main/doctor/minefield_doctor.py
python3 minefield_doctor.py --base-url http://127.0.0.1:11435/v1 --model <tag>
```

---

## 4. API-surface appendix

| Trap | Topic | Status | Where |
|------|-------|--------|-------|
| 01 | Wrong reasoning field name | `covered via doctor` | `doctorCheckReasoningField` |
| 04 | History reasoning stripping | `covered via test` / code | [`server/chat_sanitize.go`](../server/chat_sanitize.go) |
| 12 | Empty content at token ceiling | `covered via doctor` | Think roundtrip + lab clean on 0.5b @ 512 |
| 19 | Tool parsing / structured calls | `covered via doctor` | Lab clean + `doctorCheckToolCallShape` |
| 57 | Thinking kwarg truthiness | `n/a` / `partial` | Native `ThinkValue` typed; raw kwargs can still hit upstream |
| 58/64/65 | Effort / toggle / rescue | `covered via doctor` + test | [`server/runtime_v1_legacy_test.go`](../server/runtime_v1_legacy_test.go) |
| **77** | Only one request field validated | **fixed** | Unknown top-level keys on `/v1/chat/completions` → HTTP 400 (`CheckUnknownChatCompletionFields`); assert on response still required for known-but-unread knobs |
| **78** | `tool_choice` fails open | **fixed** | `tool_choice: "none"` omits tools in chat + responses conversion |

### Model config (also in native doctor)

| Trap | Status |
|------|--------|
| 10, 21, 55/61, 56 | `covered via doctor` |

---

## Updating this doc

When the broker/admission sources of free VRAM change, or a lab doctor re-run flips PROBLEMS ↔ CLEAN, update sections 1–2 and the matching regression tests under `discover/` and `openai/`.
