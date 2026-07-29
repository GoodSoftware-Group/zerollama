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

## 2. Lab minefield-doctor runs

Lab only (`OLLAMA_HOST=127.0.0.1:11435`, `ZEROLLAMA_RUNTIME=0`, `ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0`). Never production `:11434` / `:8081`.

### Thinking lane — `qwen3:0.6b` (2026-07-28)

```text
model=qwen3:0.6b  build=0.30.11  tool=minefield_doctor.py
```

| Bucket | Result |
|--------|--------|
| **PROBLEMS** | **12** (empty content at `max_tokens=512` with long reasoning — conversion floor; size budgets for thinking), **29** (default thinking-off is not a gate: client `reasoning_effort` / aliases can re-enable per request — mitigate with `ZEROLLAMA_THINKING_GATE=deny\|strip`) |
| **CLEAN** | **77**, **78**, **01**, **03** (toggle map separable), **02**, **23**, **07**, **19**, **26**, mm-* |
| Coverage | `problems 2` · `clean 9` numbered · Core executed **01, 03, 12, 19, 77** |

**Operator takeaway (12 / 29):** On thinking models, a 512 ceiling can score as capability collapse while the model is still reasoning. Treat server thinking-off as a **default**, not a hard gate, unless your gateway strips thinking kwargs.

### Non-thinking lane — `qwen2.5:0.5b` (2026-07-28)

```text
model=qwen2.5:0.5b  build=0.30.11
```

| Bucket | Result |
|--------|--------|
| **PROBLEMS** | **none** |
| **CLEAN** | **77**, **78**, **12**, **23**, **02**, **07**, **19**, **26**, mm-* |
| Coverage | **03/29** N/A (no reasoning channel; arms return but none fire) |

### Earlier baseline (pre-fix)

```text
model=qwen2.5:0.5b  build=0.30.11  (pre trap 77/78 fixes)
```

#### PROBLEMS (2) — pre-fix

| Trap | Finding |
|------|---------|
| **77** | Invented top-level field accepted with HTTP 200. **Fixed:** `/v1` + `/api/chat` reject unknown keys; `chat_template_kwargs` / `enable_thinking` mapped to `think` with nested-kwarg validation. |
| **78** | `tool_choice: "none"` accepted and ignored. **Fixed:** omits tools in chat + responses conversion. |

### Still out of reach for the upstream doctor here

- **04/20/25** — no render/`/apply-template` path on this Ollama-shaped stack
- **10/17/21** — need `--hf-repo` or a readable chat template for full checks
- Core **35 / 53 / 61** — no check in `minefield_doctor.py` (upstream leaves them as hand-runs; see §2.1). Zerollama covers **53** (serve identity) and **55/61 arithmetic** in `zerollama doctor`; **35** and the **61 behavioural ladder** stay operator-run.

Coverage lines from the tool are **not** a bill of health for the full registry (103 entries; doctor implements ~19).

### 2.1 Core hand-runs (35 / 53 / 61) on zerollama

Upstream [CORE.md](https://github.com/Blackwellboy/model-serving-minefield/blob/main/CORE.md) deliberately omits these from `minefield_doctor.py`. Map them onto this stack as follows.

#### Trap 35 — identical weights do not score identically

Not a serving bug. Before publishing small eval deltas on zerollama (ggml or runtime lane):

1. Fix one host + one binary revision as the measurement room.
2. Run the **same** items twice (same process, then fresh process) at `temperature=0`.
3. Report **per-item agreement**, not only score delta.
4. Treat the observed score spread as your minimum detectable effect; do not assemble paired arms across machines without quoting that floor.

Zerollama does not ship an agreement-floor harness; use your benchmark’s per-item JSON. Temp-0 still has a prompt-length / prefix-cache floor (traps **91/92**).

#### Trap 53 — config edit never took effect

`zerollama doctor` prints **serve identity (trap 53)**: answering `base`, `/api/version`, and (on Darwin/Linux) listener `pid` + `ps` start/etime/cmd.

After every config or binary change:

```bash
# prove the process answering is newer than the edit
lsof -nP -iTCP:11434 -sTCP:LISTEN          # production — read-only
ps -o pid,lstart,etime,cmd -p <pid>
curl -s http://127.0.0.1:11434/api/version
./zerollama doctor                         # includes serve identity
```

Kill **by port**, assert free, then start. Never trust a restart command’s exit code alone. Agents must not free production **11434/8081**; operators do that deliberately.

#### Trap 61 — advertised window fails silently

Distinct from trap **55** (quality in the trained regime / three numbers disagreeing). Trap **61** is *no error*: long prompts return HTTP 200 with exact `prompt_tokens`, yet the head may be unread.

Zerollama surfaces the **arithmetic** half:

- Manifest doctor: `trap-55/61 (context)` when advertised / `num_ctx` / GGUF trained diverge ([`internal/modelhealth/traps.go`](../internal/modelhealth/traps.go))
- Live doctor: `context ceilings … (trap 55/61)` from `/api/ps` `loaded_metadata`

The **behavioural** half remains a hand-run (cold only — warm prefix cache lies; see upstream trap 60):

1. Plant a fact at position 0, unique filler, decoy at the tail, ask for the fact.
2. Ladder depths (1k → served `num_ctx`); compare local tokenize vs response `prompt_eval_count` / usage.
3. Record recovery + `done_reason` / finish reason each rung.
4. Treat **trained** context from `loaded_metadata.train_context_length` as the supported window; advertised/served above that are capability claims, not guarantees.

```bash
# warm runner already loaded — arithmetic only
curl -s http://127.0.0.1:11434/api/ps | jq '.models[].loaded_metadata|{num_ctx,train_context_length}'
```

---

## 3. `zerollama doctor` coverage (native)

`zerollama doctor` includes minefield-style checks in-tree:

1. **Serve identity** ([`cmd/doctor_serve_identity.go`](../cmd/doctor_serve_identity.go)): **53** — who holds the port / version / start time
2. **Model config traps** ([`internal/modelhealth/traps.go`](../internal/modelhealth/traps.go)): **21**, **10**, **56**, **55/61** (arithmetic)
3. **Live serving traps** ([`cmd/doctor_serving_traps.go`](../cmd/doctor_serving_traps.go)): **55/61** ceilings on warm `/api/ps`, **01/03**, **12/64/65**, **19** — warn (not fail) when inconclusive

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
| 01 | Wrong reasoning field name | `covered via doctor` | Lab CLEAN on `qwen3:0.6b` (`reasoning`); `doctorCheckReasoningField` |
| 03 | Thinking toggle / default drift | `covered via doctor` | Lab CLEAN toggle map on `qwen3:0.6b` |
| 04 | History reasoning stripping | `covered via test` / code | [`server/chat_sanitize.go`](../server/chat_sanitize.go) |
| 12 | Empty content at token ceiling | `documented` / model-dependent | CLEAN on `qwen2.5:0.5b` @ 512; **PROBLEM** on `qwen3:0.6b` @ 512 (honest truncation into reasoning) |
| 19 | Tool parsing / structured calls | `covered via doctor` | Lab clean + `doctorCheckToolCallShape` |
| **29** | Server thinking-off is not a gate | **optional gate** | Default still allows client re-enable. Set `ZEROLLAMA_THINKING_GATE=deny` (400) or `strip` (force off) on lanes sized for non-thinking budgets ([`envconfig/thinking_gate.go`](../envconfig/thinking_gate.go)) |
| 57 | Thinking kwarg truthiness | `n/a` / `partial` | Native `ThinkValue` typed; OpenAI aliases mapped |
| 58/64/65 | Effort / toggle / rescue | `covered via doctor` + test | [`server/runtime_v1_legacy_test.go`](../server/runtime_v1_legacy_test.go) |
| **77** | Only one request field validated | **fixed** | Unknown top-level keys on `/v1`, `/api/chat`, and `/api/generate` → 400; known nested kwargs validated |
| **78** | `tool_choice` fails open | **fixed** | `tool_choice: "none"` omits tools in chat + responses conversion |

### Model config / Core gaps (also in native doctor)

| Trap | Status | Where |
|------|--------|-------|
| 10, 21, 55/61, 56 | `covered via doctor` | 55/61 = arithmetic; 61 behavioural ladder = §2.1 hand-run |
| **35** | `documented` | Agreement-floor protocol in §2.1 — no automated doctor check |
| **53** | `covered via doctor` | `doctorCheckServeIdentity` — pid/start/version of answering process |

---

## Updating this doc

When the broker/admission sources of free VRAM change, or a lab doctor re-run flips PROBLEMS ↔ CLEAN, update sections 1–2 and the matching regression tests under `discover/` and `openai/`.
