# Surfaces And Integrations Roadmap

This file holds the detailed backlog for **product-facing API surfaces and integrations**.

Research-only tracks live in [../RESEARCH_ROADMAP.md](../RESEARCH_ROADMAP.md).

---

## Zerollama remote cloud (Eliza)

Open follow-ups only:

- **Stricter response mapping** where Eliza schemas are stable enough to normalize without losing fields clients depend on.
- **Catalog collision policy** when local and remote names overlap.
- **Multipart / image route behavior** when `model` is absent from JSON and routing has to infer intent.

Current behavior is documented in [../eliza-cloud.md](../design/surfaces/eliza-cloud.md). Non-goals from that doc still apply.

---

## Structured inference surfaces

These are product-facing API surfaces that still have open backlog work, even when the underlying model or runtime experiments live elsewhere.

### Typed decisions (Laya / CLM / DiffusionGemma)

Open work only:

| Milestone | Status | Open work |
|-----------|--------|-----------|
| **LAYA1** | **Active** | `LLM_ARCH_LAYA` + GGUF convert path |
| **LAYA2** | **Active** | llama-server `--decisions` + Go `/v1/decisions` |
| **LAYA3** | **Later** | HF pull + multilingual polish |
| **LAYA4** | **Parked** | MLX sidecar / mlxrunner graph |
| **GD4** | **Partial** | CUDA/device coexistence and VRAM policy for GLiNER-decide |

Closed CLM / DiffusionGemma / GD0–GD3 history stays in the track docs and [CHANGELOG](../../CHANGELOG.md).

### Entity extract (GLiNER)

Open work only:

| Milestone | Status | Open work |
|-----------|--------|-----------|
| **GL5** | **Parked** | RelEx / bi-encoder path if GLiNER.cpp grows that surface |

### Auto-router (`model: "auto"`)

Open work only:

| Milestone | Status | Open work |
|-----------|--------|-----------|
| **AR7** | **Pending lab run** | Run the labeled eval gate and decide whether the current Jev path is production-worthy |
| **AR9** | **Deferred** | Server-side modality handoff in chat responses after AR7 evidence |
| **AR10** | **Parked** | Feedback loop into KNN corpus / Laya calibration |

---

## LM Studio integration

Open follow-ups only:

| Item | Open work |
|------|-----------|
| In-place MLX load | Avoid repack doubling disk use |
| Cross-volume symlinks | Support split `OLLAMA_MODELS` vs LM Studio cache disks cleanly |
| Quant tag parity | Fuzzy match tests and publisher naming table |
| CI fixture cache | Real regression coverage without massive weights |
| GLM / Hermes MLX runtime bugs | Compatibility matrix and upstream fixes |

History for shipped v0.0.1 behavior belongs in [CHANGELOG](../../CHANGELOG.md) and [../lmstudio-import.md](../design/surfaces/lmstudio-import.md).
