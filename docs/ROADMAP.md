# Roadmap

This file is the **executive roadmap**: a short view of what matters next.

- **Detailed product backlogs** live in [roadmaps/](roadmaps).
- **Research / exploratory / borrowings work** lives in [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md).
- **Shipped / historical detail** belongs in [CHANGELOG](../CHANGELOG.md).

---

## Now

- **Core serving:** Phase **15** page bind / native decode closure, Phase **16** edge cleanup, and Phase **11** admission polish. Detailed backlog: [core-serving.md](roadmaps/core-serving.md#local-inference--actionable-phases).
- **Apple serving:** keep product-track Apple work on **M16** Flash-MoE and **M24** GGUF MTP. Details: [core-serving.md](roadmaps/core-serving.md#apple-silicon--metal-track).
- **Surfaces and integrations:** Eliza route hardening, **LAYA1/LAYA2**, and the **AR7** auto-router gate. Details: [surfaces-and-integrations.md](roadmaps/surfaces-and-integrations.md#zerollama-remote-cloud-eliza), [structured surfaces](roadmaps/surfaces-and-integrations.md#structured-inference-surfaces).
- **Training and scheduling:** **T9 → T10 → T11 → T6**, plus **SL1/SL2** Slurm follow-through. Details: [training-and-scheduling.md](roadmaps/training-and-scheduling.md#gpu-training-fine-tuning), [Slurm](roadmaps/training-and-scheduling.md#slurm).
- **Media:** Wan cancel / TTL, `mmgp` 16 GB offload, first image workflow calibration, and mesh service bake. Details: [media.md](roadmaps/media.md#video-generation--dit-media-toolkit-wan--ltx--h3), [image](roadmaps/media.md#image-generation--mlx-fast-path--comfyui-utility-partial), [mesh](roadmaps/media.md#mesh--glb-generation--trellis-lab-http-not-started).
- **Audio:** one writer per realtime socket (six `voicepipe` tests fail on master), body caps on the speech endpoints, and Piper plus sidecar lifetime. Details: [media.md](roadmaps/media.md#speech-transcription-and-realtime-voice).

## Next

- **Core polish:** Phase **13** clamp/render cleanup after the serving-critical phases above.
- **Surfaces:** **LAYA3**, **GD4**, **AR9**, and **GL5** once the active gates are settled.
- **Scheduling:** **SL3/SL4** once the current guest / submit path stabilizes.
- **Integrations:** LM Studio parity work after serving and training queues settle.
- **Media second wave:** Wan CPU sidecar, music **m1/m2/m3**, image **v0.3**, mesh **g2**.

## Later / Parked

- **Training architecture:** **T2** only when auth is a deployment requirement; **T5** only if Python embed becomes the bottleneck.
- **Surfaces:** **LAYA4** and **AR10** stay parked until the active surfaces justify more investment.
- **Integrations:** LM Studio fixture cache and MLX compatibility matrix stay later-tier.
- **Media:** long-tail runner/value expansion and deeper native acceleration stay behind the near-term product backlog.
- **Code structure:** collapse the duplicated digest scan and split the oversized serving files (**R2 → R7**). **R1** (the unused `zzauditvp` voice clone) is done. Details: [core-serving.md](roadmaps/core-serving.md#code-structure).

## Track roadmaps

- [Core serving roadmap](roadmaps/core-serving.md)
- [Surfaces and integrations roadmap](roadmaps/surfaces-and-integrations.md)
- [Training and scheduling roadmap](roadmaps/training-and-scheduling.md)
- [Media roadmap](roadmaps/media.md)
- [Research roadmap](RESEARCH_ROADMAP.md)

---

## Compatibility anchors

These headings stay here as short pointers so older cross-doc links keep working.

## Product model: queues, stakeholders, and GPU time

Moved to [core-serving.md](roadmaps/core-serving.md#product-model-queues-stakeholders-and-gpu-time).

## Technology ladder (north star)

Moved to [core-serving.md](roadmaps/core-serving.md#technology-ladder-north-star).

## Local inference — actionable phases

Moved to [core-serving.md](roadmaps/core-serving.md#local-inference--actionable-phases).

### Phase 15 — exit criteria (Partial)

Moved to [core-serving.md](roadmaps/core-serving.md#phase-15--exit-criteria-partial).

### Phase 17 — upstream GGUF path alignment

Moved to [core-serving.md](roadmaps/core-serving.md#phase-17--upstream-gguf-path-alignment).

### Phase 16 — exit criteria (Partial)

Moved to [core-serving.md](roadmaps/core-serving.md#phase-16--exit-criteria-partial).

## Zerollama remote cloud (Eliza)

Moved to [surfaces-and-integrations.md](roadmaps/surfaces-and-integrations.md#zerollama-remote-cloud-eliza).

## Python inference runtime (GGUF + PagedAttention)

Moved to [core-serving.md](roadmaps/core-serving.md#python-inference-runtime-gguf--pagedattention).

## Apple Silicon & Metal track

Moved to [core-serving.md](roadmaps/core-serving.md#apple-silicon--metal-track). Research-track Apple items live in [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md#apple-silicon--metal-experiments).

## Fleet scheduling (multi-node)

Moved to [training-and-scheduling.md](roadmaps/training-and-scheduling.md#fleet-scheduling-multi-node).

## Slurm

Moved to [training-and-scheduling.md](roadmaps/training-and-scheduling.md#slurm).

## GPU training (fine-tuning)

Moved to [training-and-scheduling.md](roadmaps/training-and-scheduling.md#gpu-training-fine-tuning).

### Phases (training track)

Moved to [training-and-scheduling.md](roadmaps/training-and-scheduling.md#phases-training-track).

## Video generation — DiT media toolkit (Wan + LTX + H3)

Moved to [media.md](roadmaps/media.md#video-generation--dit-media-toolkit-wan--ltx--h3).

## Music generation — MiniMax Music 3

Moved to [media.md](roadmaps/media.md#music-generation--minimax-music-3).

## LM Studio integration

Moved to [surfaces-and-integrations.md](roadmaps/surfaces-and-integrations.md#lm-studio-integration).
