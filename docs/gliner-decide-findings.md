# GLiNER2.5-Decide — findings (WHYs)

Lab notes for the **GD\*** Decider track. Keep WHYs here so Decide is not bolted onto NER extract or claimed calibrated without a gate.

## Finding 1 — Decide ≠ span GLiNER

**What:** Fastino [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) is an encoder **classifier** (`classify_text`). Our C++ `gliner-server` is ONNX **span/token NER**.

**Why:** Sharing `ZEROLLAMA_GLINER_URL` or `/v1/extract` would lie about the wire and break agents that expect spans.

**Learning:** Separate URL (`ZEROLLAMA_GLINER_DECIDE_URL`), modality `gliner-decide`, and mechanical `/v1/gliner-decide`.

## Finding 2 — VRAM coexistence on 16 GB

**What:** Decide (340M) plus OpenJev `-ngl 25` or CUDA NER ORT can OOM or thrash on the 5080.

**Why:** Each stack holds its own weights; no shared VRAM broker for these siblings yet.

**Learning:** Unload OpenJev/emb/NER CUDA before Decide GPU; default sibling device `cpu` for lab smoke.

## Finding 3 — calibrated:false until fixture gate

**What:** v0 answers always set `"calibrated": false`.

**Why:** Fastino reports strong average accuracy on their suite, but we have not run an in-repo holdout / ECE gate comparable to OpenJev DG7–DG9.

**Learning:** Do not flip calibrated without GD-cal gate + operator opt-in (mirror OpenJev pattern later).

## Finding 4 — Name collision with `gliner-*` extract

**What:** Extract heuristics historically matched `gliner-*`, which would swallow `gliner-decide`.

**Why:** Prefix overlap.

**Learning:** Extract `isGlinerModelName` excludes `gliner-decide*` / `gliner2.5-decide*`; Decide heuristics own those names.
