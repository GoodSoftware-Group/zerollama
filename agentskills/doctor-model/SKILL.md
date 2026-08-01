---
name: doctor-model
description: "Diagnose a specific local model's manifest/blob health (ok/repairable/orphaned/broken) and config-level footguns (quant label, missing generation config, context mismatch, no chat template) with zerollama doctor --models and the model-serving-minefield trap registry."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, doctor, model-health, manifest, blob-integrity, minefield, repair]
    category: mlops
    related_skills: [diagnose-server-health, model-authoring, download-model, benchmark-model-speed, model-suggester]
---

# Doctor Model Skill

Diagnose whether *a specific model* is safe to serve on
[zerollama](https://github.com/GoodSoftware-Group/zerollama) — separate from
`diagnose-server-health`, which checks the runtime/toolchain. This skill is
about the model's manifest, blobs, and metadata: is it downloaded correctly,
does its config lie about quant/context, and (once a model is warm) does live
serving hit any of the ~107 [model-serving-minefield](https://github.com/Blackwellboy/model-serving-minefield)
footguns zerollama tracks.

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
zerollama <subcommand> --help            # confirm the flag/subcommand exists before scripting it
```

An unrecognized flag/subcommand, or `--help` not mentioning an option this skill relies on, means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- A model fails to load or gives a "manifest not found" / missing-blob error
- Deciding whether to trust a model's advertised quant label or context window
- Before publishing a benchmark number for a model (rule out config traps
  that would explain a bad score before blaming the checkpoint)
- After `zerollama pull` was interrupted or an LM Studio cache was reorganized
- A model works but responses seem truncated/empty at long context, or a
  reasoning model's thinking is inconsistently stripped

## Two layers of model health

### 1. Manifest / blob integrity (`zerollama doctor --models`)

Every locally registered model tag gets one status:

| Status | Meaning | Fix hint given |
|---|---|---|
| `ok` | Manifest + all blobs present | — |
| `repairable` | Blobs missing, but an LM Studio cache match exists | `zerollama list` or `zerollama pull <name>` to re-import |
| `orphaned` | Blobs missing, no known source to recover from | `zerollama rm <name>` then re-pull/re-download |
| `broken` | Manifest itself not found under any models dir | `zerollama pull <name>` |

```bash
zerollama doctor --models                 # human report, all local tags
zerollama doctor --models --json          # machine-readable
zerollama doctor --models --fix           # auto-removes ONLY orphaned tags
zerollama doctor --models --audit         # + blob storage rollup (see also `zerollama blobs audit`)
```

`--fix` with `--models` only removes `orphaned` manifests (dangling
registration with genuinely no recovery path). It never touches `repairable`
or `broken` tags — those need an explicit `pull`/re-import because deleting
first would lose the LM Studio match.

### 2. Config-level footguns (manifest metadata vs GGUF reality)

Independent of blob integrity, a model's manifest metadata can *disagree*
with what's actually in the GGUF header. `internal/modelhealth/traps.go`
checks four such traps (IDs from the upstream minefield registry) whenever a
model's blobs are intact:

| Trap ID | Topic | What it flags |
|---|---|---|
| **10** | Quant label mismatch | Tag says `q4_k_m` but GGUF `general.file_type` is a different quant — trust the GGUF, not the tag/repo name |
| **21** | No generation_config | No sampling keys (`temperature`, `top_p`, `top_k`, ...) in the params layer → server defaults silently win; not necessarily wrong, but worth knowing |
| **55/61** | Context mismatch (arithmetic half) | Advertised context, manifest `num_ctx`, and GGUF `context_length` disagree — treat GGUF trained length as the real ceiling, not the advertised number |
| **56** | No chat template | Neither a Modelfile `TEMPLATE` nor a GGUF `tokenizer.chat_template` is present — the model can't render a chat prompt at all |

These specific four checks are implemented in
[`internal/modelhealth/traps.go`](../../internal/modelhealth/traps.go) as a
library (`modelhealth.CheckConfigTrapsAll()` / `CheckName`-adjacent
`checkConfigTrapsIn`) but are **not yet wired into a `zerollama doctor` CLI
flag** — there is no `--traps` flag today. To use them from an agent:

- Read `zerollama show <model> --modelfile` and cross-check `PARAMETER
  num_ctx` / sampling lines against `zerollama show <model> --parameters`
  and the GGUF's own `general.file_type` / `*.context_length` (visible via
  `zerollama show <model> --verbose` or the pulled GGUF header) by hand, or
- Call the Go library directly if scripting inside the repo (tests in
  `internal/modelhealth/traps_test.go` show the exact call shape).

Trap **55/61**'s *behavioural* half (does a long prompt silently get
truncated rather than erroring) is not arithmetic and needs a live probe —
see `docs/model-serving-minefield.md` §2.1 for the cold/warm ladder script
(`scripts/minefield_cold_ladder.sh <model>`), lab ports only.

### 3. Live serving traps (once the model is warm)

`zerollama doctor` (no `--models`) also runs ~19 live serving-trap probes
against whatever model is currently loaded — reasoning field names, thinking
toggle drift, orphaned `</think>`, tool-call shape, streamed content
placement, kwarg deadness, latency reconciliation, serve identity (are you
even talking to the process you think you are), and empty-content-at-ceiling
(deep mode only):

```bash
zerollama run <model>              # warm the model first
zerollama doctor                   # now includes live serving probes
ZEROLLAMA_DOCTOR_DEEP=1 zerollama doctor   # + trap-12 ceiling check @ 512 tokens
```

Full trap-to-check mapping (which numbered minefield trap each doctor probe
covers, plus which traps are hand-run scripts vs fully automated) lives in
[`docs/model-serving-minefield.md`](../../docs/model-serving-minefield.md) —
read it before assuming a trap is or isn't covered.

## How to Run (typical agent flow)

```bash
# 1. Is this model's registration intact?
zerollama doctor --models --json | jq '.checks[] | select(.name == "model '"'"'<tag>'"'"'")'

# 2. If ok, warm it and run live checks
zerollama run <tag>
zerollama doctor --json | jq '.checks[] | select(.status != "ok")'

# 3. If publishing a benchmark, also rule out config traps by hand
zerollama show <tag> --parameters
zerollama show <tag> --modelfile
```

## Pitfalls

- **`--models` alone does not run config traps (10/21/55/61/56)** — it is
  blob-integrity only. Don't assume a clean `--models` report means the
  model's metadata is trustworthy; it only means the files exist.
- **`--fix --models` is scoped to `orphaned` only** — it will never delete a
  `broken` or `repairable` tag automatically; those need explicit operator
  action so a re-importable model isn't lost.
- **Trap 55/61 has two halves** — the *arithmetic* half (numbers disagree)
  is what doctor/traps.go can check offline; the *behavioural* half (a long
  prompt returns HTTP 200 with no error but the head went unread) requires a
  live cold+warm ladder, not just reading the manifest.
- **`repairable` requires `ZEROLLAMA_LMSTUDIO_IMPORT`-eligible matching** —
  if LM Studio import is disabled or the cache was moved, a model that would
  otherwise be `repairable` reports as `orphaned` instead.
- **Manifest metadata problems are a different fix path than blob repair** —
  use the `model-authoring` skill's `zerollama repair` (dry-run by default,
  `--write` to apply) to refresh manifest params/config from GGUF headers;
  `doctor --models` never rewrites metadata, only reports/removes.
- **Config traps only run when blobs are intact** — `CheckConfigTrapsAll`
  explicitly skips any manifest with missing blobs (that's `--models`'
  job), so fix blob integrity first before trusting a "no config traps"
  read on a broken model.

## Related

- `diagnose-server-health` — environment/toolchain doctor (build, venvs,
  native libs, sidecar) — run this first if the *server* itself seems unwell
- `model-authoring` — `zerollama repair` to fix manifest metadata this skill
  only diagnoses
- `download-model` — re-pulling a model that doctor reports as
  `orphaned`/`broken`
- `benchmark-model-speed` — rule out config traps before publishing a score
- `model-suggester` — verifying a candidate's health before recommending it
