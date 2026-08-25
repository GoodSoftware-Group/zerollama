---
name: configure-zerollama-env
description: "Help a user navigate zerollama's ~1000 environment variables and YAML profiles (OLLAMA_*/ZEROLLAMA_*) — discover current effective config, prefer profiles over raw flags, and verify changes instead of guessing."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, config, env-vars, yaml, profiles, tuning]
    category: mlops
    related_skills: [install-zerollama, diagnose-server-health, gpu-capability-discovery, fleet-vram-admission]
---

# Configure Zerollama Env Skill

[zerollama](https://github.com/GoodSoftware-Group/zerollama) has hundreds of
`OLLAMA_*`/`ZEROLLAMA_*` environment variables across the Go daemon and the
Python runtime sidecar, plus YAML config profiles. Nobody should memorize
all of them, and neither should you. This skill is a **procedure** for
figuring out the *few* knobs that matter for a given goal, using the tools
zerollama already ships for exactly this problem — not a flag dictionary.

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

- The user wants to tune, enable, or disable a zerollama feature and isn't
  sure which env var(s) control it
- Debugging behavior that "should" be controlled by an env var that doesn't
  seem to be taking effect
- Reviewing/cleaning up a shell profile full of stale `ZEROLLAMA_*` exports

## Core principle: discover before guessing

Never propose an env var from memory/guessing when a discovery tool exists.
Use these, in order:

```bash
# 1. Curated Go-level daemon flags with descriptions (the ~15-20 that matter most)
zerollama serve --help

# 2. Full effective runtime/sidecar config (L3, KV, VRAM, autoconfig) — no server needed
./scripts/runtime/runtime_env_doctor.sh

# 3. Same data from a live server
curl -s http://127.0.0.1:8081/health | python3 -m json.tool

# 4. Environment + build + model health in one report
zerollama doctor --json

# 5. llama.cpp vendor/patch resolution (root, binary, patch drift)
./scripts/vendor/llama_patch_doctor.sh
```

`runtime_env_doctor.sh` in particular prints the **resolved** config after
YAML + smart-default + env layering — this is what the sidecar will
actually use, not just what's currently exported in your shell.

## Prefer profiles over raw env

Zerollama's own docs call this out as the #1 anti-pattern: stacking 5-6
individual `ZEROLLAMA_*` exports when one profile does the same thing.
**Env always wins over YAML when explicitly set** — so profiles are safe
defaults you can still override surgically.

| Goal | Prefer this | Not this |
|---|---|---|
| CUDA prod throughput (L1 + safe defaults) | `ZEROLLAMA_INFERENCE_PROFILE=auto` | Stacking `GPU_PROFILE` + `LLAMA_CACHE` + `FORK=0` + `GGML_CUDA_USE_GRAPHS=0` |
| Multi-slot agent + prefix cache reuse | `ZEROLLAMA_INFERENCE_PROFILE=agent` (or `ZEROLLAMA_L3_PROFILE=agent`) | 5+ individual `ZEROLLAMA_LLAMA_CACHE*`/`ZEROLLAMA_RADIX_*` exports |
| Long-ctx VRAM (TBQ when ctx ≥ 32k) | `ZEROLLAMA_INFERENCE_PROFILE=vram` | Flipping `ZEROLLAMA_LLAMA_FORK=1` for tok/s |
| Custom runtime tuning | `ZEROLLAMA_RUNTIME_CONFIG=runtime/configs/<name>.yaml` | Dozens of `ZEROLLAMA_RUNTIME_*` exports |
| L3/prefix trace debugging | `ZEROLLAMA_DEBUG=l3` | Guessing which subsystem's debug flag to flip |
| Single discrete GPU (16GB class) | let autoconfig pick `single_gpu.yaml` | Manually setting every VRAM knob |
| Apple Silicon | let autoconfig pick `apple_silicon.yaml` | Manually replicating Mac defaults on Linux env |

Autoconfig already resolves a sane profile per platform at startup — check
`autoconfig.pick` in `/health` (see `gpu-capability-discovery`) before
assuming you need to override anything by hand.

## Category → where to actually look

Don't try to hold the full variable list in context. Route by goal instead:

| Category | Env prefix / example | Doc |
|---|---|---|
| Core daemon (host, models dir, keep-alive, parallelism, queue) | `OLLAMA_HOST`, `OLLAMA_MODELS`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_MAX_QUEUE` | `zerollama serve --help` |
| Runtime sidecar / L3 prefix cache | `ZEROLLAMA_L3_PROFILE`, `ZEROLLAMA_LLAMA_CACHE*`, `ZEROLLAMA_RADIX_*` | `docs/runtime-env.md`, `docs/gpu-profiles-l3.md` |
| Phase 15 in-process KV decode | `ZEROLLAMA_KV_NATIVE_*`, `ZEROLLAMA_KV_AUTO_BATCH*` | `docs/runtime-env.md`, `docs/phase15-native-kv.md` |
| VRAM / admission policy | `ZEROLLAMA_RUNTIME_VRAM_*`, `ZEROLLAMA_RUNTIME_INFERENCE_POLICY` | `docs/runtime-env.md`, `docs/phase13-runtime-vram.md`, `docs/scheduling-vram-policy.md` |
| Training queue / VRAM handoff | `OLLAMA_TRAINING*`, `ZEROLLAMA_TRAINING_*`, `ZEROLLAMA_BLOCK_INFERENCE_DURING_TRAINING` | `docs/gpu-training.md` |
| Multimodal backends (image/video/audio) | `OLLAMA_COMFYUI_*`, `OLLAMA_VIDEO_*`, `OLLAMA_SGLANG_URL`, `TTS`/Whisper binaries | `docs/multimodal-backends.md`, `docs/comfyui-image-backend.md`, `docs/video-understanding.md` |
| Fleet / multi-node | `ZEROLLAMA_FLEET_*` | `docs/fleet-management.md`, `docs/fleet-scheduling.md` |
| Cloud proxy (Eliza) | `ELIZACLOUD_API_KEY`, `OLLAMA_CLOUD_BASE_URL`, `OLLAMA_NO_CLOUD` | `docs/eliza-cloud.md` |
| llama.cpp vendor/build | `LLAMA_CPP_ROOT`, `LLAMA_SERVER_BIN`, `LLAMA_CPP_LIB`, `LLAMA_CPP_REPO` | `docs/runtime-env.md`, `docs/llama-cpp-unification.md` |
| LM Studio cache import | `OLLAMA_LMSTUDIO_*` | `docs/lmstudio-import.md` |
| Edge/thin deploy | `ZEROLLAMA_EDGE`, `--edge` build tag | `docs/phase16-thin-edge.md` |

`docs/README.md` in the repo is the master index if a category above isn't
enough — grep it for the feature name before inventing a variable name.

## Workflow

1. **Ask what outcome the user wants** (not which variable) — "reduce VRAM
   use," "make agent turns reuse cached prefix," "run training on a second
   GPU while chat uses the first," etc.
2. **Check current effective state first** with the discovery commands
   above. Half the time the desired behavior is already the default and the
   real problem is something else (see `diagnose-server-health`).
3. **Look up the category** in the table above; read the linked doc's
   variable table rather than guessing a name pattern.
4. **Prefer the profile/YAML knob** over a pile of individual env vars if
   one exists for the goal.
5. **Set the minimal env override** — remember env beats YAML, so you don't
   need to edit YAML files for a one-off change.
6. **Verify, don't assume** — re-run `runtime_env_doctor.sh` or check
   `/health` after restart to confirm the change actually took effect
   before declaring success.

## Pitfalls

- **Exporting Metal-tuning env then running a CUDA smoke** (or vice versa)
  — disk cache and profile context leak across runs; use profiles or
  `runtime_env_doctor.sh` to check what's actually active instead of
  trusting your shell history.
- **Six L3 env vars when `ZEROLLAMA_L3_PROFILE=agent` does the same thing**
  — always check for a profile shortcut before hand-assembling a config
  from primitives.
- **Stale `LLAMA_CPP_ROOT=../llama.cpp` overrides** — the runtime prefers
  the pinned vendor tree; a leftover manual override can silently point at
  a stale/unpatched sibling checkout. Rebuild vendor if `/kv/seq-copy`
  routes 404.
- **Assuming a variable exists because it "should"** — always confirm via
  `grep` in `envconfig/config.go` / `runtime/runtime/env.py` or the doc
  tables, not by pattern-matching a plausible-looking name.
- **Restarting the server without re-checking effective config** — env var
  changes to a *running* process don't apply; and YAML/profile resolution
  can pick something other than what you expect. Always re-verify with
  `/health` or `runtime_env_doctor.sh` post-restart.
- **Editing YAML when an env override would do** — env always wins over
  YAML when set, so a one-off tuning change rarely needs a YAML edit;
  reserve YAML edits for changes you want to persist as the new default.

## Related

- `install-zerollama` — getting the binary built and running before any of this applies
- `diagnose-server-health` — `zerollama doctor`, often the first stop when something "configured" isn't working
- `gpu-capability-discovery` — reading `/health` autoconfig/backend fields this skill also relies on
- `fleet-vram-admission` — VRAM/admission behavior driven by several of these env categories
