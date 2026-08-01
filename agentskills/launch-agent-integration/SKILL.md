---
name: launch-agent-integration
description: "Wire up a coding agent CLI (Cline, OpenCode, Droid, Pi, Hermes, etc.) to a local zerollama server via zerollama launch, using one shared model inventory instead of per-integration config hacks."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, launch, integrations, cline, opencode, droid, pi, config]
    category: mlops
    related_skills: [zerollama-integration, hermes-provider, download-model]
---

# Launch Agent Integration Skill

Configure and start a third-party coding-agent CLI against a local
[zerollama](https://github.com/GoodSoftware-Group/zerollama) server using
`zerollama launch`, instead of hand-writing that tool's config file. This
generalizes what `hermes-provider` does manually for one harness — `launch`
does it for many, backed by one shared model inventory load
(`GET /api/tags` once, not one `/api/show` per model).

## Compatibility check

This skill targets zerollama **tip/dev**, not a specific pinned
release — not every server will have every endpoint/flag below yet.
Verify before relying on this in an unattended flow, especially
against a host you don't control:

```bash
zerollama --version                      # binary build
curl -s http://localhost:11434/api/version | jq   # server build (if reachable)
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/tags   # 200/400 = route exists; 404 = missing on this build
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:11434/api/show   # 200/400 = route exists; 404 = missing on this build
```

A **404** on an endpoint above (or an unrecognized flag/subcommand) means this build predates the feature this skill
describes — check [`CHANGELOG.md`](../CHANGELOG.md) for when it
landed, or upgrade (`git pull && ./scripts/build/build_zerollama_mac.sh`)
rather than assuming the request shape is wrong.

## When to Use

- Setting up a new coding agent CLI to use local zerollama models
- The user names one of the supported integrations (Cline, OpenCode, Droid,
  Pi, Hermes, elizaOS, OpenClaw, OMP, Zoey, Copilot CLI, VS Code)
- Debugging a launched integration's config that has stale/missing model
  metadata (context length, vision/tool capability)

## Supported integrations

| Name | Aliases |
|---|---|
| `cline` | |
| `copilot` | `copilot-cli` |
| `droid` | |
| `eliza` | `elizaos` |
| `hermes` | |
| `opencode` | |
| `omp` | oh-my-pi |
| `openclaw` | `clawdbot`, `moltbot` |
| `pi` | |
| `vscode` | `code` |
| `zoey` | |

## How to Run

```bash
# Interactive menu (choose integration + model)
zerollama launch

# Launch a specific integration directly
zerollama launch opencode

# Pick a model explicitly instead of the picker
zerollama launch opencode --model qwen3-coder-next:6bit

# Write config only, don't auto-launch the integration's process
zerollama launch droid --config

# Skip interactive confirmations (non-interactive/CI use)
zerollama launch pi --yes

# Pass extra args straight through to the integration after --
zerollama launch pi -- --help
```

`zerollama launch` (no args) is equivalent to running `zerollama` directly
(the interactive menu).

## Why one inventory load per run

Before this pattern, each integration called `/api/show` per model when
writing its config — five models meant five sequential Show round trips
(5–25s), and `/api/tags` vs `/api/show` could drift, producing half-empty
configs (missing `contextWindow`, wrong vision/tool flags). `zerollama
launch` now lists tags **once**, resolves the selected names into
`[]LaunchModel` structs, and passes that same slice to every integration's
`Edit` / `ConfigureWithModels` / `Run` call.

**This only fully works for GGUF models.** `/api/tags` fills
`details.context_length` from the GGUF header at list time
(`server/model_details.go`), so GGUF-backed integrations get a real
`contextWindow` from that single load. MLX/safetensors models never get
`ContextLength` populated (`enrichModelDetailsFromSafetensors` doesn't set
it, and neither does a fallback `/api/show` call) — an integration config
generated for an MLX model may end up with `contextWindow: 0` or omitted
entirely. If a launched integration is missing context for an MLX model,
that's this gap, not a launch bug to debug further.

## Pitfalls

- **`--model` only makes sense with an integration name** — bare
  `zerollama launch --model X` without an integration argument is invalid;
  flags/extra args require naming the integration first.
- **`--config` skips auto-launch** — use it when you only want the config
  file written (e.g. to inspect or version-control it) without starting the
  integration's process.
- **Extra args need `--`** — `zerollama launch pi --help` tries to parse
  `--help` as a `launch` flag; use `zerollama launch pi -- --help` to pass
  it through to Pi instead.
- **Non-interactive contexts should pass `--yes`** — without it, a
  non-interactive session (script/CI) may hang on a confirmation prompt
  that would otherwise show a picker.
- **Config drift after pulling new models** — re-run `zerollama launch
  <integration> --config` after pulling/removing models so the written
  config's model list reflects current `/api/tags`, rather than assuming
  an old config auto-updates.
- **MLX/safetensors models get no `context_length` from `/api/tags`** —
  only GGUF models have it populated from the GGUF header at list time;
  an MLX model's launched config may show a missing/zero context window
  even though the model genuinely supports one (check its own model card).
- **Not every integration supports every zerollama feature** — QoS
  extras, `keep_alive`, and prompt-cache wiring are integration-specific;
  check that integration's own generated config for what actually got set,
  rather than assuming full parity with the `hermes-provider` skill's
  manual recipe.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `hermes-provider` — the manual version of this wiring for one specific harness (Hermes)
- `download-model` — pulling models before they show up in the launch picker
