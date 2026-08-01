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
- **Not every integration supports every zerollama feature** — QoS
  extras, `keep_alive`, and prompt-cache wiring are integration-specific;
  check that integration's own generated config for what actually got set,
  rather than assuming full parity with the `hermes-provider` skill's
  manual recipe.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `hermes-provider` — the manual version of this wiring for one specific harness (Hermes)
- `download-model` — pulling models before they show up in the launch picker
