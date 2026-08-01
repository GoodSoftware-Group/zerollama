---
name: diagnose-server-health
description: "Run zerollama doctor to diagnose local runtime readiness (venvs, libllama, MLX, sidecar health, model blob integrity) and apply safe auto-fixes."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, doctor, diagnostics, health-check, troubleshooting]
    category: mlops
    related_skills: [zerollama-integration, install-zerollama, configure-zerollama-env, gpu-capability-discovery, download-model, doctor-model]
---

# Diagnose Server Health Skill

Run `zerollama doctor` to check whether a [zerollama](https://github.com/GoodSoftware-Group/zerollama)
install is correctly set up — build toolchain, Python venvs, native
libraries, sidecar health, and local model blob integrity — before
debugging an inference failure as if it were an application bug.

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

- Before deep-diving a confusing inference/training error — rule out an
  environment problem first
- After a fresh clone/setup, to check whether Tier 0 bootstrap succeeded
- Suspecting a corrupted or orphaned local model registration
- Checking whether the Darwin runtime sidecar (`:8081`) actually started

## How to Run

```bash
# Full human-readable report
zerollama doctor

# Machine-readable (for an agent to parse programmatically)
zerollama doctor --json

# Model blob integrity only (missing/orphaned registrations)
zerollama doctor --models

# Storage rollup
zerollama doctor --audit
zerollama doctor --models --audit

# Apply safe auto-fixes (runtime venv install; on Darwin, build Metal llama.cpp if missing)
zerollama doctor --fix
```

Exit code is non-zero when any check fails (`report.OK == false` /
`Failures > 0`); a JSON caller should treat non-zero as "environment not
ready," not necessarily "request-level bug."

## What it checks

| Category | Examples |
|---|---|
| Toolchain/platform | Go build tags (edge vs full), ggml runner linkage, `zerollama` binary presence + freshness |
| llama.cpp | Unified vendor tree resolved, patches applied, `llama-server` discoverable |
| Python | `uv` on PATH, `runtime/.venv` (fastapi importable), `.venv-training` (torch/peft importable, Darwin only, optional) |
| Native libs | `libllama.{dylib,so}` found, MLX engine loadable (Darwin) |
| Serve mode | Go API reachable (`:11434`/`:8080`), runtime sidecar reachable (`:8081`), which layout matches (Mac daily vs CI/smoke) |
| Sidecar health | `/health` autoconfig pick, `llama_backend`, `vram_probe_effective`, fallback flags |
| Models | A usable local text GGUF exists; `--models` flag finds orphaned/broken/repairable manifest registrations |

## Reading a JSON report

```json
{
  "checks": [
    {"name": "libllama", "status": "fail", "detail": "Metal/CUDA libllama not found", "fix_hint": "zerollama doctor --fix ..."}
  ],
  "failures": 1,
  "warnings": 2,
  "ok": false
}
```

Each check's `status` is `ok` / `warn` / `fail`. Always prefer the
`fix_hint` field over guessing a remediation — it's generated from the same
logic that decided the check failed.

## Pitfalls

- **`--fix` is safe but not exhaustive** — it installs the runtime venv and
  (on Darwin) builds Metal llama.cpp if missing; it does **not** fix
  training venv ABI mismatches or model blob corruption. Run
  `--models --fix`-equivalent separately (orphaned model removal happens
  automatically when `--fix --models` are combined).
- **`--models` and default checks run in parallel by default** — plain
  `zerollama doctor` (no `--fix`) merges both toolchain checks *and* model
  health into one report; `--models` alone narrows to just model health.
- **Darwin-only checks are skipped on Linux** — MLX, sidecar bootstrap,
  ANE, training venv (Darwin MPS) checks only run on `darwin`; a Linux host
  gets a single "darwin runtime smoke" warning placeholder instead, which
  is expected, not a problem.
- **Sidecar "warn" often just means timing** — "Go port open, sidecar not
  up yet" during startup is transient; re-run after a few seconds before
  treating it as a real failure.
- **A stale binary can lack `doctor` itself** — if `zerollama doctor
  --help` fails against an old binary, rebuild
  (`./scripts/build/build_zerollama_mac.sh`) rather than debugging doctor's
  own output.

## Related

- `zerollama-integration` — generic API contract, sizing, pitfalls
- `install-zerollama` — the setup steps doctor is checking the result of
- `configure-zerollama-env` — fixing an env/YAML misconfiguration doctor surfaces
- `gpu-capability-discovery` — deeper dive into which GPU backend/profile got picked
- `download-model` — fixing orphaned/missing model registrations doctor finds
- `doctor-model` — deeper per-model diagnosis (config traps, minefield trap registry)
