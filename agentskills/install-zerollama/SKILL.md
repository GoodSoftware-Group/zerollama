---
name: install-zerollama
description: "Bootstrap and build zerollama from a fresh clone on macOS or Linux — prerequisites, tiered onboarding scripts, and daily-use verification."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [zerollama, install, setup, bootstrap, build, onboarding]
    category: mlops
    related_skills: [configure-zerollama-env, diagnose-server-health, gpu-capability-discovery]
---

# Install Zerollama Skill

Get a fresh [zerollama](https://github.com/GoodSoftware-Group/zerollama)
checkout building and serving locally. This is the **setup** half —
prerequisites, bootstrap scripts, tiered verification. For the ~1000-flag
environment-variable configuration surface *after* the binary builds, use
the `configure-zerollama-env` skill instead.

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

- Setting up zerollama on a new machine (Mac or Linux) for the first time
- A build is failing and you need to know which prerequisite is missing
- Verifying a fresh install actually works before touching any config

## Prerequisites (macOS)

| Tool | Install | Why |
|---|---|---|
| Go 1.24.1+ | [go.dev/dl](https://go.dev/dl/) | Matches `go.mod` — **not** 1.22 as some older docs say |
| Full Xcode.app | App Store | CGO needs `python3-embed`; CLI tools alone are usually insufficient |
| cmake | `brew install cmake` | Builds sibling Metal `libllama`/`llama-server` |
| uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | Manages `runtime/.venv` (Python 3.11+) |
| pkg-config | ships with Xcode / `brew install pkg-config` | Locates `python3-embed` |

**CLI-tools-only Macs** (no full Xcode.app): install Homebrew
`python@3.12 pkg-config cmake`, then
`eval "$(./scripts/runtime/mac_cgo_env.sh --export)"` before building.

Not required up front: pulled models, `../mlx` sibling (safetensors only),
`../bmtl` (UMA only) — `../llama.cpp` is cloned automatically by bootstrap
if missing.

## Prerequisites (Linux/CUDA)

- Go 1.24.1+
- `python3-dev`/`python3-devel` + `pkg-config` (CGO needs `python3-embed`)
- `CGO_ENABLED=1` (default on most platforms)
- CUDA toolkit matching the target GPU's compute capability (5080 =
  `120-real`, 4090 = `89`) if building `llama-server` with CUDA
- `uv` for `runtime/.venv` and (optionally) `.venv-training`

## How to Run — macOS fresh clone (recommended path)

```bash
git clone <zerollama-repo-url> zerollama
cd zerollama
./scripts/runtime/dev_bootstrap.sh
./zerollama serve
# other terminal:
./zerollama pull llama3.2:3b
./zerollama run llama3.2:3b
```

`dev_bootstrap.sh` = `mac_setup.sh` with safe defaults: clones `../llama.cpp`
at the pinned commit if missing, **skips** Metal sign-off by default, builds
`./zerollama`, `runtime/.venv`, and Metal `libllama` when the clone/build
succeed.

**Public (non-fork) llama.cpp pin** instead of the default elizaOS sibling:

```bash
LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git ./scripts/runtime/dev_bootstrap.sh
```

**With MPS LoRA training deps:**

```bash
MAC_SETUP_TRAINING=1 ./scripts/runtime/dev_bootstrap.sh
```

**Skip building llama.cpp entirely (ggml-only, chat on `:11434` works, no
MLX runtime in-process):**

```bash
MAC_SETUP_BUILD=0 MAC_SETUP_LLAMA_OPTIONAL=1 ./scripts/runtime/dev_bootstrap.sh
```

## Onboarding tiers (verify incrementally)

| Tier | Goal | Command | Needs |
|---|---|---|---|
| 0 | Build + daily serve | `./scripts/runtime/dev_bootstrap.sh` | prerequisites above |
| 1 | Pull a model + chat | `./zerollama serve` then `./zerollama pull llama3.2:3b` | Tier 0 |
| 2 | Metal sign-off (CI regression) | `MAC_SETUP_SIGNOFF=1 MAC_SETUP_GO=0 MAC_SETUP_BUILD=0 ./scripts/runtime/mac_setup.sh` | Tier 1 |
| 3 | qwen35 ggml smoke | `RUN_E2E_QWEN35_MODEL=eliza-1-2b:latest ./scripts/runtime/qwen35_mac_smoke.sh` | Tier 1 + pulled tag |

Tier 0 is the only *required* path for a new developer/agent — tiers 2–3
are opt-in signoff/regression checks.

## Script map (post-reorg — flat `scripts/*.sh` paths no longer exist)

| Need | Path |
|---|---|
| Fresh-clone bootstrap | `./scripts/runtime/dev_bootstrap.sh` |
| Mac setup knobs | `./scripts/runtime/mac_setup.sh` |
| CGO env / python-embed | `./scripts/runtime/mac_cgo_env.sh` |
| Runtime uv venv | `./scripts/runtime/runtime_uv_venv.sh` |
| Build `./zerollama` | `./scripts/build/build_zerollama_mac.sh` |
| Sibling llama.cpp | `./scripts/vendor/ensure_llama_cpp_sibling.sh` |
| Metal libllama / llama-server | `./scripts/build/build_llama_server.sh` |
| Metal sign-off | `./scripts/gpu/metal_signoff.sh` |
| Training venv | `./scripts/training/training_uv_venv.sh` |

## Ports: daily dev vs. CI/smoke

| Layout | Go API | Runtime sidecar | When |
|---|---|---|---|
| `./zerollama serve` (default) | `:11434` | `:8081` | Daily dev — **production, don't kill** |
| Sign-off / e2e smokes | `:8080` | `:8081` | `metal_signoff.sh`, `macos_metal_smoke.sh` — set `OLLAMA_HOST=http://127.0.0.1:8080` |

Never copy a `:8080` smoke curl example against a default `:11434` daily
serve without changing the host, and never kill a process listening on
`11434`/`8081` to "fix" a port conflict — treat those as production.

## Daily verification

```bash
./zerollama serve          # :11434; auto sidecar :8081 on Darwin
./zerollama doctor          # environment/build health — see diagnose-server-health skill
./zerollama doctor --fix    # uv venv + build + clone ../llama.cpp + Metal libllama
./zerollama run llama3.2:3b
```

## Pitfalls

- **Go 1.22 is not enough** — `go.mod` requires **1.24.1+**; docs that say
  otherwise are stale.
- **CLI tools alone often fail CGO** — `mac_cgo_env.sh` looks for
  `python3-embed` under `/Applications/Xcode.app/...`; without full
  Xcode.app or the Homebrew fallback, expect `python3-embed not found`.
- **Flat script paths (`scripts/dev_bootstrap.sh`, etc.) don't exist
  post-reorg** — always use the `scripts/<category>/...` paths in the table
  above; `zerollama doctor --fix` already follows the correct paths.
- **Sibling `../llama.cpp` defaults to a fork** — `ensure_llama_cpp_sibling.sh`
  clones elizaOS's fork by default (fork kernels); set `LLAMA_CPP_REPO` to
  the public `ggml-org/llama.cpp.git` if you need the vanilla pin.
- **Don't bind `:11434`/`:8081` as a "test" server** — those are reserved
  production ports; use lab ports (`11435`, `18081`, etc.) for anything
  experimental.
- **A successful build doesn't mean full functionality is configured** —
  MLX, training, and llama-server backends are each independently optional;
  run `zerollama doctor` (not just a successful build) to see what's
  actually wired up.

## Related

- `configure-zerollama-env` — the ~1000-flag environment/YAML configuration surface, once the binary is built
- `diagnose-server-health` — `zerollama doctor` deep dive
- `gpu-capability-discovery` — confirming which GPU backend got picked after install
