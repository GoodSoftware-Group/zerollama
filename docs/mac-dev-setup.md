# macOS developer setup

Short onboarding for **any Mac** (Apple Silicon or Intel), **any checkout path**.

**Full guide:** [apple-silicon-metal.md](./apple-silicon-metal.md) · **Build details:** [development.md](./development.md#macos-apple-silicon) · **Pin:** [runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md)

---

## Fresh clone (recommended)

One command — installs host deps user-local (no Homebrew/sudo required), builds ggml Metal `zerollama`, installs to `~/.local/bin`, creates `runtime/.venv`. Sibling Metal `libllama` is best-effort.

```bash
git clone <zerollama-repo-url> zerollama
cd zerollama
./scripts/runtime/dev_bootstrap.sh
zerollama serve
# other terminal:
zerollama pull llama3.2:3b
zerollama run llama3.2:3b
```

`dev_bootstrap.sh` defaults:

| Knob | Default | Meaning |
|------|---------|---------|
| `MAC_SETUP_HOST_DEPS` | `1` | Go / cmake / pkg-config / uv → `~/.local` |
| `MAC_SETUP_INSTALL` | `1` | `install_mac.sh` → `~/.local/bin/zerollama` |
| `MAC_SETUP_LLAMA_OPTIONAL` | `1` | ggml tier 0 succeeds without Metal `libllama` |
| `MAC_SETUP_LLAMA_CLONE` | `1` | try shallow clone of `../llama.cpp` |
| `MAC_SETUP_SMOKE` | `1` | curl `:11434/api/version` only if already up (never starts serve) |
| `BUILD_ANE_RESTORE` | `0` | skip ANE intree rewrite (avoids dirty `llama/llama.cpp/common`) |
| `MAC_SETUP_SIGNOFF` | `0` | metal sign-off off (needs a pulled model) |

Preview host installs without writing:

```bash
./scripts/runtime/mac_host_deps.sh --dry-run
```

Require runtime inprocess (`libllama`):

```bash
MAC_SETUP_LLAMA_OPTIONAL=0 ./scripts/runtime/dev_bootstrap.sh
```

For MPS LoRA training deps:

```bash
MAC_SETUP_TRAINING=1 ./scripts/runtime/dev_bootstrap.sh
```

---

## Prerequisites

| Need | How |
|------|-----|
| **Xcode.app** (or Homebrew `python@3.12`) | App Store — CGO `python3-embed` |
| **Xcode CLI tools** | `xcode-select --install` |
| **Go ≥ 1.24.1, cmake, pkg-config, uv** | **`./scripts/runtime/mac_host_deps.sh`** (user-local) — or Homebrew if you already use it |

**Not required up front:** pulled models, `../mlx`, `../bmtl`. Sibling `../llama.cpp` is cloned when possible; CGO vendor headers can be restored via a shallow pin without a full tree (`ensure_llama_cgo_vendor.sh`).

---

## Onboarding tiers

| Tier | Goal | Command |
|------|------|---------|
| **0** | Host deps + build + install + daily serve | `./scripts/runtime/dev_bootstrap.sh` |
| **1** | Pull a model + chat | `zerollama serve` (prints coach) then `zerollama pull llama3.2:3b` — or LAN models via `ZEROLLAMA_MDNS=1` + `storage serve` / `storage doctor` |
| **2** | Metal sign-off (CI ports) | `MAC_SETUP_SIGNOFF=1 MAC_SETUP_GO=0 MAC_SETUP_BUILD=0 ./scripts/runtime/mac_setup.sh` |
| **3** | qwen35 ggml smoke | `RUN_E2E_QWEN35_MODEL=eliza-1-2b:latest ./scripts/runtime/qwen35_mac_smoke.sh` |

---

## Script map (post-reorg)

| Need | Path |
|------|------|
| Fresh-clone bootstrap | `./scripts/runtime/dev_bootstrap.sh` |
| Host deps (Go/cmake/pkgconf/uv) | `./scripts/runtime/mac_host_deps.sh` |
| Install binary to PATH | `./scripts/install_mac.sh` |
| Mac setup knobs | `./scripts/runtime/mac_setup.sh` |
| CGO env / python-embed | `./scripts/runtime/mac_cgo_env.sh` |
| CGO vendor headers | `./scripts/vendor/ensure_llama_cgo_vendor.sh` |
| Runtime uv venv | `./scripts/runtime/runtime_uv_venv.sh` |
| Build `./zerollama` | `./scripts/build/build_zerollama_mac.sh` |
| Build script surface check | `./scripts/build/check_mac_build_scripts.sh` |
| Sibling llama.cpp | `./scripts/vendor/ensure_llama_cpp_sibling.sh` |
| Metal libllama / llama-server | `./scripts/build/build_llama_server.sh` |
| Metal sign-off | `./scripts/gpu/metal_signoff.sh` |

`zerollama doctor --fix` repairs an **already built** env (venv + sibling + rebuild). On a bare Mac, start with `dev_bootstrap.sh` (shell-only host deps — no chicken-and-egg).

---

## Ports: daily dev vs CI smokes

| Layout | Go API | Runtime sidecar | When |
|--------|--------|-----------------|------|
| **`zerollama serve`** (default) | **`:11434`** | `:8081` | Daily dev |
| **Sign-off / e2e smokes** | **`:8080`** | **`:18081`** | `metal_signoff.sh`, `macos_metal_smoke.sh` |

Agents must not kill listeners on **11434** / **8081**.

---

## Daily use

```bash
zerollama serve          # :11434; auto sidecar :8081 on Darwin
zerollama doctor
zerollama doctor --fix   # uv venv + build + clone ../llama.cpp + Metal libllama
zerollama run llama3.2:3b
```

---

## Sign-off after you have a model

```bash
zerollama pull llama3.2:3b
MAC_SETUP_SIGNOFF=1 MAC_SETUP_GO=0 MAC_SETUP_BUILD=0 ./scripts/runtime/mac_setup.sh
```

---

## If `go build` fails on your Mac

**Preferred:**

```bash
./scripts/runtime/mac_host_deps.sh   # if Go/cmake/pkg-config missing
./scripts/build/build_zerollama_mac.sh
./scripts/install_mac.sh
```

**Or load env into your shell:**

```bash
eval "$(./scripts/runtime/mac_cgo_env.sh --export)"
GOFLAGS=-mod=mod go build -o zerollama .
```

| Error | Fix |
|-------|-----|
| `go: command not found` | `./scripts/runtime/mac_host_deps.sh` |
| `pkg-config` / `cmake` missing | same |
| `python3-embed not found` | Install **full Xcode.app**, or Homebrew `python@3.12`, then `mac_cgo_env --export` |
| `stddef.h file not found` | Use `build_zerollama_mac.sh` or `mac_cgo_env --export` |
| CGO vendor / `httplib.h` missing | `./scripts/vendor/ensure_llama_cgo_vendor.sh` |
| `llama.cpp` clone EOF | retry; or `LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git` |
| MLX / safetensors | `BUILD_MLX=1 ./scripts/build/build_zerollama_mac.sh` when `../mlx` exists |

**Ggml-only (skip llama.cpp build):**

```bash
MAC_SETUP_BUILD=0 MAC_SETUP_LLAMA_OPTIONAL=1 ./scripts/runtime/dev_bootstrap.sh
```

---

## Dev vs production MLX layout

| Workflow | Build | Run from |
|----------|-------|----------|
| **Daily dev** | `./scripts/build/build_zerollama_mac.sh` + `./scripts/install_mac.sh` | `zerollama` on PATH |
| **Fresh-clone bootstrap** | `./scripts/runtime/dev_bootstrap.sh` | same |
| **ggml only (fast)** | `BUILD_MLX=0 ./scripts/build/build_zerollama_mac.sh` | repo root or install |
| **Release / dist tarball** | `./scripts/build/build_production_mac.sh` | `dist/darwin-arm64/` |

See [apple-silicon-metal.md](./apple-silicon-metal.md) for MLX pin bumps and `zerollama doctor` hints.
