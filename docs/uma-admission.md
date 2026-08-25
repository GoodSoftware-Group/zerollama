# UMA admission on Darwin (operator overview)

**Admission only** — machine-wide `uma_daemon` decides *when* units may run (`HOLD_GPU` / `HOLD_ANE` / `HOLD_AMX` + `RELEASE`). Kernels stay in MLX / ggml / llama.cpp / peer ANE-AMX owners. Zerollama Metal paths use **`HOLD_GPU`** only.

| Track | Surface | Gate |
|-------|---------|------|
| **M20** | mlxrunner | `./scripts/phase/m20_uma_signoff.sh` |
| **M21** | ollamarunner + llamarunner | `./scripts/phase/m21_ggml_uma_signoff.sh` |
| **M22** | llama-server + **runtime inprocess/subprocess** via `libllama` | `./scripts/phase/m22_llama_server_uma_signoff.sh` |
| **M23** | ANE draft eval (`HOLD_ANE`) | `./scripts/phase/m23_vendor_ane_uma_signoff.sh` (+ source / multiunit smokes) |
| **All** | operator ladder (lab) | `./scripts/phase/mac_uma_signoff.sh` (`RUN_M20=1` for heavy MLX) |

Python runtime **subprocess** (`llama-server` child) and **inprocess** (ctypes → same `libllama.dylib`) both inherit M22 when the dylib is UMA-linked. No separate Python HOLD wrap.

Client glue lives in **`x/uma`** (`libuma_embed.a` for Go `-tags uma`, `libuma_llama.a` for llama-server).

| API | Unit |
|-----|------|
| `LeaseBegin` / `RunGPU` | `HOLD_GPU` (Metal default) |
| `LeaseBeginUnit("ane"\|"amx", …)` / `RunUnit` | `HOLD_ANE` / `HOLD_AMX` (F0390; independent tickets so GPU ∥ ANE) |

Broker also advertises holds on `INFO` (`holds=…`). Optional HOLD TTL: `lease_s=` / F0395. Lab: `./scripts/phase/m23_uma_multiunit_client_smoke.sh`.

**Adopter install (bmtl toolkit):**
- Client: `make install-smoke` or `./packaging/homebrew/install_to_brew_prefix.sh`
- Broker under prefix: `make install-broker PREFIX=…` then dry-run via `./packaging/homebrew/uma-services.sh plist` (or `install_launchagent.sh --prefix=… --dry-run`); bootstrap replaces the machine agent
- Darwin dist client (metal sibling): from zerollama `./scripts/build/install_uma_into_dist.sh` or toolkit `make dist-client` → `dist/darwin-arm64/{bin,include,lib}`
- Protocol CI (no Metal): toolkit `make uma-mock-smoke` / `pip install -e python/` against `docker/uma-mock`
- Client tarball: `make client-tarball` → `dist/uma-client-*.tar.gz` (Homebrew stub: `packaging/homebrew/uma.rb`)

**M23 vendor e2e (Darwin lab):** `./scripts/phase/m23_vendor_ane_uma_signoff.sh` — vendor pin + **0095**, ANE `step_once` queues under competitor `HOLD_ANE` (`wait_ms≈3s`), then `HOLD_GPU ∥ HOLD_ANE`. Never `:11434`/`:8081`.
## Defaults

- **Runtime:** `ZEROLLAMA_UMA_SCHED` unset → **`auto`** (gate if broker up; else ungated)
- **Build:** `BUILD_UMA=auto` on Mac scripts when sibling `bmtl/.../uma_toolkit` exists

## Disable

| Knob | Effect |
|------|--------|
| `ZEROLLAMA_UMA_SCHED=off` (`0` / `false` / `disabled` / `none`) | No connect, no HOLD — **no rebuild** |
| `BUILD_UMA=0` | Compile out client (`-tags uma` / `libuma_llama.a`) |

Details: [mlx-uma-sched.md § Disabling](./mlx-uma-sched.md#disabling-uma-build--runtime).

## Lab ports

Never use production **`:11434`** / **`:8081`**. Sign-offs use `:11435` (Go) and `:18082` (llama-server).

## Docs

- [mlx-uma-sched.md](./mlx-uma-sched.md) · [ggml-uma-sched.md](./ggml-uma-sched.md) · [llama-server-uma-sched.md](./llama-server-uma-sched.md)
- Wishlist: `bmtl/.../uma_toolkit/WISHLIST_MLX.md`, `WISHLIST_GGML.md`
