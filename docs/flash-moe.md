# Flash-MoE (anemll) on zerollama

**Audience:** Mac operators running **MoE models larger than unified RAM** via [Anemll/anemll-flash-llama.cpp](https://github.com/Anemll/anemll-flash-llama.cpp) slot-bank + sidecar streaming.

**Related:** [phase17-llama-server.md](./phase17-llama-server.md), [apple-silicon-metal.md](./apple-silicon-metal.md), [ane-probe.md](./ane-probe.md).

---

## Why this exists

Stock llama.cpp loads **all** routed experts into the address space. Models like **Qwen3.5-397B-A17B** (~200GB+ experts alone) cannot load on a 48–128GB Mac even with aggressive quantization. [@anemll](https://github.com/Anemll) **Flash-MoE** splits weights:

| Weight class | Storage | Runtime |
|--------------|---------|---------|
| Dense (attn, norms, embed, shared experts) | GGUF mmap | GPU/CPU via `-ngl` |
| Routed experts (`ffn_*_exps`) | **Sidecar** dir on SSD | **Slot-bank** streamed per token |

**Why zerollama integrates via Phase 17, not ggml Metal default:**

- Flash-MoE ships in a **forked `llama-server`** with `--moe-*` flags and `-fit` VRAM budgeting — same boundary upstream Ollama uses for GGUF.
- Mac **ggml Metal** remains ~**+7% faster** for in-RAM models (M7 bench); forcing Flash-MoE into ggml would delay the anemll merge and duplicate slot-bank logic.
- Go already spawns `llama-server` when `ZEROLLAMA_LLAMA_SERVER=1`; passthrough is the smallest correct seam.

**Why a separate binary (`build/flash-moe-llama-server-darwin/`):** Flash-MoE patches are not in zerollama's vendored llama.cpp pin yet. A dedicated build keeps the main pin stable while operators experiment. ROADMAP tracks cherry-pick / upstream merge separately from L2 fork work.

---

## Architecture

```text
zerollama serve --llama-server-backend
        │
        ▼
llm/startLlamaServer()
        │  appendFlashMoEArgs() when sidecar configured
        │    --moe-mode, --moe-sidecar, --moe-slot-bank, …
        │    -fit on          (clamp dense GPU offload vs slot budget)
        │    -ub 1            (MoE prefill correctness on GPU)
        ▼
build/flash-moe-llama-server-darwin/bin/llama-server
        │
        ├── mmap dense weights from GGUF
        └── stream routed experts from sidecar SSD → GPU slot-bank
```

**Why activation requires a sidecar path:** `ZEROLLAMA_FLASH_MOE=1` alone selects the Flash-MoE binary but does not pass `--moe-*` until `ZEROLLAMA_FLASH_MOE_SIDECAR` or manifest `moe_sidecar` is set. **Why:** avoid forcing `-ub 1` / `-fit on` on non-MoE models when an operator only wanted binary discovery.

**Why manifest `moe_prefetch_temporal: false` does not enable Flash-MoE:** prefetch is a tuning knob on an already-configured sidecar run, not a standalone switch.

---

## Prerequisites

1. Clone anemll fork (Server-Flash-Moe branch):

```bash
git clone --branch Server-Flash-Moe --depth 1 \
  https://github.com/Anemll/anemll-flash-llama.cpp.git \
  ~/Sites/inference/anemll-flash-llama.cpp
```

2. Build Flash-MoE `llama-server`:

```bash
./scripts/build/build_flash_moe_llama_server.sh
# → build/flash-moe-llama-server-darwin/bin/llama-server
```

3. Extract a sidecar (example: Qwen3.5-35B-A3B):

```bash
./scripts/gpu/flash_moe_extract_sidecar.sh \
  --model ~/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --out-dir ~/Models/flash/qwen35 --force --verify
```

Or manually:
FLASH_REPO=~/Sites/inference/anemll-flash-llama.cpp
python3 "${FLASH_REPO}/tools/flashmoe-sidecar/flashmoe_sidecar.py" extract \
  --model ~/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --out-dir ~/Models/flash/qwen35 --force
```

See upstream [tools/flashmoe-sidecar/README.md](https://github.com/Anemll/anemll-flash-llama.cpp/blob/Server-Flash-Moe/tools/flashmoe-sidecar/README.md).

---

## Enable on serve

```bash
export ZEROLLAMA_FLASH_MOE=1
export ZEROLLAMA_FLASH_MOE_SIDECAR=~/Models/flash/qwen35
export ZEROLLAMA_FLASH_MOE_SLOT_BANK=64    # tune to RAM (16 on 8GB, 128 on 128GB)
export ZEROLLAMA_FLASH_MOE_TOPK=4
export ZEROLLAMA_FLASH_MOE_PREFETCH=1
export ZEROLLAMA_LLAMA_SERVER=1

./zerollama serve --llama-server-backend
```

**Doctor:**

```bash
./zerollama doctor   # reports repo/binary/sidecar even when Flash-MoE is not enabled
```

When disabled, doctor still shows **binary ready @ …** if the fork is built — **why:** operators should not enable env vars blindly without knowing the toolchain is present.

---

## Per-model manifest (Modelfile / create)

Instead of global env, set runner options on the model:

```json
{
  "options": {
    "moe_mode": "slot-bank",
    "moe_sidecar": "/Users/you/Models/flash/qwen35",
    "moe_slot_bank": 64,
    "moe_topk": 4,
    "moe_prefetch_temporal": true,
    "num_gpu": 99
  }
}
```

Zerollama automatically applies **`-ub 1`** and **`-fit on`** when a sidecar is configured — **why:** anemll documents these as required for correct MoE prefill and dense/slot VRAM balance; hiding them reduces misconfiguration.

---

## Env reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `ZEROLLAMA_FLASH_MOE` | off | Prefer flash-moe `llama-server` binary + pass moe flags when sidecar set |
| `ZEROLLAMA_FLASH_MOE_SIDECAR` | — | Sidecar manifest directory |
| `ZEROLLAMA_FLASH_MOE_MODE` | `slot-bank` | `stock`, `slot-bank`, `oracle-*`, … |
| `ZEROLLAMA_FLASH_MOE_SLOT_BANK` | omit | Resident slots per layer |
| `ZEROLLAMA_FLASH_MOE_TOPK` | omit | Routed K override |
| `ZEROLLAMA_FLASH_MOE_PREFETCH` | off | `--moe-prefetch-temporal` |
| `ZEROLLAMA_FLASH_MOE_AUTO_EXTRACT` | off | Auto-extract sidecar for MoE tags on `pull`/`create` (see below) |
| `ZEROLLAMA_FLASH_MOE_LLAMA_SERVER_BIN` | — | Override Flash-MoE binary path |
| `FLASH_MOE_REPO` | `~/Sites/inference/anemll-flash-llama.cpp` | Build script source tree |
| `LLAMA_SERVER_BIN` | — | Overrides **all** llama-server discovery when set |

---

## Auto-extract sidecar on `pull` (opt-in)

```bash
export ZEROLLAMA_FLASH_MOE_AUTO_EXTRACT=1
./zerollama pull qwen35:latest
```

**Why opt-in, not default:** extraction reads the full routed-expert GGUF
payload and can take minutes on 100GB+ MoE models. `pull` must not silently
balloon in time/disk for operators who never asked for slot-bank streaming.

**What happens:** after the manifest lands, zerollama checks whether the GGUF
looks like a MoE model (`expert_count` / arch / family) and whether a sidecar
already exists at the default `~/Models/flash/<tag>` path. If missing, it
shells out to `flashmoe_sidecar.py extract` (same tool as
`flash_moe_extract_sidecar.sh`) and, on success, writes `moe_sidecar` into the
manifest's params layer — so a later `serve` or Modelfile `moe_sidecar` value
is not required; `zerollama flash-moe-resolve` and `doctor` will report
`sidecar_ready` immediately. Extraction failures are logged only; `pull`
still succeeds either way.

Requires the anemll fork checked out at `FLASH_MOE_REPO` (default
`~/Sites/inference/anemll-flash-llama.cpp`) — same prerequisite as manual
extraction above.

---

## Slot-bank sizing (from anemll)

| Machine RAM | `--moe-slot-bank` | Notes |
|-------------|-------------------|-------|
| 8 GB | 16 | `-ngl 0` if dense does not fit GPU |
| 16 GB | 32 | |
| 36 GB | 64 | 35B may fit stock if entire GGUF in RAM |
| 128 GB | 128 | 397B requires slot-bank |

Rule of thumb: **5–15% of RAM** for slot bank.

---

## Design notes (implementation)

| Decision | Why |
|----------|-----|
| `internal/reporoots` for binary discovery | Same root-walk logic for ANE probe and Flash-MoE artifacts; avoids drift between packages |
| `FindFlashMoELlamaServer` filters `llamaCppBinaryCandidates` | Reuses Phase 17 layout (`build/flash-moe-llama-server-*`) instead of a hardcoded darwin path |
| `setLlamaServerUbatch` rewrites existing `-ub` | `appendBatchArgs` may set `-ub 64` earlier; Flash-MoE must win without duplicate flags |
| Separate fork binary vs vendor pin | Ship operator value now; merge Flash-MoE patches into `vendor/llama-cpp-*` when anemll upstream stabilizes |

---

## Smoke testing

**Why tiered:** MoE sidecars and 100GB+ GGUFs are operator-local. Tier 0 validates wiring in CI; tiers 1–2 need your model on disk.

| Tier | Env | Proves |
|------|-----|--------|
| **0** (default) | — | Go flag tests + flash-moe `llama-server` exposes `--moe-sidecar` |
| **1** | `RUN_E2E_FLASH_MOE_STARTUP=1` | Direct llama-server loads GGUF + sidecar (slot-bank reserve) |
| **2** | `RUN_E2E_FLASH_MOE=1` | Full `zerollama serve --llama-server-backend` → `/api/generate` |

```bash
# Tier 0 (CI-friendly on Mac)
./scripts/phase/flash_moe_smoke.sh

# Tier 1 — startup only (no generation)
RUN_E2E_FLASH_MOE_STARTUP=1 \
  FLASH_MOE_GGUF=~/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  FLASH_MOE_SIDECAR=~/Models/flash/qwen35 \
  ./scripts/phase/flash_moe_smoke.sh

# Auto-extract sidecar on first run
FLASH_MOE_EXTRACT=1 RUN_E2E_FLASH_MOE_STARTUP=1 FLASH_MOE_GGUF=... ./scripts/phase/flash_moe_smoke.sh

# Tier 2 — full E2E (needs pulled tag or FLASH_MOE_MODEL)
RUN_E2E_FLASH_MOE=1 \
  FLASH_MOE_GGUF=... FLASH_MOE_SIDECAR=... FLASH_MOE_MODEL=qwen35:latest \
  ./scripts/phase/flash_moe_smoke.sh
```

Disk-starved dev hosts: `FLASH_MOE_SKIP_GO_TEST=1 ./scripts/phase/flash_moe_smoke.sh`

**Auto-resolve from zerollama store:** tier 1/2 call `./zerollama flash-moe-resolve` when `FLASH_MOE_GGUF` / `FLASH_MOE_SIDECAR` are unset — scans pulled MoE tags under `~/.ollama/models`, reads manifest `moe_sidecar`, and checks `~/Models/flash/<tag>`.

```bash
./zerollama flash-moe-resolve --list          # all local MoE tags
./zerollama flash-moe-resolve --json          # best pick (sidecar-ready first, else smallest)
./zerollama flash-moe-resolve --model qwen35  # prefer a tag
```

See also [testing-smoke.md](./testing-smoke.md).

---

## Non-goals (today)

- Flash-MoE inside **ggml Metal** default path (llama-server only)
- CUDA Flash-MoE build script (upstream fork supports it; zerollama script is Darwin-only for now)
- Merging Flash-MoE patches into the main vendor pin (`vendor/llama-cpp-*`) — anemll's fork is not yet stable enough upstream; tracked separately from L2 fork-KV merge work

---

## See also

- [ANE probe](./ane-probe.md) — optional maderix/ANE bridge smoke (experimental, separate from Flash-MoE)
- [ROADMAP — M16 Flash-MoE](./ROADMAP.md#apple-silicon--metal-track)
