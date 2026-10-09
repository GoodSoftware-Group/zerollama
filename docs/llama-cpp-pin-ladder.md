# llama.cpp pin-debt ladder (L6 enabler)

**Status (Oct 2026):** **rung 5 (tip) landed** — pin **b11351** + **137** patches
(CUDA `llama-server` + Clef on tip `common_batch` API; **0136** ANE shim, **0137** Laya).
Rollback: `vendor/llama-cpp-b11232/` + `LLAMA_CPP_*.prev`.

| Field | Value |
|-------|--------|
| **Current pin** | **b11351** (`631109b34da437a3c4a5ebd75091d677671392e3`) |
| **Previous (rollback)** | **b11232** (`6f767fe9…`) — keep `vendor/llama-cpp-b11232/` |
| **Ollama tip pin** | **b11351** (matched) |
| **Clef runtime floor** | **b11232+** — tip wire uses `common_batch` / `batch.view` |
| **Patch series** | `llama/patches/` through **0137** (137 commits on b11351) |
| **Pin truth** | [runtime/LLAMA_CPP_PIN.md](../runtime/LLAMA_CPP_PIN.md) · `LLAMA_CPP_VERSION` / `LLAMA_CPP_COMMIT` · `Makefile.sync` |
| **CUDA toolkit (5080 CT)** | Prefer **CUDA 12.8** (`CUDA_HOME=/usr/local/cuda-12.8`) — CUDA 13.3 `libcudart` aborts in `cudaGetDeviceCount` on this host |

**WHY this ladder exists:** L2 deferred Clef/Strands **runtime** until the pin
moves ([system-one-score.md](./system-one-score.md)). L6 is a parallel enabler —
not a gate for L1 DX or L3 MLX.

---

## Intermediate targets (Ollama tip trail)

Rungs match Ollama’s own `LLAMA_CPP_VERSION` bumps toward tip (not every
ggml-org tag). Prefer **one rung per rebase session** so `llama/patches/`
conflicts stay reviewable.

| Rung | Tag | Commit (ggml-org) | Why stop here | Patch / product risk |
|------|-----|-------------------|---------------|----------------------|
| **0** | **b10615** | `f280b269…` | Rollback baseline | Kept as `vendor/llama-cpp-b10615/` |
| **1** | **b10864** | `5d806aa2…` | First Ollama pin after our base (`#18317`) | **Done** — kept as `vendor/llama-cpp-b10864/` rollback |
| **2** | **b10969** | `391fac16…` | Mid Sep tip (`#18446`) | **Done** — kept as `vendor/llama-cpp-b10969/` rollback |
| **3** | **b11081** | `161755f2…` | Prior “skip until rebase” gate in [upstream-ollama-diff](./upstream-ollama-diff.md) | **Done** — kept as `vendor/llama-cpp-b11081/` rollback |
| **4** | **b11232** | `6f767fe9…` | Clef floor (`#18652` / Clef `#18741`) | **Done** — kept as `vendor/llama-cpp-b11232/` rollback |
| **5 (ship / tip)** | **b11351** | `631109b3…` | Match Ollama **v0.40.x** tip (`#18761`) | **Done (Oct 2026)** — 137 patches; tip `common_batch` + Clef; lab chat smoke OK (CPU when VRAM held) |

**Do not** jump 0 → 5 in one shot unless you have a clean worktree day and a
rollback plan (`vendor/llama-cpp-b10615/` + `llama/patches/` backup).

---

## Clef / `score_fields` dependency

| Layer | On b11351 (tip) | Notes |
|-------|-----------------|-------|
| Go compile (`decision/clef.go`) | **Landed** (L2) | — |
| Create / import (`x/create/clef.go`) | **Landed** (L2) | — |
| Score client shape (`score_fields` JSON) | **Landed** (Go → llama-server) | — |
| Joint head sources (`llama/clef/*.cpp`) | **Linked** into `llama-server` | — |
| Server wire (`0135` + `llama/compat/002-clef.patch`) | **In tree** | `already_applied` on vendor; tip uses `common_batch` / `batch.view` |
| Tip shims | **0136** ANE `sync_target_cross`; **0137** Laya `llama_process` | Required after tip `common_batch` API |
| CMake link (`ZEROLLAMA_CLEF_DIR`) | **Landed** | `build_llama_server.sh` links `llama/clef/clef.cpp` into `server-context` |
| Live Clef GGUF score | Needs Clef GGUF + `--embeddings` | Head symbols present in `libllama-server-impl.so` |

**WHY b11232+:** Ollama’s first Clef commit (`2d24aa78`) already had
`LLAMA_CPP_VERSION=b11232`. The patch assumes tip-shaped
`tools/server/server-context.cpp` (embedding task fields, decision context type).
Forcing that patch onto b10615 via `llama/compat/` would break
`apply-patch.cmake` (GLOB_RECURSE `*.patch`).

Staging lives under **`llama/clef/`** (not `llama/compat/`) — see
[llama/clef/README.md](../llama/clef/README.md).

---

## Operator runbook (one rung)

### Preconditions

- No production serve from the agent (lab ports only; PVE → CT 1564).
- Clean or stashed tree; do **not** delete `llama/patches/`.
- GPU host (or Mac) free for a full `llama-server` + CGO rebuild.

### Steps

```bash
# 0) Status (read-only)
./scripts/phase/l6_pin_ladder_status.sh
./scripts/phase/l6_clef_compile_check.sh   # head-only; OK on b10615

# 1) Pick next rung (example: b10864)
NEXT=b10864
NEXT_REF=$(git ls-remote https://github.com/ggml-org/llama.cpp.git "refs/tags/${NEXT}" | awk '{print $1}')

# 2) Snapshot current pin identity
cp LLAMA_CPP_VERSION LLAMA_CPP_VERSION.prev
cp LLAMA_CPP_COMMIT LLAMA_CPP_COMMIT.prev

# 3) Point Makefile.sync + pin files at the rung
#    FETCH_HEAD / WORKDIR / FETCH_REF / BUILD_NUMBER / LLAMA_CPP_*
#    (edit deliberately; WORKDIR=vendor/llama-cpp-${NEXT})

# 4) Materialize vendor + re-apply patches (expect conflicts)
make -f Makefile.sync clean apply-patches
# Fix conflicts in vendor/llama-cpp-${NEXT}, commit per patch, then:
make -f Makefile.sync format-patches
./scripts/vendor/sync_vendor_llama.sh

# 5) Rebuild (lab — not production :11434 / CT serve unless operator owns it)
./scripts/build/build_llama_server.sh
# Darwin CGO: ./scripts/build/build_zerollama_mac.sh && ./zerollama doctor

# 6) Gates (pick host)
./scripts/phase/phase15_llama_kv_ext_pin_check.sh
./scripts/phase/l2_full_gate.sh            # Metal
# ./scripts/phase/l2_cuda_full_gate.sh     # CUDA
./scripts/vendor/llama_patch_doctor.sh

# 7) Only at b11232+: wire Clef
./scripts/phase/stage_clef_for_pin.sh --wire
# refresh llama/compat/002-clef.patch hunks if apply fails
```

### Rollback

```bash
mv LLAMA_CPP_VERSION.prev LLAMA_CPP_VERSION
mv LLAMA_CPP_COMMIT.prev LLAMA_CPP_COMMIT
# restore Makefile.sync FETCH_* to b10615 values from git
make -f Makefile.sync clean apply-patches
./scripts/vendor/sync_vendor_llama.sh
```

---

## Patch rebase risk map (high-touch series)

| Series | Patches | Risk on bump |
|--------|---------|--------------|
| Compat hooks / kv-ext / seq-copy | 0001, 0014–0022, Phase 15 | API renames in `llama.h` / ggml-backend |
| Eliza QJL/Polar/TBQ + Metal | 0026–0030, 0067–0070, 0096–0098 | Kernel / SET_ROWS drift |
| Pretok / BPE / unicode | 0106–0126 | `src/unicode.*` churn on tip |
| Laya decisions | 0124–0125 | Collides with Clef decision context types — rebase carefully at tip |
| MultiDecode | 0126–0128 | `server-context` / batch parent masks — high conflict vs Clef server hunks |
| DCA / Kokoro / Bee | 0087–0102 | Optional; fix after core apply |

Conflict budget rule of thumb: if >~15 patches need hand merges in one rung,
stop, format-patch, smoke, then continue — do not stack unresolved `git am` gaps.

---

## Scripts (enabler scaffolding)

| Script | Purpose |
|--------|---------|
| `./scripts/phase/l6_pin_ladder_status.sh` | Current pin vs ladder; Clef apply check; deferred staging presence |
| `./scripts/phase/l6_clef_compile_check.sh` | Object-compile `llama/clef/clef.cpp` against vendor includes |
| `./scripts/phase/l6_clef_live_smoke.sh` | Tip llama-server synth score (`:18086`) |
| `./scripts/phase/l6_clef_decisions_e2e.sh` | Lab Go `:11435` + tip `LLAMA_SERVER_BIN` → `POST /v1/decisions` (synth or `CLEF_E2E_PRODUCT_GGUF`) |
| `./scripts/phase/l6_clef_product_convert.sh` | Cloudflare HF → Ollama-wire Q8_0 + mmproj F16 (`CLEF_CONVERT_MMPROJ=1`) |
| `./scripts/phase/l6_clef_vl_decisions_smoke.sh` | Image `/v1/decisions` against a VL Clef tag (host already serving) |
| `./scripts/phase/l6_strands_graft_head.py` | Graft Strands pointer head + decision meta onto Qwen3.5 GGUF |
| `./scripts/phase/l6_strands_decisions_e2e.sh` | Lab Go + tip `LLAMA_SERVER_BIN` → Strands `/v1/decisions` |
| `./scripts/phase/l6_promote_tip_env.sh` | Prep tip/rollback `LLAMA_SERVER_BIN` env (no serve restart) |
| `./scripts/phase/stage_clef_for_pin.sh` | Refresh staging; `--wire` only when pin ≥ b11232 |

---

## Remaining gaps after tip

1. **Production Clef GGUF** — **done:** Cloudflare HF → Ollama-wire via `l6_clef_product_convert.sh` (text Q8_0 + mmproj F16). Tags: `clef-flash` (text), `clef-flash-vl` (text+mmproj). Note: `ggml-org/Clef-Flash-GGUF` (native `clef` + `decision.*`) is **not** interchangeable with `llama/clef/clef.cpp`.
2. **Strands PointerRows** on CUDA llama-server — **done (lab):** tip wire **0138** + `llama/strands/` pointer head; graft `l6_strands_graft_head.py` → product Q8; e2e `l6_strands_decisions_e2e.sh` PASS. MLX path unchanged.
3. **Operator rebuild:** on this 5080 CT use `CUDA_HOME=/usr/local/cuda-12.8` (default CUDA 13.3 aborts at device init). Tip Go binary for lab: rebuild after tiled/parsers CGO packages land (`/tmp/zerollama-lab`).
4. Rollback stays at `vendor/llama-cpp-b11232/` + `LLAMA_CPP_*.prev` if tip misbehaves in production.
5. **Production CT serve (cudallama)** — **promoted to tip b11351** (Oct 2026): `run/zerollama-lab` + `LLAMA_SERVER_BIN=…/llama-cpp-b11351`; `clef-flash` + `clef-flash-vl` live; text + image `/v1/decisions` smoke PASS. Rollback:
   ```bash
   ./scripts/phase/l6_promote_tip_env.sh --rollback --write
   # restart ~/bin/serve.sh with that env
   ```

### Lab: synthetic Clef score smoke

```bash
CUDA_HOME=/usr/local/cuda-12.8 CMAKE_CUDA_ARCHITECTURES=120-real \
  ./scripts/build/build_llama_server.sh
./scripts/phase/l6_clef_live_smoke.sh   # :18086 — curl + Go segment tokenize/score
# Product path (lab ports only; hides CUDA when production holds VRAM):
hf download Cloudflare/clef-flash --local-dir /root/models/clef-flash-hf
./scripts/phase/l6_clef_product_convert.sh
# also writes mmproj-clef-flash-f16.gguf (CLEF_CONVERT_MMPROJ=1)
CLEF_E2E_ZEROLLAMA=/tmp/zerollama-lab \
  CLEF_E2E_PRODUCT_GGUF=/root/models/clef-flash-gguf/clef-flash-ollama-q8_0.gguf \
  CLEF_E2E_MODEL=clef-flash-lab \
  ./scripts/phase/l6_clef_decisions_e2e.sh
# VL create in lab e2e (slow on CPU with product Q8):
CLEF_E2E_ZEROLLAMA=/tmp/zerollama-lab \
  CLEF_E2E_PRODUCT_GGUF=/root/models/clef-flash-gguf/clef-flash-ollama-q8_0.gguf \
  CLEF_E2E_MMPROJ=/root/models/clef-flash-gguf/mmproj-clef-flash-f16.gguf \
  CLEF_E2E_MODEL=clef-flash-vl-lab \
  ./scripts/phase/l6_clef_decisions_e2e.sh
# Or against already-serving host:
OLLAMA_HOST=127.0.0.1:8080 CLEF_VL_MODEL=clef-flash-vl \
  ./scripts/phase/l6_clef_vl_decisions_smoke.sh
```

Requires `llama/compat` skip of `clef.*` in `translate_metadata` (upstream parity) so the backbone loader ignores the joint head tensors.

For local `go test ./decision` / CGO after a clean tree: `./scripts/phase/ensure_ggml_version_h.sh` (writes `ml/backend/ggml/ggml/src/ggml-version.h`, gitignored).
