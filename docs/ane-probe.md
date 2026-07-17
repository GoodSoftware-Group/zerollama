# ANE probe (maderix/ane)

**Audience:** Mac operators evaluating **Apple Neural Engine (ANE)** access for future hybrid inference — separate from the ggml Metal hot path and from Flash-MoE.

**Related:** [flash-moe.md](./flash-moe.md) (RAM-busting MoE via llama-server), [apple-silicon-metal.md](./apple-silicon-metal.md), [phase17-llama-server.md](./phase17-llama-server.md), [ane-draft-inprocess.md](./ane-draft-inprocess.md) (B1–B6 in-process dflash hook — **why not this probe binary:** IOSurface handoff requires same PID as llama-server).

---

## Why this exists

Apple ships a dedicated ML accelerator (ANE) on every Apple Silicon Mac. For third-party apps, ANE is normally reachable only through **Core ML**, which adds compile latency and hides low-level scheduling. Research from [@maderix](https://github.com/maderix/ANE) reverse-engineers private `_ANEClient` / `_ANECompiler` APIs and exposes them through **`libane_bridge.dylib`** — a C-callable bridge with mega-kernel fusion to cut XPC dispatch overhead.

**Why zerollama cares:** ANE could eventually accelerate **small, fused subgraphs** (embeddings, vision front-ends, speculative draft heads) while ggml Metal keeps the main decode loop. That hybrid is not wired into inference today; we need a **safe smoke test** before committing to unstable private APIs in the main binary.

**Why a subprocess probe, not CGO in `zerollama`:**

| Approach | Why we did / didn't |
|----------|---------------------|
| CGO link `libane_bridge` into Go | Private APIs break on macOS updates; would force `-tags darwin` + ObjC in every build |
| **Subprocess `ane-probe`** | Isolates crash risk; doctor can warn without blocking serve; operators opt in |
| Core ML only | Higher compile/dispatch cost; maderix track targets direct ANE for research parity |

**Not on the inference hot path.** Successful probe ≠ faster chat. It means ANE is reachable on *this* machine *today*.

---

## Architecture

```text
zerollama doctor / zerollama ane-probe (hidden)
        │
        ▼
discover/ProbeANE()  ──exec──►  build/ane-probe-darwin/bin/ane-probe
        │                              │
        │                              ▼
        │                    libane_bridge.dylib  (from maderix/ane)
        │                              │
        │                              ▼
        │                    compile tiny 1×1 conv MIL → ANE eval
        │
        └── JSON stdout: { ok, eval_ms, compile_count, source, error? }
```

**Why `cmd.Dir = filepath.Dir(bin)`:** macOS `dyld` resolves `@loader_path/libane_bridge.dylib` relative to the probe executable. Running from the wrong cwd breaks load even when the dylib is co-located.

**Why `install_name_tool` in the Makefile:** the linker records an absolute path to the bridge at build time; rewriting to `@loader_path` lets the probe ship beside the dylib in `build/ane-probe-darwin/bin/`.

---

## Quick start

```bash
git clone https://github.com/maderix/ane ~/Sites/inference/ane
./scripts/ane/ane_probe_build.sh
./build/ane-probe-darwin/bin/ane-probe          # smoke JSON
./zerollama ane-probe                           # hidden subcommand; same JSON
./zerollama ane-bench                           # peak conv-stack TFLOPS proxy
./zerollama ane-bench --quick                   # shorter depth for CI
./zerollama ane-draft-bench                     # draft-step latency proxy
./zerollama doctor                              # includes ANE check (warn if missing)
./scripts/ane/ane_probe_smoke.sh                    # build + run end-to-end
./scripts/ane/ane_prefill_smoke.sh                  # prefill ANE vs Metal compare + sweep
```

**Bench tools (M17 follow-on):**

| Binary | Purpose |
|--------|---------|
| `ane-probe` | 1×1 conv smoke — ANE reachable |
| `ane-matmul-bench` | Stacked conv peak throughput (512ch × depth 32 default) |
| `ane-draft-bench` | Single conv at draft-like dims — baseline for DFlash-on-ANE research |
| `ane-iosurface-smoke` | IOSurface write/eval/read latency (CPU memcpy producer) |
| `ane-prefill-bench` | Dynamic matmul at IC×OC×SEQ — prefill geometry proxy (`mil_dynamic.h`) |
| `metal-prefill-bench` | Naive Metal compute matmul at IC×OC×SEQ (compare leg) |
| `metal-prefill-mps-bench` | MPS `MPSMatrixMultiplication` matmul — tuned Metal baseline (compare leg) |
| `ane-prefill-handoff-smoke` | Metal activation fill → IOSurface → ANE prefill matmul (ggml hook prototype) |
| `ane-metal-handoff-smoke` | Metal `newBufferWithBytesNoCopy` fill → ANE eval (draft conv handoff) |
| `ane-draft-daemon` | Persistent draft kernel — compile once, JSON eval/bench protocol |
| `ane-ggml-map-smoke` | Parent-side `ggml_metal_buffer_map` equivalent on IOSurface base |

```bash
./scripts/ane/ane_bridge_patch.sh                 # IOSurface export API on maderix/ane bridge (once)
./zerollama ane-handoff-smoke                 # CPU producer IOSurface path
./zerollama ane-handoff-smoke --metal         # Metal compute fill on shared IOSurface
./zerollama ane-handoff-smoke --suite         # probe + draft + iosurface + metal
./zerollama ane-draft-resolve             # list eliza draft-eagle3 / -dflash tags
./zerollama ane-draft-inspect --model eliza-1-2b-dflash
./zerollama ane-draft-smoke --model eliza-1-2b-dflash   # bench at GGUF proxy dims (256ch × 16)
./zerollama ane-hybrid-smoke --model eliza-1-2b-dflash  # Metal handoff + draft conv at proxy dims
./zerollama ane-prefill-bench --model eliza-1-2b --tokens 512   # 2048×2048×512 matmul proxy
./zerollama ane-prefill-bench --compare-metal --ic 256 --oc 256 --seq 512
./zerollama ane-prefill-bench --compare-metal --model eliza-1-2b --tokens 512
./zerollama ane-prefill-sweep --ic 256 --oc 256 --quick
./zerollama ane-model-resolve                              # all local GGUF tags + embedding_length
./zerollama ane-model-resolve --model qwen
./zerollama ane-prefill-sweep --model eliza-1-2b --quick
./zerollama ane-prefill-sweep --model qwen3.6 --quick      # any pulled GGUF tag
./zerollama ane-prefill-crossover --quick                  # ANE vs MPS width crossover @ SEQ=512
./zerollama ane-draft-surface-smoke --model qwen3.6 --quick   # surface_id + draft conv handoff
./zerollama ane-draft-daemon-smoke --model eliza-1-2b --quick  # compile-once daemon session
./zerollama ane-ggml-map-smoke --model eliza-1-2b --quick     # ggml map parent + daemon eval
./zerollama ane-ggml-hook-status                             # in-tree ggml IOSurface API readiness
./zerollama ane-draft-mil-status --model eliza-1-2b-dflash   # Eagle3 sidecar / MIL blockers
./zerollama ane-draft-mil-map --model eliza-1-2b-dflash      # tensor → MIL slot plan
./zerollama ane-draft-mil-extract --model eliza-1-2b-dflash --out /tmp/ane-draft-weight.bin
ZEROLLAMA_ANE_DRAFT=1 ./zerollama ane-draft-router-smoke --model eliza-1-2b-dflash --quick  # auto sidecar weight cache
./zerollama ane-inprocess-smoke --model eliza-1-27b-256k-dflash --quick  # same-PID ggml map + ANE eval (B1)
./zerollama ane-hybrid-smoke --model eliza-1-2b --quick     # any GGUF tag (not only -dflash)
./scripts/ane/ane_crossover_report.sh                           # ANE vs MPS crossover table
./zerollama ane-prefill-handoff-smoke --model eliza-1-2b --tokens 128 --quick
```

See [ane-hybrid-path.md](./ane-hybrid-path.md) for Metal+ANE integration plan.

`ZEROLLAMA_ANE_DRAFT=1` — reserved for future scheduler hook (default off).

`--model` prefill probes cap IC/OC at **2048** unless **`--full-embed`** is set (use for 27B / 5120-wide models).

**Bench hygiene:** MPS and naive Metal legs run on the **GPU**. If production serve or other Metal work is active, use **`--ane-only`** on sweep/crossover/lab-status (ANE engine only), or wait for an idle GPU before `--compare-metal` / crossover without `--ane-only`.

`ane-prefill-bench` reports `tflops = gflop / eval_ms`; treat as **relative** until compared against Metal at the same IC×OC×SEQ (microkernel dispatch dominates at small GFLOP counts).

`ane-matmul-bench` reports `tflops = gflop / eval_ms`; treat as a **relative** signal until calibrated against maderix `inmem_peak.m` on your chip.

---

## Env reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANE_REPO` | `~/Sites/inference/ane` | maderix/ANE checkout (`bridge/libane_bridge.dylib`) |
| `ZEROLLAMA_ANE_PROBE` | — | Override path to built `ane-probe` binary |
| `ZEROLLAMA_ANE_DRAFT` | `0` | Reserved: route speculative draft to ANE when scheduler wiring lands (lab subprocesses today) |

Both appear in `zerollama envconfig` via `envconfig.ANERepo()`.

---

## Doctor check

`zerollama doctor` runs the ANE check on **darwin only**:

1. Bridge dylib present at `$ANE_REPO/bridge/libane_bridge.dylib`
2. `ane-probe` built (`./scripts/ane/ane_probe_build.sh`)
3. Live probe succeeds (compile + eval)

**Why warn, not fail:** ANE is experimental. Missing bridge must not block Metal serve or Flash-MoE setup.

---

## Risks and non-goals

- **Private APIs** — Apple can change or block `_ANEClient` on any macOS update; treat as lab tooling.
- **No training path** — maderix also demos ANE training; zerollama does not expose it.
- **No ggml/llama-server integration** — future work (ROADMAP **M17** follow-on); see [ane-hybrid-path.md](./ane-hybrid-path.md).

---

## See also

- [Flash-MoE (anemll)](./flash-moe.md) — different track (@anemll): MoE models larger than RAM via slot-bank streaming
- [ROADMAP — M17 ANE probe](./ROADMAP.md#apple-silicon--metal-track)
