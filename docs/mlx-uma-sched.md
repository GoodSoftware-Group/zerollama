# MLX compute gated through UMA broker

**Status:** M20 production path (Darwin). **Doc:** [ROADMAP.md](./ROADMAP.md) Apple Silicon **M20**.

**Admission only:** UMA decides *when* mlxrunner may use the GPU (`HOLD_GPU` / `RELEASE`). MLX still runs Metal kernels (`mlx_eval`); math ops do **not** execute inside `uma_sched`.

mlxrunner admits MLX materialization through the **one** machine-wide `uma_daemon` — never a private `uma_sched`.

```text
mlxrunner
  LeaseBegin(load|prefill|decode) → SUBMIT HOLD_GPU → phase=holding
  mlx.Eval / AsyncEval… under lease
  mlx.Synchronize (still leased) → drain Metal
  LeaseEnd → RELEASE → WAIT
```

**Multi-unit (F0390):** `x/uma` also exposes `LeaseBeginUnit("ane"|"amx", …)` / `RunUnit` for peer ANE/AMX admission (`HOLD_ANE` / `HOLD_AMX`). Tickets are independent of GPU so `HOLD_GPU ∥ HOLD_ANE` works. Metal mlxrunner path stays on `HOLD_GPU`. Lab: `./scripts/phase/m23_uma_multiunit_client_smoke.sh`.

**GRAPH (F0624 / wishlist 0.4):** `FormatGraph` / `FormatGraphEx` / `Submit` / `Wait` / `Graph` — broker GRAPH via libuma_client (no `uma_graph.h`). Lab: `./scripts/phase/m24_uma_graph_client_smoke.sh`.

**BUF (F0627):** `BufAlloc` / `BufPut` / `BufGet` / `BufFree` / `BufExport` / `BufReclaim` for staging named buffers before GRAPH. Lab: `./scripts/phase/m27_uma_buf_graph_smoke.sh`.

**Live OptiQ chain (F0633):** C dump of converted ornith packs → Go `Buf*` + `GEMV_Q4_G64` Wz→Wo. Lab: `./scripts/phase/m28_uma_optiq_live_chain_smoke.sh` (in `mac_uma_signoff`; `SKIP_M28_OPTIQ_CHAIN=1` to skip).

**HOLD gap (F0634):** `grain=op` `RunGPU` (Eval) then GRAPH in RELEASE gap — do **not** nest Graph under RunGPU. Lab: `./scripts/phase/m29_uma_optiq_live_chain_hold_smoke.sh` (`SKIP_M29_OPTIQ_HOLD=1`).

**Serve probe (F0635/F0636):** after first decode `LeaseEnd`, optional `MaybeProbeOptiqLiveChain`. Default dump `/tmp/uma_optiq_live_dump` via `make -C …/uma_toolkit optiq-live-dump`. Env: `ZEROLLAMA_UMA_OPTIQ_GRAPH_PROBE=1|require`. Lab: `./scripts/phase/m30_uma_optiq_graph_probe_smoke.sh`.

**GRAPH generate opt-in (F0698/F0699/F0719):** after prefill, optional `RunOptiqGraphGenerate(ctx, prompt, nGen)` (in-process F0697 cascade L0..31 via `libuma_optiq_graph_gen.dylib`). Env: `ZEROLLAMA_UMA_OPTIQ_GRAPH_GENERATE=1|require`. F0719 passes request prompt tokens; freeze rematch when prompt matches dump (`GRAPH_GEN_TOKENS=[12675,248046]`). Lab: `./scripts/phase/m31_optiq_graph_generate_rematch.sh`.

**GRAPH token-tail ownership (F0687):** session-resident `GEMV_Q8`→`ARGMAX` (post-norm) or `NORM`→`GEMV`→`ARGMAX`. Dump: `make -C …/uma_toolkit optiq-token-tail-dump`. Env: `ZEROLLAMA_UMA_OPTIQ_GRAPH_TOKEN=1|require|owned` (`owned` replaces mlxrunner greedy sample). Lab: `./scripts/phase/m32_uma_optiq_token_tail_smoke.sh`.

**Eval grain (F0625 / wishlist 4.1):** `ZEROLLAMA_UMA_GRAIN=phase` (default) keeps coarse `LeaseBegin`/`LeaseEnd`. `ZEROLLAMA_UMA_GRAIN=op` makes coarse leases no-ops; each `mlx.Eval` / `RunGPU` takes one-shot HOLD so peer GRAPH can run between Evals. Lab: `./scripts/phase/m25_uma_grain_op_smoke.sh`.

**Live OptiQ freeze (F0626 / wishlist 4.6):** `./scripts/phase/m26_mlxrunner_optiq_tokens_freeze.sh` freezes greedy mlxrunner `context` tokens (`ornith-9b-optiq` default) and rematches UMA `require` vs `off`.

## Build

Mac production script links the client when the sibling toolkit exists:

```bash
./scripts/build/build_zerollama_mac.sh          # BUILD_UMA=auto (default)
BUILD_UMA=0 ./scripts/build/build_zerollama_mac.sh   # skip
```

Manual:

```bash
make -C x/uma
CGO_ENABLED=1 go build -tags uma -o zerollama .
```

Requires broker with `HOLD_GPU` (rebuild `UMAStatus.app` from bmtl `uma_toolkit`). Install the machine daemon once:

```bash
make -C ../bmtl/hardware_lab/lanes/m4/uma_toolkit uma-daemon-install
# or: open …/UMAStatus.app
```

## Disabling UMA (build + runtime)

Two independent knobs — use either or both:

| Knob | Effect |
|------|--------|
| **Runtime** `ZEROLLAMA_UMA_SCHED=off` (also `0` / `false` / `disabled` / `none`) | No broker connect, no HOLD — Metal runs ungated. Works for mlxrunner, ollamarunner, llamarunner, and UMA-linked llama-server **without rebuild**. |
| **Build** `BUILD_UMA=0` | Omit client entirely: Mac `build_zerollama_mac.sh` skips `-tags uma`; `build_llama_server.sh` skips `libuma_llama.a` / `ZEROLLAMA_UMA`. |

```bash
# daily escape hatch (any UMA-capable binary)
ZEROLLAMA_UMA_SCHED=off ./zerollama serve

# compile out
BUILD_UMA=0 ./scripts/build/build_zerollama_mac.sh
BUILD_UMA=0 ./scripts/build/build_llama_server.sh
```

Default remains **`auto`** (gate only when `uma_daemon` is up). Doctor still reports broker health; it does not force the gate on.

## Runtime (`ZEROLLAMA_UMA_SCHED`)

| Value | Behavior |
|-------|----------|
| **unset / `auto`** | **Default** — use broker if up + `HOLD_GPU`; else warn and run ungated |
| `0` / `off` / `disabled` / `none` / `false` | **Gate off** — no connect, no HOLD |
| `1` / `require` / `on` | **Require** broker; fail start or lease on error |
| `degraded` | Connect required; lease failures fall back to ungated (lab only) |

Other:

| Env | Meaning |
|-----|---------|
| `ZEROLLAMA_UMA_SCHED_LOG=1` | Lease begin/end lines (`wait_ms`, `hold_ms`, `evals`) + disconnect `stats` summary |
| `ZEROLLAMA_UMA_GRAIN` | `phase` (default) coarse leases · `op`/`eval`/`fine` per-Eval one-shot HOLD (F0625) |
| `ZEROLLAMA_UMA_OPTIQ_GRAPH_GENERATE` | `off` · `1` soft · `require` full-stack GRAPH generate after prefill (F0698/F0699) |
| `ZEROLLAMA_UMA_OPTIQ_GRAPH_TOKEN` | `off` · `1`/`require` shadow · `owned` greedy sample = GRAPH token-tail (F0687) |
| `ZEROLLAMA_UMA_OPTIQ_TOKEN_RECIPE` | `gemv_argmax` (post-norm) · `norm_gemv_argmax` (F0687 prenorm) |
| `UMA_OPTIQ_TOKEN_TAIL_DIR` | Pack dump (default `/tmp/uma_optiq_token_tail_dump`) |
| `UMA_SOCK` | Socket (default `/tmp/uma_daemon.sock`) |
| `UMA_JOB_NAME` / `UMA_PROJECT` | Ticket project base (default `mlxrunner`) |
| `UMA_PROJECT_FLAT=1` | Single project name (no `-load`/`-prefill`/`-decode` suffix) |

```bash
# lab (strict)
OLLAMA_HOST=127.0.0.1:11435 ZEROLLAMA_UMA_SCHED=require ZEROLLAMA_UMA_SCHED_LOG=1 \
  ./zerollama serve
```

`zerollama doctor` includes **uma broker** (ok when `HOLD_GPU`+`HOLD_ANE`; warn if down or ANE missing).

## Sign-off (lab)

```bash
./scripts/phase/mac_uma_signoff.sh          # M21–M23 ladder (lab ports)
RUN_M20=1 ./scripts/phase/mac_uma_signoff.sh
./scripts/phase/m20_uma_signoff.sh
# full broker TERM soak (disrupts all UMA clients):
./scripts/phase/m20_uma_restart_soak.sh
```

Uses **`:11435` only** (never production `:11434` / `:8081`). Sets `ZEROLLAMA_RUNTIME_DARWIN_SIDECAR=0`. Checks doctor, golden tokens, agent two-turn, HOLD/ATTN/`RUN_HYBRID`/RUN_NOP contention, lease `cum_*`, libuma_client half-close reconnect (6b), and optional broker TERM same-serve recover (step 7).

**Verified (Jul 22–23, lab):** full gate PASS including `RUN_HYBRID` under HOLD (M=8 prepare) and broker restart same-serve recover.

## Production enable (checklist)

1. Install machine broker: `make -C ../bmtl/hardware_lab/lanes/m4/uma_toolkit uma-daemon-install` (or open `UMAStatus.app`)
2. Build Mac binary: `./scripts/build/build_zerollama_mac.sh` (`BUILD_UMA=auto`)
3. `zerollama doctor` → `[ok] uma broker`
4. Gate: `./scripts/phase/m20_uma_signoff.sh` (lab `:11435` only)
5. Daily serve: leave `ZEROLLAMA_UMA_SCHED` unset (`auto`). Strict hosts: `require`.

## Limits

- Admission / lease only — MLX still runs kernels.
- Without `uma_daemon`, default `auto` is a no-op (ungated MLX).
