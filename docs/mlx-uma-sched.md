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

## Build

Mac production script links the client when the sibling toolkit exists:

```bash
./scripts/build/build_zerollama_mac.sh          # BUILD_UMA=auto (default)
BUILD_UMA=0 ./scripts/build/build_zerollama_mac.sh   # skip
```

Manual:

```bash
make -C x/mlxrunner/uma
CGO_ENABLED=1 go build -tags uma -o zerollama .
```

Requires broker with `HOLD_GPU` (rebuild `UMAStatus.app` from bmtl `uma_toolkit`). Install the machine daemon once:

```bash
make -C ../bmtl/hardware_lab/lanes/m4/uma_toolkit uma-daemon-install
# or: open …/UMAStatus.app
```

## Runtime (`ZEROLLAMA_UMA_SCHED`)

| Value | Behavior |
|-------|----------|
| **unset / `auto`** | **Default** — use broker if up + `HOLD_GPU`; else warn and run ungated |
| `0` / `off` | Gate off |
| `1` / `require` / `on` | **Require** broker; fail mlxrunner start or lease on error |
| `degraded` | Connect required; lease failures fall back to ungated (lab only) |

Other:

| Env | Meaning |
|-----|---------|
| `ZEROLLAMA_UMA_SCHED_LOG=1` | Lease begin/end lines (`wait_ms`, `hold_ms`, `evals`) + disconnect `stats` summary |
| `UMA_SOCK` | Socket (default `/tmp/uma_daemon.sock`) |
| `UMA_JOB_NAME` / `UMA_PROJECT` | Ticket project base (default `mlxrunner`) |
| `UMA_PROJECT_FLAT=1` | Single project name (no `-load`/`-prefill`/`-decode` suffix) |

```bash
# lab (strict)
OLLAMA_HOST=127.0.0.1:11435 ZEROLLAMA_UMA_SCHED=require ZEROLLAMA_UMA_SCHED_LOG=1 \
  ./zerollama serve
```

`zerollama doctor` includes **uma broker** (warn if down or missing `HOLD_GPU`).

## Sign-off (lab)

```bash
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
