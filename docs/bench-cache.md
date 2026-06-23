# Model bench cache (`zerollama bench` + `TOK/S` in `ls`)

**Audience:** operators choosing models on one machine; contributors touching CLI or list output.

**Related:** [phase17-llama-server.md](./phase17-llama-server.md) (engine A/B), [gpu-profiles-l1.md](./gpu-profiles-l1.md) (throughput tuning), [launch-model-inventory.md](./launch-model-inventory.md) (launch metadata — separate from bench cache).

---

## Why this exists

Operators routinely ask: *“Which of my local models is fastest on **this** GPU?”* Size and parameter count in `zerollama ls` do not answer that — a 3B Q4 model can beat a 7B Q8 on the same box depending on backend, VRAM pressure, and context.

Running a full developer benchmark (`cmd/bench/bench.go`) for every model is heavy and writes benchstat/CSV, not something you want in daily workflow. **`zerollama bench`** is the lightweight operator path:

1. Warm up + a few timed `/api/generate` epochs per model.
2. Average **decode tok/s** from server `EvalCount` / `EvalDuration`.
3. Persist to **`~/.ollama/bench.json`** keyed by **manifest digest** (not name — re-pull invalidates stale numbers automatically).
4. Show **`TOK/S`** in **`zerollama ls`** without re-running inference.

**Why client-only:** Bench runs through the same HTTP path agents use; no server schema change, no manifest layer, no fleet coupling. Results reflect *your* serve flags, GPU, and loaded backend (ggml vs llama-server).

**Why digest keys:** Tag names can be copied (`zerollama cp`) or reused; digest changes when weights change. Orphan cache entries are harmless — `ls` only looks up the current digest.

---

## Data flow

```text
zerollama bench [MODEL...]
        │
        ▼
  GET /api/tags          filter: local, completion-capable, optional name prefix
        │
        ▼
  POST /api/generate     warmup (discarded) + N timed epochs (Raw prompt, num_predict=128)
        │                EvalCount / EvalDuration → avg tok/s
        ▼
  ~/.ollama/bench.json   save after each model (Ctrl-C keeps earlier results)
        │
        ▼
zerollama ls             TOK/S column (-- when no entry for digest)
```

**Why save after each model:** Multi-model bench can take tens of minutes; partial progress should survive interruption.

**Why unload between models (`KeepAlive: 0`):** Each model gets a fair VRAM slot; otherwise a warm predecessor skews load time and scheduler state for the next tag.

---

## Cache file

**Path:** `~/.ollama/bench.json` (same directory as `config.json`).

**Shape:**

```json
{
  "sha256:abc123…": {
    "model": "llama3.2:latest",
    "tok_per_sec": 42.3,
    "benched_at": "2026-06-20T08:00:00Z"
  }
}
```

| Field | Why |
|-------|-----|
| Map key = full digest | Stable across renames; auto-invalidates on re-pull |
| `model` | Human-readable label when inspecting JSON |
| `tok_per_sec` | Generation decode rate only (not prefill, not TTFT) |
| `benched_at` | Operator audit; future TTL policies could use this |

Writes use atomic temp + rename (`fileutil.WriteWithBackup`) — same durability pattern as integration config.

---

## Command reference

```bash
zerollama bench                    # all local text models
zerollama bench llama3.2           # prefix filter (same idea as ls)
zerollama bench --force            # ignore cache
zerollama bench --epochs 5 --tokens 256 --warmup 2 --timeout 180
zerollama ls                       # TOK/S column populated from cache
```

| Flag | Default | Why |
|------|---------|-----|
| `--epochs` | 3 | Average reduces noise from scheduler jitter |
| `--tokens` | 128 | Enough decode work without dominating wall time |
| `--warmup` | 1 | Pays load + first-graph cost outside timed epochs |
| `--force` | off | Skip re-bench when digest already cached |
| `--timeout` | 120s | Large models may need long first load |

**Skipped models:** remote catalog stubs (except LM Studio imports), embedding/image/video_gen/speech-only tags, empty digest.

**Warmup failures:** logged as warnings; timed epochs still run — **why:** slow first load should not abort the whole tag when decode would succeed.

---

## What the number means (and does not)

| Measures | Does not measure |
|----------|------------------|
| Sustained **generation** tok/s on **this host** | Cloud / remote models |
| Same backend path as `zerollama run` | Prefill tok/s or TTFT |
| Raw prompt (`Raw: true`) decode throughput | Chat-template formatting quality |
| Average of N epochs with varied prompts | Agent tool-call or vision paths |

**Why `Raw: true`:** Template rendering and thinking tags add variable overhead; bench targets decode engine speed, comparable to `cmd/bench/bench.go`.

**Why varied word-list prompts per epoch:** Defeats KV prefix reuse across epochs so timed runs measure fresh decode, not cached-prefix shortcuts.

---

## Relationship to other benchmarks

| Tool | Role |
|------|------|
| **`zerollama bench`** | Operator cache → `ls` column; minimal flags |
| **`cmd/bench/bench.go`** | Developer harness; benchstat/CSV; TTFT/prefill/load columns |
| **`scripts/m4_upstream_vs_zerollama_bench.sh`** | Phase 17 ggml vs upstream llama-server decision |
| **L1 gates (`l1_cuda_full_gate.sh`)** | CI throughput regression with profile env |

Do not expect `TOK/S` in `ls` to match script benchstat lines byte-for-byte — prompts, epochs, and backends differ by design.

---

## Code map

| File | Responsibility |
|------|----------------|
| `cmd/bench_cmd.go` | Cobra command, generate loop, model filter |
| `cmd/benchcache/benchcache.go` | JSON load/save |
| `cmd/cmd.go` `ListHandler` | `TOK/S` column; soft-fail if cache missing |

**Why soft-fail in `ls`:** Listing models must never break when bench has never been run or cache is corrupt; missing entry shows `--`.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `TOK/S` always `--` | Never ran `bench`, or digest changed after re-pull |
| Bench skips model | Cached entry; use `--force` |
| `error` in summary table | Timed epoch failed (timeout, non-completion model) |
| Lower than `run --verbose` | Bench uses Raw + fixed prompt; thinking models may differ |
| Stale after backend change | Re-bench — cache does not track ggml vs llama-server |

Manual cache reset: delete `~/.ollama/bench.json` or bench with `--force`.
