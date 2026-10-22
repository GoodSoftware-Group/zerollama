# DeepInfer borrowings ladder (DI0–DI8)

**Upstream:** [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer) — production offline engine for DeepSeek-V4.1-Flash (SM100 / Ascend 950).  
**Idea donor doc:** [deepinfer-borrowings.md](./deepinfer-borrowings.md)  
**Sibling checkout:** Mac `../DeepInfer`; astra `/mnt/ssd2/src/DeepInfer`  
**ROADMAP:** **M29**

**Status (Oct 2026):** **DI0–DI1 / DI7 / DI6 / DI4 / DI5 / DI2 / DI3 done.** **DI8** standing skip. Do **not** try to serve DeepInfer on 4090 / 5080 / H200.

---

## Why this ladder exists

| Temptation | Why we do not |
|------------|----------------|
| `pip install` DeepInfer + FlashMLA into `:8081` | SM100-only kernels; ~510 GB Flash checkpoint; wrong substrate vs GGUF/`llama-server` |
| Treat H200 / 5080 as “close enough” | DeepInfer gates on `is_sm_major_10` (SM100); 5080=`sm_120`, H200=SM90, 4090=SM89 |
| Vendor MegaMoE / DeepEP / Engram | Fleet EP; already a DeepGEMM non-goal |
| Skip straight to code without DI0–DI1 | Need hardware reality + bring/watch/skip before touching adaptive-dm / L3 |

**Product shape:** steal **schedule / speculation / SWA policy** into llama-server + Python runtime. Keep DeepInfer as a read-only sibling.

---

## Capability map

| Their piece | Zerollama today | Ladder use |
|-------------|-----------------|------------|
| Prefill vs decode step cadence (DP ALL/ANY) | llama-server `-cb` | **DI7** — opt-in *slot-launch* defer while decode-hot (not DP vote, not exclusive steps) |
| ZOS + `forbid_cuda_sync` | PA tick + continuous batch | **DI6** — inventory only (no ZOS port) |
| YOCO SWA compact | L3 SWA store/resume gates | **DI4** — keep-token *budget* helper; no hybrid whole-prompt truncate |
| Batch-invariant + `seq_seed` | `ZEROLLAMA_BATCH_INVARIANT` | **DI5** — eval hygiene (force greedy+seed; not kernel batch-invariance) |
| DSpark confidence-varlen MTP | adaptive-dm (profit), ggml MTP, ANE dflash, MLX MTP | **DI2** design → **DI3** spike (accept_by_pos) |
| FlashMLA / DeepGEMM / DeepEP | [deepgemm-borrowings.md](./deepgemm-borrowings.md) | **DI8** standing skip |
| xgrammar structured output | Ollama `format` / GBNF / MLX xgrammar | Already covered — not a DI rung |

---

## Rungs

| Rung | Goal | Owner | Exit | Status |
|------|------|-------|------|--------|
| **DI0** | Sibling inventory + hardware probe | Ops | Clone; GPU CC; `is_sm_major_10` false on ship hosts | **Done (astra, Oct 2026)** |
| **DI1** | Borrowings doc + ROADMAP wire | Docs | borrowings + this ladder + **M29** + sibling map | **Done (Oct 2026)** |
| **DI7** | Prefill join cadence while decode-hot | llama-server | Opt-in env; defer slot launch; defaults off; honest vs DeepInfer | **Done (Oct 2026)** — patch **0140** (audit-fixed) |
| **DI6** | Async schedule sync audit | Runtime lab | Inventory; no ZOS/`forbid_cuda_sync` port | **Done (Oct 2026)** |
| **DI4** | YOCO keep-token budget (policy) | Python L3 | Document budget; **non-goal:** whole-prompt truncate on hybrid GGUF | **Done (policy)** — engine wire parked |
| **DI5** | Eval hygiene env (batch-invariant *idea*) | Runtime / doctor | Force greedy+seed; deny KV reuse/auto-batch; serialize generate_batch | **Done (Oct 2026)** — audit-fixed |
| **DI2** | Confidence-scheduled draft — design | Spec | Map DSpark → adaptive-dm / MTP; name accept vs confidence | **Done (Oct 2026)** — design below |
| **DI3** | Verified accept_by_pos exposure + unit | llama-server | Timings + slots export; host unit; no production default | **Done (Oct 2026)** — patch **0141** |
| **DI8** | Do not vendor DeepInfer | Policy | Standing skip | Standing |

**Priority order (historical):** DI7 → DI6 → DI4 → DI5 → DI2 → DI3. Ladder complete aside from **DI8** standing skip.

---

## DI0 — astra probe (2026-10-20)

| Item | Value | Verdict |
|------|-------|---------|
| Clone | `/mnt/ssd2/src/DeepInfer` | OK |
| Host GPUs | 2× RTX 4090 (SM89) | Cannot run DeepInfer serve |
| FlashMLA / DeepGEMM arches | `sm_100a` / `sm_103a` | Skip for 4090/5080/H200 |

---

## DI7 — prefill join cadence (decode-hot)

### What DeepInfer does

In `NORMAL` engine, each schedule step is **either** prefill **or** decode for the whole DP group:

- Most steps → decode.
- Every `prefill_attempt_interval` decode steps (**ALL** point, default **70**) → collective prefill only if **all** ranks have waiting prompts.
- Every `prefill_any_attempt_interval` (**ANY** point, default **100**) → collective prefill if **any** rank has work.
- Clock = GPU `decode_step`, reset after a prefill step.

### Wrong place (reverted)

PA `SchedulerLoop.tick()` admit gate — wrong layer/clock; `DECODE` never armed during work. Do **not** reintroduce.

### What we ship (single-node analogue)

| Piece | Detail |
|-------|--------|
| Patch | `llama/patches/0140-server-DI7-prefill-join-cadence-decode-hot.patch` |
| Hook | `process_single_task`: free slot + COMPLETION/INFILL → defer if any slot is `GENERATING` and outside window |
| Clock | Decode batches since last launch (`metrics.n_decode` delta) |
| Re-offer | `maybe_requeue_deferred_prefills()` after `update_slots` |
| Env | `ZEROLLAMA_PREFILL_ATTEMPT_INTERVAL` — **≤1 = off**; `>1` = periodic join every N decode batches since last launch |
| Starve | `ZEROLLAMA_PREFILL_ANY_ATTEMPT_INTERVAL` — if unset and attempt>1 → **2×attempt**; explicit `1` → periodic only; `>1` → that gap |
| Defaults | Unset → today's join-when-free |
| Not claimed | DP ALL/ANY voting; exclusive prefill-or-decode steps (cb may still mix tokens after launch) |

**Lab:** `:18082`, `-np≥2`, attempt/starve set, confirm startup log `DI7 prefill join cadence:…`. No production default flip.

---

## DI6 — sync / overlap audit

DeepInfer ZOS overlaps CPU `schedule(N+1)` with GPU `forward(N)` and uses `forbid_cuda_sync` on the launch path. We do **not** port ZOS.

| Layer | Behavior |
|-------|----------|
| PA `tick()` | Admit only — no CUDA sync |
| `yield_to_queue` during decode | CPU metrics/slot tasks while GPU runs — keep |
| `llama_synchronize` / logits / sampler | **Required** before host read |
| `metrics_flush_idle` | Sync on idle flush — OK |
| Python runtime | No torch sync arbiter (ggml ≠ torch) |

**Non-goals:** raise-on-sync guard; ZOS port. Optional lab `llama_synchronize` counter only if a bench shows unexplained host stalls.

---

## DI4 — YOCO keep-token budget (policy only)

DeepInfer compact mode is **layer-aware** (consumer SWA metadata / keep-tokens); full layers keep long context.

| Already | Role |
|---------|------|
| `HybridKVCacheCoordinator` | Deny resume when prefix exceeds min SWA window |
| `swa_reachable_store_mask` | Store reachable SWA tail |
| `runtime/kv/swa_compact_reprefill.py` | Numeric budget `min(n, window+backtrack)` + tests |

**Non-goal:** slicing the whole prompt before `run_prefill` / `llama_decode` on hybrid GGUF (drops full-layer prefix). Revisit only with a layer-safe ggml path or SWA-only models.

---

## DI5 — eval hygiene (`ZEROLLAMA_BATCH_INVARIANT`)

DeepInfer: `DEEPINFER_BATCH_INVARIANT=1` disables FlashMLA split-KV + pins `seq_seed` so alone vs co-batched generate matches.

Zerollama ships **eval hygiene**, not that kernel property. Env name kept for the idea; `/health.batch_invariant.mode` = `eval_hygiene`.

### What we ship (audit-fixed)

| Lever | Detail |
|-------|--------|
| Env | `ZEROLLAMA_BATCH_INVARIANT=1` (default off) |
| Sampling | **Force** greedy (`temperature=0`, `top_k=1`, `top_p=1`) — overrides client/Modelfile knobs |
| Seed | `ZEROLLAMA_SEQ_SEED` (default `0`); client `seed>=0` wins; `seed=-1` → default |
| Prefix | Force `cache_prompt=false` + Radix share off |
| Auto-batch | Skip Phase 15 auto-batch / stream auto-batch |
| `generate_batch` / stream batch | **Serialize** to per-prompt `generate` / `stream_generate` |
| Health | `/health` → `batch_invariant` (`mode`, `forces_greedy`, `serializes_generate_batch`, …) |
| Module | `runtime/runtime/batch_invariant.py` + tests |

### Honest limits

- No ggml FlashMLA split-KV switch — logits may still vary under `n_parallel>1` continuous batch.
- Strongest goldens: this env + lab ports + single-slot / `-np 1` llama-server.

---

## DI2 — confidence-scheduled draft budget (design)

### What DeepInfer does (DSpark / MTP varlen)

```text
confidence_head(hidden) → rates[seq, draft_pos]   # predicted P(accept) per draft token
                        → sort / allocate via decode_perf_curve
                        → batch total_q + seqlens_q[seq]   # how many draft tokens to verify
```

Locus: `csrc/engine/mtp_varlen.hpp` `schedule()`, confidence head in `models/mtp/dflash.py`. Uses **predicted** accept rates (N−1 / N−2 lag) plus a measured decode-time curve — not the same object as post-verify accept counts.

### What zerollama already has

| Surface | Mechanism | Signal |
|---------|-----------|--------|
| **adaptive-dm profit** (`server-adaptive-dm.h`, patch **0102**) | Clamp `draft_n_max` per slot | **Ex-post** accept count + cycle-time EWMA |
| ggml MTP / DFlash / ANE dflash / MLX MTP | Fixed or `--spec-draft-n-max` | No per-position predicted rates |
| Metrics | `n_draft_*` / accept tallies | Measured verify outcome |

Lab: `ZEROLLAMA_SPEC_DM_ADAPTIVE=profit` / `--spec-dm-adaptive profit`. Default **off**.

### Trap (naming)

| Term | Meaning | Do not confuse with |
|------|---------|---------------------|
| **confidence** (DeepInfer) | Model head: predicted P(accept) | Measured accept fraction |
| **accept_rate** (verify) | Accepted / drafted after target forward | Confidence head output |
| **adaptive-dm score** | Profit EWMA from timing + accept | Either of the above |

DI3 must keep separate fields (`predicted_accept` vs `verified_accept`) if both exist.

### Portable bring list (no new PyTorch engine)

1. **Keep adaptive-dm as the delivery vehicle** — do not vendor DSpark / tilelang confidence heads.
2. **DI3 spike (smallest):** EWMA of *per-position verified accept* as a pseudo-confidence curve to bias `decide_profit_n_max` (or expose `accept_by_pos[]` on metrics) — still ex-post, closer to DSpark’s per-depth allocation without a head.
3. **Optional later:** GGUF MTP confidence tensor → same `rates[pos]` slot — only then is it true confidence-scheduled.
4. **Non-goals:** port `schedule_mtp_varlen` total_q allocator; FlashMLA varlen Q; SM100 confidence kernels; production default flip.

### Map

| DeepInfer | Zerollama today | DI3 candidate |
|-----------|-----------------|---------------|
| `rates[seq, nextn]` from head | — | EWMA `accept_by_pos` or future GGUF head |
| `total_q` / `seqlens_q` | Fixed / `adaptive_n_max` per slot | Cohort clamp in adaptive-dm; extend with pos curve |
| Decode perf curve | Profit cycle-time EWMA | Keep; do not duplicate DeepInfer’s curve store |

**DI2 exit:** this section. **DI3** ships the per-position verified EWMA surface (below).

---

## DI3 — verified accept_by_pos (spike done)

**Why DI3 after DI2:** adaptive-dm already maintained `profit_pos_accept_ewma[]` and used it inside `decide_profit_n_max` (survival expected accept). Lab A/B could not *see* that curve on the wire — only logs/prometheus counts. DI3 closes the **observability** gap without claiming a DSpark confidence head (predicted P(accept) ≠ verified accept).

| Piece | Detail |
|-------|--------|
| Patch | `llama/patches/0141-server-DI3-verified-accept-by-pos-timings-slots.patch` |
| Timings JSON | `accept_rate`, `accept_by_pos[]`, `accept_by_pos_kind=verified` when `draft_n > 0` |
| Slots JSON | `adaptive_dm.verified_accept_by_pos_ewma` + `adaptive_n_max` when profit controller on |
| Export helper | `server_adaptive_dm_state::export_verified_accept_by_pos` |
| Host unit | `./scripts/phase/di3_accept_by_pos_unit.sh` (no GPU) |
| Lab opt-in | Existing `ZEROLLAMA_SPEC_DM_ADAPTIVE=profit` / `--spec-dm-adaptive profit` |
| Defaults | Off — no production flip |
| Not claimed | Predicted confidence head; DSpark `total_q` allocator; SM100 kernels |

**Two verified signals (do not mix):**

| Field | Scope | Meaning |
|-------|-------|---------|
| `timings.accept_by_pos` | **This request** | Count[pos] / `n_draft_verif_steps` (empirical survival) |
| `slots[].adaptive_dm.verified_accept_by_pos_ewma` | **Controller** (cross-request, profit on) | EWMA used inside `decide_profit_n_max` |

**Naming:** both are **ex-post verified** — never label them `predicted_accept` / `confidence`.

---

## DI8 — standing non-goals

- Bind DeepInfer or its vendors to `:11434` / `:8081`
- Start DeepInfer on PVE hypervisor
- Claim 5080 / H200 “DeepInfer compatible”
- Vendor Ascend / CANN
- Whole-prompt compact truncate on hybrid GGUF (DI4)

---

## Findings

| Date | Note |
|------|------|
| 2026-10-20 | DeepInfer OSS; borrowings + ladder; schedule/prefill before speculation. |
| 2026-10-20 | DI7 PA stub reverted; retargeted to llama-server continuous batch. |
| 2026-10-20 | DI7 spike **0140**; lab `:18082`. |
| 2026-10-20 | DI6 inventory: `yield_to_queue` overlap; required logits sync; no ZOS port. |
| 2026-10-20 | DI4 keep-token budget helper; hybrid whole-prompt wire marked non-goal. |
| 2026-10-20 | Audit fix-up: DI7 `any` default/`since`-based periodic; rename patch (drop ALL-ANY claim); DI4 docs/non-goal hardened. |
| 2026-10-20 | DI5: `ZEROLLAMA_BATCH_INVARIANT` eval hygiene; `/health.batch_invariant`. |
| 2026-10-20 | DI5 audit fix: **force** greedy (override client temp); serialize `generate_batch`; mode=`eval_hygiene`; prefer `-np 1` for strongest goldens. |
| 2026-10-20 | DI2 design: DSpark confidence-varlen → adaptive-dm delivery; trap = predicted confidence ≠ verified accept; DI3 = per-pos EWMA spike (parked). |
| 2026-10-21 | DI3 spike: timings `accept_rate`/`accept_by_pos` + slots EWMA export; patch **0141**; host unit `di3_accept_by_pos_unit.sh`. |
| 2026-10-21 | DI3 audit: document timings (per-request) vs slots EWMA (controller) as distinct verified signals. |
