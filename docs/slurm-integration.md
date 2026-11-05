# Slurm integration

**Status:** Design only. Nothing in this document is shipped.

Zerollama should meet [Slurm](https://github.com/SchedMD/slurm) where sites already are: **inside an allocation**, and as a **client of an existing queue**. Slurm remains the cluster workload manager. Zerollama remains the process that admits chat, generate, and training on the GPUs it was given.

This complements:

- [scheduling-vram-policy.md](./scheduling-vram-policy.md) — per-node queues and the VRAM broker
- [t6-unified-queue.md](./t6-unified-queue.md) — inference-first deferral on one machine
- [fleet-scheduling.md](./fleet-scheduling.md) — warm-model routing across zerollama nodes that are **not** a Slurm cluster
- [gpu-training.md](./gpu-training.md) — `/api/train/*`
- [ROADMAP.md](./ROADMAP.md#slurm) — milestones **SL0–SL4**

Upstream: [SchedMD/slurm](https://github.com/SchedMD/slurm) · docs [slurm.schedmd.com](https://slurm.schedmd.com/) · current release line **26.05**. License is **GPL**. Talk to it with the Slurm CLI or [slurmrestd](https://slurm.schedmd.com/slurmrestd.html) HTTP. Do not CGO-link `libslurm` and do not vendor the tree.

---

## Why this is two seams, not a second slurmctld

Slurm already does the part zerollama has refused to build: exclusive node allocation, wall-clock reservations, partitions, accounts, and a pending queue for a whole cluster. Reimplementing `slurmctld` (MUNGE, the binary RPC, SPANK, `slurmdbd`, fairshare, preemption plugins) would fork a workload manager and still lose every site’s existing `slurm.conf`.

The useful compatibility surface is the part operators and workflow tools already script:

| Operator already has | Zerollama should honor |
|----------------------|------------------------|
| `sbatch` / `salloc` / `srun --gres=gpu:N` | Run as a **guest** of that allocation |
| `squeue` / `scancel` / `scontrol show job` | Reflect those jobs when **we** submitted them |
| `slurmrestd` `POST /slurm/<parser>/job/submit` | Same submit, over HTTP, when there is no local `sbatch` |
| `SIGTERM`, then `KillWait`, then `SIGKILL` | Drain and exit before Slurm kills the step |
| `CUDA_VISIBLE_DEVICES` from GRES + cgroup | Use only those devices |

A homelab with one GPU and no Slurm install keeps today’s queues. A later facade can speak a **subset** of the slurmrestd JSON at that box; the real `sbatch` binary will still only talk to a real `slurmctld`.

---

## Who schedules what

```text
Mode G — guest                         Mode Q — queue client
---------------------                  -------------------------
sbatch / salloc                        long-lived zerollama
        │                                      │
        ▼                                      ▼
zerollama serve  (this job)            sbatch or slurmrestd
  local chat FIFO                        one cluster job per
  local train FIFO                       train or batch item
  VRAM broker                            submit host does no GPU work
        │                                      │
        ▼                                      ▼
GPUs in this allocation only           compute node runs the payload
walltime is Slurm's                    walltime is Slurm's
```

Inside one allocation the existing per-node policy still applies: many HTTP callers, one card, T6 deferral between chat and training. That policy does not reach other nodes. Slurm already picked the node and the deadline.

Fleet routing ([fleet-scheduling.md](./fleet-scheduling.md)) stays the scheduler for a set of **zerollama** hosts. It does not become a partition, and it does not grow 60s reservations to imitate Slurm. A node that is a Slurm guest should not also advertise itself as a fleet member that accepts cross-node assignments for GPUs Slurm gave someone else.

---

## Mode G — allocation guest

`sbatch` or `salloc` starts `zerollama serve`. The Slurm job **is** the server. Clients use HTTP until the job ends.

```bash
#!/bin/bash
#SBATCH --job-name=zerollama
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

export OLLAMA_HOST=0.0.0.0:11434
exec zerollama serve
```

**Detection.** `SLURM_JOB_ID` set ⇒ guest behavior on (`ZEROLLAMA_SLURM_GUEST=auto`, the default). `ZEROLLAMA_SLURM_GUEST=0` leaves the process as a normal serve, for debugging. `=1` forces the drain/status path even if the variable was stripped.

**Devices.** Trust `CUDA_VISIBLE_DEVICES` when Slurm’s cgroup constrains devices. If that variable is empty, fall back to `SLURM_STEP_GPUS`, then `SLURM_JOB_GPUS`. Never scan the machine and claim a GPU outside that set. `OLLAMA_MAX_LOADED_MODELS` still caps runners inside the visible set.

**Listen address.** Compute-node networks are often unreachable from a login node or a laptop. Bind what the site routes (`OLLAMA_HOST`), print the URL on stdout, and write `$SLURM_SUBMIT_DIR/zerollama.$SLURM_JOB_ID.url` when `SLURM_SUBMIT_DIR` is set. When the job owner can `scontrol update`, set the job comment to that URL so `squeue` shows where to connect. Zerollama does not open firewalls or allocate a routable IP.

**Drain.** Slurm sends `SIGTERM`, waits `KillWait` (commonly 30s), then `SIGKILL`. On `SIGTERM`:

1. Stop admitting (`loads_paused`, HTTP **503** on new chat/generate/train).
2. Abort in-flight work that will not finish inside the grace window.
3. Unload runners and exit 0.

Grace defaults to 20s (`ZEROLLAMA_SLURM_KILL_WAIT`) so unload still fits under a 30s `KillWait`. A site with a longer `KillWait` can raise it. A second signal means exit immediately.

**Status.** `GET /api/status` gains a `slurm` object when guest mode is on, so a caller can tell an allocation from a normal daemon:

```json
{
  "slurm": {
    "guest": true,
    "job_id": "48211",
    "job_name": "zerollama",
    "cluster": "campus",
    "gpus": "0",
    "kill_wait_secs": 20
  }
}
```

`job_id` is a string because Slurm job ids are printed that way (`48211` or `48211_2` for an array task). Cluster comes from `SLURM_CLUSTER_NAME` when present.

**What stays local.** Chat FIFO, runtime admission, T6 idle-wait, and the VRAM broker keep working for clients that share this server. They schedule **this** allocation. They do not call `sbatch`.

---

## Mode Q — queue client

A long-lived zerollama (service node, login node, or a machine that is not the GPU) submits work into a real partition. The submit host records the Slurm id and polls. The GPU work runs only on the compute node Slurm chose.

This is the integration for training and for offline batch inference. Interactive `/api/chat` stays on whatever GPU the caller already has; it is not turned into a pending Slurm job (queue latency and cold start break the stream contract in [fleet-scheduling.md](./fleet-scheduling.md#status-contract-node--agent)).

### Submit

`POST /api/train/jobs` accepts an optional `slurm` object. Presence of that object, or `ZEROLLAMA_SLURM_SUBMIT=1`, sends the job to Slurm instead of the embedded trainer on this process.

```json
{
  "kind": "train",
  "priority": "low",
  "payload": { "model_name": "llama3.2:3b", "training_data": [] },
  "slurm": {
    "partition": "gpu",
    "account": "lab",
    "qos": "normal",
    "time": "04:00:00",
    "gpus": 1,
    "cpus_per_task": 8,
    "mem": "32G",
    "constraint": "a100",
    "dependency": "afterok:48001"
  }
}
```

Response includes both ids:

```json
{
  "id": "defer-9f3…",
  "slurm_job_id": "48212",
  "state": "pending",
  "priority": "low"
}
```

`time` is required when the site has no default walltime. Pass `account`, `partition`, `qos`, and `constraint` through unchanged. If Slurm rejects them, return **400** with Slurm’s message. Do not invent accounts or rewrite QoS onto T6 priority. T6 `priority` still describes how **this** zerollama treats the job locally (`low` may defer the submit until the allowed window); the cluster’s priority is Slurm’s.

**Transport, in order:**

1. `sbatch --parsable` on the submit host, as the uid zerollama is running as (MUNGE). Preferred. No extra daemon.
2. slurmrestd `POST /slurm/v0.0.XX/job/submit` when `ZEROLLAMA_SLURMRESTD_URL` is set. Pin the data_parser version (`v0.0.46` on the 26.05 line, or whatever `slurmrestd -d` lists). Do not send `latest`.

The batch script is generated. It exports the payload (or a path to it), runs the same training entry the embed uses, and exits when that process exits. The login-node daemon must not also start that job in-process.

Batch inference (a JSONL of prompts, not a chat stream) uses the same object on a dedicated submit route once training submit works: `POST /api/generate/batch` with `slurm`. Until that route exists, a train `kind` of `run_script` whose script calls `zerollama run` is the supported batch path.

### State

Poll `squeue` / `scontrol show job --json`, or slurmrestd `GET /slurm/v0.0.XX/job/{id}`. Map Slurm states onto the train job states operators already poll:

| Slurm | `/api/train/jobs/:id` |
|-------|------------------------|
| `PENDING`, `CONFIGURING`, `REQUEUED` | `waiting` |
| `RUNNING`, `COMPLETING` | `running` |
| `COMPLETED` | `done` |
| `CANCELLED` by `scancel` / our DELETE | `cancelled` |
| `FAILED`, `TIMEOUT`, `NODE_FAIL`, `OUT_OF_MEMORY`, `PREEMPTED`, `BOOT_FAIL` | `error` (message includes the Slurm reason) |
| `SUSPENDED` | `waiting` with `slurm_state: SUSPENDED` |

`DELETE /api/train/jobs/:id` calls `scancel` on `slurm_job_id` when the job is still `PENDING` or `RUNNING`. After `COMPLETED`, DELETE is a local tombstone only.

List and status responses include `slurm_job_id`, `slurm_state`, `partition`, and `time_left` when Slurm reports them. `GET /api/status` under queue-client mode reports submit configuration, not a guest job:

```json
{
  "slurm": {
    "guest": false,
    "submit": true,
    "transport": "sbatch",
    "rest_url": "",
    "pending": 2,
    "running": 1
  }
}
```

### Identity

Submit as the unix user the daemon runs as. Sites that need per-user accounts should run per-user daemons or put a real `sbatch` in front. Do not accept a caller-supplied uid and pass `--uid` unless that caller is already a Slurm admin (`ZEROLLAMA_SLURM_ALLOW_UID=1`, default off). JWT for slurmrestd is an operator secret (`ZEROLLAMA_SLURMRESTD_JWT` or a file mode `0600`), sent as `X-SLURM-USER-TOKEN`. It is not logged.

---

## Mode F — local facade (later)

For a single machine with **no** Slurm, expose a read/write subset that matches slurmrestd job JSON closely enough that a small client can target either:

| slurmrestd | Local meaning |
|------------|----------------|
| `POST /slurm/v0.0.46/job/submit` | Enqueue a train or batch-inference job |
| `GET /slurm/v0.0.46/jobs` | List those jobs |
| `DELETE /slurm/v0.0.46/job/{id}` | Cancel a waiting or running job |
| `GET /slurm/v0.0.46/partitions` | One synthetic partition, `zerollama`, with the visible GPU count |

`script` is the payload. `tres_per_node: gres/gpu:N` must fit the local card or the submit returns Slurm-shaped errors (`errors: [{error: "..."}]`). Job id is a decimal string so clients that parse `job_id` as a number still work.

`zerollama slurm squeue|scancel|sinfo` prints the familiar columns against **this** daemon. Those subcommands are not drop-in replacements for `/usr/bin/sbatch` on `PATH`.

Mode F does not implement steps, job arrays, hetjobs, accounting, or fairshare. A site that needs those installs Slurm and uses Mode Q.

---

## Field map (the subset we claim)

| Slurm idea | Guest (G) | Queue client (Q) | Facade (F) |
|------------|-----------|------------------|------------|
| Job id | Read `SLURM_JOB_ID` | Store `sbatch --parsable` id | Mint a decimal id |
| Partition | Ignored (already chosen) | Pass-through | One name: `zerollama` |
| GRES `gpu:N` | Visible devices only | `--gpus=N` / `tres_per_node` | Admit if N fits |
| Walltime | Drain before `KillWait` | Required `time` | Optional local deadline |
| QoS / account | Ignored | Pass-through | Ignored |
| Priority / nice | Ignored | Pass-through to `sbatch` | Map onto T6 `low`/`normal`/`high` only in F |
| Dependency `afterok` | Ignored | Pass-through | Local “run after job id” |
| Cancel | `SIGTERM` | `scancel` | Drop local job |
| Arrays, hetjobs, MPI steps | Out of scope | Out of scope | Out of scope |
| Fairshare, preempt, `slurmdbd` | Out of scope | Slurm’s | Out of scope |

T6 labels stay on the zerollama side of the door. Slurm QoS stays on the cluster side. Mixing them in one field is how a typo would submit interactive work into a preemptable QoS.

---

## Environment

| Variable | Default | Mode | Role |
|----------|---------|------|------|
| `ZEROLLAMA_SLURM_GUEST` | `auto` | G | `auto`: on when `SLURM_JOB_ID` is set. `1` force. `0` off. |
| `ZEROLLAMA_SLURM_KILL_WAIT` | `20s` | G | Drain budget after `SIGTERM`. Keep it under the site `KillWait`. |
| `ZEROLLAMA_SLURM_URL_FILE` | `$SLURM_SUBMIT_DIR/zerollama.$SLURM_JOB_ID.url` | G | Where to write the listen URL. Empty disables the file. |
| `ZEROLLAMA_SLURM_SUBMIT` | `0` | Q | `1`: train submits go to Slurm even without a `slurm` object (partition/time from env). |
| `ZEROLLAMA_SLURM_PARTITION` | unset | Q | Default partition when the request omits it. |
| `ZEROLLAMA_SLURM_ACCOUNT` | unset | Q | Default account. |
| `ZEROLLAMA_SLURM_TIME` | unset | Q | Default walltime. Submit fails closed when both this and the request omit `time` and Slurm has no default. |
| `ZEROLLAMA_SLURMRESTD_URL` | unset | Q | If set, HTTP transport instead of `sbatch`. |
| `ZEROLLAMA_SLURMRESTD_VERSION` | unset | Q | Data parser, e.g. `v0.0.46`. Required when the URL is set. |
| `ZEROLLAMA_SLURMRESTD_JWT` | unset | Q | Token value or `@/path/to/file`. |
| `ZEROLLAMA_SLURM_ALLOW_UID` | `0` | Q | Allow an admin caller to set the Slurm uid. |
| `ZEROLLAMA_SLURM_FACADE` | `0` | F | Publish the local slurmrestd-shaped routes. |

Guest and queue-client may both be off. Turning both on means “this process is inside an allocation **and** it submits further jobs.” That is allowed only when the allocation is a login/submit step with no GPUs. If guest mode sees a GPU and `ZEROLLAMA_SLURM_SUBMIT=1`, refuse submit with **409**: the allocation should run work itself, not enqueue more work onto the cluster.

---

## Non-goals

| Idea | Why it stays out |
|------|------------------|
| Embed `slurmctld` / `slurmd` | GPL link, and the site already runs them |
| Replace the fleet manager with a partition | Fleet tokens are seconds; Slurm walltime is the reservation. Different callers. |
| Route `/api/chat` through `squeue` | Pending + cold load breaks the stream status contract |
| SPANK plugin required for v1 | Env + `SIGTERM` covers guest mode. A plugin can come later for sites that want a prolog to write the URL file their way. |
| Multi-node tensor parallel as one Slurm step | Zerollama serve is one process on the GPUs it can see. Multi-node is fleet or an explicit launch script the site owns. |
| Job arrays and hetjobs in v1 | One submit, one id. Arrays can wrap Mode Q later. |
| Honoring Slurm on macOS / the local lab | Slurm’s own README: tested on Linux. Guest auto-detect simply stays off when `SLURM_JOB_ID` is unset. |

---

## Milestones

See [ROADMAP — Slurm](./ROADMAP.md#slurm).

| Milestone | Summary |
|-----------|---------|
| **SL0** | This design |
| **SL1** | Guest: detect, device fence, URL file, `SIGTERM` drain, `/api/status` `slurm` object |
| **SL2** | Queue client: `sbatch --parsable` + state map + `scancel`; train payload only |
| **SL3** | slurmrestd transport for SL2, pinned data_parser |
| **SL4** | Local facade + `zerollama slurm` text commands |

SL1 is useful on its own. A campus GPU partition can `sbatch zerollama serve` before any submit path exists.
