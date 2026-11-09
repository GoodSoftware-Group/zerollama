# Training And Scheduling Roadmap

This file holds the detailed backlog for **training**, **queue policy**, and **cluster scheduling**.

---

## Fleet scheduling (multi-node)

Closed for now. The F-track shipped its initial management-node / warm-routing shape; keep history in [../fleet-scheduling.md](../design/training/fleet-scheduling.md), [../fleet-management.md](../design/training/fleet-management.md), and [CHANGELOG](../../CHANGELOG.md).

## Slurm

Open work:

| Milestone | Status | Open work |
|-----------|--------|-----------|
| **SL1** | **Open** | Allocation guest mode (`SLURM_JOB_ID`, drain-on-`SIGTERM`, GPU fence, status exposure) |
| **SL2** | **Partial** | `sbatch` submit, poll, and `scancel` on `/api/train/jobs` are in. Batch inference route and slurmrestd are not. |
| **SL3** | **Open** | `slurmrestd` transport with pinned parser version and sane auth handling |
| **SL4** | **Open** | Local slurmrestd-shaped facade over the in-process queue |

Design reference: [../slurm-integration.md](../design/training/slurm-integration.md).

---

## GPU training (fine-tuning)

**History:** shipped items such as **T1**, **T3**, **T4**, **T7**, and **T8** belong in [CHANGELOG](../../CHANGELOG.md) and [../gpu-training.md](../design/training/gpu-training.md). This section keeps only open training work.

Open borrowings / follow-ons:

| Borrow | Open work |
|--------|-----------|
| Stock Trainer efficiency | **T9** |
| GRPO / TRL RL job kind | **T10** |
| Lite data recipes | **T11** |

### Phases (training track)

| Phase | Status | Open work |
|-------|--------|-----------|
| **T2** | **Open** | Auth on `/api/train` and legacy `:9500` |
| **T5** | **Optional** | Native training only if Python embed becomes the bottleneck |
| **T6** | **Partial** | Queue policy polish: strict priority, echo, defer, window, cross-queue FIFO, optional fourth latency tier |
| **T9** | **Partial** | Stock Trainer efficiency polish and 5080 operator sign-off |
| **T10** | **Partial** | GRPO / RL job kind and 5080 production-venv sign-off |
| **T11** | **Partial** | Lite data recipes / presets / upload ergonomics |

**Suggested order:** **T9 → T10 → T11 → T6**. Do **T2** when auth becomes a deployment requirement. Keep **T5** optional unless the Python path becomes the limiting factor.

`training.py` `process_training_request` (dataset, collator, train, save in one function) is **R8** on the serving structure backlog: [core-serving.md](core-serving.md#code-structure). Split it while already editing that request path.

### Non-goals (for this track)

Keep the current non-goals from [../gpu-training.md](../design/training/gpu-training.md) and the track docs: no second in-tree training backend, no vendored Studio UI, no promise of Unsloth Core parity, and no implicit GPU sharing with a separate Studio product.
