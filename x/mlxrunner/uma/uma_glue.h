/*
 * Thin C API for mlxrunner ↔ machine-wide uma_daemon (libuma_client only).
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1 if gate is configured (require/auto/degraded), not force-off. */
int uma_mlx_runtime_enabled(void);

/*
 * Connect to broker.
 * Modes (ZEROLLAMA_UMA_SCHED):
 *   unset/auto      — connect if broker up; else warn and stay inactive (default)
 *   0/off           — no-op success
 *   1/on/require    — require broker + HOLD_GPU
 *   degraded        — require connect; lease failures fall back to ungated
 * Returns 0 ok, -1 hard failure (require/degraded cannot connect).
 */
int uma_mlx_acquire(void);

void uma_mlx_release(void);

/* Cumulative lease stats since acquire (valid after release too). */
void uma_mlx_stats(uint64_t *leases, uint64_t *evals, double *wait_ms_total,
                   double *hold_ms_total);

/* 1 if connected and gate active. */
int uma_mlx_active(void);

/*
 * Coarse GPU lease (load / prefill chunk / decode step). Nested begins are
 * refcounted. phase is for logs (e.g. "load", "prefill", "decode").
 * Returns 0 ok, -1 on require failure (caller should abort).
 */
int uma_mlx_lease_begin(const char *phase);
void uma_mlx_lease_end(void);

/*
 * Run goUmaMlxJob under an active lease, or one-shot HOLD if none.
 * On require failure, sets last error; does not run ungated unless degraded.
 */
void uma_mlx_run_gpu(void);

/* 1 if last lease/run failed in require mode (Go should surface error). */
int uma_mlx_last_failed(void);
const char *uma_mlx_last_error(void);

#ifdef __cplusplus
}
#endif
