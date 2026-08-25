/*
 * Thin C API for mlxrunner ↔ machine-wide uma_daemon (libuma_client only).
 */
#pragma once

#include <stddef.h>
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
 *   0/off/disabled  — no-op success (gate fully off)
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
 * Alias for uma_mlx_lease_begin_unit("gpu", phase).
 */
int uma_mlx_lease_begin(const char *phase);
void uma_mlx_lease_end(void);

/*
 * Multi-unit leases (F0390): unit is "gpu"|"ane"|"amx" (case-insensitive).
 * GPU / ANE / AMX tickets are independent so HOLD_GPU ∥ HOLD_ANE works.
 */
int uma_mlx_lease_begin_unit(const char *unit, const char *phase);
void uma_mlx_lease_end_unit(const char *unit);

/*
 * Run goUmaMlxJob under an active lease for unit, or one-shot HOLD_* if none.
 * On require failure, sets last error; does not run ungated unless degraded.
 */
void uma_mlx_run_gpu(void);
void uma_mlx_run_unit(const char *unit);

/* 1 if last lease/run failed in require mode (Go should surface error). */
int uma_mlx_last_failed(void);
const char *uma_mlx_last_error(void);

/*
 * Admission grain (F0625 / wishlist 4.1):
 *   ZEROLLAMA_UMA_GRAIN=phase (default) — coarse LeaseBegin/End windows
 *   ZEROLLAMA_UMA_GRAIN=op              — LeaseBegin no-op; each RunGPU one-shot
 * Returns "phase" or "op".
 */
const char *uma_mlx_grain(void);

/*
 * GRAPH helpers (wishlist GRAPH-MLX 0.4 / F0624) — libuma_client only, no
 * uma_graph.h. Requires Acquire() with an active broker connection.
 */
int uma_mlx_format_graph(char *out, size_t n, int ntok, const char *form,
                         const char *nodes);
int uma_mlx_format_graph_ex(char *out, size_t n, const char *level, int ntok,
                            const char *form, const char *nodes, int ngen,
                            int eos, const char *toks);
int uma_mlx_submit(const char *project, const char *job, uint64_t *ticket_out);
int uma_mlx_wait(uint64_t ticket, double timeout_s, char *buf, size_t n);
/* Submit + WAIT; project NULL → UMA_JOB_NAME / mlxrunner. */
int uma_mlx_graph(const char *project, const char *job, double timeout_s,
                  char *buf, size_t n);

/*
 * Named BUF helpers (F0627) — libuma_client; required for real GRAPH recipes.
 * Export returns iosurface_id and token via out-params (0 on success).
 */
int uma_mlx_buf_alloc(const char *name, size_t nbytes);
int uma_mlx_buf_free(const char *name);
int uma_mlx_buf_put(const char *name, const void *data, size_t nbytes);
int uma_mlx_buf_get(const char *name, void *dst, size_t dst_n, size_t *got);
int uma_mlx_buf_export(const char *name, uint32_t *iosurface_id_out,
                       uint32_t *token_out);
int uma_mlx_buf_reclaim(const char *name, uint32_t token);

#ifdef __cplusplus
}
#endif
