/*
 * sched_unipc.h — Flow-matching UniPC host scheduler (reference impl).
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sched_unipc sched_unipc;

sched_unipc *sched_unipc_create(int steps, float shift);
void sched_unipc_destroy(sched_unipc *s);

int sched_unipc_num_sigmas(const sched_unipc *s);
const float *sched_unipc_sigmas(const sched_unipc *s);

float sched_unipc_warp_sigma(float sigma, float shift);

void sched_unipc_cfg_combine(const float *x_uncond, const float *x_cond,
                             float *x_out, size_t n, float cfg_scale);

int sched_unipc_step(sched_unipc *s, int step, const float *model_out,
                     float *sample, size_t n);

#ifdef __cplusplus
}
#endif
