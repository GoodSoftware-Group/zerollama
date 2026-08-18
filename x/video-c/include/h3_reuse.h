/* H3 DiT velocity-reuse schedule (from antirez/h3.c h3_dit.c; host-only). */
#ifndef H3_REUSE_H
#define H3_REUSE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Mark steps that run a fresh DiT forward. Returns count, or -1 on error. */
int h3_dit_reuse_schedule(int steps, int reuse_interval, uint8_t *selected,
                          size_t selected_count);

/* Linear velocity extrapolation between two evaluated σ (antirez h3.c). */
float h3_dit_extrapolation_ratio(float current_sigma, float last_sigma,
                                 float previous_sigma, int have_previous);
void h3_dit_extrapolate_velocity(float *output, const float *last,
                                 const float *previous, size_t count,
                                 float current_sigma, float last_sigma,
                                 float previous_sigma, int have_previous);

#ifdef __cplusplus
}
#endif

#endif /* H3_REUSE_H */
