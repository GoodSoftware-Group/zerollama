/* Vendored from antirez/h3.c (MIT) — velocity reuse schedule + extrapolate. */
#include "h3_reuse.h"

#include <string.h>

int h3_dit_reuse_schedule(int steps, int reuse_interval, uint8_t *selected,
                          size_t selected_count) {
  if (steps < 1 || reuse_interval < 1 || reuse_interval > 32 || !selected ||
      selected_count < (size_t)steps)
    return -1;
  memset(selected, 0, (size_t)steps);

  int count = 0;
  for (int step = 0; step < steps; step++) {
    if (reuse_interval == 1 || step == 0 || step == steps - 1 ||
        step % reuse_interval == 0) {
      selected[step] = 1;
      count++;
    }
  }
  return count;
}

float h3_dit_extrapolation_ratio(float current_sigma, float last_sigma,
                                 float previous_sigma, int have_previous) {
  if (!have_previous)
    return 0.0f;
  float denominator = last_sigma - previous_sigma;
  float ratio = denominator != 0.0f
                    ? (current_sigma - last_sigma) / denominator
                    : 0.0f;
  if (ratio < -2.0f)
    ratio = -2.0f;
  if (ratio > 2.0f)
    ratio = 2.0f;
  return ratio;
}

void h3_dit_extrapolate_velocity(float *output, const float *last,
                                 const float *previous, size_t count,
                                 float current_sigma, float last_sigma,
                                 float previous_sigma, int have_previous) {
  if (!output || !last)
    return;
  if (!have_previous || !previous) {
    memcpy(output, last, count * sizeof(*output));
    return;
  }
  float ratio = h3_dit_extrapolation_ratio(current_sigma, last_sigma,
                                           previous_sigma, have_previous);
  for (size_t i = 0; i < count; i++)
    output[i] = last[i] + ratio * (last[i] - previous[i]);
}
