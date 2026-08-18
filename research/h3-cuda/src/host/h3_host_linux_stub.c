/* Minimal Linux stubs for symbols h3_dit.c needs from Apple h3_host.c.
 * Resize/vImage paths stay on Metal; this is compile/link coverage only. */
#include "h3_host.h"

#include <stdlib.h>
#include <string.h>

double h3_time_shift_slope(double sigma, double from_shift, double to_shift) {
  (void)from_shift;
  (void)to_shift;
  return sigma;
}

void h3_layout_free(h3_layout *layout) {
  if (!layout) return;
  free(layout->segments);
  free(layout->positions);
  memset(layout, 0, sizeof(*layout));
}

int h3_res_step(float *output, const float *sample, const float *denoised,
                const float *old_denoised, size_t count, const float *sigmas,
                int step, int total_steps) {
  (void)old_denoised;
  (void)sigmas;
  (void)step;
  (void)total_steps;
  if (!output || !sample || !denoised) return 0;
  for (size_t i = 0; i < count; i++)
    output[i] = sample[i] + (sample[i] - denoised[i]);
  return 1;
}

void h3_const_denoised_from_host_velocity(float *denoised, const float *sample,
                                          const float *velocity, size_t count,
                                          float sigma) {
  if (!denoised || !sample || !velocity) return;
  for (size_t i = 0; i < count; i++)
    denoised[i] = sample[i] + sigma * velocity[i];
}

int h3_euler_velocity_step(float *sample, const float *velocity, size_t count,
                           float sigma, float sigma_next) {
  if (!sample || !velocity) return 0;
  if (!(sigma > sigma_next)) return 0;
  float dt = sigma_next - sigma;
  for (size_t i = 0; i < count; i++) sample[i] += velocity[i] * dt;
  return 1;
}
