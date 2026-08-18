#include "sched_unipc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int approx(float a, float b) { return fabsf(a - b) < 1e-5f; }

int main(void) {
  float s0 = sched_unipc_warp_sigma(0.5f, 1.0f);
  if (!approx(s0, 0.5f)) {
    fprintf(stderr, "shift=1 warp failed: %f\n", s0);
    return 1;
  }

  float s1 = sched_unipc_warp_sigma(0.5f, 3.0f);
  float expect = 3.0f * 0.5f / (1.0f + 2.0f * 0.5f);
  if (!approx(s1, expect)) {
    fprintf(stderr, "shift=3 warp failed: got %f want %f\n", s1, expect);
    return 1;
  }

  sched_unipc *sched = sched_unipc_create(4, 2.0f);
  if (!sched) {
    fprintf(stderr, "sched create failed\n");
    return 1;
  }
  if (sched_unipc_num_sigmas(sched) != 5) {
    fprintf(stderr, "bad sigma count\n");
    return 1;
  }
  const float *sig = sched_unipc_sigmas(sched);
  if (sig[0] < sig[4]) {
    fprintf(stderr, "sigmas not decreasing\n");
    return 1;
  }
  /* Final sigma is 0 (Wan final_sigmas_type=zero). */
  if (!approx(sig[4], 0.0f)) {
    fprintf(stderr, "final sigma want 0 got %f\n", sig[4]);
    return 1;
  }
  /* First sigma is warped linspace start (=warp(1-1/1000)). */
  float w1 = sched_unipc_warp_sigma(1.0f - 1.0f / 1000.0f, 2.0f);
  if (!approx(sig[0], w1)) {
    fprintf(stderr, "sigma0 want %f got %f\n", w1, sig[0]);
    return 1;
  }

  float sample[2] = {1.0f, 2.0f};
  float vs[4][2] = {{0.5f, -0.5f}, {1.0f, 0.0f}, {0.25f, 0.75f}, {-0.1f, 0.2f}};
  for (int st = 0; st < 4; st++) {
    if (sched_unipc_step(sched, st, vs[st], sample, 2) != 0) {
      fprintf(stderr, "step%d (UniPC order≤3) failed\n", st);
      return 1;
    }
    if (!isfinite(sample[0]) || !isfinite(sample[1])) {
      fprintf(stderr, "non-finite sample after step%d\n", st);
      return 1;
    }
  }

  float a[2] = {0.0f, 1.0f};
  float b[2] = {1.0f, 2.0f};
  float out[2];
  sched_unipc_cfg_combine(a, b, out, 2, 2.0f);
  if (!approx(out[0], 2.0f) || !approx(out[1], 3.0f)) {
    fprintf(stderr, "cfg combine failed\n");
    return 1;
  }

  sched_unipc_destroy(sched);
  fprintf(stderr, "test_sched_unipc OK\n");
  return 0;
}
