/* Real-weight Qwen3-VL-4B host TE smoke (skip if pack absent). */
#include "h3_qwen_te_4b.h"
#include "h3_qwen_te_host.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
  char dir[768];
  const char *env = getenv("H3_QWEN_TE_DIR");
  if (env && env[0])
    snprintf(dir, sizeof(dir), "%s", env);
  else {
    const char *home = getenv("HOME");
    if (!home) {
      printf("test_h3_qwen_te_4b SKIP (no HOME)\n");
      return 0;
    }
    snprintf(dir, sizeof(dir), "%s/.zerollama/models/Qwen3-VL-4B-Instruct",
             home);
  }
  char shard[900];
  snprintf(shard, sizeof(shard), "%s/model-00001-of-00002.safetensors", dir);
  if (access(shard, R_OK) != 0) {
    printf("test_h3_qwen_te_4b SKIP (no weights at %s)\n", dir);
    return 0;
  }
  const uint32_t ids[] = {32, 2518, 38835, 11435, 1526, 11794};
  const size_t n = 6;
  float *h = (float *)calloc(n * (size_t)H3_QWEN_TE_HIDDEN_4B, sizeof(float));
  if (!h)
    return 1;
  char error[1024];
  if (!h3_qwen_te_4b_forward(dir, ids, n, NULL, 1, h, error, sizeof(error))) {
    fprintf(stderr, "FAIL 4B forward: %s\n", error);
    free(h);
    return 1;
  }
  double sum = 0.0, sq = 0.0;
  size_t N = n * (size_t)H3_QWEN_TE_HIDDEN_4B;
  int finite = 1, nonzero = 0;
  for (size_t i = 0; i < N; i++) {
    if (!isfinite(h[i]))
      finite = 0;
    if (fabsf(h[i]) > 1e-8f)
      nonzero = 1;
    sum += h[i];
    sq += (double)h[i] * h[i];
  }
  if (!finite || !nonzero) {
    fprintf(stderr, "FAIL hidden finite=%d nonzero=%d\n", finite, nonzero);
    free(h);
    return 1;
  }
  printf("test_h3_qwen_te_4b OK n=%zu dim=%d mean=%.6g rms=%.6g\n", n,
         H3_QWEN_TE_HIDDEN_4B, sum / (double)N, sqrt(sq / (double)N));
  free(h);
  return 0;
}
