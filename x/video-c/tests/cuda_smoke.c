/* cuBLAS GEMM smoke through wan_backend CUDA. Lab only — no prod ports. */
#include "wan_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
  const int M = 256, N = 256, K = 256;
  wan_backend *b = wan_backend_cuda_create(0);
  if (!b) {
    fprintf(stderr, "FAIL: wan_backend_cuda_create (no GPU?)\n");
    return 1;
  }

  size_t a_bytes = (size_t)M * K * sizeof(float);
  size_t b_bytes = (size_t)K * N * sizeof(float);
  size_t y_bytes = (size_t)M * N * sizeof(float);
  float *A = malloc(a_bytes);
  float *B = malloc(b_bytes);
  float *Y = malloc(y_bytes);
  float *Yref = malloc(y_bytes);
  if (!A || !B || !Y || !Yref) return 1;
  for (int i = 0; i < M * K; i++) A[i] = (float)((i % 17) - 8) * 0.01f;
  for (int i = 0; i < K * N; i++) B[i] = (float)((i % 13) - 6) * 0.01f;

  if (b->vt->buf_put(b, "A", A, a_bytes) || b->vt->buf_put(b, "B", B, b_bytes)) {
    fprintf(stderr, "FAIL buf_put\n");
    return 1;
  }

  /* Warmup: first cuBLAS call pays JIT / context tax — not a wall metric. */
  if (b->vt->gemm_f32(b, "A", "B", "Y", M, N, K) || b->vt->sync(b)) {
    fprintf(stderr, "FAIL gemm warmup\n");
    return 1;
  }

  double t0 = now_s();
  if (b->vt->gemm_f32(b, "A", "B", "Y", M, N, K) || b->vt->sync(b)) {
    fprintf(stderr, "FAIL gemm\n");
    return 1;
  }
  double ms = (now_s() - t0) * 1e3;
  if (b->vt->buf_get(b, "Y", Y, y_bytes)) {
    fprintf(stderr, "FAIL buf_get\n");
    return 1;
  }

  /* Host reference (small). */
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float s = 0;
      for (int k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
      Yref[i * N + j] = s;
    }
  float max_abs = 0;
  for (int i = 0; i < M * N; i++) {
    float d = fabsf(Y[i] - Yref[i]);
    if (d > max_abs) max_abs = d;
  }
  if (max_abs > 1e-3f) {
    fprintf(stderr, "FAIL gemm parity max_abs=%g\n", max_abs);
    return 1;
  }

  printf("ok: cuda GEMM %dx%dx%d max_abs=%g wall=%.3fms peak_dev=%zu bytes backend=%s\n",
         M, N, K, max_abs, ms, b->vt->device_bytes(b), b->vt->name);
  wan_backend_destroy(b);
  free(A);
  free(B);
  free(Y);
  free(Yref);
  return 0;
}
