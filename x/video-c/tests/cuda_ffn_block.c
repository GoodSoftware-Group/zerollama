/* DiT FFN half-block on wan_backend + dit_pager — phase-2 unlock.
 * LN → AdaLN affine → GEMM up → GELU → GEMM down → gated residual.
 * See docs/cuda-uma-toolkit.md */
#include "dit_pager.h"
#include "dit_resident.h"
#include "wan_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
  N_LAYERS = 4,
  T = 64,
  D = 256,
  FFN = 1024
};

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_mat(float *W, int rows, int cols, unsigned seed) {
  for (int i = 0; i < rows * cols; i++)
    W[i] = (float)(((int)seed * 17 + i) % 19 - 9) * 0.01f;
}

static float cosine(const float *a, const float *b, size_t n) {
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < n; i++) {
    dot += (double)a[i] * b[i];
    na += (double)a[i] * a[i];
    nb += (double)b[i] * b[i];
  }
  if (na < 1e-30 || nb < 1e-30) return 0.f;
  return (float)(dot / (sqrt(na) * sqrt(nb)));
}

static void bank_wu(char *o, size_t n, int L) { snprintf(o, n, "Wu_L%d", L); }
static void bank_wd(char *o, size_t n, int L) { snprintf(o, n, "Wd_L%d", L); }

static int ffn_half(wan_backend *b, dit_pager *pg, float *Wu, float *Wd,
                    size_t wu_b, size_t wd_b, unsigned layer) {
  dit_pager_stats before = dit_pager_get_stats(pg);
  int evicted = -1;
  int slot = dit_pager_touch(pg, layer, &evicted);
  if (slot < 0) return -1;
  dit_pager_stats after = dit_pager_get_stats(pg);
  int was_hit = after.hits > before.hits;

  char bu[64], bd[64];
  bank_wu(bu, sizeof(bu), (int)layer);
  bank_wd(bd, sizeof(bd), (int)layer);

  if (!was_hit) {
    if (evicted >= 0) {
      char ou[64], od[64];
      bank_wu(ou, sizeof(ou), evicted);
      bank_wd(od, sizeof(od), evicted);
      if (b->vt->bank_evict(b, ou) || b->vt->bank_evict(b, od)) return -2;
    }
    fill_mat(Wu, D, FFN, layer * 3 + 1);
    fill_mat(Wd, FFN, D, layer * 3 + 2);
    if (b->vt->bank_put(b, bu, Wu, wu_b) || b->vt->bank_put(b, bd, Wd, wd_b))
      return -3;
    dit_pager_set_slot_bytes(pg, slot, wu_b + wd_b);
  }

  if (b->vt->bank_bind(b, bu, "Wu") || b->vt->bank_bind(b, bd, "Wd")) return -4;

  /* LN → AdaLN → up → GELU → down → gated residual into Y */
  if (b->vt->layernorm(b, "X", "LN", NULL, T, D)) return -5;
  if (b->vt->affine_mul_add(b, "LN", "AD", "SC", "SH", T, D)) return -6;
  if (b->vt->gemm_f32(b, "AD", "Wu", "MID", T, FFN, D)) return -7;
  if (b->vt->gelu_tanh(b, "MID", "GELU", (size_t)T * FFN)) return -8;
  if (b->vt->gemm_f32(b, "GELU", "Wd", "DELTA", T, D, FFN)) return -9;
  if (b->vt->gated_residual(b, "X", "DELTA", "GATE", "Y", T, D)) return -10;
  if (b->vt->sync(b)) return -11;
  return was_hit ? 1 : 0;
}

static int run_backend(wan_backend *b, const char *tag, float *Yout) {
  int n_resident = wan_dit_resident_slots();
  if (n_resident <= 0) n_resident = 2;

  dit_pager *pg = dit_pager_create((unsigned)n_resident);
  if (!pg) return 1;

  size_t x_b = (size_t)T * D * sizeof(float);
  size_t d_b = (size_t)D * sizeof(float);
  size_t wu_b = (size_t)D * FFN * sizeof(float);
  size_t wd_b = (size_t)FFN * D * sizeof(float);
  float *X = malloc(x_b);
  float *SC = malloc(d_b);
  float *SH = malloc(d_b);
  float *GATE = malloc(d_b);
  float *Wu = malloc(wu_b);
  float *Wd = malloc(wd_b);
  float *Y = malloc(x_b);
  if (!X || !SC || !SH || !GATE || !Wu || !Wd || !Y) return 1;

  for (int i = 0; i < T * D; i++) X[i] = (float)((i % 13) - 6) * 0.02f;
  for (int i = 0; i < D; i++) {
    SC[i] = (float)((i % 5) - 2) * 0.01f;
    SH[i] = (float)((i % 7) - 3) * 0.01f;
    GATE[i] = 0.5f + (float)(i % 3) * 0.1f;
  }
  if (b->vt->buf_put(b, "X", X, x_b) || b->vt->buf_put(b, "SC", SC, d_b) ||
      b->vt->buf_put(b, "SH", SH, d_b) || b->vt->buf_put(b, "GATE", GATE, d_b))
    return 1;

  size_t weight_all = (size_t)N_LAYERS * (wu_b + wd_b);
  size_t budget = (size_t)n_resident * (wu_b + wd_b) + x_b * 6 + d_b * 3;
  /* activations: X LN AD MID GELU DELTA Y ≈ rough; peak tracked by backend */

  double t0 = now_s();
  for (int L = 0; L < 2; L++)
    if (ffn_half(b, pg, Wu, Wd, wu_b, wd_b, (unsigned)L) < 0) return 1;
  if (ffn_half(b, pg, Wu, Wd, wu_b, wd_b, 0) != 1) {
    fprintf(stderr, "FAIL %s expected hit\n", tag);
    return 1;
  }
  for (int L = 2; L < N_LAYERS; L++)
    if (ffn_half(b, pg, Wu, Wd, wu_b, wd_b, (unsigned)L) < 0) return 1;
  double ms = (now_s() - t0) * 1e3;

  if (b->vt->buf_get(b, "Y", Y, x_b)) return 1;
  memcpy(Yout, Y, x_b);

  dit_pager_stats st = dit_pager_get_stats(pg);
  size_t peak_pager = dit_pager_peak_bytes(pg);
  size_t peak_dev = b->vt->device_bytes(b);

  printf("ok: ffn-block backend=%s N=%d hits=%llu miss=%llu evict=%llu "
         "pager_peak=%.2fMiB backend_peak=%.2fMiB wall=%.3fms\n",
         tag, n_resident, (unsigned long long)st.hits,
         (unsigned long long)st.misses, (unsigned long long)st.evictions,
         peak_pager / (1024.0 * 1024.0), peak_dev / (1024.0 * 1024.0), ms);

  if (peak_pager >= (size_t)(0.8 * weight_all)) {
    fprintf(stderr, "FAIL %s pager peak\n", tag);
    return 1;
  }
  if (st.evictions < 1 || st.hits < 1) {
    fprintf(stderr, "FAIL %s residency stats\n", tag);
    return 1;
  }
  (void)budget;

  dit_pager_destroy(pg);
  free(X);
  free(SC);
  free(SH);
  free(GATE);
  free(Wu);
  free(Wd);
  free(Y);
  return 0;
}

int main(void) {
  size_t x_b = (size_t)T * D * sizeof(float);
  float *Yh = malloc(x_b);
  float *Yc = malloc(x_b);
  if (!Yh || !Yc) return 1;

  wan_backend *host = wan_backend_host_create();
  wan_backend *cuda = wan_backend_cuda_create(0);
  if (!host || !cuda) {
    fprintf(stderr, "FAIL create backends\n");
    return 1;
  }
  if (run_backend(host, "host", Yh)) return 1;
  if (run_backend(cuda, "cuda", Yc)) return 1;

  float cos = cosine(Yh, Yc, (size_t)T * D);
  printf("ok: ffn-block host↔cuda cosine=%.6f\n", cos);
  if (cos < 0.999f) {
    fprintf(stderr, "FAIL cosine %g < 0.999\n", cos);
    return 1;
  }

  wan_backend_destroy(host);
  wan_backend_destroy(cuda);
  free(Yh);
  free(Yc);
  return 0;
}
