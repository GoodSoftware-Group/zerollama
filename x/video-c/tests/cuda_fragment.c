/* DiT FFN-linear fragment + dit_pager — unlock proof with bank_evict coupling.
 * See docs/cuda-uma-toolkit.md */
#include "dit_pager.h"
#include "dit_resident.h"
#include "wan_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
  N_LAYERS = 8,
  T = 64,
  M = 256, /* out features */
  K = 256  /* in features */
};

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_layer_w(float *W, unsigned layer) {
  /* Row-major [K, M] so Y[T,M] = X[T,K] @ W[K,M]. */
  for (int k = 0; k < K; k++)
    for (int m = 0; m < M; m++)
      W[k * M + m] =
          (float)(((int)layer * 31 + k * M + m) % 19 - 9) * 0.01f;
}

static void layer_bank_name(char *out, size_t n, int layer) {
  snprintf(out, n, "W_L%d", layer);
}

/*
 * Touch layer. On miss: bank_evict(evicted) if any, bank_put(W_L{layer}).
 * Always bank_bind → gemm. Hits must not grow device peak.
 */
static int run_layer(wan_backend *b, dit_pager *pg, float *W, size_t w_bytes,
                     unsigned layer) {
  dit_pager_stats before = dit_pager_get_stats(pg);
  int evicted = -1;
  int slot = dit_pager_touch(pg, layer, &evicted);
  if (slot < 0) return -1;
  dit_pager_stats after = dit_pager_get_stats(pg);
  int was_hit = after.hits > before.hits;

  char bank[64];
  layer_bank_name(bank, sizeof(bank), (int)layer);

  if (!was_hit) {
    if (evicted >= 0) {
      char old[64];
      layer_bank_name(old, sizeof(old), evicted);
      if (b->vt->bank_evict(b, old) != 0) return -2;
    }
    fill_layer_w(W, layer);
    if (b->vt->bank_put(b, bank, W, w_bytes) != 0) return -3;
    dit_pager_set_slot_bytes(pg, slot, w_bytes);
  }

  if (b->vt->bank_bind(b, bank, "W") != 0) return -4;
  if (b->vt->gemm_f32(b, "X", "W", "Y", T, M, K) != 0) return -5;
  if (b->vt->sync(b) != 0) return -6;
  return was_hit ? 1 : 0;
}

int main(void) {
  int n_resident = wan_dit_resident_slots();
  if (n_resident <= 0) n_resident = 2; /* lab default when env unset */

  wan_backend *b = wan_backend_cuda_create(0);
  if (!b) {
    fprintf(stderr, "FAIL: no CUDA backend\n");
    return 1;
  }
  dit_pager *pg = dit_pager_create((unsigned)n_resident);
  if (!pg) return 1;

  size_t w_bytes = (size_t)M * K * sizeof(float);
  size_t x_bytes = (size_t)T * K * sizeof(float);
  size_t y_bytes = (size_t)T * M * sizeof(float);
  float *X = malloc(x_bytes);
  float *W = malloc(w_bytes);
  float *Y = malloc(y_bytes);
  if (!X || !W || !Y) return 1;
  for (int i = 0; i < T * K; i++) X[i] = (float)((i % 11) - 5) * 0.02f;
  if (b->vt->buf_put(b, "X", X, x_bytes)) return 1;

  size_t load_all = N_LAYERS * w_bytes + x_bytes + y_bytes;
  size_t weight_all = N_LAYERS * w_bytes;
  size_t budget = (size_t)n_resident * w_bytes + x_bytes + y_bytes;

  double t0 = now_s();
  for (int layer = 0; layer < 2; layer++) {
    if (run_layer(b, pg, W, w_bytes, (unsigned)layer) < 0) {
      fprintf(stderr, "FAIL warm layer %d\n", layer);
      return 1;
    }
  }
  /* Intentional hit on layer 0 while still resident. */
  {
    int rc = run_layer(b, pg, W, w_bytes, 0);
    if (rc != 1) {
      fprintf(stderr, "FAIL expected hit on layer 0 (rc=%d)\n", rc);
      return 1;
    }
  }
  for (int layer = 2; layer < N_LAYERS; layer++) {
    if (run_layer(b, pg, W, w_bytes, (unsigned)layer) < 0) {
      fprintf(stderr, "FAIL layer %d\n", layer);
      return 1;
    }
  }
  double ms = (now_s() - t0) * 1e3;
  if (b->vt->buf_get(b, "Y", Y, y_bytes)) return 1;

  dit_pager_stats st = dit_pager_get_stats(pg);
  size_t peak_pager = dit_pager_peak_bytes(pg);
  size_t peak_dev = b->vt->device_bytes(b);

  printf("ok: dit-fragment layers=%d N=%d hits=%llu misses=%llu evictions=%llu\n",
         N_LAYERS, n_resident, (unsigned long long)st.hits,
         (unsigned long long)st.misses, (unsigned long long)st.evictions);
  printf("  pager_peak=%zu bytes (%.2f MiB) backend_peak=%zu bytes (%.2f MiB)\n",
         peak_pager, peak_pager / (1024.0 * 1024.0), peak_dev,
         peak_dev / (1024.0 * 1024.0));
  printf("  load_all_est=%zu bytes (%.2f MiB) budget=%zu wall=%.3fms Y[0]=%g "
         "WAN_DIT_RESIDENT→%d\n",
         load_all, load_all / (1024.0 * 1024.0), budget, ms, Y[0], n_resident);

  if (peak_pager >= (size_t)(0.8 * weight_all)) {
    fprintf(stderr, "FAIL kill: pager_peak %zu >= 80%% of weight_all %zu\n",
            peak_pager, weight_all);
    return 1;
  }
  if (peak_dev > budget) {
    fprintf(stderr,
            "FAIL kill: backend_peak %zu > N-slot budget %zu (bank_evict not "
            "wired?)\n",
            peak_dev, budget);
    return 1;
  }
  if (st.evictions < 1) {
    fprintf(stderr, "FAIL expected evictions with N=%d and %d layers\n",
            n_resident, N_LAYERS);
    return 1;
  }
  if (st.hits < 1) {
    fprintf(stderr, "FAIL expected at least one pager hit\n");
    return 1;
  }

  dit_pager_destroy(pg);
  wan_backend_destroy(b);
  free(X);
  free(W);
  free(Y);
  return 0;
}
