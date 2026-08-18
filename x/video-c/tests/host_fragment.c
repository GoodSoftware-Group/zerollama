/* Host backend + dit_pager residency contract (no CUDA). Same kill as cuda_fragment. */
#include "dit_pager.h"
#include "dit_resident.h"
#include "wan_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_LAYERS = 8, T = 32, M = 64, K = 64 };

static void fill_layer_w(float *W, unsigned layer) {
  for (int k = 0; k < K; k++)
    for (int m = 0; m < M; m++)
      W[k * M + m] =
          (float)(((int)layer * 31 + k * M + m) % 19 - 9) * 0.01f;
}

static void layer_bank_name(char *out, size_t n, int layer) {
  snprintf(out, n, "W_L%d", layer);
}

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
  return was_hit ? 1 : 0;
}

int main(void) {
  int n_resident = wan_dit_resident_slots();
  if (n_resident <= 0) n_resident = 2;

  wan_backend *b = wan_backend_host_create();
  if (!b) return 1;
  dit_pager *pg = dit_pager_create((unsigned)n_resident);
  if (!pg) return 1;

  size_t w_bytes = (size_t)M * K * sizeof(float);
  size_t x_bytes = (size_t)T * K * sizeof(float);
  size_t y_bytes = (size_t)T * M * sizeof(float);
  float *X = malloc(x_bytes);
  float *W = malloc(w_bytes);
  if (!X || !W) return 1;
  for (int i = 0; i < T * K; i++) X[i] = 0.01f * (float)(i % 7);
  if (b->vt->buf_put(b, "X", X, x_bytes)) return 1;

  size_t weight_all = N_LAYERS * w_bytes;
  size_t budget = (size_t)n_resident * w_bytes + x_bytes + y_bytes;

  for (int layer = 0; layer < 2; layer++)
    if (run_layer(b, pg, W, w_bytes, (unsigned)layer) < 0) return 1;
  if (run_layer(b, pg, W, w_bytes, 0) != 1) {
    fprintf(stderr, "FAIL expected hit\n");
    return 1;
  }
  for (int layer = 2; layer < N_LAYERS; layer++)
    if (run_layer(b, pg, W, w_bytes, (unsigned)layer) < 0) return 1;

  dit_pager_stats st = dit_pager_get_stats(pg);
  size_t peak_pager = dit_pager_peak_bytes(pg);
  size_t peak_dev = b->vt->device_bytes(b);

  printf("ok: host-fragment N=%d hits=%llu misses=%llu evictions=%llu "
         "pager_peak=%zu backend_peak=%zu budget=%zu\n",
         n_resident, (unsigned long long)st.hits, (unsigned long long)st.misses,
         (unsigned long long)st.evictions, peak_pager, peak_dev, budget);

  if (peak_pager >= (size_t)(0.8 * weight_all)) return 1;
  if (peak_dev > budget) {
    fprintf(stderr, "FAIL backend_peak %zu > budget %zu\n", peak_dev, budget);
    return 1;
  }
  if (st.evictions < 1 || st.hits < 1) return 1;

  dit_pager_destroy(pg);
  wan_backend_destroy(b);
  free(X);
  free(W);
  return 0;
}
