/* Synthetic DiT block0: self-attn + FFN on wan_backend + dit_pager.
 * LN→AdaLN→QKV→RMS→RoPE3→SDPA→O→gate + FFN half-block.
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
  T = 16,
  H = 4,
  HD = 32,
  D = H * HD, /* 128 */
  FFN = 256,
  GT = 2,
  GH = 2,
  GW = 4 /* 2*2*4 == T */
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

static void key(char *o, size_t n, const char *pfx, int L) {
  snprintf(o, n, "%s_L%d", pfx, L);
}

static int load_layer_weights(wan_backend *b, int layer, float *tmp, size_t d2,
                              size_t df, size_t hd) {
  char k[64];
  fill_mat(tmp, D, D, (unsigned)layer * 11 + 1);
  key(k, sizeof(k), "Wq", layer);
  if (b->vt->bank_put(b, k, tmp, d2)) return -1;
  fill_mat(tmp, D, D, (unsigned)layer * 11 + 2);
  key(k, sizeof(k), "Wk", layer);
  if (b->vt->bank_put(b, k, tmp, d2)) return -1;
  fill_mat(tmp, D, D, (unsigned)layer * 11 + 3);
  key(k, sizeof(k), "Wv", layer);
  if (b->vt->bank_put(b, k, tmp, d2)) return -1;
  fill_mat(tmp, D, D, (unsigned)layer * 11 + 4);
  key(k, sizeof(k), "Wo", layer);
  if (b->vt->bank_put(b, k, tmp, d2)) return -1;
  fill_mat(tmp, D, FFN, (unsigned)layer * 11 + 5);
  key(k, sizeof(k), "Wu", layer);
  if (b->vt->bank_put(b, k, tmp, df)) return -1;
  fill_mat(tmp, FFN, D, (unsigned)layer * 11 + 6);
  key(k, sizeof(k), "Wd", layer);
  if (b->vt->bank_put(b, k, tmp, (size_t)FFN * D * sizeof(float))) return -1;
  for (int i = 0; i < HD; i++) tmp[i] = 1.f + (float)((layer + i) % 3) * 0.01f;
  key(k, sizeof(k), "Nq", layer);
  if (b->vt->bank_put(b, k, tmp, hd)) return -1;
  key(k, sizeof(k), "Nk", layer);
  if (b->vt->bank_put(b, k, tmp, hd)) return -1;
  return 0;
}

static int evict_layer(wan_backend *b, int layer) {
  const char *pfx[] = {"Wq", "Wk", "Wv", "Wo", "Wu", "Wd", "Nq", "Nk"};
  for (int i = 0; i < 8; i++) {
    char k[64];
    key(k, sizeof(k), pfx[i], layer);
    if (b->vt->bank_evict(b, k)) return -1;
  }
  return 0;
}

static int bind_layer(wan_backend *b, int layer) {
  const char *pfx[] = {"Wq", "Wk", "Wv", "Wo", "Wu", "Wd", "Nq", "Nk"};
  const char *as[] = {"Wq", "Wk", "Wv", "Wo", "Wu", "Wd", "Nq", "Nk"};
  for (int i = 0; i < 8; i++) {
    char k[64];
    key(k, sizeof(k), pfx[i], layer);
    if (b->vt->bank_bind(b, k, as[i])) return -1;
  }
  return 0;
}

static int block_once(wan_backend *b, dit_pager *pg, float *tmp, unsigned layer) {
  dit_pager_stats before = dit_pager_get_stats(pg);
  int evicted = -1;
  int slot = dit_pager_touch(pg, layer, &evicted);
  if (slot < 0) return -1;
  int was_hit = dit_pager_get_stats(pg).hits > before.hits;

  size_t d2 = (size_t)D * D * sizeof(float);
  size_t df = (size_t)D * FFN * sizeof(float);
  size_t hd = (size_t)HD * sizeof(float);
  size_t layer_bytes = 4 * d2 + df + (size_t)FFN * D * sizeof(float) + 2 * hd;

  if (!was_hit) {
    if (evicted >= 0 && evict_layer(b, evicted) != 0) return -2;
    if (load_layer_weights(b, (int)layer, tmp, d2, df, hd) != 0) return -3;
    dit_pager_set_slot_bytes(pg, slot, layer_bytes);
  }
  if (bind_layer(b, (int)layer) != 0) return -4;

  /* Self-attn */
  if (b->vt->layernorm(b, "X", "LN", NULL, T, D)) return -10;
  if (b->vt->affine_mul_add(b, "LN", "AD", "SC", "SH", T, D)) return -11;
  if (b->vt->gemm_f32(b, "AD", "Wq", "Q", T, D, D)) return -12;
  if (b->vt->gemm_f32(b, "AD", "Wk", "K", T, D, D)) return -13;
  if (b->vt->gemm_f32(b, "AD", "Wv", "V", T, D, D)) return -14;
  if (b->vt->head_rmsnorm(b, "Q", "Qr", "Nq", T, H, HD)) return -15;
  if (b->vt->head_rmsnorm(b, "K", "Kr", "Nk", T, H, HD)) return -16;
  if (b->vt->rope3(b, "Qr", "Qrope", T, H, HD, GT, GH, GW)) return -17;
  if (b->vt->rope3(b, "Kr", "Krope", T, H, HD, GT, GH, GW)) return -18;
  if (b->vt->attn_sdpa(b, "Qrope", "Krope", "V", "Attn", T, T, H, HD))
    return -19;
  if (b->vt->gemm_f32(b, "Attn", "Wo", "Dsa", T, D, D)) return -20;
  if (b->vt->gated_residual(b, "X", "Dsa", "GSA", "X1", T, D)) return -21;

  /* FFN on X1 → Y */
  if (b->vt->layernorm(b, "X1", "LN2", NULL, T, D)) return -30;
  if (b->vt->affine_mul_add(b, "LN2", "AD2", "SC2", "SH2", T, D)) return -31;
  if (b->vt->gemm_f32(b, "AD2", "Wu", "MID", T, FFN, D)) return -32;
  if (b->vt->gelu_tanh(b, "MID", "GELU", (size_t)T * FFN)) return -33;
  if (b->vt->gemm_f32(b, "GELU", "Wd", "Dff", T, D, FFN)) return -34;
  if (b->vt->gated_residual(b, "X1", "Dff", "GFF", "Y", T, D)) return -35;
  if (b->vt->sync(b)) return -36;

  /* Next block reads X — copy Y → X for multi-layer walk */
  size_t xb = (size_t)T * D * sizeof(float);
  float *yhost = malloc(xb);
  if (!yhost) return -37;
  if (b->vt->buf_get(b, "Y", yhost, xb) || b->vt->buf_put(b, "X", yhost, xb)) {
    free(yhost);
    return -38;
  }
  free(yhost);
  return was_hit ? 1 : 0;
}

static int run_backend(wan_backend *b, const char *tag, float *Yout) {
  int n_resident = wan_dit_resident_slots();
  if (n_resident <= 0) n_resident = 2;
  dit_pager *pg = dit_pager_create((unsigned)n_resident);
  if (!pg) return 1;

  size_t xb = (size_t)T * D * sizeof(float);
  size_t db = (size_t)D * sizeof(float);
  size_t tmp_b = (size_t)D * FFN * sizeof(float);
  if (tmp_b < (size_t)D * D * sizeof(float)) tmp_b = (size_t)D * D * sizeof(float);
  float *X = malloc(xb);
  float *SC = malloc(db), *SH = malloc(db), *GSA = malloc(db);
  float *SC2 = malloc(db), *SH2 = malloc(db), *GFF = malloc(db);
  float *tmp = malloc(tmp_b);
  float *Y = malloc(xb);
  if (!X || !SC || !SH || !GSA || !SC2 || !SH2 || !GFF || !tmp || !Y) return 1;

  for (int i = 0; i < T * D; i++) X[i] = (float)((i % 13) - 6) * 0.02f;
  for (int i = 0; i < D; i++) {
    SC[i] = (float)((i % 5) - 2) * 0.01f;
    SH[i] = (float)((i % 7) - 3) * 0.01f;
    GSA[i] = 0.4f + (float)(i % 3) * 0.1f;
    SC2[i] = SC[i] * 0.5f;
    SH2[i] = SH[i] * 0.5f;
    GFF[i] = 0.6f + (float)(i % 4) * 0.05f;
  }
  if (b->vt->buf_put(b, "X", X, xb) || b->vt->buf_put(b, "SC", SC, db) ||
      b->vt->buf_put(b, "SH", SH, db) || b->vt->buf_put(b, "GSA", GSA, db) ||
      b->vt->buf_put(b, "SC2", SC2, db) || b->vt->buf_put(b, "SH2", SH2, db) ||
      b->vt->buf_put(b, "GFF", GFF, db))
    return 1;

  size_t d2 = (size_t)D * D * sizeof(float);
  size_t layer_w =
      4 * d2 + (size_t)D * FFN * sizeof(float) + (size_t)FFN * D * sizeof(float) +
      2 * (size_t)HD * sizeof(float);
  size_t weight_all = (size_t)N_LAYERS * layer_w;

  double t0 = now_s();
  for (int L = 0; L < 2; L++)
    if (block_once(b, pg, tmp, (unsigned)L) < 0) {
      fprintf(stderr, "FAIL %s warm L%d\n", tag, L);
      return 1;
    }
  if (block_once(b, pg, tmp, 0) != 1) {
    fprintf(stderr, "FAIL %s expected hit\n", tag);
    return 1;
  }
  for (int L = 2; L < N_LAYERS; L++)
    if (block_once(b, pg, tmp, (unsigned)L) < 0) {
      fprintf(stderr, "FAIL %s L%d\n", tag, L);
      return 1;
    }
  double ms = (now_s() - t0) * 1e3;

  if (b->vt->buf_get(b, "Y", Y, xb)) return 1;
  memcpy(Yout, Y, xb);

  dit_pager_stats st = dit_pager_get_stats(pg);
  printf("ok: block0 backend=%s N=%d hits=%llu miss=%llu evict=%llu "
         "pager_peak=%.2fMiB backend_peak=%.2fMiB wall=%.3fms\n",
         tag, n_resident, (unsigned long long)st.hits,
         (unsigned long long)st.misses, (unsigned long long)st.evictions,
         dit_pager_peak_bytes(pg) / (1024.0 * 1024.0),
         b->vt->device_bytes(b) / (1024.0 * 1024.0), ms);

  if (dit_pager_peak_bytes(pg) >= (size_t)(0.8 * weight_all)) {
    fprintf(stderr, "FAIL %s pager peak\n", tag);
    return 1;
  }
  if (st.evictions < 1 || st.hits < 1) {
    fprintf(stderr, "FAIL %s residency\n", tag);
    return 1;
  }

  dit_pager_destroy(pg);
  free(X);
  free(SC);
  free(SH);
  free(GSA);
  free(SC2);
  free(SH2);
  free(GFF);
  free(tmp);
  free(Y);
  return 0;
}

int main(void) {
  size_t xb = (size_t)T * D * sizeof(float);
  float *Yh = malloc(xb), *Yc = malloc(xb);
  if (!Yh || !Yc) return 1;
  wan_backend *host = wan_backend_host_create();
  wan_backend *cuda = wan_backend_cuda_create(0);
  if (!host || !cuda) {
    fprintf(stderr, "FAIL backends\n");
    return 1;
  }
  if (run_backend(host, "host", Yh) || run_backend(cuda, "cuda", Yc)) return 1;
  float cos = cosine(Yh, Yc, (size_t)T * D);
  printf("ok: block0 host↔cuda cosine=%.6f (T=%d H=%d HD=%d)\n", cos, T, H, HD);
  if (cos < 0.999f) {
    fprintf(stderr, "FAIL cosine %g\n", cos);
    return 1;
  }
  wan_backend_destroy(host);
  wan_backend_destroy(cuda);
  free(Yh);
  free(Yc);
  return 0;
}
