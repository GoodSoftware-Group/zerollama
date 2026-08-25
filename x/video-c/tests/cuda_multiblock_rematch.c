/* 30-block Wan DiT trunk on CUDA + dit_pager N=2; rematch vs PyTorch fixture.
 * Optional: one Flow UniPC step on token tensor (sched_unipc).
 * Fixture: dumps/multiblock_cuda_fixture/ from tools/gen_multiblock_cuda_fixture.py
 */
#include "backend_ops.h"
#include "dit_pager.h"
#include "dit_resident.h"
#include "safetensors_min.h"
#include "sched_unipc.h"
#include "wan_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
  N_BLOCKS = 30,
  T = 32,
  TK = 512,
  H = 12,
  HD = 128,
  D = H * HD,
  FFN = 8960,
  GT = 2,
  GH = 4,
  GW = 4
};

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
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

static int read_f32(const char *path, float *dst, size_t n) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "FAIL open %s\n", path);
    return -1;
  }
  size_t got = fread(dst, sizeof(float), n, f);
  fclose(f);
  return got == n ? 0 : -1;
}

static const char *ckpt_dir(void) {
  const char *e = getenv("WAN_CKPT");
  if (e && e[0]) return e;
  return "/root/.zerollama/third_party/wan/Wan2.1-T2V-1.3B";
}

static const char *fix_dir(void) {
  const char *e = getenv("WAN_MULTIBLOCK_FIXDIR");
  if (e && e[0]) return e;
  return "dumps/multiblock_cuda_fixture";
}

static void bank_key(char *o, size_t n, const char *pfx, int block) {
  snprintf(o, n, "%s_b%d", pfx, block);
}

static int load_linear(st_file *sf, wan_backend *b, const char *st_name,
                       const char *bank, int out, int in, float *tmp_oi,
                       float *tmp_io) {
  const st_tensor_t *t = st_find_tensor(sf, st_name);
  size_t n = (size_t)out * (size_t)in;
  if (!t || st_tensor_to_f32(sf, t, tmp_oi, n) != 0) return -1;
  wan_op_transpose_oi_f32(tmp_io, tmp_oi, out, in);
  return b->vt->bank_put(b, bank, tmp_io, n * sizeof(float));
}

static int load_vec(st_file *sf, wan_backend *b, const char *st_name,
                    const char *bank, int n, float *tmp) {
  const st_tensor_t *t = st_find_tensor(sf, st_name);
  if (!t || st_tensor_to_f32(sf, t, tmp, (size_t)n) != 0) return -1;
  return b->vt->bank_put(b, bank, tmp, (size_t)n * sizeof(float));
}

static const char *BLOCK_PFX[] = {
    "Wq",  "Wk",  "Wv",  "Wo",  "Wqc", "Wkc", "Wvc", "Woc", "Wu",  "Wd",
    "Bq",  "Bk",  "Bv",  "Bo",  "Bqc", "Bkc", "Bvc", "Boc", "Bu",  "Bd",
    "Nq",  "Nk",  "Nqc", "Nkc", "N3w", "N3b", "Mod"};
enum { N_BLOCK_PFX = 27 };

static int load_block(st_file *sf, wan_backend *b, int block, float *tmp_oi,
                      float *tmp_io, float *tmp_v) {
  char st[128], bk[64];
  #define LOAD_LIN(sfx, pfx, out, in)                                          \
    do {                                                                       \
      snprintf(st, sizeof(st), "blocks.%d." sfx, block);                       \
      bank_key(bk, sizeof(bk), pfx, block);                                    \
      if (load_linear(sf, b, st, bk, out, in, tmp_oi, tmp_io)) return -1;      \
    } while (0)
  #define LOAD_V(sfx, pfx, n)                                                  \
    do {                                                                       \
      snprintf(st, sizeof(st), "blocks.%d." sfx, block);                       \
      bank_key(bk, sizeof(bk), pfx, block);                                    \
      if (load_vec(sf, b, st, bk, n, tmp_v)) return -1;                        \
    } while (0)
  LOAD_LIN("self_attn.q.weight", "Wq", D, D);
  LOAD_LIN("self_attn.k.weight", "Wk", D, D);
  LOAD_LIN("self_attn.v.weight", "Wv", D, D);
  LOAD_LIN("self_attn.o.weight", "Wo", D, D);
  LOAD_LIN("cross_attn.q.weight", "Wqc", D, D);
  LOAD_LIN("cross_attn.k.weight", "Wkc", D, D);
  LOAD_LIN("cross_attn.v.weight", "Wvc", D, D);
  LOAD_LIN("cross_attn.o.weight", "Woc", D, D);
  LOAD_LIN("ffn.0.weight", "Wu", FFN, D);
  LOAD_LIN("ffn.2.weight", "Wd", D, FFN);
  LOAD_V("self_attn.q.bias", "Bq", D);
  LOAD_V("self_attn.k.bias", "Bk", D);
  LOAD_V("self_attn.v.bias", "Bv", D);
  LOAD_V("self_attn.o.bias", "Bo", D);
  LOAD_V("cross_attn.q.bias", "Bqc", D);
  LOAD_V("cross_attn.k.bias", "Bkc", D);
  LOAD_V("cross_attn.v.bias", "Bvc", D);
  LOAD_V("cross_attn.o.bias", "Boc", D);
  LOAD_V("ffn.0.bias", "Bu", FFN);
  LOAD_V("ffn.2.bias", "Bd", D);
  LOAD_V("self_attn.norm_q.weight", "Nq", D);
  LOAD_V("self_attn.norm_k.weight", "Nk", D);
  LOAD_V("cross_attn.norm_q.weight", "Nqc", D);
  LOAD_V("cross_attn.norm_k.weight", "Nkc", D);
  LOAD_V("norm3.weight", "N3w", D);
  LOAD_V("norm3.bias", "N3b", D);
  snprintf(st, sizeof(st), "blocks.%d.modulation", block);
  {
    const st_tensor_t *t = st_find_tensor(sf, st);
    if (!t || st_tensor_to_f32(sf, t, tmp_v, (size_t)6 * D) != 0) return -1;
    bank_key(bk, sizeof(bk), "Mod", block);
    if (b->vt->bank_put(b, bk, tmp_v, (size_t)6 * D * sizeof(float))) return -1;
  }
  #undef LOAD_LIN
  #undef LOAD_V
  return 0;
}

static int evict_block(wan_backend *b, int block) {
  for (int i = 0; i < N_BLOCK_PFX; i++) {
    char k[64];
    bank_key(k, sizeof(k), BLOCK_PFX[i], block);
    if (b->vt->bank_evict(b, k)) return -1;
  }
  return 0;
}

static int bind_block(wan_backend *b, int block) {
  for (int i = 0; i < N_BLOCK_PFX; i++) {
    char k[64];
    bank_key(k, sizeof(k), BLOCK_PFX[i], block);
    if (b->vt->bank_bind(b, k, BLOCK_PFX[i])) return -1;
  }
  return 0;
}

static size_t block_weight_bytes(void) {
  size_t d2 = (size_t)D * D * sizeof(float);
  size_t df = (size_t)D * FFN * sizeof(float);
  size_t dv = (size_t)D * sizeof(float);
  size_t fv = (size_t)FFN * sizeof(float);
  return 8 * d2 + 2 * df + 8 * dv + fv + dv + 4 * dv + 2 * dv + 6 * dv;
}

static int run_block(wan_backend *b, dit_pager *pg, st_file *sf, const float *e0,
                     float *tmp_oi, float *tmp_io, float *tmp_v, float *mod6,
                     unsigned block) {
  int evicted = -1;
  dit_pager_stats before = dit_pager_get_stats(pg);
  int slot = dit_pager_touch(pg, block, &evicted);
  if (slot < 0) return -1;
  int was_hit = dit_pager_get_stats(pg).hits > before.hits;
  if (!was_hit) {
    if (evicted >= 0 && evict_block(b, evicted) != 0) return -2;
    if (load_block(sf, b, (int)block, tmp_oi, tmp_io, tmp_v) != 0) return -3;
    dit_pager_set_slot_bytes(pg, slot, block_weight_bytes());
  }
  if (bind_block(b, (int)block) != 0) return -4;

  size_t dbytes = (size_t)D * sizeof(float);
  if (b->vt->buf_get(b, "Mod", mod6, 6 * dbytes)) return -5;
  for (int i = 0; i < 6 * D; i++) mod6[i] += e0[i];
  if (b->vt->buf_put(b, "SH", mod6 + 0 * D, dbytes) ||
      b->vt->buf_put(b, "SC", mod6 + 1 * D, dbytes) ||
      b->vt->buf_put(b, "GSA", mod6 + 2 * D, dbytes) ||
      b->vt->buf_put(b, "SH2", mod6 + 3 * D, dbytes) ||
      b->vt->buf_put(b, "SC2", mod6 + 4 * D, dbytes) ||
      b->vt->buf_put(b, "GFF", mod6 + 5 * D, dbytes))
    return -6;

  if (b->vt->layernorm(b, "X", "LN", NULL, T, D) ||
      b->vt->affine_mul_add(b, "LN", "AD", "SC", "SH", T, D))
    return -10;
  if (b->vt->gemm_f32(b, "AD", "Wq", "Q0", T, D, D) ||
      b->vt->bias_add(b, "Q0", "Q", "Bq", T, D) ||
      b->vt->rmsnorm(b, "Q", "Qr", "Nq", T, D))
    return -11;
  if (b->vt->gemm_f32(b, "AD", "Wk", "K0", T, D, D) ||
      b->vt->bias_add(b, "K0", "K", "Bk", T, D) ||
      b->vt->rmsnorm(b, "K", "Kr", "Nk", T, D))
    return -12;
  if (b->vt->gemm_f32(b, "AD", "Wv", "V0", T, D, D) ||
      b->vt->bias_add(b, "V0", "V", "Bv", T, D))
    return -13;
  if (b->vt->rope3(b, "Qr", "Qrope", T, H, HD, GT, GH, GW) ||
      b->vt->rope3(b, "Kr", "Krope", T, H, HD, GT, GH, GW) ||
      b->vt->attn_sdpa(b, "Qrope", "Krope", "V", "Attn", T, T, H, HD) ||
      b->vt->gemm_f32(b, "Attn", "Wo", "O0", T, D, D) ||
      b->vt->bias_add(b, "O0", "Dsa", "Bo", T, D) ||
      b->vt->gated_residual(b, "X", "Dsa", "GSA", "X1", T, D))
    return -14;

  if (b->vt->layernorm(b, "X1", "LN3r", NULL, T, D) ||
      b->vt->scale_bias(b, "LN3r", "LN3", "N3w", "N3b", T, D) ||
      b->vt->gemm_f32(b, "LN3", "Wqc", "Qc0", T, D, D) ||
      b->vt->bias_add(b, "Qc0", "Qc", "Bqc", T, D) ||
      b->vt->rmsnorm(b, "Qc", "Qcr", "Nqc", T, D) ||
      b->vt->gemm_f32(b, "CTX", "Wkc", "Kc0", TK, D, D) ||
      b->vt->bias_add(b, "Kc0", "Kc", "Bkc", TK, D) ||
      b->vt->rmsnorm(b, "Kc", "Kcr", "Nkc", TK, D) ||
      b->vt->gemm_f32(b, "CTX", "Wvc", "Vc0", TK, D, D) ||
      b->vt->bias_add(b, "Vc0", "Vc", "Bvc", TK, D) ||
      b->vt->attn_sdpa(b, "Qcr", "Kcr", "Vc", "XAttn", T, TK, H, HD) ||
      b->vt->gemm_f32(b, "XAttn", "Woc", "Oc0", T, D, D) ||
      b->vt->bias_add(b, "Oc0", "Dxa", "Boc", T, D) ||
      b->vt->gated_residual(b, "X1", "Dxa", "ONES", "X2", T, D))
    return -20;

  if (b->vt->layernorm(b, "X2", "LN2", NULL, T, D) ||
      b->vt->affine_mul_add(b, "LN2", "AD2", "SC2", "SH2", T, D) ||
      b->vt->gemm_f32(b, "AD2", "Wu", "MID0", T, FFN, D) ||
      b->vt->bias_add(b, "MID0", "MID", "Bu", T, FFN) ||
      b->vt->gelu_tanh(b, "MID", "GELU", (size_t)T * FFN) ||
      b->vt->gemm_f32(b, "GELU", "Wd", "D0", T, D, FFN) ||
      b->vt->bias_add(b, "D0", "Dff", "Bd", T, D) ||
      b->vt->gated_residual(b, "X2", "Dff", "GFF", "Y", T, D) ||
      b->vt->sync(b))
    return -30;

  size_t xb = (size_t)T * D * sizeof(float);
  float *yhost = malloc(xb);
  if (!yhost) return -40;
  if (b->vt->buf_get(b, "Y", yhost, xb) || b->vt->buf_put(b, "X", yhost, xb)) {
    free(yhost);
    return -41;
  }
  free(yhost);
  return was_hit ? 1 : 0;
}

int main(void) {
  const char *fix = fix_dir();
  char path[1024];
  snprintf(path, sizeof(path), "%s/diffusion_pytorch_model.safetensors",
           ckpt_dir());
  st_file *sf = st_open(path);
  if (!sf) {
    fprintf(stderr, "FAIL open %s\n", path);
    return 1;
  }

  int n_resident = wan_dit_resident_slots();
  if (n_resident <= 0) n_resident = 2;
  wan_backend *b = wan_backend_cuda_create(0);
  dit_pager *pg = dit_pager_create((unsigned)n_resident);
  if (!b || !pg) return 1;

  size_t xb = (size_t)T * D;
  size_t cb = (size_t)TK * D;
  size_t e0n = (size_t)6 * D;
  size_t oi = (size_t)FFN * D * sizeof(float);
  if (oi < (size_t)D * D * sizeof(float)) oi = (size_t)D * D * sizeof(float);

  float *X = malloc(xb * sizeof(float));
  float *E0 = malloc(e0n * sizeof(float));
  float *CTX = malloc(cb * sizeof(float));
  float *ref = malloc(xb * sizeof(float));
  float *out = malloc(xb * sizeof(float));
  float *ones = malloc((size_t)D * sizeof(float));
  float *tmp_oi = malloc(oi);
  float *tmp_io = malloc(oi);
  float *tmp_v = malloc(e0n * sizeof(float));
  float *mod6 = malloc(e0n * sizeof(float));
  if (!X || !E0 || !CTX || !ref || !out || !ones || !tmp_oi || !tmp_io ||
      !tmp_v || !mod6)
    return 1;

  char p[1024];
  snprintf(p, sizeof(p), "%s/x_in.f32", fix);
  if (read_f32(p, X, xb)) return 1;
  snprintf(p, sizeof(p), "%s/e0.f32", fix);
  if (read_f32(p, E0, e0n)) return 1;
  snprintf(p, sizeof(p), "%s/context.f32", fix);
  if (read_f32(p, CTX, cb)) return 1;
  snprintf(p, sizeof(p), "%s/py_after_blocks.f32", fix);
  if (read_f32(p, ref, xb)) return 1;

  for (int i = 0; i < D; i++) ones[i] = 1.f;
  if (b->vt->buf_put(b, "X", X, xb * sizeof(float)) ||
      b->vt->buf_put(b, "CTX", CTX, cb * sizeof(float)) ||
      b->vt->buf_put(b, "ONES", ones, (size_t)D * sizeof(float)))
    return 1;

  double t0 = now_s();
  for (int bi = 0; bi < N_BLOCKS; bi++) {
    if (run_block(b, pg, sf, E0, tmp_oi, tmp_io, tmp_v, mod6, (unsigned)bi) <
        0) {
      fprintf(stderr, "FAIL block %d\n", bi);
      return 1;
    }
    if ((bi + 1) % 5 == 0 || bi + 1 == N_BLOCKS)
      fprintf(stderr, "cuda multiblock: %d/%d\n", bi + 1, N_BLOCKS);
  }
  double ms = (now_s() - t0) * 1e3;
  if (b->vt->buf_get(b, "X", out, xb * sizeof(float))) return 1;

  float cos = cosine(out, ref, xb);
  dit_pager_stats st = dit_pager_get_stats(pg);
  printf("ok: multiblock n=%d N=%d hits=%llu miss=%llu evict=%llu "
         "pager_peak=%.1fMiB wall=%.1fms cosine=%.6f\n",
         N_BLOCKS, n_resident, (unsigned long long)st.hits,
         (unsigned long long)st.misses, (unsigned long long)st.evictions,
         dit_pager_peak_bytes(pg) / (1024.0 * 1024.0), ms, cos);
  if (cos < 0.99f) {
    fprintf(stderr, "FAIL multiblock cosine %g\n", cos);
    return 1;
  }
  if (st.evictions < 1) {
    fprintf(stderr, "FAIL expected pager evictions\n");
    return 1;
  }

  /* Token-space UniPC step0 vs FlowUniPC fixture (composition rematch). */
  float *py_u = malloc(xb * sizeof(float));
  float *sample = malloc(xb * sizeof(float));
  float *pred = malloc(xb * sizeof(float));
  if (!py_u || !sample || !pred) return 1;
  snprintf(p, sizeof(p), "%s/py_unipc_step0.f32", fix);
  if (read_f32(p, py_u, xb)) {
    fprintf(stderr, "FAIL need py_unipc_step0.f32 — run make multiblock-fixture\n");
    return 1;
  }
  memcpy(sample, X, xb * sizeof(float));
  /* Isolate UniPC: drive step0 from PyTorch trunk output (already rematched). */
  memcpy(pred, ref, xb * sizeof(float));
  sched_unipc *sched = sched_unipc_create(4, 5.0f);
  if (!sched) return 1;
  if (sched_unipc_step(sched, 0, pred, sample, xb) != 0) {
    fprintf(stderr, "FAIL sched_unipc_step\n");
    return 1;
  }
  float cos_u = cosine(sample, py_u, xb);
  printf("ok: unipc step0 cosine=%.6f sample[0]=%g\n", cos_u, sample[0]);
  if (cos_u < 0.99f) {
    fprintf(stderr, "FAIL unipc step0 cosine %g\n", cos_u);
    return 1;
  }
  sched_unipc_destroy(sched);

  snprintf(p, sizeof(p), "%s/c_after_blocks.f32", fix);
  FILE *f = fopen(p, "wb");
  if (f) {
    fwrite(out, sizeof(float), xb, f);
    fclose(f);
  }

  wan_backend_destroy(b);
  dit_pager_destroy(pg);
  st_close(sf);
  free(X);
  free(E0);
  free(CTX);
  free(ref);
  free(out);
  free(ones);
  free(tmp_oi);
  free(tmp_io);
  free(tmp_v);
  free(mod6);
  free(py_u);
  free(sample);
  free(pred);
  return 0;
}
