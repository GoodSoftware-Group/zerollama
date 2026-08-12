/* Real Wan 1.3B full attention block: self + cross + FFN on wan_backend + dit_pager.
 * WAN_CKPT or default ~/.zerollama/third_party/wan/Wan2.1-T2V-1.3B
 */
#include "backend_ops.h"
#include "dit_pager.h"
#include "dit_resident.h"
#include "safetensors_min.h"
#include "wan_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
  N_BLOCKS = 4,
  T = 32,
  TK = 64, /* synthetic text tokens (Wan text_len=512; lab uses 64) */
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

static const char *ckpt_dir(void) {
  const char *e = getenv("WAN_CKPT");
  if (e && e[0]) return e;
  return "/root/.zerollama/third_party/wan/Wan2.1-T2V-1.3B";
}

static int load_linear(st_file *sf, wan_backend *b, const char *st_name,
                       const char *bank, int out, int in, float *tmp_oi,
                       float *tmp_io) {
  const st_tensor_t *t = st_find_tensor(sf, st_name);
  size_t n = (size_t)out * (size_t)in;
  if (!t || st_tensor_to_f32(sf, t, tmp_oi, n) != 0) {
    fprintf(stderr, "FAIL load %s\n", st_name);
    return -1;
  }
  wan_op_transpose_oi_f32(tmp_io, tmp_oi, out, in);
  return b->vt->bank_put(b, bank, tmp_io, n * sizeof(float));
}

static int load_vec(st_file *sf, wan_backend *b, const char *st_name,
                    const char *bank, int n, float *tmp) {
  const st_tensor_t *t = st_find_tensor(sf, st_name);
  if (!t || st_tensor_to_f32(sf, t, tmp, (size_t)n) != 0) {
    fprintf(stderr, "FAIL load %s\n", st_name);
    return -1;
  }
  return b->vt->bank_put(b, bank, tmp, (size_t)n * sizeof(float));
}

static void bank_key(char *o, size_t n, const char *pfx, int block) {
  snprintf(o, n, "%s_b%d", pfx, block);
}

static int load_block(st_file *sf, wan_backend *b, int block, float *tmp_oi,
                      float *tmp_io, float *tmp_v) {
  char st[128], bk[64];
  #define LOAD_LIN(st_sfx, pfx, out, in)                                       \
    do {                                                                       \
      snprintf(st, sizeof(st), "blocks.%d." st_sfx, block);                    \
      bank_key(bk, sizeof(bk), pfx, block);                                    \
      if (load_linear(sf, b, st, bk, out, in, tmp_oi, tmp_io)) return -1;      \
    } while (0)
  #define LOAD_V(st_sfx, pfx, n)                                               \
    do {                                                                       \
      snprintf(st, sizeof(st), "blocks.%d." st_sfx, block);                    \
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

  /* modulation [1,6,D] → 6*D */
  snprintf(st, sizeof(st), "blocks.%d.modulation", block);
  {
    const st_tensor_t *t = st_find_tensor(sf, st);
    size_t n = (size_t)6 * D;
    if (!t || st_tensor_to_f32(sf, t, tmp_v, n) != 0) return -1;
    bank_key(bk, sizeof(bk), "Mod", block);
    if (b->vt->bank_put(b, bk, tmp_v, n * sizeof(float))) return -1;
  }
  #undef LOAD_LIN
  #undef LOAD_V
  return 0;
}

static const char *BLOCK_PFX[] = {
    "Wq",  "Wk",  "Wv",  "Wo",  "Wqc", "Wkc", "Wvc", "Woc", "Wu",  "Wd",
    "Bq",  "Bk",  "Bv",  "Bo",  "Bqc", "Bkc", "Bvc", "Boc", "Bu",  "Bd",
    "Nq",  "Nk",  "Nqc", "Nkc", "N3w", "N3b", "Mod"};
enum { N_BLOCK_PFX = 27 };

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
  /* 8×D² (self+cross QKVO) + Wu+Wd + biases + norms + Mod */
  return 8 * d2 + 2 * df + 8 * dv + fv + dv + 4 * dv + 2 * dv + 6 * dv;
}

static int run_block(wan_backend *b, dit_pager *pg, st_file *sf, float *tmp_oi,
                     float *tmp_io, float *tmp_v, unsigned block) {
  dit_pager_stats before = dit_pager_get_stats(pg);
  int evicted = -1;
  int slot = dit_pager_touch(pg, block, &evicted);
  if (slot < 0) return -1;
  int was_hit = dit_pager_get_stats(pg).hits > before.hits;

  if (!was_hit) {
    if (evicted >= 0 && evict_block(b, evicted) != 0) return -2;
    if (load_block(sf, b, (int)block, tmp_oi, tmp_io, tmp_v) != 0) return -3;
    dit_pager_set_slot_bytes(pg, slot, block_weight_bytes());
  }
  if (bind_block(b, (int)block) != 0) return -4;

  /* AdaLN scales from modulation only (e0=0 lab). Mod is [6,D]. */
  size_t dbytes = (size_t)D * sizeof(float);
  float *mod = malloc(6 * dbytes);
  if (!mod || b->vt->buf_get(b, "Mod", mod, 6 * dbytes)) {
    free(mod);
    return -5;
  }
  /* e0..e5 rows; SH=e0, SC=e1, GSA=e2, SH2=e3, SC2=e4, GFF=e5 */
  if (b->vt->buf_put(b, "SH", mod + 0 * D, dbytes) ||
      b->vt->buf_put(b, "SC", mod + 1 * D, dbytes) ||
      b->vt->buf_put(b, "GSA", mod + 2 * D, dbytes) ||
      b->vt->buf_put(b, "SH2", mod + 3 * D, dbytes) ||
      b->vt->buf_put(b, "SC2", mod + 4 * D, dbytes) ||
      b->vt->buf_put(b, "GFF", mod + 5 * D, dbytes)) {
    free(mod);
    return -6;
  }
  free(mod);

  /* Self-attn (Wan: LN → AdaLN → QKV+bias → RMS → RoPE → SDPA → O+bias → gate) */
  if (b->vt->layernorm(b, "X", "LN", NULL, T, D)) return -10;
  if (b->vt->affine_mul_add(b, "LN", "AD", "SC", "SH", T, D)) return -11;
  if (b->vt->gemm_f32(b, "AD", "Wq", "Q0", T, D, D) ||
      b->vt->bias_add(b, "Q0", "Q", "Bq", T, D))
    return -12;
  if (b->vt->gemm_f32(b, "AD", "Wk", "K0", T, D, D) ||
      b->vt->bias_add(b, "K0", "K", "Bk", T, D))
    return -13;
  if (b->vt->gemm_f32(b, "AD", "Wv", "V0", T, D, D) ||
      b->vt->bias_add(b, "V0", "V", "Bv", T, D))
    return -14;
  if (b->vt->rmsnorm(b, "Q", "Qr", "Nq", T, D)) return -15;
  if (b->vt->rmsnorm(b, "K", "Kr", "Nk", T, D)) return -16;
  if (b->vt->rope3(b, "Qr", "Qrope", T, H, HD, GT, GH, GW)) return -17;
  if (b->vt->rope3(b, "Kr", "Krope", T, H, HD, GT, GH, GW)) return -18;
  if (b->vt->attn_sdpa(b, "Qrope", "Krope", "V", "Attn", T, T, H, HD))
    return -19;
  if (b->vt->gemm_f32(b, "Attn", "Wo", "O0", T, D, D) ||
      b->vt->bias_add(b, "O0", "Dsa", "Bo", T, D))
    return -20;
  if (b->vt->gated_residual(b, "X", "Dsa", "GSA", "X1", T, D)) return -21;

  /* Cross-attn: norm3 (affine LN) → Q from x, K/V from CTX → SDPA → O → resid */
  if (b->vt->layernorm(b, "X1", "LN3r", NULL, T, D) ||
      b->vt->scale_bias(b, "LN3r", "LN3", "N3w", "N3b", T, D))
    return -40;
  if (b->vt->gemm_f32(b, "LN3", "Wqc", "Qc0", T, D, D) ||
      b->vt->bias_add(b, "Qc0", "Qc", "Bqc", T, D) ||
      b->vt->rmsnorm(b, "Qc", "Qcr", "Nqc", T, D))
    return -41;
  if (b->vt->gemm_f32(b, "CTX", "Wkc", "Kc0", TK, D, D) ||
      b->vt->bias_add(b, "Kc0", "Kc", "Bkc", TK, D) ||
      b->vt->rmsnorm(b, "Kc", "Kcr", "Nkc", TK, D))
    return -42;
  if (b->vt->gemm_f32(b, "CTX", "Wvc", "Vc0", TK, D, D) ||
      b->vt->bias_add(b, "Vc0", "Vc", "Bvc", TK, D))
    return -43;
  if (b->vt->attn_sdpa(b, "Qcr", "Kcr", "Vc", "XAttn", T, TK, H, HD)) return -44;
  if (b->vt->gemm_f32(b, "XAttn", "Woc", "Oc0", T, D, D) ||
      b->vt->bias_add(b, "Oc0", "Dxa", "Boc", T, D))
    return -45;
  if (b->vt->gated_residual(b, "X1", "Dxa", "ONES", "X2", T, D)) return -46;

  /* FFN on X2 → Y */
  if (b->vt->layernorm(b, "X2", "LN2", NULL, T, D)) return -30;
  if (b->vt->affine_mul_add(b, "LN2", "AD2", "SC2", "SH2", T, D)) return -31;
  if (b->vt->gemm_f32(b, "AD2", "Wu", "MID0", T, FFN, D) ||
      b->vt->bias_add(b, "MID0", "MID", "Bu", T, FFN))
    return -32;
  if (b->vt->gelu_tanh(b, "MID", "GELU", (size_t)T * FFN)) return -33;
  if (b->vt->gemm_f32(b, "GELU", "Wd", "D0", T, D, FFN) ||
      b->vt->bias_add(b, "D0", "Dff", "Bd", T, D))
    return -34;
  if (b->vt->gated_residual(b, "X2", "Dff", "GFF", "Y", T, D)) return -35;
  if (b->vt->sync(b)) return -36;

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

static int run_backend(wan_backend *b, st_file *sf, const char *tag,
                       float *Yout) {
  int n_resident = wan_dit_resident_slots();
  if (n_resident <= 0) n_resident = 2;
  dit_pager *pg = dit_pager_create((unsigned)n_resident);
  if (!pg) return 1;

  size_t xb = (size_t)T * D * sizeof(float);
  size_t oi = (size_t)FFN * D * sizeof(float); /* largest [out,in] */
  if (oi < (size_t)D * D * sizeof(float)) oi = (size_t)D * D * sizeof(float);
  float *X = malloc(xb);
  float *tmp_oi = malloc(oi);
  float *tmp_io = malloc(oi);
  float *tmp_v = malloc(6 * (size_t)D * sizeof(float));
  float *Y = malloc(xb);
  if (!X || !tmp_oi || !tmp_io || !tmp_v || !Y) return 1;

  for (int i = 0; i < T * D; i++) X[i] = (float)((i % 17) - 8) * 0.01f;
  if (b->vt->buf_put(b, "X", X, xb)) return 1;

  /* Sticky text context + ones gate for ungated cross residual. */
  {
    size_t cb = (size_t)TK * D * sizeof(float);
    float *ctx = malloc(cb);
    float *ones = malloc((size_t)D * sizeof(float));
    if (!ctx || !ones) return 1;
    for (int i = 0; i < TK * D; i++) ctx[i] = (float)((i % 11) - 5) * 0.02f;
    for (int i = 0; i < D; i++) ones[i] = 1.f;
    if (b->vt->buf_put(b, "CTX", ctx, cb) ||
        b->vt->buf_put(b, "ONES", ones, (size_t)D * sizeof(float))) {
      free(ctx);
      free(ones);
      return 1;
    }
    free(ctx);
    free(ones);
  }

  double t0 = now_s();
  for (int bi = 0; bi < 2; bi++)
    if (run_block(b, pg, sf, tmp_oi, tmp_io, tmp_v, (unsigned)bi) < 0) return 1;
  if (run_block(b, pg, sf, tmp_oi, tmp_io, tmp_v, 0) != 1) {
    fprintf(stderr, "FAIL %s expected hit on block0\n", tag);
    return 1;
  }
  for (int bi = 2; bi < N_BLOCKS; bi++)
    if (run_block(b, pg, sf, tmp_oi, tmp_io, tmp_v, (unsigned)bi) < 0) return 1;
  double ms = (now_s() - t0) * 1e3;

  if (b->vt->buf_get(b, "Y", Y, xb)) return 1;
  memcpy(Yout, Y, xb);

  dit_pager_stats st = dit_pager_get_stats(pg);
  size_t weight_all = (size_t)N_BLOCKS * block_weight_bytes();
  printf("ok: block0-real backend=%s N=%d hits=%llu miss=%llu evict=%llu "
         "pager_peak=%.1fMiB backend_peak=%.1fMiB wall=%.1fms (cross Tk=%d)\n",
         tag, n_resident, (unsigned long long)st.hits,
         (unsigned long long)st.misses, (unsigned long long)st.evictions,
         dit_pager_peak_bytes(pg) / (1024.0 * 1024.0),
         b->vt->device_bytes(b) / (1024.0 * 1024.0), ms, TK);

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
  free(tmp_oi);
  free(tmp_io);
  free(tmp_v);
  free(Y);
  return 0;
}

int main(void) {
  char path[1024];
  snprintf(path, sizeof(path), "%s/diffusion_pytorch_model.safetensors",
           ckpt_dir());
  st_file *sf = st_open(path);
  if (!sf) {
    fprintf(stderr, "FAIL open %s (set WAN_CKPT)\n", path);
    return 1;
  }
  fprintf(stderr, "wan-c: block0-real safetensors %s (%d tensors)\n", path,
          st_tensor_count(sf));

  size_t xb = (size_t)T * D * sizeof(float);
  float *Yh = malloc(xb), *Yc = malloc(xb);
  wan_backend *host = wan_backend_host_create();
  wan_backend *cuda = wan_backend_cuda_create(0);
  if (!Yh || !Yc || !host || !cuda) return 1;

  if (run_backend(host, sf, "host", Yh)) return 1;
  if (run_backend(cuda, sf, "cuda", Yc)) return 1;

  float cos = cosine(Yh, Yc, (size_t)T * D);
  printf("ok: block0-real host↔cuda cosine=%.6f (T=%d Tk=%d D=%d FFN=%d "
         "self+cross+FFN)\n",
         cos, T, TK, D, FFN);
  if (cos < 0.999f) {
    fprintf(stderr, "FAIL cosine %g\n", cos);
    return 1;
  }

  wan_backend_destroy(host);
  wan_backend_destroy(cuda);
  st_close(sf);
  free(Yh);
  free(Yc);
  return 0;
}
