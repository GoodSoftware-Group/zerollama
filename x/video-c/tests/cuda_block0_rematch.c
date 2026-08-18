/* Rematch CUDA DiT block0 vs PyTorch fixture (gen_block0_cuda_fixture.py).
 * Reads dumps/block0_cuda_fixture/{x_in,e0,context,py_post_*}.f32 + safetensors.
 */
#include "backend_ops.h"
#include "safetensors_min.h"
#include "wan_backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
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
  if (got != n) {
    fprintf(stderr, "FAIL read %s got %zu want %zu\n", path, got, n);
    return -1;
  }
  return 0;
}

static const char *ckpt_dir(void) {
  const char *e = getenv("WAN_CKPT");
  if (e && e[0]) return e;
  return "/root/.zerollama/third_party/wan/Wan2.1-T2V-1.3B";
}

static const char *fix_dir(void) {
  const char *e = getenv("WAN_BLOCK0_FIXDIR");
  if (e && e[0]) return e;
  return "dumps/block0_cuda_fixture";
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

static int load_block0(st_file *sf, wan_backend *b, float *tmp_oi, float *tmp_io,
                       float *tmp_v) {
  #define L(st, bank, o, i)                                                    \
    if (load_linear(sf, b, "blocks.0." st, bank, o, i, tmp_oi, tmp_io)) return -1
  #define V(st, bank, n)                                                       \
    if (load_vec(sf, b, "blocks.0." st, bank, n, tmp_v)) return -1
  L("self_attn.q.weight", "Wq", D, D);
  L("self_attn.k.weight", "Wk", D, D);
  L("self_attn.v.weight", "Wv", D, D);
  L("self_attn.o.weight", "Wo", D, D);
  L("cross_attn.q.weight", "Wqc", D, D);
  L("cross_attn.k.weight", "Wkc", D, D);
  L("cross_attn.v.weight", "Wvc", D, D);
  L("cross_attn.o.weight", "Woc", D, D);
  L("ffn.0.weight", "Wu", FFN, D);
  L("ffn.2.weight", "Wd", D, FFN);
  V("self_attn.q.bias", "Bq", D);
  V("self_attn.k.bias", "Bk", D);
  V("self_attn.v.bias", "Bv", D);
  V("self_attn.o.bias", "Bo", D);
  V("cross_attn.q.bias", "Bqc", D);
  V("cross_attn.k.bias", "Bkc", D);
  V("cross_attn.v.bias", "Bvc", D);
  V("cross_attn.o.bias", "Boc", D);
  V("ffn.0.bias", "Bu", FFN);
  V("ffn.2.bias", "Bd", D);
  V("self_attn.norm_q.weight", "Nq", D);
  V("self_attn.norm_k.weight", "Nk", D);
  V("cross_attn.norm_q.weight", "Nqc", D);
  V("cross_attn.norm_k.weight", "Nkc", D);
  V("norm3.weight", "N3w", D);
  V("norm3.bias", "N3b", D);
  {
    const st_tensor_t *t = st_find_tensor(sf, "blocks.0.modulation");
    if (!t || st_tensor_to_f32(sf, t, tmp_v, (size_t)6 * D) != 0) return -1;
    if (b->vt->bank_put(b, "Mod", tmp_v, (size_t)6 * D * sizeof(float)))
      return -1;
  }
  #undef L
  #undef V
  return 0;
}

static int dump_buf(wan_backend *b, const char *name, float *host, size_t n,
                    const char *path) {
  if (b->vt->buf_get(b, name, host, n * sizeof(float))) return -1;
  FILE *f = fopen(path, "wb");
  if (!f) return -1;
  fwrite(host, sizeof(float), n, f);
  fclose(f);
  return 0;
}

static int report(const char *tag, const float *got, const float *ref, size_t n,
                  float min_cos) {
  float c = cosine(got, ref, n);
  printf("rematch %s cosine=%.6f\n", tag, c);
  if (c < min_cos) {
    fprintf(stderr, "FAIL %s cosine %g < %g\n", tag, c, min_cos);
    return -1;
  }
  return 0;
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

  wan_backend *b = wan_backend_cuda_create(0);
  if (!b) {
    fprintf(stderr, "FAIL cuda backend\n");
    return 1;
  }

  size_t xb = (size_t)T * D;
  size_t cb = (size_t)TK * D;
  size_t e0n = (size_t)6 * D;
  size_t oi = (size_t)FFN * D * sizeof(float);
  if (oi < (size_t)D * D * sizeof(float)) oi = (size_t)D * D * sizeof(float);

  float *X = malloc(xb * sizeof(float));
  float *E0 = malloc(e0n * sizeof(float));
  float *CTX = malloc(cb * sizeof(float));
  float *Mod = malloc(e0n * sizeof(float));
  float *tmp_oi = malloc(oi);
  float *tmp_io = malloc(oi);
  float *tmp_v = malloc(e0n * sizeof(float));
  float *host = malloc(xb * sizeof(float));
  float *ref = malloc(xb * sizeof(float));
  float *ones = malloc((size_t)D * sizeof(float));
  if (!X || !E0 || !CTX || !Mod || !tmp_oi || !tmp_io || !tmp_v || !host ||
      !ref || !ones)
    return 1;

  char p[1024];
  snprintf(p, sizeof(p), "%s/x_in.f32", fix);
  if (read_f32(p, X, xb)) return 1;
  snprintf(p, sizeof(p), "%s/e0.f32", fix);
  if (read_f32(p, E0, e0n)) return 1;
  snprintf(p, sizeof(p), "%s/context.f32", fix);
  if (read_f32(p, CTX, cb)) return 1;

  if (load_block0(sf, b, tmp_oi, tmp_io, tmp_v)) {
    fprintf(stderr, "FAIL load block0 weights\n");
    return 1;
  }
  if (b->vt->buf_get(b, "Mod", Mod, e0n * sizeof(float))) return 1;
  for (size_t i = 0; i < e0n; i++) Mod[i] += E0[i]; /* AdaLN: mod + e0 */

  for (int i = 0; i < D; i++) ones[i] = 1.f;
  if (b->vt->buf_put(b, "X", X, xb * sizeof(float)) ||
      b->vt->buf_put(b, "CTX", CTX, cb * sizeof(float)) ||
      b->vt->buf_put(b, "ONES", ones, (size_t)D * sizeof(float)) ||
      b->vt->buf_put(b, "SH", Mod + 0 * D, (size_t)D * sizeof(float)) ||
      b->vt->buf_put(b, "SC", Mod + 1 * D, (size_t)D * sizeof(float)) ||
      b->vt->buf_put(b, "GSA", Mod + 2 * D, (size_t)D * sizeof(float)) ||
      b->vt->buf_put(b, "SH2", Mod + 3 * D, (size_t)D * sizeof(float)) ||
      b->vt->buf_put(b, "SC2", Mod + 4 * D, (size_t)D * sizeof(float)) ||
      b->vt->buf_put(b, "GFF", Mod + 5 * D, (size_t)D * sizeof(float)))
    return 1;

  /* Self-attn */
  if (b->vt->layernorm(b, "X", "LN", NULL, T, D) ||
      b->vt->affine_mul_add(b, "LN", "AD", "SC", "SH", T, D))
    return 1;
  if (b->vt->gemm_f32(b, "AD", "Wq", "Q0", T, D, D) ||
      b->vt->bias_add(b, "Q0", "Q", "Bq", T, D) ||
      b->vt->rmsnorm(b, "Q", "Qr", "Nq", T, D))
    return 1;
  if (b->vt->gemm_f32(b, "AD", "Wk", "K0", T, D, D) ||
      b->vt->bias_add(b, "K0", "K", "Bk", T, D) ||
      b->vt->rmsnorm(b, "K", "Kr", "Nk", T, D))
    return 1;
  if (b->vt->gemm_f32(b, "AD", "Wv", "V0", T, D, D) ||
      b->vt->bias_add(b, "V0", "V", "Bv", T, D))
    return 1;
  if (b->vt->rope3(b, "Qr", "Qrope", T, H, HD, GT, GH, GW) ||
      b->vt->rope3(b, "Kr", "Krope", T, H, HD, GT, GH, GW))
    return 1;
  if (b->vt->attn_sdpa(b, "Qrope", "Krope", "V", "Attn", T, T, H, HD) ||
      b->vt->gemm_f32(b, "Attn", "Wo", "O0", T, D, D) ||
      b->vt->bias_add(b, "O0", "Dsa", "Bo", T, D) ||
      b->vt->gated_residual(b, "X", "Dsa", "GSA", "X1", T, D))
    return 1;
  if (b->vt->sync(b)) return 1;
  snprintf(p, sizeof(p), "%s/c_post_sa.f32", fix);
  if (dump_buf(b, "X1", host, xb, p)) return 1;
  snprintf(p, sizeof(p), "%s/py_post_sa.f32", fix);
  if (read_f32(p, ref, xb) || report("post_sa", host, ref, xb, 0.99f)) return 1;

  /* Cross-attn */
  if (b->vt->layernorm(b, "X1", "LN3r", NULL, T, D) ||
      b->vt->scale_bias(b, "LN3r", "LN3", "N3w", "N3b", T, D))
    return 1;
  if (b->vt->gemm_f32(b, "LN3", "Wqc", "Qc0", T, D, D) ||
      b->vt->bias_add(b, "Qc0", "Qc", "Bqc", T, D) ||
      b->vt->rmsnorm(b, "Qc", "Qcr", "Nqc", T, D))
    return 1;
  if (b->vt->gemm_f32(b, "CTX", "Wkc", "Kc0", TK, D, D) ||
      b->vt->bias_add(b, "Kc0", "Kc", "Bkc", TK, D) ||
      b->vt->rmsnorm(b, "Kc", "Kcr", "Nkc", TK, D))
    return 1;
  if (b->vt->gemm_f32(b, "CTX", "Wvc", "Vc0", TK, D, D) ||
      b->vt->bias_add(b, "Vc0", "Vc", "Bvc", TK, D))
    return 1;
  if (b->vt->attn_sdpa(b, "Qcr", "Kcr", "Vc", "XAttn", T, TK, H, HD) ||
      b->vt->gemm_f32(b, "XAttn", "Woc", "Oc0", T, D, D) ||
      b->vt->bias_add(b, "Oc0", "Dxa", "Boc", T, D) ||
      b->vt->gated_residual(b, "X1", "Dxa", "ONES", "X2", T, D) ||
      b->vt->sync(b))
    return 1;
  snprintf(p, sizeof(p), "%s/c_post_cross.f32", fix);
  if (dump_buf(b, "X2", host, xb, p)) return 1;
  snprintf(p, sizeof(p), "%s/py_post_cross.f32", fix);
  if (read_f32(p, ref, xb) || report("post_cross", host, ref, xb, 0.99f))
    return 1;

  /* FFN */
  if (b->vt->layernorm(b, "X2", "LN2", NULL, T, D) ||
      b->vt->affine_mul_add(b, "LN2", "AD2", "SC2", "SH2", T, D) ||
      b->vt->gemm_f32(b, "AD2", "Wu", "MID0", T, FFN, D) ||
      b->vt->bias_add(b, "MID0", "MID", "Bu", T, FFN) ||
      b->vt->gelu_tanh(b, "MID", "GELU", (size_t)T * FFN) ||
      b->vt->gemm_f32(b, "GELU", "Wd", "D0", T, D, FFN) ||
      b->vt->bias_add(b, "D0", "Dff", "Bd", T, D) ||
      b->vt->gated_residual(b, "X2", "Dff", "GFF", "Y", T, D) ||
      b->vt->sync(b))
    return 1;
  snprintf(p, sizeof(p), "%s/c_post_ffn.f32", fix);
  if (dump_buf(b, "Y", host, xb, p)) return 1;
  snprintf(p, sizeof(p), "%s/py_post_ffn.f32", fix);
  if (read_f32(p, ref, xb) || report("post_ffn", host, ref, xb, 0.99f)) return 1;

  printf("ok: block0 rematch CUDA↔PyTorch (T=%d Tk=%d) all stages ≥0.99\n", T,
         TK);
  wan_backend_destroy(b);
  st_close(sf);
  free(X);
  free(E0);
  free(CTX);
  free(Mod);
  free(tmp_oi);
  free(tmp_io);
  free(tmp_v);
  free(host);
  free(ref);
  free(ones);
  return 0;
}
