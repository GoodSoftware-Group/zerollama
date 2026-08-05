#include "wan_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * DiT denoise (F0789–F0882 + quality slice):
 *   Real AdaLN from time_embedding/time_projection + block.modulation
 *   Cross-attn uses dit.blocks.*.cross_attn.* weights
 *   FFN uses host GELU(tanh) (Wan) instead of SILU_MUL scaffold
 * Scaffold (eye) when GGUF tensors missing.
 */

static float silu_f(float x) { return x / (1.f + expf(-x)); }

static void gelu_tanh_inplace(float *x, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float v = x[i];
    float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    x[i] = 0.5f * v * (1.f + tanhf(c));
  }
}

/* Block-0 stage dumps for DiT A/B (WAN_DUMP_DIR). */
static void dit_dump_named_buf(wan_ctx *ctx, const char *buf, size_t nbytes,
                               const char *fname) {
  const char *dump = getenv("WAN_DUMP_DIR");
  if (!dump || !dump[0] || !ctx || !ctx->uma || !buf || !fname)
    return;
  float *tmp = malloc(nbytes);
  if (!tmp)
    return;
  char resp[256];
  size_t got = 0;
  if (uma_client_buf_get(ctx->uma, buf, tmp, nbytes, &got, resp, sizeof(resp)) ==
          0 &&
      got == nbytes) {
    char path[768];
    snprintf(path, sizeof(path), "%s/%s", dump, fname);
    FILE *f = fopen(path, "wb");
    if (f) {
      fwrite(tmp, 1, nbytes, f);
      fclose(f);
      fprintf(stderr, "wan-c: dumped %s (%zu bytes)\n", fname, nbytes);
    }
  }
  free(tmp);
}

/* Wan sinusoidal_embedding_1d: [cos|sin] halves (not interleaved).
 * Compute freqs in double like Wan (float64 outer) for head AdaLN parity. */
static void wan_sinusoid_1d(float *out, float t, int dim) {
  int half = dim / 2;
  double td = (double)t;
  for (int i = 0; i < half; i++) {
    double freq = pow(10000.0, -(double)i / (double)half);
    double ang = td * freq;
    out[i] = (float)cos(ang);
    out[half + i] = (float)sin(ang);
  }
}

/*
 * e0_6d[6*D] = time_projection(SiLU(time_embedding(sinu(t)))).
 * If e_out != NULL, also writes time_embedding output [D] for Head.
 * Matches Wan2.x WanModel forward (freq_dim=256).
 */
static int dit_time_proj6(wan_ctx *ctx, int step, int D, float *e0_6d,
                          float *e_out) {
  const int freq_dim = 256;
  if (!ctx || !e0_6d || D < 1)
    return -1;
  if (!wan_gguf_has(ctx, "dit.time_embedding.0.weight") ||
      !wan_gguf_has(ctx, "dit.time_projection.1.weight"))
    return -1;

  size_t n0 = 0, nb0 = 0, n2 = 0, nb2 = 0, np = 0, nbp = 0;
  float *W0 = wan_load_tensor_f32(ctx, "dit.time_embedding.0.weight", &n0);
  float *b0 = wan_load_tensor_f32(ctx, "dit.time_embedding.0.bias", &nb0);
  float *W2 = wan_load_tensor_f32(ctx, "dit.time_embedding.2.weight", &n2);
  float *b2 = wan_load_tensor_f32(ctx, "dit.time_embedding.2.bias", &nb2);
  float *Wp = wan_load_tensor_f32(ctx, "dit.time_projection.1.weight", &np);
  float *bp = wan_load_tensor_f32(ctx, "dit.time_projection.1.bias", &nbp);
  float *sinu = calloc((size_t)freq_dim, sizeof(float));
  float *h = calloc((size_t)D, sizeof(float));
  float *h2 = calloc((size_t)D, sizeof(float));
  int ok = 0;
  if (W0 && W2 && Wp && sinu && h && h2 &&
      n0 == (size_t)D * (size_t)freq_dim && n2 == (size_t)D * (size_t)D &&
      np == (size_t)(6 * D) * (size_t)D) {
    /* Flow continuous t: prefer pipeline-set gen_t (sigma*1000). */
    float t = ctx->gen_t;
    if (t <= 0.f)
      t = 1000.f * fmaxf(0.f, 1.f - (float)step / 25.f);
    wan_sinusoid_1d(sinu, t, freq_dim);
    uma_wan_gemm_f32(h, sinu, W0, 1, D, freq_dim);
    if (b0 && nb0 == (size_t)D)
      for (int i = 0; i < D; i++)
        h[i] += b0[i];
    for (int i = 0; i < D; i++)
      h[i] = silu_f(h[i]);
    uma_wan_gemm_f32(h2, h, W2, 1, D, D);
    if (b2 && nb2 == (size_t)D)
      for (int i = 0; i < D; i++)
        h2[i] += b2[i];
    for (int i = 0; i < D; i++)
      h2[i] = silu_f(h2[i]);
    if (e_out)
      memcpy(e_out, h2, (size_t)D * sizeof(float));
    uma_wan_gemm_f32(e0_6d, h2, Wp, 1, 6 * D, D);
    if (bp && nbp == (size_t)(6 * D))
      for (int i = 0; i < 6 * D; i++)
        e0_6d[i] += bp[i];
    ok = 1;
  }
  free(W0);
  free(b0);
  free(W2);
  free(b2);
  free(Wp);
  free(bp);
  free(sinu);
  free(h);
  free(h2);
  return ok ? 0 : -1;
}

/* e6 = e0 + blocks.N.modulation ; scale/shift + residual gates e2/e5. */
static int dit_mod_scales(wan_ctx *ctx, int block, int D, const float *e0_6d,
                          float *sc, float *sh, float *sc2, float *sh2,
                          float *gate_sa, float *gate_ff) {
  char name[96];
  snprintf(name, sizeof(name), "dit.blocks.%d.modulation", block);
  size_t nm = 0;
  float *mod = wan_load_tensor_f32(ctx, name, &nm);
  if (!mod || !e0_6d || nm != (size_t)(6 * D)) {
    free(mod);
    return -1;
  }
  for (int i = 0; i < D; i++) {
    /* AFFINE is x*(1+s)+t; Wan uses *(1+e1)+e0 → pass s=e1, t=e0. */
    float e0 = e0_6d[0 * D + i] + mod[0 * D + i];
    float e1 = e0_6d[1 * D + i] + mod[1 * D + i];
    float e2 = e0_6d[2 * D + i] + mod[2 * D + i];
    float e3 = e0_6d[3 * D + i] + mod[3 * D + i];
    float e4 = e0_6d[4 * D + i] + mod[4 * D + i];
    float e5 = e0_6d[5 * D + i] + mod[5 * D + i];
    sh[i] = e0;
    sc[i] = e1;
    sh2[i] = e3;
    sc2[i] = e4;
    if (gate_sa)
      gate_sa[i] = e2;
    if (gate_ff)
      gate_ff[i] = e5;
  }
  free(mod);
  return 0;
}

/* bs += delta * gate[d]  (token-major [T,D]). */
static int dit_gated_residual_host(wan_ctx *ctx, const char *bs, const char *bdelta,
                                   int T, int D, const float *gate) {
  size_t nbytes = (size_t)T * (size_t)D * 4;
  float *x = calloc((size_t)T * (size_t)D, sizeof(float));
  float *y = calloc((size_t)T * (size_t)D, sizeof(float));
  if (!x || !y) {
    free(x);
    free(y);
    return -1;
  }
  char resp[256];
  size_t got = 0;
  if (uma_client_buf_get(ctx->uma, bs, x, nbytes, &got, resp, sizeof(resp)) !=
          0 ||
      got != nbytes ||
      uma_client_buf_get(ctx->uma, bdelta, y, nbytes, &got, resp,
                         sizeof(resp)) != 0 ||
      got != nbytes) {
    free(x);
    free(y);
    return -1;
  }
  for (int t = 0; t < T; t++)
    for (int d = 0; d < D; d++)
      x[(size_t)t * (size_t)D + (size_t)d] +=
          y[(size_t)t * (size_t)D + (size_t)d] * gate[d];
  int rc = uma_buf_pool_put(ctx->bufs, bs, x, nbytes);
  free(x);
  free(y);
  return rc;
}

/* WanRMSNorm over last dim: y = x * rsqrt(mean(x^2)+eps) * weight. */
static void dit_rmsnorm_rows(float *x, const float *w, int rows, int D,
                             float eps) {
  for (int r = 0; r < rows; r++) {
    float *row = x + (size_t)r * (size_t)D;
    float acc = 0.f;
    for (int i = 0; i < D; i++)
      acc += row[i] * row[i];
    float inv = 1.f / sqrtf(acc / (float)D + eps);
    for (int i = 0; i < D; i++)
      row[i] = row[i] * inv * (w ? w[i] : 1.f);
  }
}

/*
 * After Q/K/V GEMM: add Linear biases; RMSNorm Q/K (Wan qk_norm).
 * weight_prefix e.g. "dit.blocks.%d.self_attn" or "...cross_attn".
 */
static int dit_qk_bias_norm_host(wan_ctx *ctx, const char *bq, const char *bk,
                                 const char *bv, int T, int Tk, int D,
                                 const char *weight_prefix) {
  if (!ctx || !bq || !bk || !weight_prefix || T < 1 || D < 1)
    return -1;
  char nq[128], nk[128], nbq[128], nbk[128], nbv[128];
  snprintf(nq, sizeof(nq), "%s.norm_q.weight", weight_prefix);
  snprintf(nk, sizeof(nk), "%s.norm_k.weight", weight_prefix);
  snprintf(nbq, sizeof(nbq), "%s.q.bias", weight_prefix);
  snprintf(nbk, sizeof(nbk), "%s.k.bias", weight_prefix);
  snprintf(nbv, sizeof(nbv), "%s.v.bias", weight_prefix);

  size_t nwq = 0, nwk = 0, nbbq = 0, nbbk = 0, nbbv = 0;
  float *wq = wan_load_tensor_f32(ctx, nq, &nwq);
  float *wk = wan_load_tensor_f32(ctx, nk, &nwk);
  float *bq_b = wan_load_tensor_f32(ctx, nbq, &nbbq);
  float *bk_b = wan_load_tensor_f32(ctx, nbk, &nbbk);
  float *bv_b = bv ? wan_load_tensor_f32(ctx, nbv, &nbbv) : NULL;
  if (!wq || !wk || nwq != (size_t)D || nwk != (size_t)D) {
    free(wq);
    free(wk);
    free(bq_b);
    free(bk_b);
    free(bv_b);
    return -1;
  }

  size_t nbytes = (size_t)T * (size_t)D * 4;
  size_t kbytes = (size_t)(Tk > 0 ? Tk : T) * (size_t)D * 4;
  float *q = calloc((size_t)T * (size_t)D, sizeof(float));
  float *k = calloc((size_t)(Tk > 0 ? Tk : T) * (size_t)D, sizeof(float));
  float *v = bv ? calloc((size_t)(Tk > 0 ? Tk : T) * (size_t)D, sizeof(float))
                : NULL;
  if (!q || !k || (bv && !v)) {
    free(wq);
    free(wk);
    free(bq_b);
    free(bk_b);
    free(bv_b);
    free(q);
    free(k);
    free(v);
    return -1;
  }
  char resp[256];
  size_t got = 0;
  int krows = Tk > 0 ? Tk : T;
  if (uma_client_buf_get(ctx->uma, bq, q, nbytes, &got, resp, sizeof(resp)) !=
          0 ||
      got != nbytes ||
      uma_client_buf_get(ctx->uma, bk, k, kbytes, &got, resp, sizeof(resp)) !=
          0 ||
      got != kbytes ||
      (bv && (uma_client_buf_get(ctx->uma, bv, v, kbytes, &got, resp,
                                 sizeof(resp)) != 0 ||
              got != kbytes))) {
    free(wq);
    free(wk);
    free(bq_b);
    free(bk_b);
    free(bv_b);
    free(q);
    free(k);
    free(v);
    return -1;
  }
  if (bq_b && nbbq == (size_t)D)
    for (int t = 0; t < T; t++)
      for (int d = 0; d < D; d++)
        q[(size_t)t * (size_t)D + (size_t)d] += bq_b[d];
  if (bk_b && nbbk == (size_t)D)
    for (int t = 0; t < krows; t++)
      for (int d = 0; d < D; d++)
        k[(size_t)t * (size_t)D + (size_t)d] += bk_b[d];
  if (v && bv_b && nbbv == (size_t)D)
    for (int t = 0; t < krows; t++)
      for (int d = 0; d < D; d++)
        v[(size_t)t * (size_t)D + (size_t)d] += bv_b[d];
  dit_rmsnorm_rows(q, wq, T, D, 1e-6f);
  dit_rmsnorm_rows(k, wk, krows, D, 1e-6f);
  int rc = 0;
  if (uma_buf_pool_put(ctx->bufs, bq, q, nbytes) != 0 ||
      uma_buf_pool_put(ctx->bufs, bk, k, kbytes) != 0 ||
      (bv && uma_buf_pool_put(ctx->bufs, bv, v, kbytes) != 0))
    rc = -1;
  free(wq);
  free(wk);
  free(bq_b);
  free(bk_b);
  free(bv_b);
  free(q);
  free(k);
  free(v);
  return rc;
}

/* Add bias[D] to each row of buf [rows,D]. Soft-fail if bias missing. */
static int dit_add_bias_host(wan_ctx *ctx, const char *buf, int rows, int D,
                             const char *bias_name) {
  if (!ctx || !buf || !bias_name || rows < 1 || D < 1)
    return -1;
  size_t nb = 0;
  float *bias = wan_load_tensor_f32(ctx, bias_name, &nb);
  if (!bias || nb != (size_t)D) {
    free(bias);
    return -1;
  }
  size_t nbytes = (size_t)rows * (size_t)D * 4;
  float *x = calloc((size_t)rows * (size_t)D, sizeof(float));
  if (!x) {
    free(bias);
    return -1;
  }
  char resp[256];
  size_t got = 0;
  if (uma_client_buf_get(ctx->uma, buf, x, nbytes, &got, resp, sizeof(resp)) !=
          0 ||
      got != nbytes) {
    free(bias);
    free(x);
    return -1;
  }
  for (int r = 0; r < rows; r++)
    for (int d = 0; d < D; d++)
      x[(size_t)r * (size_t)D + (size_t)d] += bias[d];
  int rc = uma_buf_pool_put(ctx->bufs, buf, x, nbytes);
  free(bias);
  free(x);
  return rc;
}

/* WanLayerNorm (+ optional affine) on bx → by. */
static int dit_layernorm_host(wan_ctx *ctx, const char *bx, const char *by,
                              int rows, int D, const char *wname,
                              const char *bname) {
  if (!ctx || !bx || !by || rows < 1 || D < 1)
    return -1;
  size_t nw = 0, nb = 0;
  float *w = wname ? wan_load_tensor_f32(ctx, wname, &nw) : NULL;
  float *b = bname ? wan_load_tensor_f32(ctx, bname, &nb) : NULL;
  if (wname && (!w || nw != (size_t)D)) {
    free(w);
    free(b);
    return -1;
  }
  if (bname && (!b || nb != (size_t)D)) {
    free(w);
    free(b);
    return -1;
  }
  size_t nbytes = (size_t)rows * (size_t)D * 4;
  float *x = calloc((size_t)rows * (size_t)D, sizeof(float));
  float *y = calloc((size_t)rows * (size_t)D, sizeof(float));
  if (!x || !y) {
    free(w);
    free(b);
    free(x);
    free(y);
    return -1;
  }
  char resp[256];
  size_t got = 0;
  if (uma_client_buf_get(ctx->uma, bx, x, nbytes, &got, resp, sizeof(resp)) !=
          0 ||
      got != nbytes) {
    free(w);
    free(b);
    free(x);
    free(y);
    return -1;
  }
  uma_wan_layernorm_f32(y, x, w, b, rows, D, 1e-6f);
  int rc = uma_buf_pool_put(ctx->bufs, by, y, nbytes);
  free(w);
  free(b);
  free(x);
  free(y);
  return rc;
}

static void timestep_scale_shift(wan_ctx *ctx, int step, int block, int D,
                                 float *scale, float *shift) {
  float t = (float)step + 0.37f * (float)block;
  /* F0906 fallback when real time MLP missing. */
  if (ctx && ctx->uma && ctx->bufs && ctx->caps.sinusoid && D >= 2 &&
      (D % 2) == 0) {
    const char *bt = "x_dit_ts";
    const char *be = "x_dit_temb";
    float ts[1] = {t};
    float *emb = calloc((size_t)D, sizeof(float));
    if (emb &&
        uma_buf_pool_alloc(ctx->bufs, bt, sizeof(float)) == 0 &&
        uma_buf_pool_alloc(ctx->bufs, be, (size_t)D * 4) == 0 &&
        uma_buf_pool_put(ctx->bufs, bt, ts, sizeof(float)) == 0 &&
        wan_graph_sinusoid(ctx, bt, be, 1, D) == 0) {
      char resp[256];
      size_t got = 0;
      if (uma_client_buf_get(ctx->uma, be, emb, (size_t)D * 4, &got, resp,
                             sizeof(resp)) == 0 &&
          got == (size_t)D * 4) {
        for (int i = 0; i < D; i++) {
          scale[i] = 0.05f * emb[i];
          shift[i] = 0.02f * emb[(i + 1) % D];
        }
        free(emb);
        return;
      }
    }
    free(emb);
  }
  if (D >= 2 && (D % 2) == 0) {
    float ts[1] = {t};
    float *emb = calloc((size_t)D, sizeof(float));
    if (emb) {
      uma_wan_sinusoid_f32(emb, ts, 1, D);
      for (int i = 0; i < D; i++) {
        scale[i] = 0.05f * emb[i];
        shift[i] = 0.02f * emb[(i + 1) % D];
      }
      free(emb);
      return;
    }
  }
  for (int i = 0; i < D; i++) {
    float a = sinf(0.01f * t + 0.1f * (float)i);
    float b = cosf(0.01f * t + 0.07f * (float)i);
    scale[i] = 0.05f * a;
    shift[i] = 0.02f * b;
  }
}

static int dit_dims(size_t n, int z_channels, int *rows_out, int *D_out) {
  int D = z_channels > 0 ? z_channels : 16;
  if (n < (size_t)D || (n % (size_t)D) != 0)
    return -1;
  *rows_out = (int)(n / (size_t)D);
  *D_out = D;
  return 0;
}

/* Pick H/KV/HD with H==KV, HD>=4, H*HD==D. */
static void dit_head_geom(int D, int *H, int *KV, int *HD) {
  int hd = 8;
  while (hd >= 4 && (D % hd) != 0)
    hd /= 2;
  if (hd < 4)
    hd = (D >= 4 && (D % 4) == 0) ? 4 : D;
  *HD = hd;
  *H = D / hd;
  if (*H < 1)
    *H = 1;
  *KV = *H;
}

static int dit_host(wan_ctx *ctx, float *latent, size_t n, int step) {
  int rows, D;
  if (dit_dims(n, ctx->cfg.z_channels, &rows, &D) != 0)
    return -1;

  float *tmp = calloc(n, sizeof(float));
  float *scale = calloc((size_t)D, sizeof(float));
  float *shift = calloc((size_t)D, sizeof(float));
  float *w = calloc((size_t)D * (size_t)D, sizeof(float));
  float *out = calloc(n, sizeof(float));
  if (!tmp || !scale || !shift || !w || !out) {
    free(tmp);
    free(scale);
    free(shift);
    free(w);
    free(out);
    return -1;
  }

  timestep_scale_shift(ctx, step, 0, D, scale, shift);
  wan_fill_eye_nt(w, D, D);
  uma_wan_layernorm_f32(tmp, latent, NULL, NULL, rows, D, 1e-6f);
  uma_wan_affine_mul_add_f32(tmp, tmp, scale, shift, rows, D);
  {
    int gt = ctx->gen_tp > 0 ? ctx->gen_tp : rows;
    int gh = ctx->gen_hp > 0 ? ctx->gen_hp : 1;
    int gw = ctx->gen_wp > 0 ? ctx->gen_wp : 1;
    if ((size_t)gt * (size_t)gh * (size_t)gw != (size_t)rows) {
      gt = rows;
      gh = 1;
      gw = 1;
    }
    if (wan_rope3_tokens_grid(tmp, rows, 1, D, gt, gh, gw) != 0) {
      free(tmp);
      free(scale);
      free(shift);
      free(w);
      free(out);
      return -1;
    }
  }
  uma_wan_gemm_f32(out, tmp, w, rows, D, D);
  memcpy(latent, out, n * sizeof(float));

  free(out);
  free(tmp);
  free(scale);
  free(shift);
  free(w);
  return 0;
}

/* unused — kept for local scaffold tips if reintroduced */
#if 0
static int put_ones(wan_ctx *ctx, const char *name, int n) {
  float *o = calloc((size_t)n, sizeof(float));
  if (!o)
    return -1;
  for (int i = 0; i < n; i++)
    o[i] = 1.f;
  int rc = uma_buf_pool_alloc(ctx->bufs, name, (size_t)n * 4) ||
           uma_buf_pool_put(ctx->bufs, name, o, (size_t)n * 4);
  free(o);
  return rc;
}
#endif

/* One F0791 DiT block into buffer `bs` (in/out). e0_6d may be NULL → sinusoid AdaLN.
 * token_mirror / text_mirror: host copies; re-PUT after BANK weight loads. */
static int dit_block_broker(wan_ctx *ctx, const char *bs, int T, int D, int H,
                            int KV, int HD, int Ffn, int step, int block,
                            const char *text_ctx, int Tk, const float *e0_6d,
                            const float *token_mirror,
                            const float *text_mirror) {
  char nodes[4096];
  /* Broker AdaLN scale names — only used when real_mod=0 (sinusoid fallback). */
  const char *scn = "x_dit_sc";
  const char *shn = "x_dit_sh";
  const char *sc2 = "x_dit_sc2";
  const char *sh2 = "x_dit_sh2";

#define DIT_FAIL(stage)                                                        \
  do {                                                                         \
    fprintf(stderr, "wan-c: DiT block %d fail at %s\n", block, (stage));       \
    return -1;                                                                 \
  } while (0)

  float sc_buf[1536], sh_buf[1536], scb_buf[1536], shb_buf[1536];
  float gate_sa_buf[1536], gate_ff_buf[1536];
  if (D > 1536)
    DIT_FAIL("dim>1536");
  float *sc = sc_buf;
  float *sh = sh_buf;
  float *scb = scb_buf;
  float *shb = shb_buf;
  float *gate_sa = gate_sa_buf;
  float *gate_ff = gate_ff_buf;
  int real_mod = 0;
  if (e0_6d && dit_mod_scales(ctx, block, D, e0_6d, sc, sh, scb, shb, gate_sa,
                              gate_ff) == 0) {
    real_mod = 1;
    static int logged_mod;
    if (!logged_mod) {
      fprintf(stderr,
              "wan-c: AdaLN time_mlp+modulation+gates D=%d (real)\n", D);
      logged_mod = 1;
    }
  } else {
    timestep_scale_shift(ctx, step, block, D, sc, sh);
    timestep_scale_shift(ctx, step, block + 10, D, scb, shb);
    for (int i = 0; i < D; i++) {
      gate_sa[i] = 1.f;
      gate_ff[i] = 1.f;
    }
  }

  size_t dbytes = (size_t)D * 4;
  size_t nbytes = (size_t)T * (size_t)D * 4;
  size_t nff = (size_t)T * (size_t)Ffn * 4;
  size_t tbytes = Tk > 0 ? (size_t)Tk * (size_t)D * 4 : 0;

  /* Host AdaLN when real_mod — skip broker scale slots (long-CFG pressure). */
  if (!real_mod) {
    if (uma_buf_pool_alloc(ctx->bufs, scn, dbytes) != 0 ||
        uma_buf_pool_alloc(ctx->bufs, shn, dbytes) != 0 ||
        uma_buf_pool_alloc(ctx->bufs, sc2, dbytes) != 0 ||
        uma_buf_pool_alloc(ctx->bufs, sh2, dbytes) != 0 ||
        uma_buf_pool_put(ctx->bufs, scn, sc, dbytes) != 0 ||
        uma_buf_pool_put(ctx->bufs, shn, sh, dbytes) != 0 ||
        uma_buf_pool_put(ctx->bufs, sc2, scb, dbytes) != 0 ||
        uma_buf_pool_put(ctx->bufs, sh2, shb, dbytes) != 0) {
      DIT_FAIL("adaln_put");
    }
  } else {
    static int logged_hadaln;
    if (!logged_hadaln) {
      fprintf(stderr,
              "wan-c: DiT AdaLN on host (LN no-affine + modulate; no x_dit_sc*)\n");
      logged_hadaln = 1;
    }
  }

  const char *bt = "x_dit_t";
  const char *ba = "x_dit_a";
  const char *bq = "x_dit_q";
  const char *bk = "x_dit_k";
  const char *bv = "x_dit_v";
  const char *bo = "x_dit_ao";
  const char *bff = "x_dit_ff";
  const char *bwq = "x_dit_Wq";
  const char *bwk = "x_dit_Wk";
  const char *bwv = "x_dit_Wv";
  const char *bwo = "x_dit_Wo";
  const char *bwu = "x_dit_Wu";
  const char *bwd = "x_dit_Wd";
  const char *bwqc = "x_dit_Wqc";
  const char *bwkc = "x_dit_Wkc";
  const char *bwvc = "x_dit_Wvc";
  const char *bwoc = "x_dit_Woc";
  const char *bkc = "x_dit_kc";
  const char *bvc = "x_dit_vc";

  char gname[128];
  int have_cross = text_ctx && Tk > 0 && ctx->caps.attn_full &&
                   wan_gguf_has(ctx, "dit.blocks.0.cross_attn.q.weight");

  /* When real_mod, dense GEMMs run on host — only bind cross weights on broker. */
  if (!real_mod) {
    snprintf(gname, sizeof(gname), "dit.blocks.%d.self_attn.q.weight", block);
    if (wan_put_weight_or_eye(ctx, bwq, "dit.Wq", gname, D, D) != 0)
      DIT_FAIL("Wq");
    snprintf(gname, sizeof(gname), "dit.blocks.%d.self_attn.k.weight", block);
    if (wan_put_weight_or_eye(ctx, bwk, "dit.Wk", gname, D, D) != 0)
      DIT_FAIL("Wk");
    snprintf(gname, sizeof(gname), "dit.blocks.%d.self_attn.v.weight", block);
    if (wan_put_weight_or_eye(ctx, bwv, "dit.Wv", gname, D, D) != 0)
      DIT_FAIL("Wv");
    snprintf(gname, sizeof(gname), "dit.blocks.%d.self_attn.o.weight", block);
    if (wan_put_weight_or_eye(ctx, bwo, "dit.Wo", gname, D, D) != 0)
      DIT_FAIL("Wo");
    snprintf(gname, sizeof(gname), "dit.blocks.%d.ffn.0.weight", block);
    if (wan_put_weight_or_eye(ctx, bwu, "dit.Wu", gname, Ffn, D) != 0)
      DIT_FAIL("Wu");
    snprintf(gname, sizeof(gname), "dit.blocks.%d.ffn.2.weight", block);
    if (wan_put_weight_or_eye(ctx, bwd, "dit.Wd", gname, D, Ffn) != 0)
      DIT_FAIL("Wd");
  }

  if (have_cross) {
    snprintf(gname, sizeof(gname), "dit.blocks.%d.cross_attn.q.weight", block);
    if (wan_put_weight_or_eye(ctx, bwqc, "dit.Wqc", gname, D, D) != 0)
      DIT_FAIL("Wqc");
    snprintf(gname, sizeof(gname), "dit.blocks.%d.cross_attn.k.weight", block);
    if (wan_put_weight_or_eye(ctx, bwkc, "dit.Wkc", gname, D, D) != 0)
      DIT_FAIL("Wkc");
    snprintf(gname, sizeof(gname), "dit.blocks.%d.cross_attn.v.weight", block);
    if (wan_put_weight_or_eye(ctx, bwvc, "dit.Wvc", gname, D, D) != 0)
      DIT_FAIL("Wvc");
    snprintf(gname, sizeof(gname), "dit.blocks.%d.cross_attn.o.weight", block);
    if (wan_put_weight_or_eye(ctx, bwoc, "dit.Woc", gname, D, D) != 0)
      DIT_FAIL("Woc");
    static int logged_x;
    if (!logged_x) {
      fprintf(stderr, "wan-c: DiT cross_attn weights enabled\n");
      logged_x = 1;
    }
  }

  if (uma_buf_pool_alloc(ctx->bufs, bt, nbytes) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, ba, nbytes) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bq, nbytes) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bk, nbytes) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bv, nbytes) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bo, nbytes) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bff, nff) != 0)
    DIT_FAIL("act_alloc");
  if (have_cross &&
      (uma_buf_pool_alloc(ctx->bufs, bkc, tbytes) != 0 ||
       uma_buf_pool_alloc(ctx->bufs, bvc, tbytes) != 0))
    DIT_FAIL("cross_kv_alloc");

  /* Weight BANK/BUF puts can drop tokens / text — restore before graphs. */
#define DIT_RESTORE_HOT()                                                      \
  do {                                                                         \
    if (!real_mod) {                                                           \
      if (uma_buf_pool_ensure_put(ctx->bufs, scn, sc, dbytes) != 0 ||          \
          uma_buf_pool_ensure_put(ctx->bufs, shn, sh, dbytes) != 0 ||          \
          uma_buf_pool_ensure_put(ctx->bufs, sc2, scb, dbytes) != 0 ||         \
          uma_buf_pool_ensure_put(ctx->bufs, sh2, shb, dbytes) != 0)           \
        DIT_FAIL("restore_adaln");                                             \
    }                                                                          \
    if (token_mirror &&                                                        \
        uma_buf_pool_ensure_put(ctx->bufs, bs, token_mirror, nbytes) != 0)     \
      DIT_FAIL("restore_x_dit_s");                                             \
    if (have_cross && text_mirror && tbytes > 0 &&                              \
        uma_buf_pool_ensure_put(ctx->bufs, text_ctx, text_mirror, tbytes) !=   \
            0)                                                                 \
      DIT_FAIL("restore_x_dit_tctx");                                          \
  } while (0)
  DIT_RESTORE_HOT();

  int flat = T * D;

  /* Self-attn (+ RoPE on Q/K every block when caps allow). */
  {
    float *hq = NULL, *hk = NULL, *hv = NULL;
    if (real_mod) {
      /* Host LN+AdaLN+QKV GEMM — avoids BANK weight ↔ hot buf eviction. */
      float *ha = calloc((size_t)T * (size_t)D, sizeof(float));
      hq = calloc((size_t)T * (size_t)D, sizeof(float));
      hk = calloc((size_t)T * (size_t)D, sizeof(float));
      hv = calloc((size_t)T * (size_t)D, sizeof(float));
      if (!ha || !hq || !hk || !hv) {
        free(ha);
        free(hq);
        free(hk);
        free(hv);
        DIT_FAIL("host_qkv_alloc");
      }
      if (token_mirror)
        memcpy(ha, token_mirror, nbytes);
      else {
        char resp[256];
        size_t got = 0;
        if (uma_client_buf_get(ctx->uma, bs, ha, nbytes, &got, resp,
                               sizeof(resp)) != 0 ||
            got != nbytes) {
          free(ha);
          free(hq);
          free(hk);
          free(hv);
          DIT_FAIL("host_qkv_get");
        }
      }
      {
        float *tmp = calloc((size_t)T * (size_t)D, sizeof(float));
        if (!tmp) {
          free(ha);
          free(hq);
          free(hk);
          free(hv);
          DIT_FAIL("host_qkv_tmp");
        }
        uma_wan_layernorm_f32(tmp, ha, NULL, NULL, T, D, 1e-6f);
        for (int t = 0; t < T; t++)
          for (int d = 0; d < D; d++) {
            size_t i = (size_t)t * (size_t)D + (size_t)d;
            ha[i] = tmp[i] * (1.f + sc[d]) + sh[d];
          }
        free(tmp);
      }
      size_t nw = 0;
      snprintf(gname, sizeof(gname), "dit.blocks.%d.self_attn.q.weight", block);
      const float *Wq = wan_borrow_tensor_f32(ctx, gname, &nw);
      snprintf(gname, sizeof(gname), "dit.blocks.%d.self_attn.k.weight", block);
      const float *Wk = wan_borrow_tensor_f32(ctx, gname, &nw);
      snprintf(gname, sizeof(gname), "dit.blocks.%d.self_attn.v.weight", block);
      const float *Wv = wan_borrow_tensor_f32(ctx, gname, &nw);
      if (!Wq || !Wk || !Wv) {
        free(ha);
        free(hq);
        free(hk);
        free(hv);
        DIT_FAIL("host_qkv_W");
      }
      uma_wan_gemm_f32(hq, ha, Wq, T, D, D);
      uma_wan_gemm_f32(hk, ha, Wk, T, D, D);
      uma_wan_gemm_f32(hv, ha, Wv, T, D, D);
      free(ha);
      if (uma_buf_pool_ensure_put(ctx->bufs, bq, hq, nbytes) != 0 ||
          uma_buf_pool_ensure_put(ctx->bufs, bk, hk, nbytes) != 0 ||
          uma_buf_pool_ensure_put(ctx->bufs, bv, hv, nbytes) != 0) {
        free(hq);
        free(hk);
        free(hv);
        DIT_FAIL("host_qkv_put");
      }
      static int logged_hqkv;
      if (!logged_hqkv) {
        fprintf(stderr, "wan-c: DiT self-attn QKV GEMM on host (AdaLN+LN)\n");
        logged_hqkv = 1;
      }
    } else {
      hq = hk = hv = NULL;
      char qkv[1536];
      snprintf(qkv, sizeof(qkv),
               "LAYERNORM_MUL@CPU! x=%s y=%s N=%d D=%d ; "
               "AFFINE_MUL_ADD@CPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
               "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
               "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
               "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; MARK@CPU?",
               bs, bt, T, D, bt, ba, scn, shn, T, D, ba, bq, bwq, T, D, D, ba,
               bk, bwk, T, D, D, ba, bv, bwv, T, D, D);
      if (wan_submit_graph(ctx->uma, qkv) != 0)
        DIT_FAIL("qkv");
    }

    {
      char pref[96];
      snprintf(pref, sizeof(pref), "dit.blocks.%d.self_attn", block);
      if (dit_qk_bias_norm_host(ctx, bq, bk, bv, T, T, D, pref) == 0) {
        static int logged_qk;
        if (!logged_qk) {
          fprintf(stderr, "wan-c: DiT QK RMSNorm + Linear biases (self)\n");
          logged_qk = 1;
        }
      }
    }

    const char *q_attn = bq;
    const char *k_attn = bk;
    if (ctx->caps.rope3) {
      /* Reuse fixed names — per-block allocs exhaust broker buf slots mid-CFG. */
      const char *bqr = "x_dit_qr";
      const char *bkr = "x_dit_kr";
      int a_qr = uma_buf_pool_alloc(ctx->bufs, bqr, nbytes);
      int a_kr = uma_buf_pool_alloc(ctx->bufs, bkr, nbytes);
      int r_q = (a_qr == 0) ? wan_graph_rope3(ctx, bq, bqr, T, H, HD) : -1;
      int r_k = (a_kr == 0) ? wan_graph_rope3(ctx, bk, bkr, T, H, HD) : -1;
      if (a_qr == 0 && a_kr == 0 && r_q == 0 && r_k == 0) {
        q_attn = bqr;
        k_attn = bkr;
        static int logged_rope;
        if (!logged_rope) {
          fprintf(stderr,
                  "wan-c: DiT RoPE on self-attn Q/K (all blocks)%s\n",
                  (ctx->gen_tp > 0 && ctx->gen_hp > 0 && ctx->gen_wp > 0)
                      ? " +3D grid"
                      : "");
          logged_rope = 1;
        }
        if (block == 0) {
          dit_dump_named_buf(ctx, q_attn, nbytes, "b0_q_rope.f32");
          dit_dump_named_buf(ctx, k_attn, nbytes, "b0_k_rope.f32");
        }
      } else {
        static int logged_rope_fail;
        if (!logged_rope_fail) {
          fprintf(stderr,
                  "wan-c: DiT RoPE3D SKIPPED (alloc q=%d k=%d rope q=%d k=%d) "
                  "— quality gap\n",
                  a_qr, a_kr, r_q, r_k);
          logged_rope_fail = 1;
        }
      }
    }

    if (real_mod) {
      /* Refresh host Q/K/V after bias/norm/RoPE may have mutated broker slots. */
      if (hq && hk && hv) {
        char resp[256];
        size_t got = 0;
        (void)uma_client_buf_get(ctx->uma, q_attn, hq, nbytes, &got, resp,
                                 sizeof(resp));
        (void)uma_client_buf_get(ctx->uma, k_attn, hk, nbytes, &got, resp,
                                 sizeof(resp));
        (void)uma_client_buf_get(ctx->uma, bv, hv, nbytes, &got, resp,
                                 sizeof(resp));
        (void)uma_buf_pool_ensure_put(ctx->bufs, q_attn, hq, nbytes);
        (void)uma_buf_pool_ensure_put(ctx->bufs, k_attn, hk, nbytes);
        (void)uma_buf_pool_ensure_put(ctx->bufs, bv, hv, nbytes);
      }
      if (uma_buf_pool_alloc(ctx->bufs, bo, nbytes) != 0) {
        free(hq);
        free(hk);
        free(hv);
        DIT_FAIL("attn_ao_alloc");
      }
      int n = snprintf(
          nodes, sizeof(nodes),
          "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
          "HD=%d kind=full ; MARK@CPU?",
          q_attn, k_attn, bv, bo, T, T, H, KV, HD);
      if (n < 0 || (size_t)n >= sizeof(nodes) ||
          wan_submit_graph(ctx->uma, nodes) != 0) {
        free(hq);
        free(hk);
        free(hv);
        DIT_FAIL("attn_sa");
      }
      free(hq);
      free(hk);
      free(hv);
      hq = hk = hv = NULL;
      {
        float *ao = calloc((size_t)T * (size_t)D, sizeof(float));
        float *out = calloc((size_t)T * (size_t)D, sizeof(float));
        if (!ao || !out) {
          free(ao);
          free(out);
          DIT_FAIL("host_o_alloc");
        }
        char resp[256];
        size_t got = 0;
        if (uma_client_buf_get(ctx->uma, bo, ao, nbytes, &got, resp,
                               sizeof(resp)) != 0 ||
            got != nbytes) {
          free(ao);
          free(out);
          DIT_FAIL("host_o_get");
        }
        size_t nw = 0;
        snprintf(gname, sizeof(gname), "dit.blocks.%d.self_attn.o.weight",
                 block);
        const float *Wo = wan_borrow_tensor_f32(ctx, gname, &nw);
        if (!Wo || nw != (size_t)D * (size_t)D) {
          free(ao);
          free(out);
          DIT_FAIL("host_o_W");
        }
        uma_wan_gemm_f32(out, ao, Wo, T, D, D);
        free(ao);
        if (uma_buf_pool_ensure_put(ctx->bufs, bt, out, nbytes) != 0) {
          free(out);
          DIT_FAIL("host_o_put");
        }
        free(out);
        {
          char ob[96];
          snprintf(ob, sizeof(ob), "dit.blocks.%d.self_attn.o.bias", block);
          (void)dit_add_bias_host(ctx, bt, T, D, ob);
        }
      }
      if (dit_gated_residual_host(ctx, bs, bt, T, D, gate_sa) != 0)
        DIT_FAIL("gate_sa");
      if (block == 0)
        dit_dump_named_buf(ctx, bs, nbytes, "b0_post_sa.f32");
    } else {
      int n = snprintf(
          nodes, sizeof(nodes),
          "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
          "HD=%d kind=full ; "
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
          "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@CPU?",
          q_attn, k_attn, bv, bo, T, T, H, KV, HD, bo, bt, bwo, T, D, D, bt, bs,
          flat);
      if (n < 0 || (size_t)n >= sizeof(nodes) ||
          wan_submit_graph(ctx->uma, nodes) != 0)
        return -1;
    }
  }

  /* Cross-attn: norm3(x) → Q; K/V from text (Wan cross_attn_norm). */
  if (have_cross) {
    char nw3[96], nb3[96];
    snprintf(nw3, sizeof(nw3), "dit.blocks.%d.norm3.weight", block);
    snprintf(nb3, sizeof(nb3), "dit.blocks.%d.norm3.bias", block);
    if (dit_layernorm_host(ctx, bs, bt, T, D, nw3, nb3) == 0) {
      static int logged_n3;
      if (!logged_n3) {
        fprintf(stderr, "wan-c: DiT cross_attn norm3 (affine LN)\n");
        logged_n3 = 1;
      }
    } else {
      char ln[256];
      snprintf(ln, sizeof(ln),
               "LAYERNORM_MUL@CPU! x=%s y=%s N=%d D=%d ; MARK@CPU?", bs, bt, T,
               D);
      if (wan_submit_graph(ctx->uma, ln) != 0)
        DIT_FAIL("cross_ln");
    }
    int n = snprintf(
        nodes, sizeof(nodes),
        "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
        "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
        "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; MARK@CPU?",
        bt, bq, bwqc, T, D, D, text_ctx, bkc, bwkc, Tk, D, D, text_ctx, bvc,
        bwvc, Tk, D, D);
    if (n < 0 || (size_t)n >= sizeof(nodes) ||
        wan_submit_graph(ctx->uma, nodes) != 0)
      return -1;
    {
      char pref[96];
      snprintf(pref, sizeof(pref), "dit.blocks.%d.cross_attn", block);
      if (dit_qk_bias_norm_host(ctx, bq, bkc, bvc, T, Tk, D, pref) == 0) {
        static int logged_xqk;
        if (!logged_xqk) {
          fprintf(stderr, "wan-c: DiT QK RMSNorm + Linear biases (cross)\n");
          logged_xqk = 1;
        }
      }
    }
    n = snprintf(
        nodes, sizeof(nodes),
        "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
        "HD=%d kind=full ; "
        "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; MARK@CPU?",
        bq, bkc, bvc, bo, T, Tk, H, KV, HD, bo, bt, bwoc, T, D, D);
    if (n < 0 || (size_t)n >= sizeof(nodes) ||
        wan_submit_graph(ctx->uma, nodes) != 0)
      return -1;
    {
      char ob[96];
      snprintf(ob, sizeof(ob), "dit.blocks.%d.cross_attn.o.bias", block);
      (void)dit_add_bias_host(ctx, bt, T, D, ob);
    }
    n = snprintf(nodes, sizeof(nodes),
                 "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@CPU?", bt, bs, flat);
    if (n < 0 || (size_t)n >= sizeof(nodes) ||
        wan_submit_graph(ctx->uma, nodes) != 0)
      return -1;
    if (block == 0)
      dit_dump_named_buf(ctx, bs, nbytes, "b0_post_cross.f32");
  } else if (text_ctx && Tk > 0 && ctx->caps.attn_full) {
    /* Scaffold: reuse self Wq/Wo, treat text as K/V. */
    int n = snprintf(
        nodes, sizeof(nodes),
        "LAYERNORM_MUL@CPU! x=%s y=%s N=%d D=%d ; "
        "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
        "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
        "HD=%d kind=full ; "
        "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
        "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@CPU?",
        bs, bt, T, D, bt, bq, bwq, T, D, D, bq, text_ctx, text_ctx, bo, T, Tk, H,
        KV, HD, bo, bt, bwo, T, D, D, bt, bs, flat);
    if (n < 0 || (size_t)n >= sizeof(nodes) ||
        wan_submit_graph(ctx->uma, nodes) != 0)
      return -1;
  }

  /* Dense FFN: GEMM↑ → host GELU(tanh) → GEMM↓ + residual. */
  {
    if (real_mod) {
      float *hx = calloc((size_t)T * (size_t)D, sizeof(float));
      float *ha = calloc((size_t)T * (size_t)D, sizeof(float));
      float *ff = calloc((size_t)T * (size_t)Ffn, sizeof(float));
      float *hout = calloc((size_t)T * (size_t)D, sizeof(float));
      if (!hx || !ha || !ff || !hout) {
        free(hx);
        free(ha);
        free(ff);
        free(hout);
        DIT_FAIL("host_ff_alloc");
      }
      char resp[256];
      size_t got = 0;
      if (uma_client_buf_get(ctx->uma, bs, hx, nbytes, &got, resp,
                             sizeof(resp)) != 0 ||
          got != nbytes) {
        free(hx);
        free(ha);
        free(ff);
        free(hout);
        DIT_FAIL("host_ff_get");
      }
      uma_wan_layernorm_f32(ha, hx, NULL, NULL, T, D, 1e-6f);
      for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++) {
          size_t i = (size_t)t * (size_t)D + (size_t)d;
          ha[i] = ha[i] * (1.f + scb[d]) + shb[d];
        }
      size_t nw = 0;
      snprintf(gname, sizeof(gname), "dit.blocks.%d.ffn.0.weight", block);
      const float *Wu = wan_borrow_tensor_f32(ctx, gname, &nw);
      snprintf(gname, sizeof(gname), "dit.blocks.%d.ffn.2.weight", block);
      const float *Wd = wan_borrow_tensor_f32(ctx, gname, &nw);
      if (!Wu || !Wd) {
        free(hx);
        free(ha);
        free(ff);
        free(hout);
        DIT_FAIL("host_ff_W");
      }
      uma_wan_gemm_f32(ff, ha, Wu, T, Ffn, D);
      {
        char ub[96];
        snprintf(ub, sizeof(ub), "dit.blocks.%d.ffn.0.bias", block);
        size_t nbb = 0;
        const float *ubias = wan_borrow_tensor_f32(ctx, ub, &nbb);
        if (ubias && nbb == (size_t)Ffn) {
          for (int t = 0; t < T; t++)
            for (int d = 0; d < Ffn; d++)
              ff[(size_t)t * (size_t)Ffn + (size_t)d] += ubias[d];
        }
      }
      gelu_tanh_inplace(ff, (size_t)T * (size_t)Ffn);
      uma_wan_gemm_f32(hout, ff, Wd, T, D, Ffn);
      free(ff);
      free(ha);
      {
        char db[96];
        snprintf(db, sizeof(db), "dit.blocks.%d.ffn.2.bias", block);
        size_t nbb = 0;
        const float *dbias = wan_borrow_tensor_f32(ctx, db, &nbb);
        if (dbias && nbb == (size_t)D) {
          for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++)
              hout[(size_t)t * (size_t)D + (size_t)d] += dbias[d];
        }
      }
      for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++) {
          size_t i = (size_t)t * (size_t)D + (size_t)d;
          hx[i] += hout[i] * gate_ff[d];
        }
      free(hout);
      if (uma_buf_pool_ensure_put(ctx->bufs, bs, hx, nbytes) != 0) {
        free(hx);
        DIT_FAIL("host_ff_put");
      }
      free(hx);
      if (block == 0)
        dit_dump_named_buf(ctx, bs, nbytes, "b0_post_ffn.f32");
      static int logged_hff;
      if (!logged_hff) {
        fprintf(stderr, "wan-c: DiT FFN AdaLN+GEMM+GELU on host\n");
        logged_hff = 1;
      }
    } else {
      int n = snprintf(
          nodes, sizeof(nodes),
          "LAYERNORM_MUL@CPU! x=%s y=%s N=%d D=%d ; "
          "AFFINE_MUL_ADD@CPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; MARK@CPU?",
          bs, bt, T, D, bt, ba, sc2, sh2, T, D, ba, bff, bwu, T, Ffn, D);
      if (n < 0 || (size_t)n >= sizeof(nodes) ||
          wan_submit_graph(ctx->uma, nodes) != 0)
        DIT_FAIL("ffn_up");

      float *ff = calloc((size_t)T * (size_t)Ffn, sizeof(float));
      if (!ff)
        return -1;
      char resp[256];
      size_t got = 0;
      if (uma_client_buf_get(ctx->uma, bff, ff, nff, &got, resp, sizeof(resp)) !=
              0 ||
          got != nff) {
        free(ff);
        return -1;
      }
      {
        char ub[96];
        snprintf(ub, sizeof(ub), "dit.blocks.%d.ffn.0.bias", block);
        size_t nbb = 0;
        float *ubias = wan_load_tensor_f32(ctx, ub, &nbb);
        if (ubias && nbb == (size_t)Ffn) {
          for (int t = 0; t < T; t++)
            for (int d = 0; d < Ffn; d++)
              ff[(size_t)t * (size_t)Ffn + (size_t)d] += ubias[d];
          static int logged_fb;
          if (!logged_fb) {
            fprintf(stderr, "wan-c: DiT FFN Linear biases\n");
            logged_fb = 1;
          }
        }
        free(ubias);
      }
      gelu_tanh_inplace(ff, (size_t)T * (size_t)Ffn);
      static int logged_gelu;
      if (!logged_gelu) {
        fprintf(stderr, "wan-c: DiT FFN GELU(tanh) host\n");
        logged_gelu = 1;
      }
      if (uma_buf_pool_put(ctx->bufs, bff, ff, nff) != 0) {
        free(ff);
        return -1;
      }
      free(ff);

      n = snprintf(nodes, sizeof(nodes),
                   "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
                   "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@CPU?",
                   bff, bt, bwd, T, D, Ffn, bt, bs, flat);
      if (n < 0 || (size_t)n >= sizeof(nodes) ||
          wan_submit_graph(ctx->uma, nodes) != 0)
        return -1;
    }
  }

  return 0;
#undef DIT_RESTORE_HOT
#undef DIT_FAIL
}

static int dit_patch_host(wan_ctx *ctx, const float *latent, size_t latent_n,
                          float **tok_out, int *T_out, int *D_out, int *lt_out,
                          int *lh_out, int *lw_out, int *ptp_out, int *php_out,
                          int *pwp_out) {
  const wan_model_config *c = &ctx->cfg;
  int C = c->z_channels;
  int D = c->dim;
  int pt = c->patch_t, ph = c->patch_h, pw = c->patch_w;
  int lt = ctx->gen_lt, lh = ctx->gen_lh, lw = ctx->gen_lw;
  if (lt < 1 || lh < 1 || lw < 1)
    return -1;
  if ((size_t)C * (size_t)lt * (size_t)lh * (size_t)lw != latent_n)
    return -1;
  if ((lt % pt) || (lh % ph) || (lw % pw))
    return -1;

  size_t wne = 0, nbi = 0;
  float *w = wan_load_tensor_f32(ctx, "dit.patch_embedding.weight", &wne);
  float *bias = wan_load_tensor_f32(ctx, "dit.patch_embedding.bias", &nbi);
  size_t expect_w =
      (size_t)D * (size_t)C * (size_t)pt * (size_t)ph * (size_t)pw;
  if (!w || wne != expect_w) {
    fprintf(stderr,
            "wan-c: DiT patch load failed name=dit.patch_embedding.weight "
            "got=%zu want=%zu\n",
            wne, expect_w);
    free(w);
    free(bias);
    return -1;
  }

  int tp = lt / pt, hp = lh / ph, wp = lw / pw;
  size_t vol_out = (size_t)D * (size_t)tp * (size_t)hp * (size_t)wp;
  float *vol = calloc(vol_out, sizeof(float));
  float *tok = calloc((size_t)tp * (size_t)hp * (size_t)wp * (size_t)D,
                      sizeof(float));
  if (!vol || !tok) {
    free(w);
    free(bias);
    free(vol);
    free(tok);
    return -1;
  }
  uma_wan_conv3d_f32(vol, latent, w, (bias && nbi == (size_t)D) ? bias : NULL,
                     1, C, lt, lh, lw, D, pt, ph, pw, pt, ph, pw, 0, 0, 0);
  free(w);
  free(bias);
  if (nbi == (size_t)D) {
    static int logged_pb;
    if (!logged_pb) {
      fprintf(stderr, "wan-c: DiT patch_embedding bias\n");
      logged_pb = 1;
    }
  }
  uma_wan_ncdhw_to_tokens_f32(tok, vol, 1, D, tp, hp, wp);
  free(vol);
  *tok_out = tok;
  *T_out = tp * hp * wp;
  *D_out = D;
  *lt_out = lt;
  *lh_out = lh;
  *lw_out = lw;
  *ptp_out = tp;
  *php_out = hp;
  *pwp_out = wp;
  ctx->gen_tp = tp;
  ctx->gen_hp = hp;
  ctx->gen_wp = wp;
  return 0;
}

static int dit_unpatch_host(wan_ctx *ctx, const float *tok, int T, int D,
                            int lt, int lh, int lw, int tp, int hp, int wp,
                            float *latent, size_t latent_n,
                            const float *e_time) {
  const wan_model_config *c = &ctx->cfg;
  int C = c->z_channels;
  int pt = c->patch_t, ph = c->patch_h, pw = c->patch_w;
  int out_per = C * pt * ph * pw; /* 64 */
  if (T != tp * hp * wp || D < 1)
    return -1;
  if ((size_t)C * (size_t)lt * (size_t)lh * (size_t)lw != latent_n)
    return -1;

  size_t wne = 0, nmod = 0, nb = 0;
  float *w = wan_load_tensor_f32(ctx, "dit.head.head.weight", &wne);
  float *bias = wan_load_tensor_f32(ctx, "dit.head.head.bias", &nb);
  float *mod = wan_load_tensor_f32(ctx, "dit.head.modulation", &nmod);
  if (!w || wne != (size_t)out_per * (size_t)D) {
    free(w);
    free(bias);
    free(mod);
    return -1;
  }
  float *normed = calloc((size_t)T * (size_t)D, sizeof(float));
  float *head = calloc((size_t)T * (size_t)out_per, sizeof(float));
  float *vol = calloc(latent_n, sizeof(float));
  if (!normed || !head || !vol) {
    free(w);
    free(bias);
    free(mod);
    free(normed);
    free(head);
    free(vol);
    return -1;
  }

  /* Head: LN → AdaLN(mod+e) → Linear(+bias). */
  uma_wan_layernorm_f32(normed, tok, NULL, NULL, T, D, 1e-6f);
  if (mod && nmod == (size_t)(2 * D) && e_time) {
    for (int t = 0; t < T; t++) {
      float *row = normed + (size_t)t * (size_t)D;
      for (int i = 0; i < D; i++) {
        float sh = mod[0 * D + i] + e_time[i];
        float sc = mod[1 * D + i] + e_time[i];
        row[i] = row[i] * (1.f + sc) + sh;
      }
    }
    static int logged_hm;
    if (!logged_hm) {
      fprintf(stderr, "wan-c: DiT head.modulation + LN AdaLN\n");
      logged_hm = 1;
    }
  }
  uma_wan_gemm_f32(head, normed, w, T, out_per, D);
  free(w);
  free(normed);
  free(mod);
  if (bias && nb == (size_t)out_per) {
    for (int t = 0; t < T; t++)
      for (int i = 0; i < out_per; i++)
        head[(size_t)t * (size_t)out_per + (size_t)i] += bias[i];
  }
  free(bias);

  /* Wan unpatchify: view [F,H,W,pt,ph,pw,C] then einsum fhwpqrc→cfphqwr.
   * Head vector packing is patch-major with channel innermost (not C-major). */
  memset(vol, 0, latent_n * sizeof(float));
  for (int tt = 0; tt < tp; tt++)
    for (int th = 0; th < hp; th++)
      for (int tw = 0; tw < wp; tw++) {
        size_t ti =
            (((size_t)tt * (size_t)hp + (size_t)th) * (size_t)wp + (size_t)tw);
        const float *vec = head + ti * (size_t)out_per;
        for (int pti = 0; pti < pt; pti++)
          for (int phi = 0; phi < ph; phi++)
            for (int pwi = 0; pwi < pw; pwi++)
              for (int c = 0; c < C; c++) {
                size_t vi =
                    (((((size_t)pti * (size_t)ph + (size_t)phi) * (size_t)pw +
                       (size_t)pwi) *
                      (size_t)C) +
                     (size_t)c);
                int t = tt * pt + pti;
                int h = th * ph + phi;
                int ww = tw * pw + pwi;
                size_t oi =
                    (((((size_t)c * (size_t)lt + (size_t)t) * (size_t)lh +
                       (size_t)h) *
                      (size_t)lw) +
                     (size_t)ww);
                vol[oi] = vec[vi];
              }
      }
  free(head);
  memcpy(latent, vol, latent_n * sizeof(float));
  free(vol);
  return 0;
}

/* Pack text into x_dit_tctx; *text_mirror_out owns host copy for re-PUT. */
static int dit_pack_text(wan_ctx *ctx, const float *text_emb, size_t text_n,
                         int D, const char **tk_out, const char **tv_out,
                         int *Tk_out, float **text_mirror_out) {
  *tk_out = NULL;
  *tv_out = NULL;
  *Tk_out = 0;
  if (text_mirror_out)
    *text_mirror_out = NULL;
  if (!text_emb || text_n < 1 || D < 1)
    return 0;

  int Td = ctx->cfg.text_dim;
  float *proj = NULL;
  size_t proj_n = 0;
  const float *src = text_emb;
  size_t src_n = text_n;

  /* Project text_dim → dim when text_embedding present.
   * Wan pads T5 u[:seq] to text_len with zeros *before* the MLP — bias on
   * padded rows is intentional; do not trim pads pre-MLP. */
  if (Td > 0 && text_n >= (size_t)Td &&
      wan_gguf_has(ctx, "dit.text_embedding.0.weight")) {
    int active = (int)(text_n / (size_t)Td);
    if (active < 1)
      active = 1;
    if (active > ctx->cfg.text_len)
      active = ctx->cfg.text_len;
    /* Drop only trailing all-zero rows beyond real T5 seq (already trimmed
     * by T5 encode); then pad up to text_len for the MLP. */
    while (active > 1) {
      const float *row = text_emb + (size_t)(active - 1) * (size_t)Td;
      int nonzero = 0;
      for (int d = 0; d < Td; d++) {
        if (row[d] != 0.f) {
          nonzero = 1;
          break;
        }
      }
      if (nonzero)
        break;
      active--;
    }
    int pad_rows = ctx->cfg.text_len > 0 ? ctx->cfg.text_len : active;
    if (pad_rows < active)
      pad_rows = active;
    float *padded = calloc((size_t)pad_rows * (size_t)Td, sizeof(float));
    if (!padded)
      return -1;
    memcpy(padded, text_emb, (size_t)active * (size_t)Td * sizeof(float));

    size_t wne = 0, wne2 = 0, nb0 = 0, nb2 = 0;
    float *W = wan_load_tensor_f32(ctx, "dit.text_embedding.0.weight", &wne);
    float *W2 = wan_load_tensor_f32(ctx, "dit.text_embedding.2.weight", &wne2);
    float *b0 = wan_load_tensor_f32(ctx, "dit.text_embedding.0.bias", &nb0);
    float *b2 = wan_load_tensor_f32(ctx, "dit.text_embedding.2.bias", &nb2);
    if (W && wne == (size_t)D * (size_t)Td) {
      proj = calloc((size_t)pad_rows * (size_t)D, sizeof(float));
      float *mid = calloc((size_t)pad_rows * (size_t)D, sizeof(float));
      if (proj && mid) {
        uma_wan_gemm_f32(mid, padded, W, pad_rows, D, Td);
        if (b0 && nb0 == (size_t)D)
          for (int r = 0; r < pad_rows; r++)
            for (int d = 0; d < D; d++)
              mid[(size_t)r * (size_t)D + (size_t)d] += b0[d];
        gelu_tanh_inplace(mid, (size_t)pad_rows * (size_t)D);
        if (W2 && wne2 == (size_t)D * (size_t)D) {
          uma_wan_gemm_f32(proj, mid, W2, pad_rows, D, D);
          if (b2 && nb2 == (size_t)D)
            for (int r = 0; r < pad_rows; r++)
              for (int d = 0; d < D; d++)
                proj[(size_t)r * (size_t)D + (size_t)d] += b2[d];
          static int logged_te;
          if (!logged_te) {
            fprintf(stderr,
                    "wan-c: dit.text_embedding MLP pad=%d active=%d "
                    "(Linear→GELU→Linear)+bias\n",
                    pad_rows, active);
            logged_te = 1;
          }
        } else {
          memcpy(proj, mid, (size_t)pad_rows * (size_t)D * sizeof(float));
        }
        src = proj;
        src_n = (size_t)pad_rows * (size_t)D;
        proj_n = src_n;
      } else {
        free(proj);
        proj = NULL;
      }
      free(mid);
    }
    free(padded);
    free(W);
    free(W2);
    free(b0);
    free(b2);
  }

  int Tk = (int)(src_n / (size_t)D);
  if (Tk < 1 && src_n >= (size_t)D)
    Tk = 1;
  int tk_cap = ctx->cfg.text_len > 0 ? ctx->cfg.text_len : 512;
  const char *etk = getenv("WAN_DIT_TEXT_TOK");
  if (etk && etk[0]) {
    int c = atoi(etk);
    if (c >= 1 && c <= 512)
      tk_cap = c;
  }
  if (Tk > tk_cap)
    Tk = tk_cap;
  /* After Wan-style pad+MLP, do not trim — pad rows are non-zero via bias. */
  if (Tk < 1) {
    free(proj);
    return 0;
  }
  size_t tbytes = (size_t)Tk * (size_t)D * sizeof(float);
  float *pack = calloc((size_t)Tk * (size_t)D, sizeof(float));
  if (!pack) {
    free(proj);
    return -1;
  }
  size_t copy = tbytes;
  if (copy > src_n * sizeof(float))
    copy = src_n * sizeof(float);
  memcpy(pack, src, copy);
  free(proj);
  (void)proj_n;

  const char *tk = "x_dit_tctx";
  if (uma_buf_pool_alloc(ctx->bufs, tk, tbytes) != 0 ||
      uma_buf_pool_put(ctx->bufs, tk, pack, tbytes) != 0) {
    free(pack);
    return -1;
  }
  /* Keep host mirror — BANK weight churn can drop x_dit_tctx mid-CFG. */
  if (text_mirror_out)
    *text_mirror_out = pack;
  else
    free(pack);
  *tk_out = tk;
  *tv_out = tk; /* single context buffer; K/V projected per-block */
  *Tk_out = Tk;
  return 0;
}

static int dit_broker(wan_ctx *ctx, float *latent, size_t n, int step,
                      const float *text_emb, size_t text_n) {
  if (!ctx->caps.attn_full || !ctx->caps.gemm_f16 || !ctx->caps.affine)
    return dit_host(ctx, latent, n, step);

  int real = wan_gguf_has(ctx, "dit.blocks.0.self_attn.q.weight");
  int rows, D, H, KV, HD, Ffn;
  float *tok = NULL;
  int tp = 0, hp = 0, wp = 0, lt = 0, lh = 0, lw = 0;
  int use_real_geom = 0;

  if (real && ctx->gen_lt > 0 &&
      dit_patch_host(ctx, latent, n, &tok, &rows, &D, &lt, &lh, &lw, &tp, &hp,
                     &wp) == 0) {
    use_real_geom = 1;
    Ffn = ctx->cfg.ffn_dim;
    if (ctx->cfg.num_heads > 0 && (D % ctx->cfg.num_heads) == 0) {
      H = ctx->cfg.num_heads;
      HD = D / H;
      KV = H;
    } else {
      dit_head_geom(D, &H, &KV, &HD);
    }
    fprintf(stderr,
            "wan-c: DiT real weights dim=%d ffn=%d heads=%d tokens=%d "
            "grid=%dx%dx%d→%dx%dx%d\n",
            D, Ffn, H, rows, lt, lh, lw, tp, hp, wp);
  } else {
    free(tok);
    tok = NULL;
    if (dit_dims(n, ctx->cfg.z_channels, &rows, &D) != 0)
      return -1;
    dit_head_geom(D, &H, &KV, &HD);
    Ffn = D * 2;
    if (Ffn < 16)
      Ffn = 16;
  }

  size_t nbytes = (size_t)rows * (size_t)D * sizeof(float);
  const char *bs = "x_dit_s";
  if (uma_buf_pool_alloc(ctx->bufs, bs, nbytes) != 0)
    return -1;
  if (use_real_geom) {
    if (uma_buf_pool_put(ctx->bufs, bs, tok, nbytes) != 0) {
      free(tok);
      return -1;
    }
    /* Stage dump for DiT A/B (patch tokens). */
    if (step == 0) {
      const char *dump = getenv("WAN_DUMP_DIR");
      if (dump && dump[0]) {
        char path[768];
        snprintf(path, sizeof(path), "%s/patch_tok.f32", dump);
        FILE *f = fopen(path, "wb");
        if (f) {
          fwrite(tok, sizeof(float), (size_t)rows * (size_t)D, f);
          fclose(f);
          fprintf(stderr, "wan-c: dumped patch_tok [%d,%d]\n", rows, D);
        }
      }
    }
  } else {
    /* Scaffold path: optional TOK3 rematch on CTHW-as-tokens. */
    const char *bvol = "x_dit_vol";
    char kind[64];
    int glt = ctx->gen_lt, glh = ctx->gen_lh, glw = ctx->gen_lw;
    if (glt < 1 || glh < 1 || glw < 1 ||
        (size_t)D * (size_t)glt * (size_t)glh * (size_t)glw != n) {
      glt = rows;
      glh = 1;
      glw = 1;
    }
    snprintf(kind, sizeof(kind), "1_%d_%d_%d_%d", D, glt, glh, glw);
    int use_layout = ctx->caps.tok3 && ctx->caps.ncdhw3 &&
                     ((size_t)D * (size_t)glt * (size_t)glh * (size_t)glw == n);
    if (use_layout) {
      if (uma_buf_pool_alloc(ctx->bufs, bvol, n * 4) != 0 ||
          uma_buf_pool_put(ctx->bufs, bvol, latent, n * 4) != 0 ||
          wan_graph_tok3(ctx, bvol, bs, kind) != 0) {
        if (uma_buf_pool_put(ctx->bufs, bs, latent, n * 4) != 0)
          return -1;
      }
    } else if (uma_buf_pool_put(ctx->bufs, bs, latent, n * 4) != 0) {
      return -1;
    }
  }

  const char *tk = NULL;
  const char *tv = NULL;
  int Tk = 0;
  float *text_mirror = NULL;
  if (dit_pack_text(ctx, text_emb, text_n, D, &tk, &tv, &Tk, &text_mirror) !=
      0) {
    free(tok);
    return -1;
  }

  float *e0 = calloc((size_t)6 * (size_t)D, sizeof(float));
  float *e_time = calloc((size_t)D, sizeof(float));
  int have_time = 0;
  if (e0 && use_real_geom &&
      dit_time_proj6(ctx, step, D, e0, e_time) == 0) {
    have_time = 1;
    static int logged_tp;
    if (!logged_tp) {
      fprintf(stderr, "wan-c: DiT time_embedding→time_projection ok\n");
      logged_tp = 1;
    }
    if (step == 0) {
      const char *dump = getenv("WAN_DUMP_DIR");
      if (dump && dump[0]) {
        char path[768];
        FILE *f;
        snprintf(path, sizeof(path), "%s/time_e.f32", dump);
        f = fopen(path, "wb");
        if (f) {
          fwrite(e_time, sizeof(float), (size_t)D, f);
          fclose(f);
        }
        snprintf(path, sizeof(path), "%s/time_e0.f32", dump);
        f = fopen(path, "wb");
        if (f) {
          fwrite(e0, sizeof(float), (size_t)6 * (size_t)D, f);
          fclose(f);
        }
        fprintf(stderr, "wan-c: dumped time_e[%d] time_e0[6,%d] t=%.2f\n", D, D,
                ctx->gen_t);
      }
    }
  } else {
    free(e_time);
    e_time = NULL;
  }

  int nblocks = 5;
  /* Cap to blocks present in weight sources (safetensors → up to 30). */
  if (use_real_geom) {
    int maxb = 0;
    for (int b = 0; b < 30; b++) {
      char nm[96];
      snprintf(nm, sizeof(nm), "dit.blocks.%d.self_attn.q.weight", b);
      if (!wan_gguf_has(ctx, nm))
        break;
      maxb = b + 1;
    }
    /* Prefer deeper when full DiT is mmapped; WAN_DIT_BLOCKS overrides. */
    if (maxb > 0) {
      nblocks = maxb;
      if (ctx->st && nblocks > 20)
        nblocks = 30; /* full DiT when mmapped; WAN_DIT_BLOCKS overrides */
    }
  }
  const char *envb = getenv("WAN_DIT_BLOCKS");
  if (envb && envb[0]) {
    int nb = atoi(envb);
    if (nb >= 1 && nb <= 30)
      nblocks = nb;
  }
  if (use_real_geom) {
    static int logged_nb;
    if (!logged_nb) {
      fprintf(stderr, "wan-c: DiT running %d blocks\n", nblocks);
      logged_nb = 1;
    }
  }

  float *mirror = NULL;
  char resp[512];
  if (use_real_geom) {
    mirror = tok; /* patch tokens; refreshed via BUF_GET after each block */
  } else {
    mirror = calloc(nbytes / sizeof(float), sizeof(float));
    if (!mirror)
      return -1;
    memcpy(mirror, latent, nbytes);
  }

  for (int b = 0; b < nblocks; b++) {
    /* Re-PUT tokens / text each block — broker may drop under CFG multi-step. */
    if (uma_buf_pool_ensure_put(ctx->bufs, bs, mirror, nbytes) != 0) {
      fprintf(stderr, "wan-c: DiT ensure x_dit_s failed before block %d\n", b);
      free(e0);
      free(e_time);
      free(text_mirror);
      if (!use_real_geom)
        free(mirror);
      free(tok);
      return -1;
    }
    if (text_mirror && tk && Tk > 0) {
      size_t tb = (size_t)Tk * (size_t)D * sizeof(float);
      if (uma_buf_pool_ensure_put(ctx->bufs, tk, text_mirror, tb) != 0) {
        fprintf(stderr, "wan-c: DiT ensure x_dit_tctx failed before block %d\n",
                b);
        free(e0);
        free(e_time);
        free(text_mirror);
        if (!use_real_geom)
          free(mirror);
        free(tok);
        return -1;
      }
    }
    if (dit_block_broker(ctx, bs, rows, D, H, KV, HD, Ffn, step, b, tk, Tk,
                         have_time ? e0 : NULL, mirror, text_mirror) != 0) {
      fprintf(stderr, "wan-c: DiT block %d failed\n", b);
      free(e0);
      free(e_time);
      free(text_mirror);
      if (!use_real_geom)
        free(mirror);
      free(tok);
      return -1;
    }
    size_t got_m = 0;
    if (uma_client_buf_get(ctx->uma, bs, mirror, nbytes, &got_m, resp,
                           sizeof(resp)) != 0 ||
        got_m != nbytes) {
      fprintf(stderr, "wan-c: DiT mirror GET failed block %d: %.120s\n", b,
              resp);
      free(e0);
      free(e_time);
      free(text_mirror);
      if (!use_real_geom)
        free(mirror);
      free(tok);
      return -1;
    }
  }
  free(e0);

  size_t got = 0;
  if (use_real_geom) {
    if (dit_unpatch_host(ctx, mirror, rows, D, lt, lh, lw, tp, hp, wp, latent,
                         n, e_time) != 0) {
      fprintf(stderr, "wan-c: DiT unpatch/head failed\n");
      free(e_time);
      free(text_mirror);
      free(tok);
      return -1;
    }
    free(e_time);
    free(text_mirror);
    free(tok);
    return 0;
  }

  free(mirror);

  free(e_time);
  free(text_mirror);
  free(tok);
  if (uma_client_buf_get(ctx->uma, bs, latent, n * 4, &got, resp,
                         sizeof(resp)) != 0 ||
      got != n * 4) {
    fprintf(stderr, "wan-c: DiT BUF_GET failed: %.160s\n", resp);
    return -1;
  }
  return 0;
}

int wan_dit_denoise(wan_ctx *ctx, float *latent, size_t n, int step,
                    const float *text_emb, size_t text_n) {
  if (!ctx || !latent || n < 1)
    return -1;
  if (wan_env_local() || ctx->local_mode)
    return dit_host(ctx, latent, n, step);
  if (!ctx->uma || !ctx->bufs) {
    fprintf(stderr, "wan-c: DiT needs UMA client (or UMA_WAN_LOCAL=1)\n");
    return -1;
  }
  return dit_broker(ctx, latent, n, step, text_emb, text_n);
}
