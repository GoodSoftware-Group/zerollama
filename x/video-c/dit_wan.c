#include "wan_internal.h"
#include "wan_profile.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * DiT denoise (F0789–F0882 + quality slice):
 *   Real AdaLN from time_embedding/time_projection + block.modulation
 *   Cross-attn uses dit.blocks.*.cross_attn.* weights
 *   FFN: F0993 FFN_GELU when persist BANK (WAN_DIT_HOST_FFN=1 → full host,
 *   kills persist; WAN_DIT_HOST_FFN_CORE=1 → broker LN/AdaLN/resid + host
 *   Accelerate FFN_GELU, rematch-safe, no bias — Brick 13).
 *   Self-attn: F0992 ATTN_NAMED t= windows (WAN_DIT_QCHUNK, default 5460).
 *   F0994: bank block weights once; bind per block (WAN_DIT_NO_PERSIST=1 off).
 *   Persist keeps x_dit_s on broker (WAN_DIT_MIRROR=1 restores per-block shuttle).
 *   Cross norm3 LN+AFFINE on broker when persist.
 * Scaffold (eye) when GGUF tensors missing.
 */

static int dit_env_truthy(const char *name) {
  const char *e = getenv(name);
  return e && e[0] && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y' ||
                       e[0] == 't' || e[0] == 'T');
}

/* Rows per ATTN Q window. Default: full T; if T>5460 and unset → 5460 (F0995). */
static int dit_qchunk_rows(int T) {
  const char *e = getenv("WAN_DIT_QCHUNK");
  int c = (e && e[0]) ? atoi(e) : 0;
  if (c < 1)
    c = (T > 5460) ? 5460 : T;
  if (c > T)
    c = T;
  if (c < 1)
    c = T;
  return c;
}

/* Rows per FFN_GELU window. Default: full T; if T>4096 and unset → 4096. */
static int dit_ffn_chunk_rows(int T) {
  const char *e = getenv("WAN_DIT_FFN_CHUNK");
  int c = (e && e[0]) ? atoi(e) : 0;
  if (c < 1)
    c = (T > 4096) ? 4096 : T;
  if (c > T)
    c = T;
  if (c < 1)
    c = T;
  return c;
}

static int dit_persist_enabled(const wan_ctx *ctx) {
  if (!ctx || !ctx->bufs || ctx->local_mode)
    return 0;
  if (dit_env_truthy("WAN_DIT_NO_PERSIST") || dit_env_truthy("WAN_DIT_HOST_FFN"))
    return 0;
  return ctx->caps.ffn_gelu && ctx->caps.gemm_f16 && ctx->caps.attn_full;
}

static int dit_bank_put_tensor(wan_ctx *ctx, const char *bank_key,
                               const char *gguf_name, size_t expect) {
  size_t nw = 0;
  const float *w = wan_borrow_tensor_f32(ctx, gguf_name, &nw);
  if (!w || (expect > 0 && nw != expect))
    return -1;
  return uma_buf_pool_bank_put(ctx->bufs, bank_key, w, nw * sizeof(float));
}

/* F0994: BANK_PUT all block dense weights once (keys blocks.{i}.*). */
static int dit_persist_bank_all(wan_ctx *ctx, int nblocks, int D, int Ffn) {
  if (!ctx || nblocks < 1 || D < 1 || Ffn < 1)
    return -1;
  if (ctx->dit_persist_ready && ctx->dit_persist_blocks >= nblocks)
    return 0;
  /* Cross-process: if daemon already has the bank from a prior video-cli
   * invocation, skip the 5.6 GiB re-PUT. */
  {
    char resp[512] = {0};
    if (ctx->uma &&
        uma_client_bank_status(ctx->uma, "wanc", resp, sizeof(resp)) == 0) {
      const char *kp = strstr(resp, "keys=");
      if (kp) {
        int k = atoi(kp + 5);
        if (k >= nblocks * 10) {
          (void)uma_buf_pool_ensure_bank_open(ctx->bufs);
          ctx->dit_persist_ready = 1;
          ctx->dit_persist_blocks = nblocks;
          fprintf(stderr,
                  "wan-c: DiT BANK_PUT skip daemon already has %d keys\n", k);
          return 0;
        }
      }
    }
  }
  size_t dd = (size_t)D * (size_t)D;
  size_t fd = (size_t)Ffn * (size_t)D;
  size_t df = (size_t)D * (size_t)Ffn;
  size_t nbytes = 0;
  int nkeys = 0;
  fprintf(stderr, "wan-c: DiT BANK_PUT persist %d blocks …\n", nblocks);
  for (int li = 0; li < nblocks; li++) {
    char gname[128], key[160];
    struct {
      const char *suf;
      const char *gguf_fmt;
      size_t ne;
    } items[] = {
        {"self_attn.q.weight", "dit.blocks.%d.self_attn.q.weight", dd},
        {"self_attn.k.weight", "dit.blocks.%d.self_attn.k.weight", dd},
        {"self_attn.v.weight", "dit.blocks.%d.self_attn.v.weight", dd},
        {"self_attn.o.weight", "dit.blocks.%d.self_attn.o.weight", dd},
        {"cross_attn.q.weight", "dit.blocks.%d.cross_attn.q.weight", dd},
        {"cross_attn.k.weight", "dit.blocks.%d.cross_attn.k.weight", dd},
        {"cross_attn.v.weight", "dit.blocks.%d.cross_attn.v.weight", dd},
        {"cross_attn.o.weight", "dit.blocks.%d.cross_attn.o.weight", dd},
        {"ffn.0.weight", "dit.blocks.%d.ffn.0.weight", fd},
        {"ffn.2.weight", "dit.blocks.%d.ffn.2.weight", df},
    };
    for (size_t i = 0; i < sizeof(items) / sizeof(items[0]); i++) {
      snprintf(gname, sizeof(gname), items[i].gguf_fmt, li);
      if (!wan_gguf_has(ctx, gname)) {
        if (i < 4 || i >= 8) /* self + ffn required */
          return -1;
        continue; /* cross optional */
      }
      snprintf(key, sizeof(key), "blocks.%d.%s", li, items[i].suf);
      if (dit_bank_put_tensor(ctx, key, gname, items[i].ne) != 0) {
        fprintf(stderr, "wan-c: BANK_PUT fail %s\n", key);
        return -1;
      }
      nbytes += items[i].ne * sizeof(float);
      nkeys++;
    }
  }
  ctx->dit_persist_blocks = nblocks;
  ctx->dit_persist_ready = 1;
  fprintf(stderr,
          "wan-c: DiT BANK_PUT persist OK keys=%d ~%.1f MiB (F0994)\n", nkeys,
          nbytes / (1024.0 * 1024.0));
  return 0;
}

static int dit_persist_bind_block(wan_ctx *ctx, int block, int have_cross) {
  if (!ctx || !ctx->dit_persist_ready)
    return -1;
  /* F0703: one BANK_BINDS IPC per block (was 10× BANK_BIND). */
  char pairs[768];
  int n = snprintf(
      pairs, sizeof(pairs),
      "blocks.%d.self_attn.q.weight:x_dit_Wq,"
      "blocks.%d.self_attn.k.weight:x_dit_Wk,"
      "blocks.%d.self_attn.v.weight:x_dit_Wv,"
      "blocks.%d.self_attn.o.weight:x_dit_Wo,"
      "blocks.%d.ffn.0.weight:x_dit_Wu,"
      "blocks.%d.ffn.2.weight:x_dit_Wd",
      block, block, block, block, block, block);
  if (n < 0 || (size_t)n >= sizeof(pairs))
    return -1;
  if (have_cross) {
    int n2 = snprintf(
        pairs + n, sizeof(pairs) - (size_t)n,
        ",blocks.%d.cross_attn.q.weight:x_dit_Wqc,"
        "blocks.%d.cross_attn.k.weight:x_dit_Wkc,"
        "blocks.%d.cross_attn.v.weight:x_dit_Wvc,"
        "blocks.%d.cross_attn.o.weight:x_dit_Woc",
        block, block, block, block);
    if (n2 < 0 || (size_t)n + (size_t)n2 >= sizeof(pairs))
      return -1;
  }
  if (uma_buf_pool_bank_binds(ctx->bufs, pairs) != 0) {
    char resp2[512] = {0};
    // Try direct client call to get detailed error
    if (ctx->uma) {
      uma_client_bank_binds(ctx->uma, "wanc", pairs, resp2, sizeof(resp2));
      fprintf(stderr, "wan-c: BANK_BINDS fail block=%d pairs=%.80s resp=%.120s\n",
              block, pairs, resp2);
    } else {
      fprintf(stderr, "wan-c: BANK_BINDS fail block=%d\n", block);
    }
    return -1;
  }
  return 0;
}

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
  const float *mod = wan_borrow_tensor_f32(ctx, name, &nm);
  if (!mod || !e0_6d || nm != (size_t)(6 * D))
    return -1;
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

/* F0994: Q/K/V bias via AFFINE + HEAD_RMSNORM (no host GET/PUT of acts). */
static int dit_qk_bias_norm_broker(wan_ctx *ctx, const char *bq, const char *bk,
                                   const char *bv, int T, int Tk, int D,
                                   const char *weight_prefix, const char *bz) {
  if (!ctx || !ctx->bufs || !bq || !bk || !weight_prefix || !bz || T < 1 ||
      D < 1 || !ctx->caps.affine || !ctx->caps.head_rmsnorm)
    return -1;
  int krows = Tk > 0 ? Tk : T;
  char nq[128], nk[128], nbq[128], nbk[128], nbv[128];
  snprintf(nq, sizeof(nq), "%s.norm_q.weight", weight_prefix);
  snprintf(nk, sizeof(nk), "%s.norm_k.weight", weight_prefix);
  snprintf(nbq, sizeof(nbq), "%s.q.bias", weight_prefix);
  snprintf(nbk, sizeof(nbk), "%s.k.bias", weight_prefix);
  snprintf(nbv, sizeof(nbv), "%s.v.bias", weight_prefix);
  size_t n1 = 0, n2 = 0, nbb = 0;
  const float *wq = wan_borrow_tensor_f32(ctx, nq, &n1);
  const float *wk = wan_borrow_tensor_f32(ctx, nk, &n2);
  if (!wq || !wk || n1 != (size_t)D || n2 != (size_t)D)
    return -1;
  size_t dbytes = (size_t)D * 4;
  const char *bbq = "x_dit_bq";
  const char *bbk = "x_dit_bk";
  const char *bbv = "x_dit_bv";
  const char *bnq = "x_dit_nq";
  const char *bnk = "x_dit_nk";
  if (uma_buf_pool_ensure_put(ctx->bufs, bnq, wq, dbytes) != 0 ||
      uma_buf_pool_ensure_put(ctx->bufs, bnk, wk, dbytes) != 0)
    return -1;
  char nodes[1536];
  int n = 0, off = 0;
  const float *bq_b = wan_borrow_tensor_f32(ctx, nbq, &nbb);
  if (bq_b && nbb == (size_t)D) {
    if (uma_buf_pool_ensure_put(ctx->bufs, bbq, bq_b, dbytes) != 0)
      return -1;
    n = snprintf(nodes + off, sizeof(nodes) - (size_t)off,
                 "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; ", bq,
                 bq, bz, bbq, T, D);
    if (n < 0 || (size_t)n >= sizeof(nodes) - (size_t)off)
      return -1;
    off += n;
  }
  const float *bk_b = wan_borrow_tensor_f32(ctx, nbk, &nbb);
  if (bk_b && nbb == (size_t)D) {
    if (uma_buf_pool_ensure_put(ctx->bufs, bbk, bk_b, dbytes) != 0)
      return -1;
    n = snprintf(nodes + off, sizeof(nodes) - (size_t)off,
                 "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; ", bk,
                 bk, bz, bbk, krows, D);
    if (n < 0 || (size_t)n >= sizeof(nodes) - (size_t)off)
      return -1;
    off += n;
  }
  if (bv) {
    const float *bv_b = wan_borrow_tensor_f32(ctx, nbv, &nbb);
    if (bv_b && nbb == (size_t)D) {
      if (uma_buf_pool_ensure_put(ctx->bufs, bbv, bv_b, dbytes) != 0)
        return -1;
      n = snprintf(nodes + off, sizeof(nodes) - (size_t)off,
                   "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; ", bv,
                   bv, bz, bbv, krows, D);
      if (n < 0 || (size_t)n >= sizeof(nodes) - (size_t)off)
        return -1;
      off += n;
    }
  }
  n = snprintf(nodes + off, sizeof(nodes) - (size_t)off,
               "HEAD_RMSNORM@GPU! x=%s w=%s H=%d HD=%d ; "
               "HEAD_RMSNORM@GPU! x=%s w=%s H=%d HD=%d ; MARK@GPU?",
               bq, bnq, T, D, bk, bnk, krows, D);
  if (n < 0 || (size_t)n >= sizeof(nodes) - (size_t)off)
    return -1;
  return wan_submit_graph(ctx->uma, nodes);
}

/* AFFINE bias add: x += bias (gate=zeros). */
static int dit_add_bias_broker(wan_ctx *ctx, const char *buf, int rows, int D,
                               const char *bias_name, const char *bz) {
  if (!ctx || !buf || !bias_name || !bz || rows < 1 || D < 1 || !ctx->caps.affine)
    return -1;
  size_t nb = 0;
  const float *bias = wan_borrow_tensor_f32(ctx, bias_name, &nb);
  if (!bias || nb != (size_t)D)
    return -1;
  const char *bb = "x_dit_obias";
  if (uma_buf_pool_ensure_put(ctx->bufs, bb, bias, (size_t)D * 4) != 0)
    return -1;
  char nodes[320];
  snprintf(nodes, sizeof(nodes),
           "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; MARK@GPU?",
           buf, buf, bz, bb, rows, D);
  return wan_submit_graph(ctx->uma, nodes);
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
  int persist = ctx->dit_persist_ready;
  int have_cross = text_ctx && Tk > 0 && ctx->caps.attn_full &&
                   wan_gguf_has(ctx, "dit.blocks.0.cross_attn.q.weight");

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
  const char *bg = "x_dit_g";   /* gated delta before residual */
  const char *bgm = "x_dit_gm";   /* gate_sa - 1 for AFFINE */
  const char *bgs = "x_dit_gsa";  /* full gate_sa (gated RESIDUAL_ADD) */
  const char *bgf = "x_dit_gf";   /* gate_ff - 1 (legacy AFFINE path) */
  const char *bgff = "x_dit_gff"; /* full gate_ff (gated RESIDUAL_ADD) */
  const char *bz = "x_dit_z";     /* zeros */
  const char *bln = "x_dit_ln";   /* ones (LN weight) */
  /* F1156: BF16 TensorOps flash when the daemon advertises ATTN_NAMED_tc. */
  const char *tc = ctx->caps.attn_tc ? " tc=1" : "";

  /* AdaLN scale/shift on broker when persist (F0994 smoke recipe). */
  if (persist && real_mod) {
    float zbuf[1536], ones[1536], gm[1536], gf[1536];
    if (D > 1536)
      DIT_FAIL("dim>1536_mod");
    for (int i = 0; i < D; i++) {
      zbuf[i] = 0.f;
      ones[i] = 1.f;
      gm[i] = gate_sa[i] - 1.f;
      gf[i] = gate_ff[i] - 1.f;
    }
    if (uma_buf_pool_ensure_put(ctx->bufs, scn, sc, dbytes) != 0 ||
        uma_buf_pool_ensure_put(ctx->bufs, shn, sh, dbytes) != 0 ||
        uma_buf_pool_ensure_put(ctx->bufs, sc2, scb, dbytes) != 0 ||
        uma_buf_pool_ensure_put(ctx->bufs, sh2, shb, dbytes) != 0 ||
        uma_buf_pool_ensure_put(ctx->bufs, bgm, gm, dbytes) != 0 ||
        uma_buf_pool_ensure_put(ctx->bufs, bgs, gate_sa, dbytes) != 0 ||
        uma_buf_pool_ensure_put(ctx->bufs, bgf, gf, dbytes) != 0 ||
        uma_buf_pool_ensure_put(ctx->bufs, bgff, gate_ff, dbytes) != 0 ||
        uma_buf_pool_ensure_put(ctx->bufs, bz, zbuf, dbytes) != 0 ||
        uma_buf_pool_ensure_put(ctx->bufs, bln, ones, dbytes) != 0)
      DIT_FAIL("persist_adaln_put");
    static int logged_padaln;
    if (!logged_padaln) {
      fprintf(stderr,
              "wan-c: DiT AdaLN+gates on broker (AFFINE; persist)\n");
      logged_padaln = 1;
    }
  } else if (!real_mod) {
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

  char gname[128];

  if (persist) {
    double t_bind = wan_profile_on() ? wan_profile_now_ms() : 0.0;
    if (dit_persist_bind_block(ctx, block, have_cross) != 0)
      DIT_FAIL("persist_bind");
    if (wan_profile_on())
      wan_profile_add_ms("dit_bind", wan_profile_now_ms() - t_bind);
    static int logged_pb;
    if (!logged_pb) {
      fprintf(stderr, "wan-c: DiT persist BANK_BINDS per block (F0703/F0994)\n");
      logged_pb = 1;
    }
  } else if (!real_mod) {
    /* Scaffold / non-AdaLN: PUT weights each block. */
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

  if (have_cross && !persist) {
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
  } else if (have_cross && persist) {
    static int logged_xp;
    if (!logged_xp) {
      fprintf(stderr, "wan-c: DiT cross_attn via persist BANK_BIND\n");
      logged_xp = 1;
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
  if (persist && uma_buf_pool_alloc(ctx->bufs, bg, nbytes) != 0)
    DIT_FAIL("gate_alloc");
  if (have_cross &&
      (uma_buf_pool_alloc(ctx->bufs, bkc, tbytes) != 0 ||
       uma_buf_pool_alloc(ctx->bufs, bvc, tbytes) != 0))
    DIT_FAIL("cross_kv_alloc");

  /* AdaLN / text restore. Token mirror only when not persist — F0994 keeps
   * x_dit_s on the broker; re-PUTting a stale host copy would wipe progress.
   * Text ctx is put once per forward before the block loop — skip re-PUT. */
  int force_mirror = dit_env_truthy("WAN_DIT_MIRROR");
#define DIT_RESTORE_HOT()                                                      \
  do {                                                                         \
    if (!real_mod || persist) {                                                \
      if (uma_buf_pool_ensure_put(ctx->bufs, scn, sc, dbytes) != 0 ||          \
          uma_buf_pool_ensure_put(ctx->bufs, shn, sh, dbytes) != 0 ||          \
          uma_buf_pool_ensure_put(ctx->bufs, sc2, scb, dbytes) != 0 ||         \
          uma_buf_pool_ensure_put(ctx->bufs, sh2, shb, dbytes) != 0)           \
        DIT_FAIL("restore_adaln");                                             \
    }                                                                          \
    if ((!persist || force_mirror) && token_mirror &&                          \
        uma_buf_pool_ensure_put(ctx->bufs, bs, token_mirror, nbytes) != 0)     \
      DIT_FAIL("restore_x_dit_s");                                             \
    if ((!persist || force_mirror) && have_cross && text_mirror &&              \
        tbytes > 0 &&                                                          \
        uma_buf_pool_ensure_put(ctx->bufs, text_ctx, text_mirror, tbytes) !=   \
            0)                                                                 \
      DIT_FAIL("restore_x_dit_tctx");                                          \
  } while (0)
  DIT_RESTORE_HOT();
  if (persist && !force_mirror) {
    static int logged_nomirror;
    if (!logged_nomirror) {
      fprintf(stderr,
              "wan-c: DiT skip per-block token/text mirror (persist; "
              "WAN_DIT_MIRROR=1 to force)\n");
      logged_nomirror = 1;
    }
  }

  int flat = T * D;

  /* Self-attn (+ RoPE on Q/K every block when caps allow). */
  double t_self0 = wan_profile_on() ? wan_profile_now_ms() : 0.0;
  {
    float *hq = NULL, *hk = NULL, *hv = NULL;
    int use_self_fuse =
        real_mod && persist && !dit_env_truthy("WAN_DIT_NO_SELF_FUSE") &&
        ctx->caps.rope3 && ctx->caps.head_rmsnorm && ctx->caps.affine &&
        ctx->caps.attn_full;
    if (use_self_fuse) {
      /* Brick 5: LN→AdaLN→QKV→bias→RMS→RoPE then ATTN→O→gated (2 GRAPHs). */
      const char *bqr = "x_dit_qr";
      const char *bkr = "x_dit_kr";
      const char *bbq = "x_dit_bq";
      const char *bbk = "x_dit_bk";
      const char *bbv = "x_dit_bv";
      const char *bnq = "x_dit_nq";
      const char *bnk = "x_dit_nk";
      const char *bft = "x_rope_ft";
      const char *bfh = "x_rope_fh";
      const char *bfw = "x_rope_fw";
      char pref[96], nq[128], nk[128], nbq[128], nbk[128], nbv[128];
      snprintf(pref, sizeof(pref), "dit.blocks.%d.self_attn", block);
      snprintf(nq, sizeof(nq), "%s.norm_q.weight", pref);
      snprintf(nk, sizeof(nk), "%s.norm_k.weight", pref);
      snprintf(nbq, sizeof(nbq), "%s.q.bias", pref);
      snprintf(nbk, sizeof(nbk), "%s.k.bias", pref);
      snprintf(nbv, sizeof(nbv), "%s.v.bias", pref);
      size_t n1 = 0, n2 = 0, nbb = 0;
      const float *wq = wan_borrow_tensor_f32(ctx, nq, &n1);
      const float *wk = wan_borrow_tensor_f32(ctx, nk, &n2);
      if (!wq || !wk || n1 != (size_t)D || n2 != (size_t)D ||
          uma_buf_pool_ensure_put(ctx->bufs, bnq, wq, dbytes) != 0 ||
          uma_buf_pool_ensure_put(ctx->bufs, bnk, wk, dbytes) != 0 ||
          uma_buf_pool_alloc(ctx->bufs, bqr, nbytes) != 0 ||
          uma_buf_pool_alloc(ctx->bufs, bkr, nbytes) != 0 ||
          uma_buf_pool_alloc(ctx->bufs, bo, nbytes) != 0)
        DIT_FAIL("self_fuse_alloc");
      int gt = 0, gh = 0, gw = 0;
      if (wan_rope3_ensure_freqs(ctx, T, HD, &gt, &gh, &gw) != 0)
        DIT_FAIL("self_fuse_freqs");

      char pre[3072];
      int off = 0, n;
      n = snprintf(
          pre + off, sizeof(pre) - (size_t)off,
          "LAYERNORM_MUL@GPU! x=%s y=%s w=%s N=%d D=%d ; "
          "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; ",
          bs, bt, bln, T, D, bt, ba, scn, shn, T, D, ba, bq, bwq, T, D, D, ba,
          bk, bwk, T, D, D, ba, bv, bwv, T, D, D);
      if (n < 0 || (size_t)n >= sizeof(pre) - (size_t)off)
        DIT_FAIL("self_fuse_pre");
      off += n;
      const float *bq_b = wan_borrow_tensor_f32(ctx, nbq, &nbb);
      if (bq_b && nbb == (size_t)D &&
          uma_buf_pool_ensure_put(ctx->bufs, bbq, bq_b, dbytes) == 0) {
        n = snprintf(pre + off, sizeof(pre) - (size_t)off,
                     "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; ",
                     bq, bq, bz, bbq, T, D);
        if (n < 0 || (size_t)n >= sizeof(pre) - (size_t)off)
          DIT_FAIL("self_fuse_pre");
        off += n;
      }
      const float *bk_b = wan_borrow_tensor_f32(ctx, nbk, &nbb);
      if (bk_b && nbb == (size_t)D &&
          uma_buf_pool_ensure_put(ctx->bufs, bbk, bk_b, dbytes) == 0) {
        n = snprintf(pre + off, sizeof(pre) - (size_t)off,
                     "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; ",
                     bk, bk, bz, bbk, T, D);
        if (n < 0 || (size_t)n >= sizeof(pre) - (size_t)off)
          DIT_FAIL("self_fuse_pre");
        off += n;
      }
      const float *bv_b = wan_borrow_tensor_f32(ctx, nbv, &nbb);
      if (bv_b && nbb == (size_t)D &&
          uma_buf_pool_ensure_put(ctx->bufs, bbv, bv_b, dbytes) == 0) {
        n = snprintf(pre + off, sizeof(pre) - (size_t)off,
                     "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; ",
                     bv, bv, bz, bbv, T, D);
        if (n < 0 || (size_t)n >= sizeof(pre) - (size_t)off)
          DIT_FAIL("self_fuse_pre");
        off += n;
      }
      if (getenv("WAN_SELF_SUB_TIME")) {
        double t_sub = wan_profile_now_ms();
        n = snprintf(
            pre + off, sizeof(pre) - (size_t)off,
            "HEAD_RMSNORM@GPU! x=%s w=%s H=%d HD=%d ; "
            "HEAD_RMSNORM@GPU! x=%s w=%s H=%d HD=%d ; "
            "ROPE3@GPU! x=%s y=%s w=%s gate=%s up=%s T=%d H=%d HD=%d "
            "Gt=%d Gh=%d Gw=%d ; "
            "ROPE3@GPU! x=%s y=%s w=%s gate=%s up=%s T=%d H=%d HD=%d "
            "Gt=%d Gh=%d Gw=%d ; MARK@GPU?",
            bq, bnq, T, D, bk, bnk, T, D, bq, bqr, bft, bfh, bfw, T, H, HD, gt,
            gh, gw, bk, bkr, bft, bfh, bfw, T, H, HD, gt, gh, gw);
        if (n < 0 || (size_t)n >= sizeof(pre) - (size_t)off ||
            wan_submit_graph(ctx->uma, pre) != 0)
          DIT_FAIL("self_fuse_pre");
        double t_e = wan_profile_now_ms();
        if (wan_profile_on())
          fprintf(stderr, "wan-c: self_pre prep=%.1fms submit=%.1fms (block=%d)\n",
                  t_sub - t_self0, t_e - t_sub, block);
      } else {
        n = snprintf(
            pre + off, sizeof(pre) - (size_t)off,
            "HEAD_RMSNORM@GPU! x=%s w=%s H=%d HD=%d ; "
            "HEAD_RMSNORM@GPU! x=%s w=%s H=%d HD=%d ; "
            "ROPE3@GPU! x=%s y=%s w=%s gate=%s up=%s T=%d H=%d HD=%d "
            "Gt=%d Gh=%d Gw=%d ; "
            "ROPE3@GPU! x=%s y=%s w=%s gate=%s up=%s T=%d H=%d HD=%d "
            "Gt=%d Gh=%d Gw=%d ; MARK@GPU?",
            bq, bnq, T, D, bk, bnk, T, D, bq, bqr, bft, bfh, bfw, T, H, HD, gt,
            gh, gw, bk, bkr, bft, bfh, bfw, T, H, HD, gt, gh, gw);
        if (n < 0 || (size_t)n >= sizeof(pre) - (size_t)off ||
            wan_submit_graph(ctx->uma, pre) != 0)
          DIT_FAIL("self_fuse_pre");
      }
      static int logged_sf;
      if (!logged_sf) {
        fprintf(stderr,
                "wan-c: DiT self-attn LN→QKV→RMS→RoPE fused (persist; "
                "grid=%d×%d×%d)\n",
                gt, gh, gw);
        logged_sf = 1;
      }

      double t_attn0 = wan_profile_on() ? wan_profile_now_ms() : 0.0;
      const char *q_attn = bqr;
      const char *k_attn = bkr;
      int qchunk = dit_qchunk_rows(T);
      if (qchunk < T) {
        for (int r0 = 0; r0 < T; r0 += qchunk) {
          int nr = T - r0;
          if (nr > qchunk)
            nr = qchunk;
          if (wan_graph_attn_full_row(ctx, q_attn, k_attn, bv, bo, nr, T, H, KV,
                                      HD, r0) != 0)
            DIT_FAIL("self_fuse_attn_win");
        }
        /* O+gated after windowed ATTN */
        char ob[96];
        snprintf(ob, sizeof(ob), "dit.blocks.%d.self_attn.o.bias", block);
        size_t nob = 0;
        const float *obias = wan_borrow_tensor_f32(ctx, ob, &nob);
        const char *bob = "x_dit_obias";
        int have_ob = obias && nob == (size_t)D &&
                      uma_buf_pool_ensure_put(ctx->bufs, bob, obias, dbytes) ==
                          0;
        int use_gated = !dit_env_truthy("WAN_DIT_NO_GATED_RESID");
        char gr[768];
        if (use_gated && have_ob)
          n = snprintf(
              gr, sizeof(gr),
              "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
              "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
              "RESIDUAL_ADD@GPU! y=%s x=%s D=%d N=%d gate=%s ; MARK@GPU?",
              bo, bt, bwo, T, D, D, bt, bt, bz, bob, T, D, bs, bt, D, T, bgs);
        else if (use_gated)
          n = snprintf(
              gr, sizeof(gr),
              "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
              "RESIDUAL_ADD@GPU! y=%s x=%s D=%d N=%d gate=%s ; MARK@GPU?",
              bo, bt, bwo, T, D, D, bs, bt, D, T, bgs);
        else if (have_ob)
          n = snprintf(
              gr, sizeof(gr),
              "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
              "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
              "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
              "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@GPU?",
              bo, bt, bwo, T, D, D, bt, bt, bz, bob, T, D, bt, bg, bgm, bz, T,
              D, bg, bs, flat);
        else
          n = snprintf(
              gr, sizeof(gr),
              "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
              "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
              "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@GPU?",
              bo, bt, bwo, T, D, D, bt, bg, bgm, bz, T, D, bg, bs, flat);
        if (n < 0 || (size_t)n >= sizeof(gr) ||
            wan_submit_graph(ctx->uma, gr) != 0)
          DIT_FAIL("self_fuse_o");
      } else {
        char ob[96];
        snprintf(ob, sizeof(ob), "dit.blocks.%d.self_attn.o.bias", block);
        size_t nob = 0;
        const float *obias = wan_borrow_tensor_f32(ctx, ob, &nob);
        const char *bob = "x_dit_obias";
        int have_ob = obias && nob == (size_t)D &&
                      uma_buf_pool_ensure_put(ctx->bufs, bob, obias, dbytes) ==
                          0;
        int use_gated = !dit_env_truthy("WAN_DIT_NO_GATED_RESID");
        char gr[1024];
        if (use_gated && have_ob)
          n = snprintf(
              gr, sizeof(gr),
              "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
              "HD=%d kind=full%s ; "
              "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
              "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
              "RESIDUAL_ADD@GPU! y=%s x=%s D=%d N=%d gate=%s ; MARK@GPU?",
              q_attn, k_attn, bv, bo, T, T, H, KV, HD, tc, bo, bt, bwo, T, D, D, bt,
              bt, bz, bob, T, D, bs, bt, D, T, bgs);
        else if (use_gated)
          n = snprintf(
              gr, sizeof(gr),
              "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
              "HD=%d kind=full%s ; "
              "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
              "RESIDUAL_ADD@GPU! y=%s x=%s D=%d N=%d gate=%s ; MARK@GPU?",
              q_attn, k_attn, bv, bo, T, T, H, KV, HD, tc, bo, bt, bwo, T, D, D, bs,
              bt, D, T, bgs);
        else if (have_ob)
          n = snprintf(
              gr, sizeof(gr),
              "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
              "HD=%d kind=full%s ; "
              "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
              "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
              "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
              "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@GPU?",
              q_attn, k_attn, bv, bo, T, T, H, KV, HD, tc, bo, bt, bwo, T, D, D, bt,
              bt, bz, bob, T, D, bt, bg, bgm, bz, T, D, bg, bs, flat);
        else
          n = snprintf(
              gr, sizeof(gr),
              "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
              "HD=%d kind=full%s ; "
              "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
              "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
              "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@GPU?",
              q_attn, k_attn, bv, bo, T, T, H, KV, HD, tc, bo, bt, bwo, T, D, D, bt,
              bg, bgm, bz, T, D, bg, bs, flat);
        if (n < 0 || (size_t)n >= sizeof(gr) ||
            wan_submit_graph(ctx->uma, gr) != 0)
          DIT_FAIL("self_fuse_attn_o");
        static int logged_sfo;
        if (!logged_sfo) {
          fprintf(stderr,
                  "wan-c: DiT self-attn ATTN→O→gated fused (persist)\n");
          logged_sfo = 1;
        }
      }
      if (wan_profile_on())
        wan_profile_add_ms("dit_self_attn", wan_profile_now_ms() - t_attn0);
      if (block == 0)
        dit_dump_named_buf(ctx, bs, nbytes, "b0_post_sa.f32");
    } else if (real_mod && persist) {
      /* Broker LN+AdaLN+QKV (scales already on broker). */
      char qkv[1024];
      int n = snprintf(
          qkv, sizeof(qkv),
          "LAYERNORM_MUL@GPU! x=%s y=%s w=%s N=%d D=%d ; "
          "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; MARK@GPU?",
          bs, bt, bln, T, D, bt, ba, scn, shn, T, D, ba, bq, bwq, T, D, D, ba,
          bk, bwk, T, D, D, ba, bv, bwv, T, D, D);
      if (n < 0 || (size_t)n >= sizeof(qkv) ||
          wan_submit_graph(ctx->uma, qkv) != 0)
        DIT_FAIL("persist_qkv");
      static int logged_pqkv;
      if (!logged_pqkv) {
        fprintf(stderr,
                "wan-c: DiT self-attn LN+AdaLN+QKV on broker (persist)\n");
        logged_pqkv = 1;
      }
    } else if (real_mod) {
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
               "LAYERNORM_MUL@GPU! x=%s y=%s N=%d D=%d ; "
               "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
               "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
               "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
               "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; MARK@GPU?",
               bs, bt, T, D, bt, ba, scn, shn, T, D, ba, bq, bwq, T, D, D, ba,
               bk, bwk, T, D, D, ba, bv, bwv, T, D, D);
      if (wan_submit_graph(ctx->uma, qkv) != 0)
        DIT_FAIL("qkv");
    }

    if (!use_self_fuse) {
    {
      char pref[96];
      snprintf(pref, sizeof(pref), "dit.blocks.%d.self_attn", block);
      int qk_ok = -1;
      if (persist)
        qk_ok = dit_qk_bias_norm_broker(ctx, bq, bk, bv, T, T, D, pref, bz);
      if (qk_ok != 0)
        qk_ok = dit_qk_bias_norm_host(ctx, bq, bk, bv, T, T, D, pref);
      if (qk_ok == 0) {
        static int logged_qk;
        if (!logged_qk) {
          fprintf(stderr,
                  "wan-c: DiT QK RMSNorm + Linear biases (self)%s\n",
                  persist && ctx->caps.head_rmsnorm ? " [broker]" : "");
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
      int r_qk = -1;
      if (a_qr == 0 && a_kr == 0)
        r_qk = wan_graph_rope3_qk(ctx, bq, bqr, bk, bkr, T, H, HD);
      if (r_qk != 0 && a_qr == 0 && a_kr == 0) {
        /* Fallback: two single-op GRAPHs. */
        int r_q = wan_graph_rope3(ctx, bq, bqr, T, H, HD);
        int r_k = wan_graph_rope3(ctx, bk, bkr, T, H, HD);
        r_qk = (r_q == 0 && r_k == 0) ? 0 : -1;
      }
      if (a_qr == 0 && a_kr == 0 && r_qk == 0) {
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
                  "wan-c: DiT RoPE3D SKIPPED (alloc q=%d k=%d rope=%d) "
                  "— quality gap\n",
                  a_qr, a_kr, r_qk);
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
      {
        int qchunk = dit_qchunk_rows(T);
        /* F0992: ATTN_NAMED t= windows into full-seq q/out (no ROW_COPY). */
        if (qchunk < T) {
          static int logged_qc;
          if (!logged_qc) {
            fprintf(stderr,
                    "wan-c: DiT self-attn ATTN t= windows (chunk=%d T=%d)\n",
                    qchunk, T);
            logged_qc = 1;
          }
          for (int r0 = 0; r0 < T; r0 += qchunk) {
            int nr = T - r0;
            if (nr > qchunk)
              nr = qchunk;
            if (wan_graph_attn_full_row(ctx, q_attn, k_attn, bv, bo, nr, T, H,
                                        KV, HD, r0) != 0) {
              free(hq);
              free(hk);
              free(hv);
              DIT_FAIL("attn_t_window");
            }
          }
        } else if (wan_graph_attn_full(ctx, q_attn, k_attn, bv, bo, T, T, H, KV,
                                       HD) != 0) {
          free(hq);
          free(hk);
          free(hv);
          DIT_FAIL("attn_sa");
        }
      }
      free(hq);
      free(hk);
      free(hv);
      hq = hk = hv = NULL;
      if (persist) {
        /* O-proj + bias + gated resid — one sticky GPU GRAPH. */
        char ob[96];
        snprintf(ob, sizeof(ob), "dit.blocks.%d.self_attn.o.bias", block);
        size_t nob = 0;
        const float *obias = wan_borrow_tensor_f32(ctx, ob, &nob);
        const char *bob = "x_dit_obias";
        int have_ob = obias && nob == (size_t)D &&
                      uma_buf_pool_ensure_put(ctx->bufs, bob, obias, dbytes) ==
                          0;
        int use_gated = !dit_env_truthy("WAN_DIT_NO_GATED_RESID");
        char gr[768];
        int n;
        if (use_gated && have_ob)
          n = snprintf(
              gr, sizeof(gr),
              "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
              "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
              "RESIDUAL_ADD@GPU! y=%s x=%s D=%d N=%d gate=%s ; MARK@GPU?",
              bo, bt, bwo, T, D, D, bt, bt, bz, bob, T, D, bs, bt, D, T, bgs);
        else if (use_gated)
          n = snprintf(
              gr, sizeof(gr),
              "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
              "RESIDUAL_ADD@GPU! y=%s x=%s D=%d N=%d gate=%s ; MARK@GPU?",
              bo, bt, bwo, T, D, D, bs, bt, D, T, bgs);
        else if (have_ob)
          n = snprintf(
              gr, sizeof(gr),
              "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
              "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
              "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
              "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@GPU?",
              bo, bt, bwo, T, D, D, bt, bt, bz, bob, T, D, bt, bg, bgm, bz, T,
              D, bg, bs, flat);
        else
          n = snprintf(
              gr, sizeof(gr),
              "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
              "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
              "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@GPU?",
              bo, bt, bwo, T, D, D, bt, bg, bgm, bz, T, D, bg, bs, flat);
        if (n < 0 || (size_t)n >= sizeof(gr) ||
            wan_submit_graph(ctx->uma, gr) != 0)
          DIT_FAIL("persist_o_gate");
        static int logged_po;
        if (!logged_po) {
          fprintf(stderr,
                  "wan-c: DiT self-attn O+gate residual on broker (persist%s)\n",
                  use_gated ? "; gated RESIDUAL_ADD; fused O" : "; fused O");
          logged_po = 1;
        }
      } else {
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
        if (dit_gated_residual_host(ctx, bs, bt, T, D, gate_sa) != 0)
          DIT_FAIL("gate_sa");
      }
      if (block == 0)
        dit_dump_named_buf(ctx, bs, nbytes, "b0_post_sa.f32");
    } else {
      int n = snprintf(
          nodes, sizeof(nodes),
          "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
          "HD=%d kind=full%s ; "
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
          "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@CPU?",
          q_attn, k_attn, bv, bo, T, T, H, KV, HD, tc, bo, bt, bwo, T, D, D, bt, bs,
          flat);
      if (n < 0 || (size_t)n >= sizeof(nodes) ||
          wan_submit_graph(ctx->uma, nodes) != 0)
        return -1;
    }
    } /* !use_self_fuse */
  }
  if (wan_profile_on())
    wan_profile_add_ms("dit_self", wan_profile_now_ms() - t_self0);

  /* Cross-attn: norm3(x) → Q; K/V from text (Wan cross_attn_norm). */
  double t_cross0 = wan_profile_on() ? wan_profile_now_ms() : 0.0;
  if (have_cross) {
    int use_cross_fuse =
        persist && !dit_env_truthy("WAN_DIT_NO_CROSS_FUSE") &&
        ctx->caps.affine && ctx->caps.head_rmsnorm && ctx->caps.attn_full;
    char nw3[96], nb3[96];
    snprintf(nw3, sizeof(nw3), "dit.blocks.%d.norm3.weight", block);
    snprintf(nb3, sizeof(nb3), "dit.blocks.%d.norm3.bias", block);

    if (use_cross_fuse) {
      /* Brick 6: LN→AFFINE→QKV→bias→RMS then ATTN→O→resid (2 GRAPHs). */
      const char *bn3w = "x_dit_n3w";
      const char *bn3b = "x_dit_n3b";
      const char *bbq = "x_dit_bq";
      const char *bbk = "x_dit_bk";
      const char *bbv = "x_dit_bv";
      const char *bnq = "x_dit_nq";
      const char *bnk = "x_dit_nk";
      const char *bob = "x_dit_cobias";
      size_t nw = 0, nb = 0, n1 = 0, n2 = 0, nbb = 0, nob = 0;
      const float *w3 = wan_borrow_tensor_f32(ctx, nw3, &nw);
      const float *b3 = wan_borrow_tensor_f32(ctx, nb3, &nb);
      if (!w3 || !b3 || nw != (size_t)D || nb != (size_t)D ||
          uma_buf_pool_ensure_put(ctx->bufs, bn3w, w3, dbytes) != 0 ||
          uma_buf_pool_ensure_put(ctx->bufs, bn3b, b3, dbytes) != 0)
        DIT_FAIL("cross_fuse_n3");

      char pref[96], nq[128], nk[128], nbq[128], nbk[128], nbv[128], ob[96];
      snprintf(pref, sizeof(pref), "dit.blocks.%d.cross_attn", block);
      snprintf(nq, sizeof(nq), "%s.norm_q.weight", pref);
      snprintf(nk, sizeof(nk), "%s.norm_k.weight", pref);
      snprintf(nbq, sizeof(nbq), "%s.q.bias", pref);
      snprintf(nbk, sizeof(nbk), "%s.k.bias", pref);
      snprintf(nbv, sizeof(nbv), "%s.v.bias", pref);
      snprintf(ob, sizeof(ob), "%s.o.bias", pref);
      const float *wq = wan_borrow_tensor_f32(ctx, nq, &n1);
      const float *wk = wan_borrow_tensor_f32(ctx, nk, &n2);
      if (!wq || !wk || n1 != (size_t)D || n2 != (size_t)D ||
          uma_buf_pool_ensure_put(ctx->bufs, bnq, wq, dbytes) != 0 ||
          uma_buf_pool_ensure_put(ctx->bufs, bnk, wk, dbytes) != 0)
        DIT_FAIL("cross_fuse_rms");

      char pre[3072];
      int off = 0, n;
      n = snprintf(
          pre + off, sizeof(pre) - (size_t)off,
          "LAYERNORM_MUL@GPU! x=%s y=%s w=%s N=%d D=%d ; "
          "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; ",
          bs, bt, bn3w, T, D, bt, bt, bz, bn3b, T, D, bt, bq, bwqc, T, D, D,
          text_ctx, bkc, bwkc, Tk, D, D, text_ctx, bvc, bwvc, Tk, D, D);
      if (n < 0 || (size_t)n >= sizeof(pre) - (size_t)off)
        DIT_FAIL("cross_fuse_pre");
      off += n;
      const float *bq_b = wan_borrow_tensor_f32(ctx, nbq, &nbb);
      if (bq_b && nbb == (size_t)D &&
          uma_buf_pool_ensure_put(ctx->bufs, bbq, bq_b, dbytes) == 0) {
        n = snprintf(pre + off, sizeof(pre) - (size_t)off,
                     "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; ",
                     bq, bq, bz, bbq, T, D);
        if (n < 0 || (size_t)n >= sizeof(pre) - (size_t)off)
          DIT_FAIL("cross_fuse_pre");
        off += n;
      }
      const float *bk_b = wan_borrow_tensor_f32(ctx, nbk, &nbb);
      if (bk_b && nbb == (size_t)D &&
          uma_buf_pool_ensure_put(ctx->bufs, bbk, bk_b, dbytes) == 0) {
        n = snprintf(pre + off, sizeof(pre) - (size_t)off,
                     "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; ",
                     bkc, bkc, bz, bbk, Tk, D);
        if (n < 0 || (size_t)n >= sizeof(pre) - (size_t)off)
          DIT_FAIL("cross_fuse_pre");
        off += n;
      }
      const float *bv_b = wan_borrow_tensor_f32(ctx, nbv, &nbb);
      if (bv_b && nbb == (size_t)D &&
          uma_buf_pool_ensure_put(ctx->bufs, bbv, bv_b, dbytes) == 0) {
        n = snprintf(pre + off, sizeof(pre) - (size_t)off,
                     "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; ",
                     bvc, bvc, bz, bbv, Tk, D);
        if (n < 0 || (size_t)n >= sizeof(pre) - (size_t)off)
          DIT_FAIL("cross_fuse_pre");
        off += n;
      }
      n = snprintf(
          pre + off, sizeof(pre) - (size_t)off,
          "HEAD_RMSNORM@GPU! x=%s w=%s H=%d HD=%d ; "
          "HEAD_RMSNORM@GPU! x=%s w=%s H=%d HD=%d ; MARK@GPU?",
          bq, bnq, T, D, bkc, bnk, Tk, D);
      if (n < 0 || (size_t)n >= sizeof(pre) - (size_t)off ||
          wan_submit_graph(ctx->uma, pre) != 0)
        DIT_FAIL("cross_fuse_pre");
      static int logged_xf;
      if (!logged_xf) {
        fprintf(stderr,
                "wan-c: DiT cross-attn LN→QKV→RMS fused (persist)\n");
        logged_xf = 1;
      }

      const float *obias = wan_borrow_tensor_f32(ctx, ob, &nob);
      int have_ob = obias && nob == (size_t)D &&
                    uma_buf_pool_ensure_put(ctx->bufs, bob, obias, dbytes) == 0;
      if (have_ob)
        n = snprintf(
            nodes, sizeof(nodes),
            "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
            "HD=%d kind=full%s ; "
            "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
            "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
            "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@GPU?",
            bq, bkc, bvc, bo, T, Tk, H, KV, HD, tc, bo, bt, bwoc, T, D, D, bt, bt,
            bz, bob, T, D, bt, bs, flat);
      else
        n = snprintf(
            nodes, sizeof(nodes),
            "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
            "HD=%d kind=full%s ; "
            "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
            "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@GPU?",
            bq, bkc, bvc, bo, T, Tk, H, KV, HD, tc, bo, bt, bwoc, T, D, D, bt, bs,
            flat);
      if (n < 0 || (size_t)n >= sizeof(nodes) ||
          wan_submit_graph(ctx->uma, nodes) != 0)
        DIT_FAIL("cross_fuse_attn_o");
      static int logged_xfo;
      if (!logged_xfo) {
        fprintf(stderr,
                "wan-c: DiT cross-attn ATTN→O→resid fused (persist)\n");
        logged_xfo = 1;
      }
      if (block == 0)
        dit_dump_named_buf(ctx, bs, nbytes, "b0_post_cross.f32");
    } else {
    int n3_ok = -1;
    if (persist) {
      size_t nw = 0, nb = 0;
      const float *w3 = wan_borrow_tensor_f32(ctx, nw3, &nw);
      const float *b3 = wan_borrow_tensor_f32(ctx, nb3, &nb);
      const char *bn3w = "x_dit_n3w";
      const char *bn3b = "x_dit_n3b";
      if (w3 && b3 && nw == (size_t)D && nb == (size_t)D &&
          uma_buf_pool_ensure_put(ctx->bufs, bn3w, w3, dbytes) == 0 &&
          uma_buf_pool_ensure_put(ctx->bufs, bn3b, b3, dbytes) == 0) {
        char ln[384];
        int nn = snprintf(
            ln, sizeof(ln),
            "LAYERNORM_MUL@GPU! x=%s y=%s w=%s N=%d D=%d ; "
            "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; MARK@GPU?",
            bs, bt, bn3w, T, D, bt, bt, bz, bn3b, T, D);
        if (nn > 0 && (size_t)nn < sizeof(ln) &&
            wan_submit_graph(ctx->uma, ln) == 0)
          n3_ok = 0;
      }
      if (n3_ok == 0) {
        static int logged_n3b;
        if (!logged_n3b) {
          fprintf(stderr,
                  "wan-c: DiT cross_attn norm3 on broker (LN+AFFINE; persist)\n");
          logged_n3b = 1;
        }
      }
    }
    if (n3_ok != 0 && dit_layernorm_host(ctx, bs, bt, T, D, nw3, nb3) == 0) {
      static int logged_n3;
      if (!logged_n3) {
        fprintf(stderr, "wan-c: DiT cross_attn norm3 (affine LN)\n");
        logged_n3 = 1;
      }
      n3_ok = 0;
    }
    if (n3_ok != 0) {
      char ln[256];
      snprintf(ln, sizeof(ln),
               "LAYERNORM_MUL@GPU! x=%s y=%s N=%d D=%d ; MARK@GPU?", bs, bt, T,
               D);
      if (wan_submit_graph(ctx->uma, ln) != 0)
        DIT_FAIL("cross_ln");
    }
    {
      int n = snprintf(
          nodes, sizeof(nodes),
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; MARK@GPU?",
          bt, bq, bwqc, T, D, D, text_ctx, bkc, bwkc, Tk, D, D, text_ctx, bvc,
          bwvc, Tk, D, D);
      if (n < 0 || (size_t)n >= sizeof(nodes) ||
          wan_submit_graph(ctx->uma, nodes) != 0)
        return -1;
    }
    {
      char pref[96];
      snprintf(pref, sizeof(pref), "dit.blocks.%d.cross_attn", block);
      int qk_ok = -1;
      if (persist)
        qk_ok = dit_qk_bias_norm_broker(ctx, bq, bkc, bvc, T, Tk, D, pref, bz);
      if (qk_ok != 0)
        qk_ok = dit_qk_bias_norm_host(ctx, bq, bkc, bvc, T, Tk, D, pref);
      if (qk_ok == 0) {
        static int logged_xqk;
        if (!logged_xqk) {
          fprintf(stderr,
                  "wan-c: DiT QK RMSNorm + Linear biases (cross)%s\n",
                  persist && ctx->caps.head_rmsnorm ? " [broker]" : "");
          logged_xqk = 1;
        }
      }
    }
    {
      /* ATTN + O + bias + residual — one sticky GPU GRAPH when bias on broker. */
      char ob[96];
      snprintf(ob, sizeof(ob), "dit.blocks.%d.cross_attn.o.bias", block);
      size_t nob = 0;
      const float *obias = wan_borrow_tensor_f32(ctx, ob, &nob);
      const char *bob = "x_dit_cobias";
      int have_ob =
          persist && obias && nob == (size_t)D &&
          uma_buf_pool_ensure_put(ctx->bufs, bob, obias, dbytes) == 0;
      int n;
      if (have_ob) {
        n = snprintf(
            nodes, sizeof(nodes),
            "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
            "HD=%d kind=full%s ; "
            "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
            "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
            "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@GPU?",
            bq, bkc, bvc, bo, T, Tk, H, KV, HD, tc, bo, bt, bwoc, T, D, D, bt, bt,
            bz, bob, T, D, bt, bs, flat);
        if (n < 0 || (size_t)n >= sizeof(nodes) ||
            wan_submit_graph(ctx->uma, nodes) != 0)
          return -1;
      } else {
        n = snprintf(
            nodes, sizeof(nodes),
            "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
            "HD=%d kind=full%s ; "
            "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; MARK@GPU?",
            bq, bkc, bvc, bo, T, Tk, H, KV, HD, tc, bo, bt, bwoc, T, D, D);
        if (n < 0 || (size_t)n >= sizeof(nodes) ||
            wan_submit_graph(ctx->uma, nodes) != 0)
          return -1;
        if (persist) {
          if (dit_add_bias_broker(ctx, bt, T, D, ob, bz) != 0)
            (void)dit_add_bias_host(ctx, bt, T, D, ob);
        } else {
          (void)dit_add_bias_host(ctx, bt, T, D, ob);
        }
        n = snprintf(nodes, sizeof(nodes),
                     "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@GPU?", bt, bs,
                     flat);
        if (n < 0 || (size_t)n >= sizeof(nodes) ||
            wan_submit_graph(ctx->uma, nodes) != 0)
          return -1;
      }
      static int logged_xo;
      if (!logged_xo) {
        fprintf(stderr,
                "wan-c: DiT cross_attn ATTN+O+resid on broker%s\n",
                have_ob ? " (fused+bias)" : "");
        logged_xo = 1;
      }
    }
    if (block == 0)
      dit_dump_named_buf(ctx, bs, nbytes, "b0_post_cross.f32");
    } /* !use_cross_fuse */
  } else if (text_ctx && Tk > 0 && ctx->caps.attn_full) {
    /* Scaffold: reuse self Wq/Wo, treat text as K/V. */
    int n = snprintf(
        nodes, sizeof(nodes),
        "LAYERNORM_MUL@GPU! x=%s y=%s N=%d D=%d ; "
        "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
        "ATTN_NAMED@GPU! q=%s k=%s v=%s out=%s B=1 T=%d Tk=%d H=%d KV=%d "
        "HD=%d kind=full%s ; "
        "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
        "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@GPU?",
        bs, bt, T, D, bt, bq, bwq, T, D, D, bq, text_ctx, text_ctx, bo, T, Tk, H,
        KV, HD, tc, bo, bt, bwo, T, D, D, bt, bs, flat);
    if (n < 0 || (size_t)n >= sizeof(nodes) ||
        wan_submit_graph(ctx->uma, nodes) != 0)
      return -1;
  }
  if (wan_profile_on())
    wan_profile_add_ms("dit_cross", wan_profile_now_ms() - t_cross0);

  /* Dense FFN: F0993 FFN_GELU when persist BANK; else host Accelerate.
   * Never FFN_SILU — Wan needs GELU(tanh). FFN Linear biases omitted in
   * FFN_GELU (matches UMA F0994 smoke); down-bias still applied on host. */
  double t_ffn0 = wan_profile_on() ? wan_profile_now_ms() : 0.0;
  {
    int force_host_ffn = dit_env_truthy("WAN_DIT_HOST_FFN");
    int use_ffn_gelu =
        real_mod && persist && ctx->caps.ffn_gelu && !force_host_ffn;
    if (use_ffn_gelu) {
      int fchunk = dit_ffn_chunk_rows(T);
      size_t mid_bytes = (size_t)fchunk * (size_t)Ffn * sizeof(float);
      const char *bmid = "x_dit_ffc";
      if (uma_buf_pool_alloc(ctx->bufs, bmid, mid_bytes) != 0)
        DIT_FAIL("ffn_gelu_mid");

      char db[96];
      snprintf(db, sizeof(db), "dit.blocks.%d.ffn.2.bias", block);
      size_t nbb = 0;
      const float *dbias = wan_borrow_tensor_f32(ctx, db, &nbb);
      const char *bfdb = "x_dit_fdb";
      int use_gated = !dit_env_truthy("WAN_DIT_NO_GATED_RESID");
      int have_bias = dbias && nbb == (size_t)D;
      if (have_bias &&
          uma_buf_pool_ensure_put(ctx->bufs, bfdb, dbias, dbytes) != 0)
        DIT_FAIL("ffn_bias_put");

      if (use_gated && fchunk >= T) {
        /* Brick 13: broker LN/AdaLN + host Accelerate FFN_GELU (no bias, matches
         * F0993/F0994 broker) + broker gated resid. Opt-in; keeps persist BANK. */
        int host_ffn_core = dit_env_truthy("WAN_DIT_HOST_FFN_CORE");
        if (host_ffn_core) {
          char pre[512];
          int n = snprintf(
              pre, sizeof(pre),
              "LAYERNORM_MUL@GPU! x=%s y=%s w=%s N=%d D=%d ; "
              "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; MARK@GPU?",
              bs, bt, bln, T, D, bt, ba, sc2, sh2, T, D);
          if (n < 0 || (size_t)n >= sizeof(pre) ||
              wan_submit_graph(ctx->uma, pre) != 0)
            DIT_FAIL("ffn_core_pre");
          float *hx = malloc(nbytes);
          float *hy = malloc(nbytes);
          float *mid = malloc((size_t)T * (size_t)Ffn * sizeof(float));
          if (!hx || !hy || !mid) {
            free(hx);
            free(hy);
            free(mid);
            DIT_FAIL("ffn_core_alloc");
          }
          char resp[256];
          size_t got = 0;
          if (uma_client_buf_get(ctx->uma, ba, hx, nbytes, &got, resp,
                                 sizeof(resp)) != 0 ||
              got != nbytes) {
            free(hx);
            free(hy);
            free(mid);
            DIT_FAIL("ffn_core_get");
          }
          size_t nw = 0;
          snprintf(gname, sizeof(gname), "dit.blocks.%d.ffn.0.weight", block);
          const float *Wu = wan_borrow_tensor_f32(ctx, gname, &nw);
          snprintf(gname, sizeof(gname), "dit.blocks.%d.ffn.2.weight", block);
          const float *Wd = wan_borrow_tensor_f32(ctx, gname, &nw);
          if (!Wu || !Wd) {
            free(hx);
            free(hy);
            free(mid);
            DIT_FAIL("ffn_core_W");
          }
          uma_wan_ffn_gelu_f32(hy, mid, hx, Wu, Wd, T, D, Ffn);
          free(mid);
          free(hx);
          if (uma_buf_pool_ensure_put(ctx->bufs, bt, hy, nbytes) != 0) {
            free(hy);
            DIT_FAIL("ffn_core_put");
          }
          free(hy);
          char gr[384];
          if (have_bias)
            n = snprintf(gr, sizeof(gr),
                         "RESIDUAL_ADD@GPU! y=%s x=%s D=%d N=%d gate=%s up=%s ; "
                         "MARK@GPU?",
                         bs, bt, D, T, bgff, bfdb);
          else
            n = snprintf(gr, sizeof(gr),
                         "RESIDUAL_ADD@GPU! y=%s x=%s D=%d N=%d gate=%s ; "
                         "MARK@GPU?",
                         bs, bt, D, T, bgff);
          if (n < 0 || (size_t)n >= sizeof(gr) ||
              wan_submit_graph(ctx->uma, gr) != 0)
            DIT_FAIL("ffn_core_gated");
          static int logged_core;
          if (!logged_core) {
            fprintf(stderr,
                    "wan-c: DiT FFN_GELU on host Accelerate (Brick 13 CORE; "
                    "LN/AdaLN/resid on broker)\n");
            logged_core = 1;
          }
        } else {
          /* LN+AdaLN+FFN+gated resid — one sticky GPU GRAPH (DIT half-block). */
          char gr[1024];
          int n;
          if (have_bias)
            n = snprintf(
                gr, sizeof(gr),
                "LAYERNORM_MUL@GPU! x=%s y=%s w=%s N=%d D=%d ; "
                "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
                "FFN_GELU@GPU! x=%s y=%s wu=%s wd=%s mid=%s M=%d D=%d ffn=%d ; "
                "RESIDUAL_ADD@GPU! y=%s x=%s D=%d N=%d gate=%s up=%s ; MARK@GPU?",
                bs, bt, bln, T, D, bt, ba, sc2, sh2, T, D, ba, bt, bwu, bwd, bmid,
                T, D, Ffn, bs, bt, D, T, bgff, bfdb);
          else
            n = snprintf(
                gr, sizeof(gr),
                "LAYERNORM_MUL@GPU! x=%s y=%s w=%s N=%d D=%d ; "
                "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
                "FFN_GELU@GPU! x=%s y=%s wu=%s wd=%s mid=%s M=%d D=%d ffn=%d ; "
                "RESIDUAL_ADD@GPU! y=%s x=%s D=%d N=%d gate=%s ; MARK@GPU?",
                bs, bt, bln, T, D, bt, ba, sc2, sh2, T, D, ba, bt, bwu, bwd, bmid,
                T, D, Ffn, bs, bt, D, T, bgff);
          if (n < 0 || (size_t)n >= sizeof(gr) ||
              wan_submit_graph(ctx->uma, gr) != 0)
            DIT_FAIL("ffn_gated");
        }
      } else {
        char fpre[512];
        int n = snprintf(
            fpre, sizeof(fpre),
            "LAYERNORM_MUL@GPU! x=%s y=%s w=%s N=%d D=%d ; "
            "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; MARK@GPU?",
            bs, bt, bln, T, D, bt, ba, sc2, sh2, T, D);
        if (n < 0 || (size_t)n >= sizeof(fpre) ||
            wan_submit_graph(ctx->uma, fpre) != 0)
          DIT_FAIL("ffn_gelu_pre");
        for (int r0 = 0; r0 < T; r0 += fchunk) {
          int nr = T - r0;
          if (nr > fchunk)
            nr = fchunk;
          if (wan_graph_ffn_gelu(ctx, ba, bt, bwu, bwd, bmid, nr, D, Ffn, r0) !=
              0)
            DIT_FAIL("ffn_gelu");
        }
        if (use_gated) {
          char gr[384];
          if (have_bias)
            n = snprintf(gr, sizeof(gr),
                         "RESIDUAL_ADD@GPU! y=%s x=%s D=%d N=%d gate=%s up=%s ; "
                         "MARK@GPU?",
                         bs, bt, D, T, bgff, bfdb);
          else
            n = snprintf(gr, sizeof(gr),
                         "RESIDUAL_ADD@GPU! y=%s x=%s D=%d N=%d gate=%s ; "
                         "MARK@GPU?",
                         bs, bt, D, T, bgff);
          if (n < 0 || (size_t)n >= sizeof(gr) ||
              wan_submit_graph(ctx->uma, gr) != 0)
            DIT_FAIL("ffn_gated");
        } else {
          if (have_bias) {
            char ab[320];
            n = snprintf(ab, sizeof(ab),
                         "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d "
                         "; MARK@GPU?",
                         bt, bt, bz, bfdb, T, D);
            if (n < 0 || (size_t)n >= sizeof(ab) ||
                wan_submit_graph(ctx->uma, ab) != 0)
              DIT_FAIL("ffn_bias");
          }
          {
            char gr[384];
            n = snprintf(
                gr, sizeof(gr),
                "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
                "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@GPU?",
                bt, bg, bgf, bz, T, D, bg, bs, flat);
            if (n < 0 || (size_t)n >= sizeof(gr) ||
                wan_submit_graph(ctx->uma, gr) != 0)
              DIT_FAIL("ffn_gate");
          }
        }
      }
      if (block == 0)
        dit_dump_named_buf(ctx, bs, nbytes, "b0_post_ffn.f32");
      static int logged_fg;
      if (!logged_fg) {
        fprintf(stderr,
                "wan-c: DiT FFN LN+AdaLN+FFN_GELU+gate on broker (persist; "
                "chunk=%d%s)\n",
                dit_ffn_chunk_rows(T),
                use_gated ? "; gated+GPU LN/AdaLN" : "; GPU LN/AdaLN");
        logged_fg = 1;
      }
    } else if (real_mod) {
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
          "LAYERNORM_MUL@GPU! x=%s y=%s N=%d D=%d ; "
          "AFFINE_MUL_ADD@GPU! x=%s y=%s gate=%s up=%s N=%d D=%d ; "
          "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; MARK@GPU?",
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
  if (wan_profile_on())
    wan_profile_add_ms("dit_ffn", wan_profile_now_ms() - t_ffn0);

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

/* Pack text into sticky x_dit_tctx{0,1}; *text_mirror_out owns host copy. */
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

  /* Sticky slots: cond + uncond projected ctx survive across UniPC steps. */
  int slot = -1;
  for (int i = 0; i < 2; i++) {
    if (ctx->dit_tctx_src[i] == text_emb && ctx->dit_tctx_src_n[i] == text_n &&
        ctx->dit_tctx_pack[i] && ctx->dit_tctx_D[i] == D &&
        ctx->dit_tctx_Tk[i] > 0) {
      slot = i;
      break;
    }
  }
  if (slot >= 0) {
    int Tk = ctx->dit_tctx_Tk[slot];
    size_t tbytes = (size_t)Tk * (size_t)D * sizeof(float);
    float *pack = malloc(tbytes);
    if (!pack)
      return -1;
    memcpy(pack, ctx->dit_tctx_pack[slot], tbytes);
    static char names[2][32];
    snprintf(names[slot], sizeof(names[slot]), "x_dit_tctx%d", slot);
    if (!ctx->dit_tctx_on_broker[slot]) {
      if (uma_buf_pool_alloc(ctx->bufs, names[slot], tbytes) != 0 ||
          uma_buf_pool_put(ctx->bufs, names[slot], pack, tbytes) != 0) {
        free(pack);
        return -1;
      }
      ctx->dit_tctx_on_broker[slot] = 1;
    }
    if (text_mirror_out)
      *text_mirror_out = pack;
    else
      free(pack);
    *tk_out = names[slot];
    *tv_out = names[slot];
    *Tk_out = Tk;
    static int logged_hit;
    if (!logged_hit) {
      fprintf(stderr, "wan-c: DiT text ctx sticky hit (slot=%d Tk=%d)\n", slot,
              Tk);
      logged_hit = 1;
    }
    if (wan_profile_on())
      wan_profile_add_count("tctx_hit", 1);
    return 0;
  }

  for (int i = 0; i < 2; i++) {
    if (!ctx->dit_tctx_pack[i]) {
      slot = i;
      break;
    }
  }
  if (slot < 0)
    slot = 0; /* replace oldest slot if >2 distinct texts */

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

  free(ctx->dit_tctx_pack[slot]);
  ctx->dit_tctx_pack[slot] = malloc(tbytes);
  if (!ctx->dit_tctx_pack[slot]) {
    free(pack);
    return -1;
  }
  memcpy(ctx->dit_tctx_pack[slot], pack, tbytes);
  ctx->dit_tctx_src[slot] = text_emb;
  ctx->dit_tctx_src_n[slot] = text_n;
  ctx->dit_tctx_Tk[slot] = Tk;
  ctx->dit_tctx_D[slot] = D;
  ctx->dit_tctx_on_broker[slot] = 0;

  static char names[2][32];
  snprintf(names[slot], sizeof(names[slot]), "x_dit_tctx%d", slot);
  if (uma_buf_pool_alloc(ctx->bufs, names[slot], tbytes) != 0 ||
      uma_buf_pool_put(ctx->bufs, names[slot], pack, tbytes) != 0) {
    free(pack);
    return -1;
  }
  ctx->dit_tctx_on_broker[slot] = 1;
  /* Keep host mirror — BANK weight churn can drop tctx mid-CFG. */
  if (text_mirror_out)
    *text_mirror_out = pack;
  else
    free(pack);
  *tk_out = names[slot];
  *tv_out = names[slot]; /* single context buffer; K/V projected per-block */
  *Tk_out = Tk;
  if (wan_profile_on())
    wan_profile_add_count("tctx_miss", 1);
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

  if (use_real_geom && dit_persist_enabled(ctx)) {
    if (dit_persist_bank_all(ctx, nblocks, D, Ffn) != 0) {
      fprintf(stderr,
              "wan-c: DiT persist BANK failed — falling back to host FFN\n");
      ctx->dit_persist_ready = 0;
      ctx->dit_persist_blocks = 0;
    }
  }

  float *mirror = NULL;
  char resp[512];
  if (use_real_geom) {
    mirror = tok; /* patch tokens; optional per-block GET when WAN_DIT_MIRROR */
  } else {
    mirror = calloc(nbytes / sizeof(float), sizeof(float));
    if (!mirror)
      return -1;
    memcpy(mirror, latent, nbytes);
  }

  int persist_hot = use_real_geom && ctx->dit_persist_ready &&
                    !dit_env_truthy("WAN_DIT_MIRROR");
  /* Seed tokens / text once; persist keeps x_dit_s on broker across blocks.
   * Fix 2026-08-22: ensure_put goes via BANK for >1M and corrupts the
   * token buffer (patch vs init corr 0.965). Use direct put. */
  if (uma_buf_pool_put(ctx->bufs, bs, mirror, nbytes) != 0) {
    fprintf(stderr, "wan-c: DiT ensure x_dit_s failed before blocks\n");
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
    int skip_put = 0;
    if (strncmp(tk, "x_dit_tctx", 10) == 0 && tk[10] >= '0' && tk[10] <= '1' &&
        tk[11] == '\0') {
      int s = tk[10] - '0';
      if (ctx->dit_tctx_on_broker[s])
        skip_put = 1;
    }
    if (!skip_put &&
        uma_buf_pool_ensure_put(ctx->bufs, tk, text_mirror, tb) != 0) {
      fprintf(stderr, "wan-c: DiT ensure x_dit_tctx failed before blocks\n");
      free(e0);
      free(e_time);
      free(text_mirror);
      if (!use_real_geom)
        free(mirror);
      free(tok);
      return -1;
    }
  }

  for (int b = 0; b < nblocks; b++) {
    if (!persist_hot) {
      /* Legacy: re-PUT each block — broker may drop under CFG multi-step. */
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
          fprintf(stderr,
                  "wan-c: DiT ensure x_dit_tctx failed before block %d\n", b);
          free(e0);
          free(e_time);
          free(text_mirror);
          if (!use_real_geom)
            free(mirror);
          free(tok);
          return -1;
        }
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
    if (!persist_hot) {
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
  }
  free(e0);

  /* Persist path: one final GET for unpatch / head. */
  if (persist_hot) {
    size_t got_m = 0;
    if (uma_client_buf_get(ctx->uma, bs, mirror, nbytes, &got_m, resp,
                           sizeof(resp)) != 0 ||
        got_m != nbytes) {
      fprintf(stderr, "wan-c: DiT final token GET failed: %.120s\n", resp);
      free(e_time);
      free(text_mirror);
      if (!use_real_geom)
        free(mirror);
      free(tok);
      return -1;
    }
  }

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
