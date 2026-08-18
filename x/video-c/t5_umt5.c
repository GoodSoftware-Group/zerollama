#include "wan_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * T5 encode: token_embedding gather from SPM ids; real UMT5 blocks
 * (attn + gated GELU FFN, matching Wan T5FeedForward) when indexed;
 * else GEMM+LN scaffold.
 */

static int t5_env_truthy(const char *name);
static void text_to_act(const char *text, float *x, int K) {
  unsigned h = 2166136261u;
  for (const unsigned char *p = (const unsigned char *)text; *p; p++)
    h = (h ^ *p) * 16777619u;
  for (int i = 0; i < K; i++) {
    h = h * 1664525u + 1013904223u;
    x[i] = ((float)(h & 0xffff) / 32768.0f) - 1.0f;
  }
}

static void rms_norm_rows(float *y, const float *x, const float *w, int rows,
                          int D, float eps) {
  for (int r = 0; r < rows; r++) {
    const float *xr = x + (size_t)r * (size_t)D;
    float *yr = y + (size_t)r * (size_t)D;
    float m2 = 0.f;
    for (int i = 0; i < D; i++)
      m2 += xr[i] * xr[i];
    m2 = 1.f / sqrtf(m2 / (float)D + eps);
    for (int i = 0; i < D; i++) {
      float g = w ? w[i] : 1.f;
      yr[i] = xr[i] * m2 * g;
    }
  }
}

/* Wan GELU (tanh approx) — T5FeedForward.gate, not SiLU/SwiGLU. */
static void gelu_tanh_inplace(float *x, size_t n) {
  for (size_t i = 0; i < n; i++) {
    float v = x[i];
    float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    x[i] = 0.5f * v * (1.f + tanhf(c));
  }
}

/* T5 relative-position bucket (bidirectional, matches Wan T5RelativeEmbedding). */
static int t5_rel_bucket(int rel, int num_buckets, int max_dist) {
  int half = num_buckets / 2;
  int rel_buckets = 0;
  if (rel > 0)
    rel_buckets = half;
  int rp = rel < 0 ? -rel : rel;
  int nb = half;
  int max_exact = nb / 2;
  int bucket;
  if (rp < max_exact) {
    bucket = rp;
  } else {
    float ratio = (float)rp / (float)max_exact;
    float den = logf((float)max_dist / (float)max_exact);
    int large =
        max_exact + (int)(logf(ratio) / den * (float)(nb - max_exact));
    if (large > nb - 1)
      large = nb - 1;
    if (large < 0)
      large = 0;
    bucket = large;
  }
  return rel_buckets + bucket;
}

/* Build [H,T,T] relative bias from embedding [num_buckets, H]. */
static void t5_fill_rel_bias(float *bias, const float *emb, int T, int H,
                             int num_buckets) {
  for (int i = 0; i < T; i++)
    for (int j = 0; j < T; j++) {
      int b = t5_rel_bucket(j - i, num_buckets, 128);
      if (b < 0)
        b = 0;
      if (b >= num_buckets)
        b = num_buckets - 1;
      for (int h = 0; h < H; h++)
        bias[((size_t)h * (size_t)T + (size_t)i) * (size_t)T + (size_t)j] =
            emb[(size_t)b * (size_t)H + (size_t)h];
    }
}

/* Host MHA; T5 uses no QK scale. Optional bias is [H,T,T].
 * n_valid>0 masks keys j>=n_valid (Wan encoder attention_mask). */
static void attn_mha_host(float *out, const float *q, const float *k,
                          const float *v, int T, int D, int H,
                          const float *bias_htt, int n_valid) {
  int HD = D / H;
  float *scores = calloc((size_t)T * (size_t)T, sizeof(float));
  if (!scores) {
    memcpy(out, v, (size_t)T * (size_t)D * sizeof(float));
    return;
  }
  if (n_valid < 1 || n_valid > T)
    n_valid = T;
  memset(out, 0, (size_t)T * (size_t)D * sizeof(float));
  for (int h = 0; h < H; h++) {
    for (int i = 0; i < T; i++) {
      float maxv = -1e30f;
      for (int j = 0; j < T; j++) {
        float s;
        if (j >= n_valid) {
          s = -1e30f;
        } else {
          s = 0.f;
          const float *qi = q + (size_t)i * (size_t)D + (size_t)h * (size_t)HD;
          const float *kj = k + (size_t)j * (size_t)D + (size_t)h * (size_t)HD;
          for (int d = 0; d < HD; d++)
            s += qi[d] * kj[d];
          /* UMT5: no 1/sqrt(HD) scale; add relative bias. */
          if (bias_htt)
            s += bias_htt[((size_t)h * (size_t)T + (size_t)i) * (size_t)T +
                          (size_t)j];
        }
        scores[i * T + j] = s;
        if (s > maxv)
          maxv = s;
      }
      float sum = 0.f;
      for (int j = 0; j < T; j++) {
        float e = (j >= n_valid) ? 0.f : expf(scores[i * T + j] - maxv);
        scores[i * T + j] = e;
        sum += e;
      }
      float inv = sum > 0.f ? 1.f / sum : 0.f;
      for (int j = 0; j < T; j++)
        scores[i * T + j] *= inv;
      float *oi = out + (size_t)i * (size_t)D + (size_t)h * (size_t)HD;
      for (int j = 0; j < n_valid; j++) {
        float a = scores[i * T + j];
        const float *vj = v + (size_t)j * (size_t)D + (size_t)h * (size_t)HD;
        for (int d = 0; d < HD; d++)
          oi[d] += a * vj[d];
      }
    }
  }
  free(scores);
}

static int t5_block_real(wan_ctx *ctx, float *x, int T, int D, int block,
                         int n_valid) {
  char n1[96], nq[96], nk[96], nv[96], no[96], n2[96], ng[96], nu[96], nd[96],
      npos[96];
  snprintf(n1, sizeof(n1), "t5.blocks.%d.norm1.weight", block);
  snprintf(nq, sizeof(nq), "t5.blocks.%d.attn.q.weight", block);
  snprintf(nk, sizeof(nk), "t5.blocks.%d.attn.k.weight", block);
  snprintf(nv, sizeof(nv), "t5.blocks.%d.attn.v.weight", block);
  snprintf(no, sizeof(no), "t5.blocks.%d.attn.o.weight", block);
  snprintf(n2, sizeof(n2), "t5.blocks.%d.norm2.weight", block);
  snprintf(ng, sizeof(ng), "t5.blocks.%d.ffn.gate.0.weight", block);
  snprintf(nu, sizeof(nu), "t5.blocks.%d.ffn.fc1.weight", block);
  snprintf(nd, sizeof(nd), "t5.blocks.%d.ffn.fc2.weight", block);
  snprintf(npos, sizeof(npos), "t5.blocks.%d.pos_embedding.embedding.weight",
           block);
  if (!wan_gguf_has(ctx, nq) || !wan_gguf_has(ctx, ng))
    return -1;

  const int Ffn = 10240;
  const int H = 64;
  const int num_buckets = 32;
  if (D % H != 0)
    return -1;

  size_t nxd = (size_t)T * (size_t)D;
  size_t nff = (size_t)T * (size_t)Ffn;
  float *h = calloc(nxd, sizeof(float));
  float *q = calloc(nxd, sizeof(float));
  float *k = calloc(nxd, sizeof(float));
  float *v = calloc(nxd, sizeof(float));
  float *o = calloc(nxd, sizeof(float));
  float *gate = calloc(nff, sizeof(float));
  float *up = calloc(nff, sizeof(float));
  float *bias = calloc((size_t)H * (size_t)T * (size_t)T, sizeof(float));
  if (!h || !q || !k || !v || !o || !gate || !up || !bias) {
    free(h);
    free(q);
    free(k);
    free(v);
    free(o);
    free(gate);
    free(up);
    free(bias);
    return -1;
  }

  size_t nw = 0;
  float *w_n1 = wan_load_tensor_f32(ctx, n1, &nw);
  rms_norm_rows(h, x, w_n1, T, D, 1e-6f);
  free(w_n1);

  float *Wq = wan_load_tensor_f32(ctx, nq, &nw);
  if (!Wq || nw != (size_t)D * (size_t)D)
    goto fail;
  uma_wan_gemm_f32(q, h, Wq, T, D, D);
  free(Wq);
  Wq = NULL;

  float *Wk = wan_load_tensor_f32(ctx, nk, &nw);
  if (!Wk || nw != (size_t)D * (size_t)D)
    goto fail;
  uma_wan_gemm_f32(k, h, Wk, T, D, D);
  free(Wk);
  Wk = NULL;

  float *Wv = wan_load_tensor_f32(ctx, nv, &nw);
  if (!Wv || nw != (size_t)D * (size_t)D)
    goto fail;
  uma_wan_gemm_f32(v, h, Wv, T, D, D);
  free(Wv);
  Wv = NULL;

  const float *bias_arg = NULL;
  float *pos = wan_load_tensor_f32(ctx, npos, &nw);
  if (pos && nw == (size_t)num_buckets * (size_t)H) {
    t5_fill_rel_bias(bias, pos, T, H, num_buckets);
    bias_arg = bias;
    static int logged_rel;
    if (!logged_rel) {
      fprintf(stderr, "wan-c: T5 relative pos bias (buckets=%d heads=%d)\n",
              num_buckets, H);
      logged_rel = 1;
    }
  }
  free(pos);

  attn_mha_host(o, q, k, v, T, D, H, bias_arg, n_valid);

  float *Wo = wan_load_tensor_f32(ctx, no, &nw);
  if (!Wo || nw != (size_t)D * (size_t)D)
    goto fail;
  uma_wan_gemm_f32(h, o, Wo, T, D, D);
  free(Wo);
  Wo = NULL;
  for (size_t i = 0; i < nxd; i++)
    x[i] += h[i];

  float *w_n2 = wan_load_tensor_f32(ctx, n2, &nw);
  rms_norm_rows(h, x, w_n2, T, D, 1e-6f);
  free(w_n2);

  float *Wg = wan_load_tensor_f32(ctx, ng, &nw);
  if (!Wg || nw != (size_t)Ffn * (size_t)D)
    goto fail;
  uma_wan_gemm_f32(gate, h, Wg, T, Ffn, D);
  free(Wg);
  Wg = NULL;

  float *Wu = wan_load_tensor_f32(ctx, nu, &nw);
  if (!Wu || nw != (size_t)Ffn * (size_t)D)
    goto fail;
  uma_wan_gemm_f32(up, h, Wu, T, Ffn, D);
  free(Wu);
  Wu = NULL;

  /* Wan: fc1(x) * GELU(gate_linear(x)) */
  gelu_tanh_inplace(gate, nff);
  for (size_t i = 0; i < nff; i++)
    gate[i] *= up[i];
  {
    static int logged_ffn;
    if (!logged_ffn) {
      fprintf(stderr, "wan-c: T5 FFN gated GELU(tanh) (Wan T5FeedForward)\n");
      logged_ffn = 1;
    }
  }

  float *Wd = wan_load_tensor_f32(ctx, nd, &nw);
  if (!Wd || nw != (size_t)D * (size_t)Ffn)
    goto fail;
  uma_wan_gemm_f32(h, gate, Wd, T, D, Ffn);
  free(Wd);
  Wd = NULL;
  for (size_t i = 0; i < nxd; i++)
    x[i] += h[i];

  free(h);
  free(q);
  free(k);
  free(v);
  free(o);
  free(gate);
  free(up);
  free(bias);
  return 0;

fail:
  free(h);
  free(q);
  free(k);
  free(v);
  free(o);
  free(gate);
  free(up);
  free(bias);
  return -1;
}

static int t5_want_blocks(void) {
  const char *e = getenv("WAN_T5_BLOCKS");
  if (e && e[0]) {
    int n = atoi(e);
    if (n < 0)
      n = 0;
    if (n > 24)
      n = 24;
    return n;
  }
  return 24; /* full UMT5 when indexed; WAN_T5_BLOCKS=N to cap for lab */
}

static int t5_active_rows(int rows, size_t n_ids) {
  /*
   * Wan T5EncoderModel pads to text_len with a mask, then returns u[:seq_len]
   * for DiT. For batch=1, running only real tokens == pad+mask+trim.
   * WAN_T5_PAD=1 forces pad to WAN_T5_SEQ (default 256) with attention mask.
   */
  int cap = 256;
  const char *e = getenv("WAN_T5_SEQ");
  if (e && e[0]) {
    int c = atoi(e);
    if (c >= 1 && c <= 512)
      cap = c;
  }
  int active = rows;
  if (active > cap)
    active = cap;

  const char *pad = getenv("WAN_T5_PAD");
  int force_pad = pad && pad[0] == '1';
  if (!force_pad && n_ids > 0 && (int)n_ids < active)
    active = (int)n_ids;

  if (active < 1)
    active = 1;
  return active;
}

static int t5_apply_real_blocks(wan_ctx *ctx, float *x, int active, int D,
                                int n_valid) {
  int want = t5_want_blocks();
  int ran = 0;
  if (n_valid < 1 || n_valid > active)
    n_valid = active;
  for (int b = 0; b < want; b++) {
    if (t5_block_real(ctx, x, active, D, b, n_valid) != 0)
      break;
    ran++;
  }
  if (ran > 0) {
    static int logged;
    if (!logged) {
      fprintf(stderr,
              "wan-c: T5 UMT5 real blocks=%d seq=%d valid=%d dim=%d\n", ran,
              active, n_valid, D);
      logged = 1;
    }
  }
  return ran;
}

/* ---------- GPU (broker GRAPH) T5 path — F1020 ----------------------------
 * Mirrors t5_block_real but ships the weight-heavy GEMMs/attn/FFN to the
 * daemon. RMS norms stay host (semantics differ from LAYERNORM_MUL and the
 * activations are tiny at T≤8). Weights are BANK_PUT once per block and
 * rebound per call, so the ~16GB/2-pass mmap weight reload disappears.
 * Rollback: WAN_T5_GPU=0 (host). Gated by caps: attn_bias + gemm_f16 +
 * gelu_tanh_mul. */

static int t5_env_truthy(const char *name) {
  const char *e = getenv(name);
  return e && e[0] && !(e[0] == '0');
}

static int t5_gpu_enabled(const wan_ctx *ctx) {
  if (!ctx || !ctx->bufs || !ctx->uma || ctx->local_mode)
    return 0;
  /* F1020: opt-in (WAN_T5_GPU=1). Dispatches T5 blocks to the daemon GPU;
   * correct (rematch ~1e-6) but SLOWER than host at T≤8 (IPC round-trips
   * dominate; host cblas + page-cached mmap weights is fast). */
  if (!t5_env_truthy("WAN_T5_GPU"))
    return 0;
  if (t5_env_truthy("WAN_T5_NO_GPU"))
    return 0;
  return ctx->caps.attn_bias && ctx->caps.gemm_f16 &&
         ctx->caps.gelu_tanh_mul;
}

static int t5_bank_put_tensor(wan_ctx *ctx, const char *bank_key,
                              const char *t5_name, size_t expect) {
  size_t nw = 0;
  const float *w = wan_borrow_tensor_f32(ctx, t5_name, &nw);
  if (!w || (expect > 0 && nw != expect))
    return -1;
  return uma_buf_pool_bank_put(ctx->bufs, bank_key, w, nw * sizeof(float));
}

/* BANK_PUT all 24 blocks' dense weights once (keys t5.blocks.{i}.*). */
static int t5_bank_persist_all(wan_ctx *ctx, int nblocks, int D, int Ffn) {
  if (!ctx || nblocks < 1 || D < 1 || Ffn < 1)
    return -1;
  if (ctx->t5_persist_ready && ctx->t5_persist_blocks >= nblocks)
    return 0;
  size_t dd = (size_t)D * (size_t)D;
  size_t fd = (size_t)Ffn * (size_t)D;
  size_t df = (size_t)D * (size_t)Ffn;
  size_t nbytes = 0;
  int nkeys = 0;
  fprintf(stderr, "wan-c: T5 BANK_PUT persist %d blocks …\n", nblocks);
  for (int li = 0; li < nblocks; li++) {
    char tname[128], key[160];
    struct {
      const char *suf;
      const char *fmt;
      size_t ne;
    } items[] = {
        {"q.weight", "t5.blocks.%d.attn.q.weight", dd},
        {"k.weight", "t5.blocks.%d.attn.k.weight", dd},
        {"v.weight", "t5.blocks.%d.attn.v.weight", dd},
        {"o.weight", "t5.blocks.%d.attn.o.weight", dd},
        {"ffn.gate.0.weight", "t5.blocks.%d.ffn.gate.0.weight", fd},
        {"ffn.fc1.weight", "t5.blocks.%d.ffn.fc1.weight", fd},
        {"ffn.fc2.weight", "t5.blocks.%d.ffn.fc2.weight", df},
    };
    for (size_t i = 0; i < sizeof(items) / sizeof(items[0]); i++) {
      snprintf(tname, sizeof(tname), items[i].fmt, li);
      if (!wan_gguf_has(ctx, tname))
        return -1;
      snprintf(key, sizeof(key), "t5.blocks.%d.%s", li, items[i].suf);
      if (t5_bank_put_tensor(ctx, key, tname, items[i].ne) != 0) {
        fprintf(stderr, "wan-c: T5 BANK_PUT fail %s\n", key);
        return -1;
      }
      nbytes += items[i].ne * sizeof(float);
      nkeys++;
    }
  }
  ctx->t5_persist_blocks = nblocks;
  ctx->t5_persist_ready = 1;
  fprintf(stderr,
          "wan-c: T5 BANK_PUT persist OK keys=%d ~%.1f MiB\n", nkeys,
          nbytes / (1024.0 * 1024.0));
  return 0;
}

/* One BANK_BINDS IPC: bind this block's banked weights as x_t5_W{...}. */
static int t5_bind_block(wan_ctx *ctx, int block) {
  if (!ctx || !ctx->t5_persist_ready)
    return -1;
  char pairs[768];
  int n = snprintf(
      pairs, sizeof(pairs),
      "t5.blocks.%d.q.weight:x_t5_Wq,"
      "t5.blocks.%d.k.weight:x_t5_Wk,"
      "t5.blocks.%d.v.weight:x_t5_Wv,"
      "t5.blocks.%d.o.weight:x_t5_Wo,"
      "t5.blocks.%d.ffn.gate.0.weight:x_t5_Wg,"
      "t5.blocks.%d.ffn.fc1.weight:x_t5_Wu,"
      "t5.blocks.%d.ffn.fc2.weight:x_t5_Wd",
      block, block, block, block, block, block, block);
  if (n < 0 || (size_t)n >= sizeof(pairs))
    return -1;
  if (uma_buf_pool_bank_binds(ctx->bufs, pairs) != 0) {
    fprintf(stderr, "wan-c: T5 BANK_BINDS fail block=%d\n", block);
    return -1;
  }
  return 0;
}

/* GPU T5 block (F1020). Weights banked; norms host; attn kind=bias (unscaled);
 * FFN = GELU_TANH_MUL(gate,up) then down GEMM; both residuals on GPU. */
/* GPU T5 block (F1020). Weights banked once; RMS norms + gated GELU on
 * host (activations tiny at T<=8; cheap); Q/K/V/O + gate/up/down GEMMs and
 * biased (unscaled) attention on the daemon GPU. Correct but IPC-bound. */
static int t5_block_gpu(wan_ctx *ctx, float *x, int T, int D, int block,
                        int n_valid) {
  char n1[96], n2[96], npos[96];
  snprintf(n1, sizeof(n1), "t5.blocks.%d.norm1.weight", block);
  snprintf(n2, sizeof(n2), "t5.blocks.%d.norm2.weight", block);
  snprintf(npos, sizeof(npos), "t5.blocks.%d.pos_embedding.embedding.weight",
           block);
  const int H = 64;
  const int num_buckets = 32;
  const int Ffn = 10240; /* UMT5 text FFN intermed (Wan T5FeedForward) */
  if (D % H != 0 || D < 1 || T < 1)
    return -1;
  int HD = D / H;

  size_t nxd = (size_t)T * (size_t)D;
  float *h = calloc(nxd, sizeof(float));
  float *bias = calloc((size_t)H * (size_t)T * (size_t)T, sizeof(float));
  if (!h || !bias) {
    free(h);
    free(bias);
    return -1;
  }
  size_t nw = 0;
  float *w_n1 = wan_load_tensor_f32(ctx, n1, &nw);
  rms_norm_rows(h, x, w_n1, T, D, 1e-6f);
  free(w_n1);
  float *pos = wan_load_tensor_f32(ctx, npos, &nw);
  if (pos && nw == (size_t)num_buckets * (size_t)H)
    t5_fill_rel_bias(bias, pos, T, H, num_buckets);
  free(pos);

  if (t5_bind_block(ctx, block) != 0) {
    free(h);
    free(bias);
    return -1;
  }

  size_t nx = nxd * sizeof(float);
  size_t nff = (size_t)T * (size_t)Ffn * sizeof(float);
  size_t nff_elems = (size_t)T * (size_t)Ffn;
  size_t nbias = (size_t)H * (size_t)T * (size_t)T * sizeof(float);
  const char *bx = "x_t5_x", *bh = "x_t5_h", *bq = "x_t5_q", *bk = "x_t5_k";
  const char *bv = "x_t5_v", *bo = "x_t5_o", *bg = "x_t5_gate";
  const char *bu = "x_t5_up", *bb = "x_t5_bias", *bh2 = "x_t5_h2";
  if (uma_buf_pool_alloc(ctx->bufs, bx, nx) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bh, nx) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bb, nbias) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bq, nx) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bk, nx) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bv, nx) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bo, nx) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bh2, nx) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bg, nff) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bu, nff) != 0 ||
      uma_buf_pool_put(ctx->bufs, bx, x, nx) != 0 ||
      uma_buf_pool_put(ctx->bufs, bh, h, nx) != 0 ||
      uma_buf_pool_put(ctx->bufs, bb, bias, nbias) != 0) {
    free(h);
    free(bias);
    return -1;
  }
  free(bias); /* copied into bb; h stays live for norm2 */

  int rc = -1;
  char nodes[1024];
  int n = snprintf(
      nodes, sizeof(nodes),
      "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
      "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
      "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; MARK@GPU?",
      bh, bq, "x_t5_Wq", T, D, D, bh, bk, "x_t5_Wk", T, D, D, bh, bv,
      "x_t5_Wv", T, D, D);
  if (n < 0 || (size_t)n >= sizeof(nodes) ||
      wan_submit_graph(ctx->uma, nodes) != 0)
    goto out;
  if (wan_graph_attn_bias(ctx, bq, bk, bv, bb, bo, T, T, H, H, HD) != 0)
    goto out;
  n = snprintf(
      nodes, sizeof(nodes),
      "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
      "RESIDUAL_ADD@GPU! y=%s x=%s D=%d ; MARK@GPU?",
      bo, bh2, "x_t5_Wo", T, D, D, bx, bh2, T * D);
  if (n < 0 || (size_t)n >= sizeof(nodes) ||
      wan_submit_graph(ctx->uma, nodes) != 0)
    goto out;

  /* fetch x (attn residual done on GPU), host norm2, push back */
  {
    char resp[256];
    size_t got = 0;
    if (uma_client_buf_get(ctx->uma, bx, x, nx, &got, resp, sizeof(resp)) != 0 ||
        got != nx)
      goto out;
  }
  {
    float *w_n2 = wan_load_tensor_f32(ctx, n2, &nw);
    rms_norm_rows(h, x, w_n2, T, D, 1e-6f);
    free(w_n2);
  }
  if (uma_buf_pool_put(ctx->bufs, bh, h, nx) != 0)
    goto out;

  n = snprintf(
      nodes, sizeof(nodes),
      "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
      "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; MARK@GPU?",
      bh, bg, "x_t5_Wg", T, Ffn, D, bh, bu, "x_t5_Wu", T, Ffn, D);
  if (n < 0 || (size_t)n >= sizeof(nodes) ||
      wan_submit_graph(ctx->uma, nodes) != 0)
    goto out;

  /* Host gated GELU: gate=gelu(Wg@h), gate*=Wu@h. GEMMs stay on GPU; the
   * elementwise is cheap and avoids daemon GELU_TANH_MUL CPU/GPU sync. */
  {
    char resp[256];
    size_t got = 0;
    float *gatev = malloc(nff);
    float *upv = malloc(nff);
    if (!gatev || !upv) {
      free(gatev);
      free(upv);
      goto out;
    }
    if (uma_client_buf_get(ctx->uma, bg, gatev, nff, &got, resp,
                           sizeof(resp)) != 0 ||
        got != nff ||
        uma_client_buf_get(ctx->uma, bu, upv, nff, &got, resp, sizeof(resp)) !=
            0 ||
        got != nff) {
      free(gatev);
      free(upv);
      goto out;
    }
    gelu_tanh_inplace(gatev, nff_elems);
    for (size_t i = 0; i < nff_elems; i++)
      gatev[i] *= upv[i];
    free(upv);
    if (uma_buf_pool_put(ctx->bufs, bg, gatev, nff) != 0) {
      free(gatev);
      goto out;
    }
    free(gatev);
  }
  n = snprintf(
      nodes, sizeof(nodes),
      "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
      "RESIDUAL_ADD@GPU! y=%s x=%s D=%d ; MARK@GPU?",
      bg, bh2, "x_t5_Wd", T, D, Ffn, bx, bh2, T * D);
  if (n < 0 || (size_t)n >= sizeof(nodes) ||
      wan_submit_graph(ctx->uma, nodes) != 0)
    goto out;

  {
    char resp[256];
    size_t got = 0;
    if (uma_client_buf_get(ctx->uma, bx, x, nx, &got, resp, sizeof(resp)) != 0 ||
        got != nx)
      goto out;
  }
  rc = 0;
out:
  free(h);
  return rc;
}
static int t5_apply_gpu_blocks(wan_ctx *ctx, float *x, int active, int D,
                               int n_valid) {
  int want = t5_want_blocks();
  int ran = 0;
  if (n_valid < 1 || n_valid > active)
    n_valid = active;
  if (!t5_gpu_enabled(ctx))
    return -1;
  if (t5_bank_persist_all(ctx, want, D, ctx->cfg.text_ffn) != 0)
    return -1;  for (int b = 0; b < want; b++) {
    if (t5_block_gpu(ctx, x, active, D, b, n_valid) != 0) {
      fprintf(stderr, "wan-c: T5 GPU block %d failed — fallback host\n", b);
      break;
    }
    ran++;
  }
  if (ran > 0) {
    static int logged;
    if (!logged) {
      fprintf(stderr, "wan-c: T5 GPU UMT5 blocks=%d seq=%d valid=%d dim=%d\n",
              ran, active, n_valid, D);
      logged = 1;
    }
  }
  return ran;
}

static int t5_embed_ids(wan_ctx *ctx, const int32_t *ids, size_t n_ids,
                        float *out, int rows, int D) {
  size_t wne = 0;
  float *emb = wan_load_tensor_f32(ctx, "t5.token_embedding.weight", &wne);
  if (!emb)
    emb = wan_load_tensor_f32(ctx, "t5.shared.weight", &wne);
  if (!emb || D < 1 || rows < 1)
    return -1;
  size_t vocab = wne / (size_t)D;
  if (vocab < 1 || wne != vocab * (size_t)D) {
    free(emb);
    return -1;
  }
  memset(out, 0, (size_t)rows * (size_t)D * sizeof(float));
  size_t use = n_ids;
  if (use > (size_t)rows)
    use = (size_t)rows;
  for (size_t i = 0; i < use; i++) {
    int id = ids[i];
    if (id < 0 || (size_t)id >= vocab)
      id = 0;
    memcpy(out + i * (size_t)D, emb + (size_t)id * (size_t)D,
           (size_t)D * sizeof(float));
  }
  free(emb);
  return 0;
}

static int t5_host(wan_ctx *ctx, const char *text, const int32_t *ids,
                   size_t n_ids, float *out, size_t n) {
  int D = ctx->cfg.text_dim;
  int rows = 1;
  if (D > 0 && n >= (size_t)D && (n % (size_t)D) == 0)
    rows = (int)(n / (size_t)D);
  if (rows < 1 || D < 1)
    return -1;

  float *x = calloc((size_t)rows * (size_t)D, sizeof(float));
  float *w = calloc((size_t)D * (size_t)D, sizeof(float));
  float *y = calloc((size_t)rows * (size_t)D, sizeof(float));
  if (!x || !w || !y) {
    free(x);
    free(w);
    free(y);
    return -1;
  }

  int have_emb = 0;
  if (ids && n_ids > 0 &&
      (wan_gguf_has(ctx, "t5.token_embedding.weight") ||
       wan_gguf_has(ctx, "t5.shared.weight"))) {
    have_emb = t5_embed_ids(ctx, ids, n_ids, x, rows, D) == 0;
  }
  if (!have_emb) {
    if (text)
      text_to_act(text, x, D);
    else
      memset(x, 0, (size_t)D * sizeof(float));
    if (rows > 1)
      for (int r = 1; r < rows; r++)
        memcpy(x + (size_t)r * (size_t)D, x, (size_t)D * sizeof(float));
  }

  int active = t5_active_rows(rows, n_ids);
  int n_valid = (n_ids > 0 && (int)n_ids < active) ? (int)n_ids : active;
  int nreal = t5_apply_real_blocks(ctx, x, active, D, n_valid);

  size_t wn = 0;
  float *loaded = wan_load_tensor_f32(ctx, "t5.norm.weight", &wn);
  if (!loaded)
    loaded = wan_load_tensor_f32(ctx, "t5.encoder.final_layer_norm.weight", &wn);
  if (nreal > 0) {
    /* Real stack: final RMSNorm on active rows (Wan returns u[:seq_len]). */
    memset(out, 0, (size_t)rows * (size_t)D * sizeof(float));
    rms_norm_rows(out, x, loaded, active, D, 1e-6f);
  } else {
    wan_fill_eye_nt(w, D, D);
    uma_wan_gemm_f32(y, x, w, rows, D, D);
    uma_wan_layernorm_f32(out, y, loaded, NULL, rows, D, 1e-6f);
  }
  free(loaded);
  free(x);
  free(w);
  free(y);
  return 0;
}

static int t5_broker(wan_ctx *ctx, const char *text, const int32_t *ids,
                     size_t n_ids, float *out, size_t n) {
  int D = ctx->cfg.text_dim;
  int rows = 1;
  if (D > 0 && n >= (size_t)D && (n % (size_t)D) == 0)
    rows = (int)(n / (size_t)D);
  if (rows < 1 || D < 1)
    return -1;

  char resp[512];
  size_t got = 0;
  float *x = calloc((size_t)rows * (size_t)D, sizeof(float));
  if (!x)
    return -1;

  int have_emb = 0;
  if (ids && n_ids > 0 &&
      (wan_gguf_has(ctx, "t5.token_embedding.weight") ||
       wan_gguf_has(ctx, "t5.shared.weight"))) {
    have_emb = t5_embed_ids(ctx, ids, n_ids, x, rows, D) == 0;
    if (have_emb)
      fprintf(stderr, "wan-c: T5 token_embedding gather ids=%zu rows=%d\n",
              n_ids, rows);
  }
  if (!have_emb) {
    if (text)
      text_to_act(text, x, D);
    else
      memset(x, 0, (size_t)D * sizeof(float));
    if (rows > 1)
      for (int r = 1; r < rows; r++)
        memcpy(x + (size_t)r * (size_t)D, x, (size_t)D * sizeof(float));
  }

  /* Real UMT5 blocks are host-side (weights too large for tip BANK). */
  int active = t5_active_rows(rows, n_ids);
  int n_valid = (n_ids > 0 && (int)n_ids < active) ? (int)n_ids : active;
  int nreal = t5_apply_gpu_blocks(ctx, x, active, D, n_valid);
  if (nreal <= 0)
    nreal = t5_apply_real_blocks(ctx, x, active, D, n_valid);
  if (nreal > 0) {
    size_t nw_ln = 0;
    float *ln_w = wan_load_tensor_f32(ctx, "t5.norm.weight", &nw_ln);
    if (!ln_w)
      ln_w = wan_load_tensor_f32(ctx, "t5.encoder.final_layer_norm.weight",
                                 &nw_ln);
    memset(out, 0, (size_t)rows * (size_t)D * sizeof(float));
    rms_norm_rows(out, x, ln_w, active, D, 1e-6f);
    free(ln_w);
    free(x);
    return 0;
  }

  const char *bx = "x_t5_x";
  const char *by = "x_t5_y";
  const char *bw = "x_t5_w";
  size_t nx = (size_t)rows * (size_t)D * sizeof(float);
  size_t ny = nx;
  size_t nw = (size_t)D * (size_t)D * sizeof(float);

  float *w = calloc((size_t)D * (size_t)D, sizeof(float));
  if (!w) {
    free(x);
    return -1;
  }
  wan_fill_eye_nt(w, D, D);

  size_t nw_ln = 0;
  float *ln_w = wan_load_tensor_f32(ctx, "t5.norm.weight", &nw_ln);
  if (!ln_w)
    ln_w = wan_load_tensor_f32(ctx, "t5.encoder.final_layer_norm.weight", &nw_ln);
  const char *bln = "x_t5_lnw";
  int have_ln = ln_w && nw_ln == (size_t)D;
  if (have_ln) {
    if (uma_buf_pool_put_weight(ctx->bufs, bln, "t5.ln", ln_w, nw_ln * 4) != 0)
      have_ln = 0;
    else
      fprintf(stderr, "wan-c: T5 norm.weight applied\n");
  }
  free(ln_w);

  if (uma_buf_pool_alloc(ctx->bufs, bx, nx) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, by, ny) != 0 ||
      uma_buf_pool_put(ctx->bufs, bx, x, nx) != 0 ||
      uma_buf_pool_put_weight(ctx->bufs, bw, "t5.W", w, nw) != 0) {
    fprintf(stderr, "wan-c: T5 BUF/BANK put failed\n");
    free(x);
    free(w);
    return -1;
  }
  memset(out, 0, ny);
  if (uma_buf_pool_put(ctx->bufs, by, out, ny) != 0) {
    free(x);
    free(w);
    return -1;
  }
  free(w);
  free(x);

  int layers = 1; /* keep shallow when real embed present; eye stack adds little */
  if (!have_emb) {
    layers = 4;
    const char *el = getenv("WAN_T5_LAYERS");
    if (el && el[0]) {
      int L = atoi(el);
      if (L >= 1 && L <= 8)
        layers = L;
    }
  } else {
    const char *el = getenv("WAN_T5_LAYERS");
    if (el && el[0]) {
      int L = atoi(el);
      if (L >= 1 && L <= 8)
        layers = L;
    }
  }

  if (!(ctx->caps.prefer_ext && ctx->caps.ext_ready)) {
    const char *by2 = "x_t5_y2";
    const char *bw2 = "x_t5_w2";
    if (uma_buf_pool_alloc(ctx->bufs, by2, ny) != 0 ||
        uma_buf_pool_put(ctx->bufs, by2, out, ny) != 0) {
      return -1;
    }
    if (layers > 1) {
      float *w2 = calloc((size_t)D * (size_t)D, sizeof(float));
      if (!w2)
        return -1;
      wan_fill_eye_nt(w2, D, D);
      if (uma_buf_pool_put_weight(ctx->bufs, bw2, "t5.W2", w2,
                                  (size_t)D * (size_t)D * 4) != 0) {
        free(w2);
        return -1;
      }
      free(w2);
    }
    char nodes[4096];
    int npos;
    if (have_ln)
      npos = snprintf(nodes, sizeof(nodes),
                      "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
                      "LAYERNORM_MUL@CPU! x=%s y=%s w=%s N=%d D=%d ; ",
                      bx, by, bw, rows, D, D, by, by, bln, rows, D);
    else
      npos = snprintf(nodes, sizeof(nodes),
                      "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
                      "LAYERNORM_MUL@CPU! x=%s y=%s N=%d D=%d ; ",
                      bx, by, bw, rows, D, D, by, by, rows, D);
    if (npos < 0 || (size_t)npos >= sizeof(nodes))
      return -1;
    for (int L = 1; L < layers; L++) {
      int m = snprintf(nodes + npos, sizeof(nodes) - (size_t)npos,
                       "GEMM_F16@GPU! x=%s y=%s w=%s M=%d N=%d K=%d ; "
                       "LAYERNORM_MUL@CPU! x=%s y=%s N=%d D=%d ; ",
                       by, by2, bw2, rows, D, D, by2, by, rows, D);
      if (m < 0 || (size_t)npos + (size_t)m >= sizeof(nodes))
        return -1;
      npos += m;
    }
    if (snprintf(nodes + npos, sizeof(nodes) - (size_t)npos, "MARK@GPU?") < 0)
      return -1;
    if (wan_submit_graph(ctx->uma, nodes) != 0)
      return -1;
  } else {
    if (wan_graph_gemm_f32(ctx, bx, by, bw, rows, D, D) != 0 ||
        wan_graph_layernorm(ctx, by, by, have_ln ? bln : NULL, rows, D) != 0)
      return -1;
  }

  if (uma_client_buf_get(ctx->uma, by, out, ny, &got, resp, sizeof(resp)) !=
          0 ||
      got != ny) {
    fprintf(stderr, "wan-c: T5 BUF_GET failed: %.160s\n", resp);
    return -1;
  }
  return 0;
}

int wan_t5_encode(wan_ctx *ctx, const char *text, float *out, size_t n) {
  if (!ctx || !out || n < 1)
    return -1;
  if (wan_env_local() || ctx->local_mode)
    return t5_host(ctx, text, NULL, 0, out, n);
  if (!ctx->uma || !ctx->bufs) {
    fprintf(stderr, "wan-c: T5 needs UMA client (or UMA_WAN_LOCAL=1)\n");
    return -1;
  }
  return t5_broker(ctx, text, NULL, 0, out, n);
}

int wan_t5_encode_ids(wan_ctx *ctx, const int32_t *ids, size_t n_ids, float *out,
                      size_t n) {
  if (!ctx || !ids || n_ids < 1 || !out || n < 1)
    return -1;
  if (wan_env_local() || ctx->local_mode)
    return t5_host(ctx, NULL, ids, n_ids, out, n);
  if (!ctx->uma || !ctx->bufs) {
    fprintf(stderr, "wan-c: T5 needs UMA client (or UMA_WAN_LOCAL=1)\n");
    return -1;
  }
  return t5_broker(ctx, NULL, ids, n_ids, out, n);
}
