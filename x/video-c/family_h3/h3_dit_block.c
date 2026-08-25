#include "h3_dit_block.h"

#include "h3_adaln_host.h"
#include "h3_convrot.h"
#include "h3_dit_host.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int load_n(const h3_st_store *st, const char *name, float *dst, size_t n,
                  char *error, size_t error_size) {
  return h3_st_store_load_f32(st, name, dst, n, error, error_size);
}

static int dit_debug_level(void);

/* Comfy loads rope.inv_freq from the pack; theta formula is the fallback. */
static int rope_inv_from_store(const h3_st_store *store, float *inv, char *error,
                               size_t error_size) {
  static float cached[H3_DIT_ROPE_INV_FREQ_LEN];
  static int have;
  if (have) {
    memcpy(inv, cached, sizeof(cached));
    return 0;
  }
  if (store && h3_st_store_find(store, "rope.inv_freq", NULL) &&
      load_n(store, "rope.inv_freq", cached, H3_DIT_ROPE_INV_FREQ_LEN, error,
             error_size) == 0) {
    have = 1;
    memcpy(inv, cached, sizeof(cached));
    if (dit_debug_level() >= 1)
      fprintf(stderr, "video-c: rope.inv_freq from pack [0]=%.6g [15]=%.6g\n",
              cached[0], cached[H3_DIT_ROPE_INV_FREQ_LEN - 1]);
    return 0;
  }
  int rc = h3_dit_rope_inv_freq(inv, H3_DIT_ROPE_INV_FREQ_LEN, H3_DIT_ROPE_THETA);
  if (rc == 0) {
    memcpy(cached, inv, sizeof(cached));
    have = 1;
  }
  return rc;
}

static int want_bf16_act(void) {
  const char *e = getenv("H3_DIT_BF16_ACT");
  return e && e[0] && e[0] != '0';
}

static void cast_bf16_n(float *x, size_t n) {
  for (size_t i = 0; i < n; i++) {
    union {
      float f;
      uint32_t u;
    } v;
    v.f = x[i];
    v.u += 0x7FFFu + ((v.u >> 16) & 1u);
    v.u &= 0xFFFF0000u;
    x[i] = v.f;
  }
}

static int dit_debug_level(void) {
  const char *e = getenv("H3_DIT_DEBUG");
  if (!e || !e[0] || strcmp(e, "0") == 0)
    return 0;
  int n = atoi(e);
  return n > 0 ? n : 1;
}

/* Scratch arena shared by all blocks of one forward pass. Sized once from
 * max_seq; the block path otherwise mallocs ~800 MB per block-step and Darwin
 * page-faults every fresh large allocation. */
struct h3_dit_ws {
  int max_seq;
  float *h;      /* max_seq * H */
  float *branch; /* max_seq * H */
  float *qkv;    /* max_seq * 3 * inner */
  float *q, *k, *v, *attn; /* max_seq * inner */
  float *fused;  /* max_seq * 2 * ffn */
  float *hid;    /* max_seq * ffn */
  float *cos, *sin; /* max_seq * H3_DIT_ROPE_DIM */
  int *idx, *tslots; /* max_seq ints */
};

h3_dit_ws *h3_dit_ws_create(int max_seq) {
  if (max_seq < 1)
    return NULL;
  h3_dit_ws *ws = (h3_dit_ws *)calloc(1, sizeof(*ws));
  if (!ws)
    return NULL;
  ws->max_seq = max_seq;
  const size_t S = (size_t)max_seq;
  const size_t H = H3_DIT_HIDDEN_SIZE, inner = H3_DIT_INNER_DIM,
               ffn = H3_DIT_FFN_HIDDEN, RD = H3_DIT_ROPE_DIM;
  ws->h = (float *)malloc(S * H * sizeof(float));
  ws->branch = (float *)malloc(S * H * sizeof(float));
  ws->qkv = (float *)malloc(S * 3 * inner * sizeof(float));
  ws->q = (float *)malloc(S * inner * sizeof(float));
  ws->k = (float *)malloc(S * inner * sizeof(float));
  ws->v = (float *)malloc(S * inner * sizeof(float));
  ws->attn = (float *)malloc(S * inner * sizeof(float));
  ws->fused = (float *)malloc(S * 2 * ffn * sizeof(float));
  ws->hid = (float *)malloc(S * ffn * sizeof(float));
  ws->cos = (float *)malloc(S * RD * sizeof(float));
  ws->sin = (float *)malloc(S * RD * sizeof(float));
  ws->idx = (int *)malloc(S * sizeof(int));
  ws->tslots = (int *)malloc(S * sizeof(int));
  if (!ws->h || !ws->branch || !ws->qkv || !ws->q || !ws->k || !ws->v ||
      !ws->attn || !ws->fused || !ws->hid || !ws->cos || !ws->sin ||
      !ws->idx || !ws->tslots) {
    h3_dit_ws_free(ws);
    return NULL;
  }
  return ws;
}

void h3_dit_ws_free(h3_dit_ws *ws) {
  if (!ws)
    return;
  free(ws->h);
  free(ws->branch);
  free(ws->qkv);
  free(ws->q);
  free(ws->k);
  free(ws->v);
  free(ws->attn);
  free(ws->fused);
  free(ws->hid);
  free(ws->cos);
  free(ws->sin);
  free(ws->idx);
  free(ws->tslots);
  free(ws);
}

float *h3_dit_ws_cos(h3_dit_ws *ws) { return ws ? ws->cos : NULL; }
float *h3_dit_ws_sin(h3_dit_ws *ws) { return ws ? ws->sin : NULL; }

int h3_dit_rope_tables(const h3_st_store *store, const float *position_ids,
                       int seq, float *cos, float *sin, char *error,
                       size_t error_size) {
  if (!position_ids || !cos || !sin || seq < 1)
    return -1;
  float inv[H3_DIT_ROPE_INV_FREQ_LEN];
  int rc = rope_inv_from_store(store, inv, error, error_size);
  if (rc == 0)
    rc = h3_dit_rope_from_positions(position_ids, seq, inv,
                                    H3_DIT_ROPE_INV_FREQ_LEN, cos, sin);
  return rc;
}

typedef struct {
  double self_w;
  double mass_vid;
  double mass_txt;
  double max_w;
  double attn_branch_rms;
  double gate_msa_rms;
  double qk_vid;
  double qk_txt;
  double qk_aud;
  double k_rms_vid;
  double k_rms_txt;
  double k_rms_aud;
  double q_cos;
  double v_cos;
  double br_cos;
  int nprobe;
} h3_attn_probe;

static double vid_token_cos(const float *tok, int dim, const int *vrows, int nvi) {
  if (!tok || nvi < 2 || dim < 1)
    return 0.0;
  double cos_sum = 0;
  int np = 0;
  int nuse = nvi < 8 ? nvi : 8;
  for (int i = 0; i < nuse; i++) {
    const float *a = tok + (size_t)vrows[i] * (size_t)dim;
    double na = 0;
    for (int d = 0; d < dim; d++)
      na += (double)a[d] * (double)a[d];
    na = sqrt(na);
    for (int j = i + 1; j < nuse; j++) {
      const float *b = tok + (size_t)vrows[j] * (size_t)dim;
      double nb = 0, dot = 0;
      for (int d = 0; d < dim; d++) {
        dot += (double)a[d] * (double)b[d];
        nb += (double)b[d] * (double)b[d];
      }
      nb = sqrt(nb);
      if (na > 1e-12 && nb > 1e-12)
        cos_sum += dot / (na * nb);
      np++;
    }
  }
  return np ? cos_sum / (double)np : 0.0;
}

/* Head-0 softmax over up to 8 video query rows. Cheap at lab seq. */
static void attn_probe_video(h3_attn_probe *p, const float *q, const float *k,
                             const float *v, int seq, int heads, int hd,
                             float attn_scale, const int *tags, const int *idx,
                             const float *gate_msa, const float *branch, int H) {
  memset(p, 0, sizeof(*p));
  if (!q || !k || !tags || seq < 1 || heads < 1 || hd < 1)
    return;
  if (branch) {
    double br = 0;
    size_t n = (size_t)seq * (size_t)H;
    for (size_t i = 0; i < n; i++)
      br += (double)branch[i] * (double)branch[i];
    p->attn_branch_rms = sqrt(br / (double)n);
  }
  if (gate_msa && idx) {
    double gacc = 0;
    int nv = 0;
    for (int s = 0; s < seq; s++) {
      if (tags[s] != H3_ADALN_TAG_VIDEO)
        continue;
      const float *g = gate_msa + (size_t)idx[s] * (size_t)H;
      double gs = 0;
      for (int i = 0; i < H; i++)
        gs += (double)g[i] * (double)g[i];
      gacc += sqrt(gs / (double)H);
      nv++;
    }
    p->gate_msa_rms = nv ? gacc / (double)nv : 0.0;
  }
  float *scores = (float *)malloc((size_t)seq * sizeof(float));
  if (!scores)
    return;
  int vrows[8];
  int nvrows = 0;
  double kvid = 0, ktxt = 0, kaud = 0;
  int nkv = 0, nkt = 0, nka = 0;
  for (int s = 0; s < seq; s++) {
    const float *kr = k + ((size_t)s * (size_t)heads + 0) * (size_t)hd;
    double acc = 0;
    for (int d = 0; d < hd; d++)
      acc += (double)kr[d] * (double)kr[d];
    acc = sqrt(acc / (double)hd);
    if (tags[s] == H3_ADALN_TAG_VIDEO) {
      kvid += acc;
      nkv++;
      if (nvrows < 8)
        vrows[nvrows++] = s;
    } else if (tags[s] == H3_ADALN_TAG_TEXT) {
      ktxt += acc;
      nkt++;
    } else if (tags[s] == H3_ADALN_TAG_AUDIO) {
      kaud += acc;
      nka++;
    }
  }
  p->k_rms_vid = nkv ? kvid / (double)nkv : 0.0;
  p->k_rms_txt = nkt ? ktxt / (double)nkt : 0.0;
  p->k_rms_aud = nka ? kaud / (double)nka : 0.0;
  int h = 0;
  double qk_v = 0, qk_t = 0, qk_a = 0;
  int nqv = 0, nqt = 0, nqa = 0;
  for (int pi = 0; pi < nvrows; pi++) {
    int row = vrows[pi];
    const float *qr =
        q + ((size_t)row * (size_t)heads + (size_t)h) * (size_t)hd;
    float m = -1e30f;
    for (int col = 0; col < seq; col++) {
      const float *kr =
          k + ((size_t)col * (size_t)heads + (size_t)h) * (size_t)hd;
      double dot = 0.0;
      for (int d = 0; d < hd; d++)
        dot += (double)qr[d] * (double)kr[d];
      float s = (float)dot * attn_scale;
      scores[col] = s;
      if (s > m)
        m = s;
      if (tags[col] == H3_ADALN_TAG_VIDEO) {
        qk_v += s;
        nqv++;
      } else if (tags[col] == H3_ADALN_TAG_TEXT) {
        qk_t += s;
        nqt++;
      } else if (tags[col] == H3_ADALN_TAG_AUDIO) {
        qk_a += s;
        nqa++;
      }
    }
    double l = 0.0;
    for (int col = 0; col < seq; col++) {
      scores[col] = expf(scores[col] - m);
      l += scores[col];
    }
    float inv = (float)(1.0 / l);
    float mx = 0, sw = 0, vw = 0, tw = 0;
    for (int col = 0; col < seq; col++) {
      float w = scores[col] * inv;
      if (w > mx)
        mx = w;
      if (col == row)
        sw = w;
      if (tags[col] == H3_ADALN_TAG_VIDEO)
        vw += w;
      else if (tags[col] == H3_ADALN_TAG_TEXT)
        tw += w;
    }
    p->self_w += sw;
    p->mass_vid += vw;
    p->mass_txt += tw;
    p->max_w += mx;
    p->nprobe++;
  }
  free(scores);
  if (p->nprobe > 0) {
    double invn = 1.0 / (double)p->nprobe;
    p->self_w *= invn;
    p->mass_vid *= invn;
    p->mass_txt *= invn;
    p->max_w *= invn;
  }
  p->qk_vid = nqv ? qk_v / (double)nqv : 0.0;
  p->qk_txt = nqt ? qk_t / (double)nqt : 0.0;
  p->qk_aud = nqa ? qk_a / (double)nqa : 0.0;
  {
    int inner = heads * hd;
    p->q_cos = vid_token_cos(q, inner, vrows, nvrows);
    p->v_cos = v ? vid_token_cos(v, inner, vrows, nvrows) : 0.0;
    p->br_cos = branch ? vid_token_cos(branch, H, vrows, nvrows) : 0.0;
  }
}

/* Attention collapse vs gated-out: mosaic of 32px DiT patches is either. */
static void dit_debug_mix(const char *prefix, const float *q, const float *k,
                          int seq, int heads, int hd, float attn_scale,
                          const int *tags, const int *idx, const float *gate_msa,
                          const float *x, const float *branch, int H) {
  if (dit_debug_level() < 1 || !q || !k || !gate_msa || !idx || !tags)
    return;
  if (dit_debug_level() < 2 && prefix && strstr(prefix, "blocks.0.") == NULL)
    return;
  double g_vid = 0, g_txt = 0, n_vid = 0, n_txt = 0;
  double br = 0, xr = 0;
  for (int s = 0; s < seq; s++) {
    const float *g = gate_msa + (size_t)idx[s] * (size_t)H;
    double acc = 0;
    for (int i = 0; i < H; i++)
      acc += (double)g[i] * (double)g[i];
    acc = sqrt(acc / (double)H);
    if (tags[s] == H3_ADALN_TAG_VIDEO) {
      g_vid += acc;
      n_vid += 1;
    } else if (tags[s] == H3_ADALN_TAG_TEXT) {
      g_txt += acc;
      n_txt += 1;
    }
    if (x && branch) {
      const float *xr0 = x + (size_t)s * (size_t)H;
      const float *br0 = branch + (size_t)s * (size_t)H;
      double ax = 0, ab = 0;
      for (int i = 0; i < H; i++) {
        ax += (double)xr0[i] * (double)xr0[i];
        ab += (double)br0[i] * (double)br0[i];
      }
      xr += sqrt(ax / (double)H);
      br += sqrt(ab / (double)H);
    }
  }
  int h = 0;
  float *scores = (float *)malloc((size_t)seq * sizeof(float));
  if (!scores)
    return;
  double self_w = 0, vid_w = 0, txt_w = 0, max_w = 0;
  int nprobe = 0;
  int vrows[32];
  int nvrows = 0;
  for (int s = 0; s < seq && nvrows < 32; s++) {
    if (tags[s] == H3_ADALN_TAG_VIDEO)
      vrows[nvrows++] = s;
  }
  int nuse = nvrows < 8 ? nvrows : 8;
  for (int pi = 0; pi < nuse; pi++) {
    int row = vrows[pi];
    const float *qr =
        q + ((size_t)row * (size_t)heads + (size_t)h) * (size_t)hd;
    float m = -1e30f;
    for (int col = 0; col < seq; col++) {
      const float *kr =
          k + ((size_t)col * (size_t)heads + (size_t)h) * (size_t)hd;
      double dot = 0.0;
      for (int d = 0; d < hd; d++)
        dot += (double)qr[d] * (double)kr[d];
      float s = (float)dot * attn_scale;
      scores[col] = s;
      if (s > m)
        m = s;
    }
    double l = 0.0;
    for (int col = 0; col < seq; col++) {
      scores[col] = expf(scores[col] - m);
      l += scores[col];
    }
    float inv = (float)(1.0 / l);
    float mx = 0, sw = 0, vw = 0, tw = 0;
    for (int col = 0; col < seq; col++) {
      float w = scores[col] * inv;
      if (w > mx)
        mx = w;
      if (col == row)
        sw = w;
      if (tags[col] == H3_ADALN_TAG_VIDEO)
        vw += w;
      else if (tags[col] == H3_ADALN_TAG_TEXT)
        tw += w;
    }
    self_w += sw;
    vid_w += vw;
    txt_w += tw;
    max_w += mx;
    nprobe++;
  }
  free(scores);
  fprintf(stderr,
          "video-c: dit-debug %sgate_msa_rms vid=%.4g txt=%.4g "
          "x_rms=%.4g attn_branch_rms=%.4g video-rows head0 self=%.3f "
          "mass_vid=%.3f mass_txt=%.3f max=%.3f (S=%d nvid=%d)\n",
          prefix ? prefix : "", n_vid ? g_vid / n_vid : 0.0,
          n_txt ? g_txt / n_txt : 0.0, seq ? xr / seq : 0.0,
          seq ? br / seq : 0.0, nprobe ? self_w / nprobe : 0.0,
          nprobe ? vid_w / nprobe : 0.0, nprobe ? txt_w / nprobe : 0.0,
          nprobe ? max_w / nprobe : 0.0, seq, nvrows);
}

static const float *maybe_act_int8(const float *x, int rows, int in_dim,
                                   float **owned) {
  *owned = NULL;
  const char *e = getenv("H3_DIT_ACT_INT8");
  if (!e || !e[0] || e[0] == '0' || in_dim < 1 || (in_dim % 256) != 0)
    return x;
  size_t n = (size_t)rows * (size_t)in_dim;
  float *xq = (float *)malloc(n * sizeof(float));
  if (!xq)
    return x;
  memcpy(xq, x, n * sizeof(float));
  if (h3_convrot_fakequant_act(xq, rows, in_dim, 256) != 0) {
    free(xq);
    return x;
  }
  *owned = xq;
  return xq;
}

static int gemm_weight(const h3_st_store *st, const char *name, const float *x,
                       int rows, int in_dim, int out_dim, const float *bias,
                       float *y, char *error, size_t error_size) {
  float *xq = NULL;
  x = maybe_act_int8(x, rows, in_dim, &xq);
  size_t nw = (size_t)out_dim * (size_t)in_dim;
  const float *W = h3_st_store_get_f32(st, name, NULL, error, error_size);
  int rc = 0;
  if (W) {
    if (dit_debug_level() >= 1 && strstr(name, "blocks.0.mlp.fc")) {
      double acc = 0;
      for (size_t i = 0; i < nw; i++)
        acc += (double)W[i] * (double)W[i];
      fprintf(stderr, "video-c: dit-debug %s rms=%.4g shape=%dx%d\n", name,
              sqrt(acc / (double)nw), out_dim, in_dim);
    }
    rc = h3_dit_linear(x, rows, in_dim, out_dim, W, bias, y);
    free(xq);
    return rc;
  }
  float *Wc = (float *)malloc(nw * sizeof(float));
  if (!Wc) {
    free(xq);
    return -1;
  }
  rc = load_n(st, name, Wc, nw, error, error_size);
  if (rc == 0 && dit_debug_level() >= 1 && strstr(name, "blocks.0.mlp.fc")) {
    double acc = 0;
    for (size_t i = 0; i < nw; i++)
      acc += (double)Wc[i] * (double)Wc[i];
    fprintf(stderr, "video-c: dit-debug %s rms=%.4g shape=%dx%d\n", name,
            sqrt(acc / (double)nw), out_dim, in_dim);
  }
  if (rc == 0)
    rc = h3_dit_linear(x, rows, in_dim, out_dim, Wc, bias, y);
  free(Wc);
  free(xq);
  return rc;
}

int h3_dit_linear_named(const h3_st_store *store, const char *weight_name,
                        const char *bias_name, const float *x, int rows,
                        int in_dim, int out_dim, float *y, char *error,
                        size_t error_size) {
  float *bias = NULL;
  int rc = 0;
  if (bias_name) {
    bias = (float *)malloc((size_t)out_dim * sizeof(float));
    if (!bias)
      return -1;
    rc = load_n(store, bias_name, bias, (size_t)out_dim, error, error_size);
  }
  if (rc == 0)
    rc = gemm_weight(store, weight_name, x, rows, in_dim, out_dim, bias, y,
                     error, error_size);
  free(bias);
  return rc;
}

static int residual_attn_mlp(const h3_st_store *store, const char *prefix,
                             const float *x, int seq, const float *position_ids,
                             const float *gate_msa, const float *gate_mlp,
                             const int *idx, float *out, h3_dit_ws *ws,
                             char *error, size_t error_size) {
  const int H = H3_DIT_HIDDEN_SIZE;
  const int heads = H3_DIT_NUM_HEADS;
  const int hd = H3_DIT_HEAD_DIM;
  const int inner = H3_DIT_INNER_DIM;
  const int ffn = H3_DIT_FFN_HIDDEN;
  char name[96];
  int rc = 0;
  float *norm_w = (float *)malloc((size_t)H * sizeof(float));
  float *h = ws->h;
  float *branch = ws->branch;
  if (!norm_w)
    rc = -1;
  snprintf(name, sizeof(name), "%snorm1.weight", prefix);
  if (rc == 0)
    rc = load_n(store, name, norm_w, (size_t)H, error, error_size);
  if (rc == 0)
    rc = h3_dit_rmsnorm(x, seq, H, H3_DIT_NORM_EPS, norm_w, h);

  float *qkv = ws->qkv;
  float *q = ws->q;
  float *k = ws->k;
  float *v = ws->v;
  float *attn = ws->attn;
  snprintf(name, sizeof(name), "%sattn.qkv_proj.weight", prefix);
  if (rc == 0)
    rc = gemm_weight(store, name, h, seq, H, 3 * inner, NULL, qkv, error,
                     error_size);
  if (rc == 0)
    rc = h3_dit_qkv_split(qkv, seq, heads, hd, q, k, v);

  float *qn = (float *)malloc((size_t)hd * sizeof(float));
  float *kn = (float *)malloc((size_t)hd * sizeof(float));
  if (!qn || !kn)
    rc = -1;
  snprintf(name, sizeof(name), "%sattn.q_norm.weight", prefix);
  if (rc == 0)
    rc = load_n(store, name, qn, (size_t)hd, error, error_size);
  snprintf(name, sizeof(name), "%sattn.k_norm.weight", prefix);
  if (rc == 0)
    rc = load_n(store, name, kn, (size_t)hd, error, error_size);
  if (rc == 0)
    rc = h3_dit_rmsnorm(q, seq * heads, hd, H3_DIT_NORM_EPS, qn, q);
  if (rc == 0)
    rc = h3_dit_rmsnorm(k, seq * heads, hd, H3_DIT_NORM_EPS, kn, k);
  free(qn);
  free(kn);

  if (position_ids) {
    float inv[H3_DIT_ROPE_INV_FREQ_LEN];
    float *cos = ws->cos;
    float *sin = ws->sin;
    if (rc == 0)
      rc = rope_inv_from_store(store, inv, error, error_size);
    if (rc == 0)
      rc = h3_dit_rope_from_positions(position_ids, seq, inv,
                                      H3_DIT_ROPE_INV_FREQ_LEN, cos, sin);
    if (rc == 0)
      rc = h3_dit_apply_rotary_heads_inplace(q, seq, heads, hd, cos, sin,
                                             H3_DIT_ROPE_DIM);
    if (rc == 0)
      rc = h3_dit_apply_rotary_heads_inplace(k, seq, heads, hd, cos, sin,
                                             H3_DIT_ROPE_DIM);
  }
  float scale = 1.0f / sqrtf((float)hd);
  if (rc == 0)
    rc = h3_dit_sdpa_f32(attn, q, k, v, seq, heads, hd, scale);
  snprintf(name, sizeof(name), "%sattn.out_proj.weight", prefix);
  if (rc == 0)
    rc = gemm_weight(store, name, attn, seq, inner, H, NULL, branch, error,
                     error_size);
  if (rc == 0) {
    if (gate_msa && idx)
      rc = h3_dit_gated_residual_indexed(x, branch, gate_msa, idx, seq, H, out);
    else {
      for (int i = 0; i < seq * H; i++)
        out[i] = x[i] + branch[i];
    }
  }

  snprintf(name, sizeof(name), "%snorm2.weight", prefix);
  if (rc == 0)
    rc = load_n(store, name, norm_w, (size_t)H, error, error_size);
  if (rc == 0)
    rc = h3_dit_rmsnorm(out, seq, H, H3_DIT_NORM_EPS, norm_w, h);

  float *fused = ws->fused;
  float *hid = ws->hid;
  snprintf(name, sizeof(name), "%smlp.fc1.weight", prefix);
  if (rc == 0)
    rc = gemm_weight(store, name, h, seq, H, ffn * 2, NULL, fused, error,
                     error_size);
  if (rc == 0) {
    for (int r = 0; r < seq; r++) {
      const float *gate = fused + (size_t)r * (size_t)ffn * 2;
      h3_dit_silu_mul(gate, gate + ffn, hid + (size_t)r * (size_t)ffn,
                      (size_t)ffn);
    }
  }
  snprintf(name, sizeof(name), "%smlp.fc2.weight", prefix);
  if (rc == 0)
    rc = gemm_weight(store, name, hid, seq, ffn, H, NULL, branch, error,
                     error_size);
  if (rc == 0) {
    if (gate_mlp && idx)
      rc = h3_dit_gated_residual_indexed(out, branch, gate_mlp, idx, seq, H,
                                         out);
    else {
      for (int i = 0; i < seq * H; i++)
        out[i] += branch[i];
    }
  }
  free(norm_w);
  return rc;
}

int h3_dit_plain_block_forward(const h3_st_store *store, const char *prefix,
                               const float *x, int seq, float *out,
                               h3_dit_ws *ws, char *error, size_t error_size) {
  if (!store || !prefix || !x || !out || seq < 1)
    return -1;
  h3_dit_ws *owned = NULL;
  if (!ws) {
    owned = h3_dit_ws_create(seq);
    if (!owned)
      return -1;
    ws = owned;
  }
  int rc = residual_attn_mlp(store, prefix, x, seq, NULL, NULL, NULL, NULL, out,
                             ws, error, error_size);
  h3_dit_ws_free(owned);
  return rc;
}

int h3_dit_block_forward(const h3_st_store *store, int block, const float *x,
                         int seq, const int *tags, float timestep,
                         const float *row_t, const float *position_ids,
                         const float *rope_cos, const float *rope_sin,
                         const float *table, int grid, int rank, float *out,
                         h3_dit_ws *ws, char *error, size_t error_size) {
  if (error && error_size)
    error[0] = 0;
  if (!store || !x || !tags || !position_ids || !out || seq < 1 || block < 0 ||
      block >= H3_DIT_NUM_LAYERS)
    return -1;
  h3_dit_ws *owned = NULL;
  if (!ws) {
    owned = h3_dit_ws_create(seq);
    if (!owned)
      return -1;
    ws = owned;
  }

  const int H = H3_DIT_HIDDEN_SIZE;
  const int heads = H3_DIT_NUM_HEADS;
  const int hd = H3_DIT_HEAD_DIM;
  const int inner = H3_DIT_INNER_DIM;
  const int ffn = H3_DIT_FFN_HIDDEN;
  const int M = H3_ADALN_MODALITY_NUM;
  const int K = H3_ADALN_TENSORS_PER_BLOCK;
  char prefix[64];
  snprintf(prefix, sizeof(prefix), "blocks.%d.", block);

  float *owned_table = NULL;
  float *h_rms = NULL;
  if (!table) {
    grid = H3_ADALN_TABLE_GRID;
    rank = H3_ADALN_TABLE_RANK;
    owned_table = (float *)malloc((size_t)grid * (size_t)rank * sizeof(float));
    if (!owned_table) {
      h3_dit_ws_free(owned);
      return -1;
    }
    if (load_n(store, "adaln_t_table", owned_table, (size_t)grid * (size_t)rank,
               error, error_size) != 0) {
      free(owned_table);
      h3_dit_ws_free(owned);
      return -1;
    }
    table = owned_table;
  }
  if (grid < 2 || rank < 1) {
    free(owned_table);
    h3_dit_ws_free(owned);
    return -1;
  }

  float uniq[8];
  int *tslots = ws->tslots;
  int nuniq = h3_adaln_collect_timesteps(row_t, timestep, seq, uniq, 8, tslots);
  if (nuniq < 1) {
    h3_dit_ws_free(owned);
    free(owned_table);
    return -1;
  }

  float *emb = (float *)malloc((size_t)nuniq * (size_t)rank * sizeof(float));
  if (!emb || rank > 128) {
    free(emb);
    h3_dit_ws_free(owned);
    free(owned_table);
    return -1;
  }
  int rc = 0;
  for (int u = 0; u < nuniq && rc == 0; u++)
    rc = h3_adaln_table_embed(table, grid, rank, uniq[u],
                              emb + (size_t)u * (size_t)rank);

  char name[96];
  snprintf(name, sizeof(name), "%sadaln_proj.linear.weight", prefix);
  size_t adaln_n = (size_t)H3_ADALN_OUT_FEATURES * (size_t)rank;
  /* Cached-pointer read (no per-call 49.5 MB memcpy like load_n). Falls back
   * to a fresh decode when the weight cache is disabled or full. */
  float *adaln_owned = NULL;
  const float *adaln_w =
      rc == 0 ? h3_st_store_get_f32(store, name, NULL, error, error_size)
              : NULL;
  if (rc == 0 && !adaln_w) {
    adaln_owned = (float *)malloc(adaln_n * sizeof(float));
    if (adaln_owned &&
        load_n(store, name, adaln_owned, adaln_n, error, error_size) == 0)
      adaln_w = adaln_owned;
  }
  float *adaln_b = (float *)malloc((size_t)H3_ADALN_OUT_FEATURES * sizeof(float));
  float *proj =
      (float *)malloc((size_t)nuniq * (size_t)H3_ADALN_OUT_FEATURES * sizeof(float));
  float *six = (float *)malloc((size_t)K * (size_t)nuniq * (size_t)M * (size_t)H *
                               sizeof(float));
  if (!adaln_w || !adaln_b || !proj || !six)
    rc = -1;
  snprintf(name, sizeof(name), "%sadaln_proj.linear.bias", prefix);
  if (rc == 0)
    rc = load_n(store, name, adaln_b, (size_t)H3_ADALN_OUT_FEATURES, error,
                error_size);
  if (rc == 0) {
    /* Comfy curve-table: lerp(adaln_t_table) is already silu(temb) in rank-k;
       apply_silu is false. H3_ADALN_SILU=1 restores a second SiLU (wide-path). */
    const char *silu = getenv("H3_ADALN_SILU");
    if (silu && silu[0] && silu[0] != '0')
      h3_dit_silu(emb, emb, (size_t)nuniq * (size_t)rank);
  }
  if (rc == 0)
    rc = h3_dit_linear(emb, nuniq, rank, H3_ADALN_OUT_FEATURES, adaln_w, adaln_b,
                       proj);
  free(adaln_owned);
  free(adaln_b);
  free(emb);
  if (rc == 0)
    rc = h3_adaln_split_block(proj, nuniq, H, six);
  free(proj);
  size_t per = (size_t)nuniq * (size_t)M * (size_t)H;
  float *shift_msa = NULL, *scale_msa = NULL, *gate_msa = NULL;
  float *shift_mlp = NULL, *scale_mlp = NULL, *gate_mlp = NULL;
  if (six) {
    shift_msa = six + 0 * per;
    scale_msa = six + 1 * per;
    gate_msa = six + 2 * per;
    shift_mlp = six + 3 * per;
    scale_mlp = six + 4 * per;
    gate_mlp = six + 5 * per;
    {
      const char *gc = getenv("H3_GATE_CLIP");
      if (gc && gc[0] && gc[0] != '0') {
        float clip = strtof(gc, NULL);
        if (clip > 0.f) {
          for (size_t i = 0; i < per; i++) {
            if (gate_msa[i] > clip)
              gate_msa[i] = clip;
            else if (gate_msa[i] < -clip)
              gate_msa[i] = -clip;
            if (gate_mlp[i] > clip)
              gate_mlp[i] = clip;
            else if (gate_mlp[i] < -clip)
              gate_mlp[i] = -clip;
          }
        }
      }
    }
  }

  int *idx = ws->idx;
  float *norm_w = (float *)malloc((size_t)H * sizeof(float));
  float *h = ws->h;
  float *branch = ws->branch;
  if (!norm_w)
    rc = -1;
  for (int s = 0; s < seq && rc == 0; s++) {
    idx[s] = h3_adaln_modality_row(tslots[s], tags[s]);
    if (idx[s] < 0)
      rc = -1;
  }

  if (rc == 0 && dit_debug_level() >= 1 &&
      (dit_debug_level() >= 2 || strstr(prefix, "blocks.0."))) {
    int v0 = -1;
    for (int s = 0; s < seq; s++) {
      if (tags[s] == H3_ADALN_TAG_VIDEO) {
        v0 = s;
        break;
      }
    }
    if (v0 >= 0) {
      const float *mods[6] = {shift_msa, scale_msa, gate_msa,
                              shift_mlp, scale_mlp, gate_mlp};
      const char *mn[6] = {"shift_msa", "scale_msa", "gate_msa",
                           "shift_mlp", "scale_mlp", "gate_mlp"};
      fprintf(stderr, "video-c: dit-debug %sadaln video-row", prefix);
      for (int k = 0; k < 6; k++) {
        const float *g = mods[k] + (size_t)idx[v0] * (size_t)H;
        double acc = 0;
        for (int i = 0; i < H; i++)
          acc += (double)g[i] * (double)g[i];
        fprintf(stderr, " %s_rms=%.4g", mn[k], sqrt(acc / (double)H));
      }
      fprintf(stderr, "\n");
    }
  }

  snprintf(name, sizeof(name), "%snorm1.weight", prefix);
  if (rc == 0)
    rc = load_n(store, name, norm_w, (size_t)H, error, error_size);
  if (rc == 0)
    rc = h3_dit_rmsnorm(x, seq, H, H3_DIT_NORM_EPS, norm_w, h);
  int hvrow[8];
  int nhv = 0;
  for (int s = 0; s < seq && nhv < 8; s++) {
    if (tags[s] == H3_ADALN_TAG_VIDEO)
      hvrow[nhv++] = s;
  }
  double h_rms_cos = 0, h_mod_cos = 0;
  if (rc == 0)
    h_rms_cos = vid_token_cos(h, H, hvrow, nhv);
  if (rc == 0 && block == 0) {
    const char *dump0 = getenv("H3_DUMP_L0");
    if (dump0 && dump0[0] && dump0[0] != '0') {
      h_rms = (float *)malloc((size_t)seq * (size_t)H * sizeof(float));
      if (h_rms)
        memcpy(h_rms, h, (size_t)seq * (size_t)H * sizeof(float));
    }
  }
  if (rc == 0)
    rc = h3_dit_modulate_indexed(h, shift_msa, scale_msa, idx, seq, H, h);
  if (rc == 0)
    h_mod_cos = vid_token_cos(h, H, hvrow, nhv);

  float *qkv = ws->qkv;
  float *q = ws->q;
  float *k = ws->k;
  float *v = ws->v;
  float *attn = ws->attn;
  snprintf(name, sizeof(name), "%sattn.qkv_proj.weight", prefix);
  if (rc == 0)
    rc = gemm_weight(store, name, h, seq, H, 3 * inner, NULL, qkv, error,
                     error_size);
  if (rc == 0)
    rc = h3_dit_qkv_split(qkv, seq, heads, hd, q, k, v);
  if (rc == 0 && block == 0) {
    const char *dump = getenv("H3_DUMP_L0");
    if (dump && dump[0] && dump[0] != '0') {
      const char *dir = (dump[0] == '1' && dump[1] == 0) ? "/tmp/h3_l0" : dump;
      char cmd[320];
      snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", dir);
      if (system(cmd) == 0) {
        char path[768];
        FILE *f;
        snprintf(path, sizeof(path), "%s/meta.txt", dir);
        f = fopen(path, "w");
        if (f) {
          fprintf(f, "seq=%d H=%d inner=%d heads=%d hd=%d\n", seq, H, inner,
                  heads, hd);
          fprintf(f, "tags");
          for (int s = 0; s < seq; s++)
            fprintf(f, " %d", tags[s]);
          fprintf(f, "\nuniq");
          for (int u = 0; u < nuniq; u++)
            fprintf(f, " %.9g", uniq[u]);
          fprintf(f, "\nrow_t");
          for (int s = 0; s < seq; s++)
            fprintf(f, " %.9g", row_t ? row_t[s] : timestep);
          fprintf(f, "\nidx");
          for (int s = 0; s < seq; s++)
            fprintf(f, " %d", idx[s]);
          fprintf(f, "\n");
          fclose(f);
        }
        if (h_rms) {
          snprintf(path, sizeof(path), "%s/h_rms.bin", dir);
          f = fopen(path, "wb");
          if (f) {
            fwrite(h_rms, sizeof(float), (size_t)seq * (size_t)H, f);
            fclose(f);
          }
        }
        snprintf(path, sizeof(path), "%s/x.bin", dir);
        f = fopen(path, "wb");
        if (f) {
          fwrite(x, sizeof(float), (size_t)seq * (size_t)H, f);
          fclose(f);
        }
        snprintf(path, sizeof(path), "%s/pos.bin", dir);
        f = fopen(path, "wb");
        if (f) {
          fwrite(position_ids, sizeof(float), (size_t)seq * 3, f);
          fclose(f);
        }
        snprintf(path, sizeof(path), "%s/h.bin", dir);
        f = fopen(path, "wb");
        if (f) {
          fwrite(h, sizeof(float), (size_t)seq * (size_t)H, f);
          fclose(f);
        }
        snprintf(path, sizeof(path), "%s/qkv.bin", dir);
        f = fopen(path, "wb");
        if (f) {
          fwrite(qkv, sizeof(float), (size_t)seq * 3 * (size_t)inner, f);
          fclose(f);
        }
        snprintf(path, sizeof(path), "%s/v.bin", dir);
        f = fopen(path, "wb");
        if (f) {
          fwrite(v, sizeof(float), (size_t)seq * (size_t)inner, f);
          fclose(f);
        }
        snprintf(path, sizeof(path), "%s/q.bin", dir);
        f = fopen(path, "wb");
        if (f) {
          fwrite(q, sizeof(float), (size_t)seq * (size_t)inner, f);
          fclose(f);
        }
        fprintf(stderr, "video-c: dumped L0 h/qkv/q/v to %s\n", dir);
      }
    }
  }
  float *qn = (float *)malloc((size_t)hd * sizeof(float));
  float *kn = (float *)malloc((size_t)hd * sizeof(float));
  if (!qn || !kn)
    rc = -1;
  snprintf(name, sizeof(name), "%sattn.q_norm.weight", prefix);
  if (rc == 0)
    rc = load_n(store, name, qn, (size_t)hd, error, error_size);
  snprintf(name, sizeof(name), "%sattn.k_norm.weight", prefix);
  if (rc == 0)
    rc = load_n(store, name, kn, (size_t)hd, error, error_size);
  if (rc == 0)
    rc = h3_dit_rmsnorm(q, seq * heads, hd, H3_DIT_NORM_EPS, qn, q);
  if (rc == 0)
    rc = h3_dit_rmsnorm(k, seq * heads, hd, H3_DIT_NORM_EPS, kn, k);
  free(qn);
  free(kn);

  /* RoPE tables are identical for every block and step — shared via the
   * caller (h3_dit_rope_tables once per forward) or built here as fallback. */
  float *cos = ws->cos;
  float *sin = ws->sin;
  if (rc == 0 && (!rope_cos || !rope_sin)) {
    float inv[H3_DIT_ROPE_INV_FREQ_LEN];
    rc = rope_inv_from_store(store, inv, error, error_size);
    if (rc == 0)
      rc = h3_dit_rope_from_positions(position_ids, seq, inv,
                                      H3_DIT_ROPE_INV_FREQ_LEN, cos, sin);
    rope_cos = cos;
    rope_sin = sin;
  }
  if (rc == 0)
    rc = h3_dit_apply_rotary_heads_inplace(q, seq, heads, hd, rope_cos,
                                           rope_sin, H3_DIT_ROPE_DIM);
  if (rc == 0)
    rc = h3_dit_apply_rotary_heads_inplace(k, seq, heads, hd, rope_cos,
                                           rope_sin, H3_DIT_ROPE_DIM);
  float scale = 1.0f / sqrtf((float)hd);
  if (rc == 0)
    rc = h3_dit_sdpa_f32(attn, q, k, v, seq, heads, hd, scale);
  snprintf(name, sizeof(name), "%sattn.out_proj.weight", prefix);
  if (rc == 0)
    rc = gemm_weight(store, name, attn, seq, inner, H, NULL, branch, error,
                     error_size);
  h3_attn_probe ap;
  memset(&ap, 0, sizeof(ap));
  if (rc == 0 && dit_debug_level() >= 1)
    attn_probe_video(&ap, q, k, v, seq, heads, hd, scale, tags, idx, gate_msa,
                     branch, H);
  if (rc == 0)
    dit_debug_mix(prefix, q, k, seq, heads, hd, scale, tags, idx, gate_msa, x,
                  branch, H);
  if (rc == 0)
    rc = h3_dit_gated_residual_indexed(x, branch, gate_msa, idx, seq, H, out);
  if (rc == 0 && want_bf16_act())
    cast_bf16_n(out, (size_t)seq * (size_t)H);

  int skip_mlp = 0;
  {
    const char *e = getenv("H3_DIT_SKIP_MLP_AFTER");
    if (e && e[0]) {
      int cut = atoi(e);
      if (cut >= 0 && block > cut)
        skip_mlp = 1;
    }
  }
  if (skip_mlp) {
    if (rc == 0) {
      size_t n = (size_t)seq * (size_t)H;
      double xin = 0, xout = 0;
      for (size_t i = 0; i < n; i++) {
        xin += (double)x[i] * (double)x[i];
        xout += (double)out[i] * (double)out[i];
      }
      fprintf(stderr,
              "video-c: dit-layer %d x_in_rms=%.4g mlp_in_rms=0 "
              "mlp_branch_rms=0 gate_mlp_rms=0 x_out_rms=%.4g "
              "attn_br=%.4g gate_msa=%.4g self=%.3f mass_vid=%.3f "
              "mass_txt=%.3f max=%.3f (skip-mlp)\n",
              block, sqrt(xin / (double)n), sqrt(xout / (double)n),
              ap.attn_branch_rms, ap.gate_msa_rms, ap.self_w, ap.mass_vid,
              ap.mass_txt, ap.max_w);
      fflush(stderr);
    }
    goto block_done;
  }

  snprintf(name, sizeof(name), "%snorm2.weight", prefix);
  if (rc == 0)
    rc = load_n(store, name, norm_w, (size_t)H, error, error_size);
  if (rc == 0)
    rc = h3_dit_rmsnorm(out, seq, H, H3_DIT_NORM_EPS, norm_w, h);
  if (rc == 0)
    rc = h3_dit_modulate_indexed(h, shift_mlp, scale_mlp, idx, seq, H, h);
  double mlp_in_rms = 0, mlp_in_max = 0, hid_rms = 0, hid_max = 0,
         fc1_rms = 0, fc1_max = 0;
  if (rc == 0) {
    size_t hn = (size_t)seq * (size_t)H;
    for (size_t i = 0; i < hn; i++) {
      double v = (double)h[i];
      mlp_in_rms += v * v;
      double a = fabs(v);
      if (a > mlp_in_max)
        mlp_in_max = a;
    }
    mlp_in_rms = sqrt(mlp_in_rms / (double)hn);
  }

  float *fused = ws->fused;
  float *hid = ws->hid;
  snprintf(name, sizeof(name), "%smlp.fc1.weight", prefix);
  if (rc == 0)
    rc = gemm_weight(store, name, h, seq, H, ffn * 2, NULL, fused, error,
                     error_size);
  if (rc == 0 && fused) {
    size_t fn = (size_t)seq * 2 * (size_t)ffn;
    for (size_t i = 0; i < fn; i++) {
      double v = (double)fused[i];
      fc1_rms += v * v;
      double a = fabs(v);
      if (a > fc1_max)
        fc1_max = a;
    }
    fc1_rms = sqrt(fc1_rms / (double)fn);
  }
  {
    const char *clip_e = getenv("H3_FFN_CLIP");
    const char *clip_t = getenv("H3_FFN_CLIP_TEXT");
    if (rc == 0 && fused) {
      float clip_all = (clip_e && clip_e[0] && clip_e[0] != '0')
                           ? strtof(clip_e, NULL)
                           : 0.f;
      float clip_txt = (clip_t && clip_t[0] && clip_t[0] != '0')
                           ? strtof(clip_t, NULL)
                           : 0.f;
      if (clip_all > 0.f || clip_txt > 0.f) {
        for (int r = 0; r < seq; r++) {
          float clip = clip_all;
          if (clip_txt > 0.f && tags && tags[r] == H3_ADALN_TAG_TEXT)
            clip = (clip_all > 0.f && clip_all < clip_txt) ? clip_all : clip_txt;
          else if (clip_all <= 0.f)
            continue;
          if (clip <= 0.f)
            continue;
          float *row = fused + (size_t)r * 2 * (size_t)ffn;
          for (int i = 0; i < ffn * 2; i++) {
            if (row[i] > clip)
              row[i] = clip;
            else if (row[i] < -clip)
              row[i] = -clip;
          }
        }
      }
    }
  }
  if (rc == 0) {
    int swap = 0;
    const char *sw = getenv("H3_SWIGLU_SWAP");
    if (sw && sw[0] && sw[0] != '0')
      swap = 1;
    for (int r = 0; r < seq; r++) {
      const float *gate = fused + (size_t)r * (size_t)ffn * 2;
      if (swap)
        h3_dit_silu_mul(gate + ffn, gate, hid + (size_t)r * (size_t)ffn,
                        (size_t)ffn);
      else
        h3_dit_silu_mul(gate, gate + ffn, hid + (size_t)r * (size_t)ffn,
                        (size_t)ffn);
    }
  }
  {
    const char *clip_e = getenv("H3_FFN_CLIP");
    const char *clip_t = getenv("H3_FFN_CLIP_TEXT");
    if (rc == 0 && hid) {
      float clip_all = (clip_e && clip_e[0] && clip_e[0] != '0')
                           ? strtof(clip_e, NULL)
                           : 0.f;
      float clip_txt = (clip_t && clip_t[0] && clip_t[0] != '0')
                           ? strtof(clip_t, NULL)
                           : 0.f;
      if (clip_all > 0.f || clip_txt > 0.f) {
        for (int r = 0; r < seq; r++) {
          float clip = clip_all;
          if (clip_txt > 0.f && tags && tags[r] == H3_ADALN_TAG_TEXT)
            clip = (clip_all > 0.f && clip_all < clip_txt) ? clip_all : clip_txt;
          else if (clip_all <= 0.f)
            continue;
          if (clip <= 0.f)
            continue;
          float *row = hid + (size_t)r * (size_t)ffn;
          for (int i = 0; i < ffn; i++) {
            if (row[i] > clip)
              row[i] = clip;
            else if (row[i] < -clip)
              row[i] = -clip;
          }
        }
      }
    }
  }
  if (rc == 0 && hid) {
    size_t hdn = (size_t)seq * (size_t)ffn;
    for (size_t i = 0; i < hdn; i++) {
      double v = (double)hid[i];
      hid_rms += v * v;
      double a = fabs(v);
      if (a > hid_max)
        hid_max = a;
    }
    hid_rms = sqrt(hid_rms / (double)hdn);
  }
  double hid_max_v = 0, hid_max_t = 0, hid_max_a = 0;
  if (rc == 0 && hid && tags) {
    for (int r = 0; r < seq; r++) {
      const float *row = hid + (size_t)r * (size_t)ffn;
      double am = 0;
      for (int i = 0; i < ffn; i++) {
        double a = fabs((double)row[i]);
        if (a > am)
          am = a;
      }
      if (tags[r] == H3_ADALN_TAG_VIDEO && am > hid_max_v)
        hid_max_v = am;
      else if (tags[r] == H3_ADALN_TAG_TEXT && am > hid_max_t)
        hid_max_t = am;
      else if (tags[r] == H3_ADALN_TAG_AUDIO && am > hid_max_a)
        hid_max_a = am;
    }
  }
  if (rc == 0 && dit_debug_level() >= 1 &&
      (dit_debug_level() >= 2 || strstr(prefix, "blocks.0."))) {
    double hr = 0, fr = 0, hidr = 0;
    size_t hn = (size_t)seq * (size_t)H;
    size_t fn = (size_t)seq * 2 * (size_t)ffn;
    size_t hdn = (size_t)seq * (size_t)ffn;
    for (size_t i = 0; i < hn; i++)
      hr += (double)h[i] * (double)h[i];
    if (fused) {
      for (size_t i = 0; i < fn; i++)
        fr += (double)fused[i] * (double)fused[i];
    }
    if (hid) {
      for (size_t i = 0; i < hdn; i++)
        hidr += (double)hid[i] * (double)hid[i];
    }
    fprintf(stderr,
            "video-c: dit-debug %smlp_in_rms=%.4g fc1_rms=%.4g swiglu_hid_rms=%.4g\n",
            prefix, sqrt(hr / (double)hn), sqrt(fr / (double)fn),
            sqrt(hidr / (double)hdn));
  }
  snprintf(name, sizeof(name), "%smlp.fc2.weight", prefix);
  if (rc == 0)
    rc = gemm_weight(store, name, hid, seq, ffn, H, NULL, branch, error,
                     error_size);
  if (rc == 0 && dit_debug_level() >= 1 &&
      (dit_debug_level() >= 2 || strstr(prefix, "blocks.0."))) {
    double br = 0;
    for (int i = 0; i < seq * H; i++)
      br += (double)branch[i] * (double)branch[i];
    fprintf(stderr, "video-c: dit-debug %smlp_branch_rms=%.4g\n", prefix,
            sqrt(br / (double)(seq * H)));
  }
  if (rc == 0)
    rc = h3_dit_gated_residual_indexed(out, branch, gate_mlp, idx, seq, H, out);
  if (rc == 0 && want_bf16_act())
    cast_bf16_n(out, (size_t)seq * (size_t)H);

  if (rc == 0 && dit_debug_level() >= 1) {
    size_t n = (size_t)seq * (size_t)H;
    double xin = 0, xout = 0, br = 0, gacc = 0;
    int nv = 0;
    int vrow[8];
    int nvi = 0;
    for (size_t i = 0; i < n; i++) {
      xin += (double)x[i] * (double)x[i];
      xout += (double)out[i] * (double)out[i];
      br += (double)branch[i] * (double)branch[i];
    }
    for (int s = 0; s < seq; s++) {
      if (tags[s] != H3_ADALN_TAG_VIDEO)
        continue;
      const float *g = gate_mlp + (size_t)idx[s] * (size_t)H;
      double gs = 0;
      for (int i = 0; i < H; i++)
        gs += (double)g[i] * (double)g[i];
      gacc += sqrt(gs / (double)H);
      nv++;
      if (nvi < 8)
        vrow[nvi++] = s;
    }
    double cos_sum = 0;
    int np = 0;
    double vrms_lo = 1e300, vrms_hi = 0;
    for (int i = 0; i < nvi; i++) {
      const float *a = out + (size_t)vrow[i] * (size_t)H;
      double na = 0;
      for (int d = 0; d < H; d++)
        na += (double)a[d] * (double)a[d];
      na = sqrt(na);
      if (na < vrms_lo)
        vrms_lo = na;
      if (na > vrms_hi)
        vrms_hi = na;
      for (int j = i + 1; j < nvi; j++) {
        const float *b = out + (size_t)vrow[j] * (size_t)H;
        double nb = 0, dot = 0;
        for (int d = 0; d < H; d++) {
          dot += (double)a[d] * (double)b[d];
          nb += (double)b[d] * (double)b[d];
        }
        nb = sqrt(nb);
        if (na > 1e-12 && nb > 1e-12)
          cos_sum += dot / (na * nb);
        np++;
      }
    }
    fprintf(stderr,
            "video-c: dit-layer %d x_in_rms=%.4g mlp_in_rms=%.4g "
            "mlp_branch_rms=%.4g gate_mlp_rms=%.4g x_out_rms=%.4g vid_cos=%.4f "
            "attn_br=%.4g gate_msa=%.4g self=%.3f mass_vid=%.3f mass_txt=%.3f "
            "max=%.3f qk_v/t/a=%.3f/%.3f/%.3f k_rms_v/t/a=%.3f/%.3f/%.3f "
            "q_cos=%.3f v_cos=%.3f br_cos=%.3f h_rms_cos=%.3f h_mod_cos=%.3f "
            "mlp_in_max=%.4g fc1_rms=%.4g fc1_max=%.4g hid_rms=%.4g hid_max=%.4g "
            "hid_max_v/t/a=%.4g/%.4g/%.4g vid_rms=%.4g/%.4g\n",
            block, sqrt(xin / (double)n), mlp_in_rms, sqrt(br / (double)n),
            nv ? gacc / (double)nv : 0.0, sqrt(xout / (double)n),
            np ? cos_sum / (double)np : 0.0, ap.attn_branch_rms,
            ap.gate_msa_rms, ap.self_w, ap.mass_vid, ap.mass_txt, ap.max_w,
            ap.qk_vid, ap.qk_txt, ap.qk_aud, ap.k_rms_vid, ap.k_rms_txt,
            ap.k_rms_aud, ap.q_cos, ap.v_cos, ap.br_cos, h_rms_cos, h_mod_cos,
            mlp_in_max, fc1_rms, fc1_max, hid_rms, hid_max, hid_max_v,
            hid_max_t, hid_max_a, nvi ? vrms_lo : 0.0, nvi ? vrms_hi : 0.0);
    fflush(stderr);
  }

  if (rc == 0 && block == 0) {
    const char *dump = getenv("H3_DUMP_L0");
    if (dump && dump[0] && dump[0] != '0') {
      const char *dir = (dump[0] == '1' && dump[1] == 0) ? "/tmp/h3_l0" : dump;
      char path[768];
      snprintf(path, sizeof(path), "%s/y.bin", dir);
      FILE *f = fopen(path, "wb");
      if (f) {
        fwrite(out, sizeof(float), (size_t)seq * (size_t)H, f);
        fclose(f);
        fprintf(stderr, "video-c: dumped L0 y to %s\n", dir);
      }
    }
  }

  if (rc != 0 && error && error_size && !error[0])
    snprintf(error, error_size, "h3_dit_block_forward: block %d failed", block);

block_done:
  free(owned_table);
  free(h_rms);
  free(six);
  free(norm_w);
  h3_dit_ws_free(owned);
  return rc;
}
