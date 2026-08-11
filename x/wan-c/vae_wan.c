#include "wan_internal.h"
#include "wan_profile.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#define WAN_VAE_ACCEL 1
#endif

/* Brick 11: exclusive-ish stage timers under WAN_VAE_STAGE_PROF=1. */
#define VAE_STAGE_T0()                                                         \
  (wan_profile_vae_stage_on() ? wan_profile_now_ms() : 0.0)
#define VAE_STAGE_ADD(name, t0)                                                \
  do {                                                                         \
    if (wan_profile_vae_stage_on())                                            \
      wan_profile_add_ms((name), wan_profile_now_ms() - (t0));                 \
  } while (0)

/*
 * VAE decode (F0781–F0784):
 *   Prefer in-daemon: GROUP_NORM → [SILU_MUL] → CONV2D → [RESIDUAL] → UNPATCHIFY3D
 *   EXT path when UMA_WAN_EXT=1; else host GN + host 1×1 C→3 + upsample.
 *
 * Real-weight tip: conv2→conv1→middle residuals→nearest×2→pool→head.2
 * Channel-change C→3 RGB stays host when tip unavailable.
 */

static void silu_inplace(float *x, size_t n) {
  for (size_t i = 0; i < n; i++)
    x[i] = x[i] / (1.f + expf(-x[i]));
}

/*
 * Wan RMS_norm(images=False): F.normalize(x, dim=C) * √C * gamma
 * (== channel L2-normalize then scale). Match PyTorch clamp_min(eps).
 */
static void rms_norm_ncdhw(float *x, const float *gamma, int C, int lt, int lh,
                           int lw, float eps) {
  size_t spat = (size_t)lt * (size_t)lh * (size_t)lw;
  float scale = sqrtf((float)C);
  for (size_t s = 0; s < spat; s++) {
    float n2 = 0.f;
    for (int c = 0; c < C; c++) {
      float v = x[(size_t)c * spat + s];
      n2 += v * v;
    }
    float nrm = sqrtf(n2);
    if (nrm < eps)
      nrm = eps;
    float inv = scale / nrm;
    for (int c = 0; c < C; c++) {
      float g = gamma ? gamma[c] : 1.f;
      x[(size_t)c * spat + s] *= inv * g;
    }
  }
}

#define VAE_CACHE_T 2
#define VAE_CACHE_MAX 96

typedef struct {
  char key[128];
  float *data; /* [C, T<=2, H, W] */
  int C, T, H, W;
  int rep; /* Wan upsample3d first-hit sentinel ('Rep') */
  int on_broker; /* F1012: sticky mid/out already seeded on daemon */
} vae_cache_ent;

typedef struct {
  vae_cache_ent e[VAE_CACHE_MAX];
  int n;
} vae_ccache;

static void vae_ccache_clear(vae_ccache *c) {
  if (!c)
    return;
  for (int i = 0; i < c->n; i++) {
    free(c->e[i].data);
    c->e[i].data = NULL;
  }
  c->n = 0;
}

static vae_cache_ent *vae_ccache_find(vae_ccache *c, const char *key) {
  if (!c || !key)
    return NULL;
  for (int i = 0; i < c->n; i++)
    if (strcmp(c->e[i].key, key) == 0)
      return &c->e[i];
  if (c->n >= VAE_CACHE_MAX)
    return NULL;
  vae_cache_ent *e = &c->e[c->n++];
  memset(e, 0, sizeof(*e));
  snprintf(e->key, sizeof(e->key), "%s", key);
  return e;
}

static int vae_ccache_has_data(vae_ccache *c, const char *key) {
  if (!c || !key)
    return 0;
  for (int i = 0; i < c->n; i++)
    if (strcmp(c->e[i].key, key) == 0)
      return c->e[i].data && c->e[i].T > 0;
  return 0;
}

static void vae_ccache_clear_key(vae_ccache *c, const char *key) {
  if (!c || !key)
    return;
  for (int i = 0; i < c->n; i++) {
    if (strcmp(c->e[i].key, key) == 0) {
      free(c->e[i].data);
      c->e[i].data = NULL;
      c->e[i].T = 0;
      return;
    }
  }
}

/* Mirror host causal_conv cache bookkeeping after broker HEADT (no feat_cache). */
static void vae_ccache_poke_from_in(vae_ccache *c, const char *key,
                                    const float *in, int Cin, int lt, int lh,
                                    int lw) {
  if (!c || !key || !in || lt < 1)
    return;
  vae_cache_ent *ent = vae_ccache_find(c, key);
  if (!ent)
    return;
  int store_t = lt < VAE_CACHE_T ? lt : VAE_CACHE_T;
  int have_prev = ent->data && ent->C == Cin && ent->H == lh && ent->W == lw &&
                  ent->T > 0;
  int next_t = store_t;
  if (store_t < VAE_CACHE_T && have_prev) {
    next_t = store_t + 1;
    if (next_t > VAE_CACHE_T)
      next_t = VAE_CACHE_T;
  }
  float *next =
      calloc((size_t)Cin * (size_t)next_t * (size_t)lh * (size_t)lw,
             sizeof(float));
  if (!next)
    return;
  int dst0 = 0;
  if (store_t < VAE_CACHE_T && have_prev) {
    for (int ch = 0; ch < Cin; ch++)
      for (int h = 0; h < lh; h++)
        for (int w = 0; w < lw; w++)
          next[((((size_t)ch * next_t + 0) * lh + h) * lw + w)] =
              ent->data[((((size_t)ch * ent->T + (ent->T - 1)) * lh + h) * lw +
                         w)];
    dst0 = 1;
  }
  int src0 = lt - store_t;
  for (int ti = 0; ti < store_t; ti++)
    for (int ch = 0; ch < Cin; ch++)
      for (int h = 0; h < lh; h++)
        for (int w = 0; w < lw; w++)
          next[((((size_t)ch * next_t + (dst0 + ti)) * lh + h) * lw + w)] =
              in[((((size_t)ch * lt + (src0 + ti)) * lh + h) * lw + w)];
  free(ent->data);
  ent->data = next;
  ent->C = Cin;
  ent->T = next_t;
  ent->H = lh;
  ent->W = lw;
}

static int vae_env_truthy(const char *name) {
  const char *e = getenv(name);
  return e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y' || e[0] == 't' ||
               e[0] == 'T');
}

/* F1012 broker warm remains opt-in (Brick 8 default-on miss: rematch+wall).
 * Enable: WAN_VAE_WARM_HEADT=1. Force off: WAN_VAE_NO_WARM_HEADT=1. */
static int vae_warm_broker_enabled(void) {
  if (vae_env_truthy("WAN_VAE_NO_WARM_HEADT"))
    return 0;
  return vae_env_truthy("WAN_VAE_WARM_HEADT");
}

static int vae_headt_enabled(const wan_ctx *ctx) {
  if (!ctx || !ctx->uma || !ctx->bufs || ctx->local_mode)
    return 0;
  if (vae_env_truthy("WAN_VAE_NO_HEADT") || vae_env_truthy("WAN_VAE_HOST"))
    return 0;
  return ctx->caps.channel_rms && ctx->caps.silu && ctx->caps.causal_pad3d &&
         ctx->caps.conv3d;
}

static int vae_bank_put_named(wan_ctx *ctx, const char *bank_key,
                              const char *tensor_name, size_t expect) {
  size_t nw = 0;
  const float *w = wan_borrow_tensor_f32(ctx, tensor_name, &nw);
  if (!w || (expect > 0 && nw != expect))
    return -1;
  return uma_buf_pool_bank_put(ctx->bufs, bank_key, w, nw * sizeof(float));
}

/* F1002: bank tip weights once (head + resample.11 + u14 residual). */
static int vae_headt_bank_all(wan_ctx *ctx) {
  if (!ctx || ctx->vae_headt_ready)
    return 0;
  if (!vae_headt_enabled(ctx))
    return -1;
  struct {
    const char *bank;
    const char *tensor;
    size_t ne;
  } items[] = {
      {"decoder.head.2.weight", "vae.decoder.head.2.weight",
       (size_t)3 * 96 * 3 * 3 * 3},
      {"decoder.conv1.weight", "vae.decoder.conv1.weight",
       (size_t)384 * 16 * 3 * 3 * 3},
      {"decoder.upsamples.11.resample.1.weight",
       "vae.decoder.upsamples.11.resample.1.weight", (size_t)96 * 192 * 3 * 3},
      {"decoder.upsamples.3.resample.1.weight",
       "vae.decoder.upsamples.3.resample.1.weight", (size_t)192 * 384 * 3 * 3},
      {"decoder.upsamples.7.resample.1.weight",
       "vae.decoder.upsamples.7.resample.1.weight", (size_t)192 * 384 * 3 * 3},
      /* ups.4 expand 192→384 + 1×1 shortcut */
      {"decoder.upsamples.4.residual.2.weight",
       "vae.decoder.upsamples.4.residual.2.weight",
       (size_t)384 * 192 * 3 * 3 * 3},
      {"decoder.upsamples.4.residual.6.weight",
       "vae.decoder.upsamples.4.residual.6.weight",
       (size_t)384 * 384 * 3 * 3 * 3},
      {"decoder.upsamples.4.shortcut.weight",
       "vae.decoder.upsamples.4.shortcut.weight", (size_t)384 * 192},
      /* middle.1 AttentionBlock */
      {"decoder.middle.1.to_qkv.weight", "vae.decoder.middle.1.to_qkv.weight",
       (size_t)(3 * 384) * 384},
      {"decoder.middle.1.proj.weight", "vae.decoder.middle.1.proj.weight",
       (size_t)384 * 384},
      /* 384-ch: middle + early ups */
      {"decoder.middle.0.residual.2.weight",
       "vae.decoder.middle.0.residual.2.weight", (size_t)384 * 384 * 3 * 3 * 3},
      {"decoder.middle.0.residual.6.weight",
       "vae.decoder.middle.0.residual.6.weight", (size_t)384 * 384 * 3 * 3 * 3},
      {"decoder.middle.2.residual.2.weight",
       "vae.decoder.middle.2.residual.2.weight", (size_t)384 * 384 * 3 * 3 * 3},
      {"decoder.middle.2.residual.6.weight",
       "vae.decoder.middle.2.residual.6.weight", (size_t)384 * 384 * 3 * 3 * 3},
      {"decoder.upsamples.0.residual.2.weight",
       "vae.decoder.upsamples.0.residual.2.weight",
       (size_t)384 * 384 * 3 * 3 * 3},
      {"decoder.upsamples.0.residual.6.weight",
       "vae.decoder.upsamples.0.residual.6.weight",
       (size_t)384 * 384 * 3 * 3 * 3},
      {"decoder.upsamples.1.residual.2.weight",
       "vae.decoder.upsamples.1.residual.2.weight",
       (size_t)384 * 384 * 3 * 3 * 3},
      {"decoder.upsamples.1.residual.6.weight",
       "vae.decoder.upsamples.1.residual.6.weight",
       (size_t)384 * 384 * 3 * 3 * 3},
      {"decoder.upsamples.2.residual.2.weight",
       "vae.decoder.upsamples.2.residual.2.weight",
       (size_t)384 * 384 * 3 * 3 * 3},
      {"decoder.upsamples.2.residual.6.weight",
       "vae.decoder.upsamples.2.residual.6.weight",
       (size_t)384 * 384 * 3 * 3 * 3},
      {"decoder.upsamples.5.residual.2.weight",
       "vae.decoder.upsamples.5.residual.2.weight",
       (size_t)384 * 384 * 3 * 3 * 3},
      {"decoder.upsamples.5.residual.6.weight",
       "vae.decoder.upsamples.5.residual.6.weight",
       (size_t)384 * 384 * 3 * 3 * 3},
      {"decoder.upsamples.6.residual.2.weight",
       "vae.decoder.upsamples.6.residual.2.weight",
       (size_t)384 * 384 * 3 * 3 * 3},
      {"decoder.upsamples.6.residual.6.weight",
       "vae.decoder.upsamples.6.residual.6.weight",
       (size_t)384 * 384 * 3 * 3 * 3},
      /* 192-ch: up8–10 (F1006/F1007) */
      {"decoder.upsamples.8.residual.2.weight",
       "vae.decoder.upsamples.8.residual.2.weight",
       (size_t)192 * 192 * 3 * 3 * 3},
      {"decoder.upsamples.8.residual.6.weight",
       "vae.decoder.upsamples.8.residual.6.weight",
       (size_t)192 * 192 * 3 * 3 * 3},
      {"decoder.upsamples.9.residual.2.weight",
       "vae.decoder.upsamples.9.residual.2.weight",
       (size_t)192 * 192 * 3 * 3 * 3},
      {"decoder.upsamples.9.residual.6.weight",
       "vae.decoder.upsamples.9.residual.6.weight",
       (size_t)192 * 192 * 3 * 3 * 3},
      {"decoder.upsamples.10.residual.2.weight",
       "vae.decoder.upsamples.10.residual.2.weight",
       (size_t)192 * 192 * 3 * 3 * 3},
      {"decoder.upsamples.10.residual.6.weight",
       "vae.decoder.upsamples.10.residual.6.weight",
       (size_t)192 * 192 * 3 * 3 * 3},
      /* 96-ch: u12–14 (F1005) */
      {"decoder.upsamples.12.residual.2.weight",
       "vae.decoder.upsamples.12.residual.2.weight",
       (size_t)96 * 96 * 3 * 3 * 3},
      {"decoder.upsamples.12.residual.6.weight",
       "vae.decoder.upsamples.12.residual.6.weight",
       (size_t)96 * 96 * 3 * 3 * 3},
      {"decoder.upsamples.13.residual.2.weight",
       "vae.decoder.upsamples.13.residual.2.weight",
       (size_t)96 * 96 * 3 * 3 * 3},
      {"decoder.upsamples.13.residual.6.weight",
       "vae.decoder.upsamples.13.residual.6.weight",
       (size_t)96 * 96 * 3 * 3 * 3},
      {"decoder.upsamples.14.residual.2.weight",
       "vae.decoder.upsamples.14.residual.2.weight",
       (size_t)96 * 96 * 3 * 3 * 3},
      {"decoder.upsamples.14.residual.6.weight",
       "vae.decoder.upsamples.14.residual.6.weight",
       (size_t)96 * 96 * 3 * 3 * 3},
  };
  int nkeys = 0;
  for (size_t i = 0; i < sizeof(items) / sizeof(items[0]); i++) {
    if (!wan_gguf_has(ctx, items[i].tensor))
      continue;
    if (vae_bank_put_named(ctx, items[i].bank, items[i].tensor, items[i].ne) !=
        0) {
      fprintf(stderr, "wan-c: VAE BANK_PUT fail %s\n", items[i].bank);
      return -1;
    }
    nkeys++;
  }
  if (nkeys < 1)
    return -1;
  ctx->vae_headt_ready = 1;
  fprintf(stderr, "wan-c: VAE tip BANK_PUT OK keys=%d (middle→HEADT)\n", nkeys);
  return 0;
}

/* Map tip residual prefix → sticky BANK keys. Longer tags first (.10 before .1). */
static int vae_tip_residual_bank(const char *prefix, const char **bank2,
                                 const char **bank6) {
  *bank2 = *bank6 = NULL;
  if (!prefix)
    return 0;
  static const struct {
    const char *tag;
    const char *b2;
    const char *b6;
    int id;
  } tab[] = {
      {"middle.0", "decoder.middle.0.residual.2.weight",
       "decoder.middle.0.residual.6.weight", 100},
      {"middle.2", "decoder.middle.2.residual.2.weight",
       "decoder.middle.2.residual.6.weight", 102},
      {"upsamples.14", "decoder.upsamples.14.residual.2.weight",
       "decoder.upsamples.14.residual.6.weight", 14},
      {"upsamples.13", "decoder.upsamples.13.residual.2.weight",
       "decoder.upsamples.13.residual.6.weight", 13},
      {"upsamples.12", "decoder.upsamples.12.residual.2.weight",
       "decoder.upsamples.12.residual.6.weight", 12},
      {"upsamples.10", "decoder.upsamples.10.residual.2.weight",
       "decoder.upsamples.10.residual.6.weight", 10},
      {"upsamples.9", "decoder.upsamples.9.residual.2.weight",
       "decoder.upsamples.9.residual.6.weight", 9},
      {"upsamples.8", "decoder.upsamples.8.residual.2.weight",
       "decoder.upsamples.8.residual.6.weight", 8},
      {"upsamples.6", "decoder.upsamples.6.residual.2.weight",
       "decoder.upsamples.6.residual.6.weight", 6},
      {"upsamples.5", "decoder.upsamples.5.residual.2.weight",
       "decoder.upsamples.5.residual.6.weight", 5},
      {"upsamples.2", "decoder.upsamples.2.residual.2.weight",
       "decoder.upsamples.2.residual.6.weight", 2},
      {"upsamples.1", "decoder.upsamples.1.residual.2.weight",
       "decoder.upsamples.1.residual.6.weight", 1},
      {"upsamples.0", "decoder.upsamples.0.residual.2.weight",
       "decoder.upsamples.0.residual.6.weight", 0},
  };
  for (size_t i = 0; i < sizeof(tab) / sizeof(tab[0]); i++) {
    if (strstr(prefix, tab[i].tag)) {
      *bank2 = tab[i].b2;
      *bank6 = tab[i].b6;
      return tab[i].id;
    }
  }
  return 0;
}

/* Build causal time cat (feat_cache) + left zero-pad; spatial pad via CONV3D. */
static float *vae_build_causal_work(const float *sil, int Cin, int lt, int lh,
                                    int lw, vae_ccache *cache,
                                    const char *cache_key, int *Twork_out) {
  if (!sil || !Twork_out || lt < 1 || lh < 1 || lw < 1)
    return NULL;
  vae_cache_ent *ent = NULL;
  if (cache && cache_key) {
    for (int i = 0; i < cache->n; i++) {
      if (strcmp(cache->e[i].key, cache_key) == 0) {
        ent = &cache->e[i];
        break;
      }
    }
  }
  int have_prev = ent && ent->data && ent->C == Cin && ent->H == lh &&
                  ent->W == lw && ent->T > 0;
  int prev_t = have_prev ? ent->T : 0;
  if (prev_t > VAE_CACHE_T)
    prev_t = VAE_CACHE_T;
  int cat_t = prev_t;
  int pad_left = 2 - cat_t;
  if (pad_left < 0)
    pad_left = 0;
  int Twork = pad_left + cat_t + lt;
  float *work =
      calloc((size_t)Cin * (size_t)Twork * (size_t)lh * (size_t)lw, sizeof(float));
  if (!work)
    return NULL;
  for (int ti = 0; ti < cat_t; ti++)
    for (int c = 0; c < Cin; c++)
      for (int h = 0; h < lh; h++)
        for (int w = 0; w < lw; w++)
          work[((((size_t)c * Twork + (pad_left + ti)) * lh + h) * lw + w)] =
              ent->data[((((size_t)c * ent->T + ti) * lh + h) * lw + w)];
  for (int ti = 0; ti < lt; ti++)
    for (int c = 0; c < Cin; c++)
      for (int h = 0; h < lh; h++)
        for (int w = 0; w < lw; w++)
          work[((((size_t)c * Twork + (pad_left + cat_t + ti)) * lh + h) * lw +
                w)] = sil[((((size_t)c * lt + ti) * lh + h) * lw + w)];
  *Twork_out = Twork;
  return work;
}

/* Sticky broker feat_cache name from Wan residual cache_key (short + unique). */
static void vae_broker_cache_name(char *dst, size_t n, const char *cache_key) {
  if (!dst || n < 12)
    return;
  unsigned h = 2166136261u;
  if (cache_key) {
    for (const unsigned char *p = (const unsigned char *)cache_key; *p; p++) {
      h ^= (unsigned)*p;
      h *= 16777619u;
    }
  }
  snprintf(dst, n, "vae_fc_%08x", h);
}

/*
 * Broker RMS→SILU→causal CONV3D.
 * Cold+warm: F1012 CAUSAL_PAD3D mid/out sticky feat_cache (no sil GET/PUT).
 * Legacy sil-shuttle warm kept behind WAN_VAE_WARM_HEADT_SHUTTLE=1.
 */
static int vae_broker_headt(wan_ctx *ctx, float *out, const float *in,
                            const float *gamma, const float *bias, int Cin,
                            int Cout, int lt, int lh, int lw,
                            const char *w_bank_key, const char *w_tensor,
                            vae_ccache *cache, const char *cache_key,
                            int update_cache) {
  if (!vae_headt_enabled(ctx) || !out || !in || !gamma || !w_tensor)
    return -1;
  int have_cache =
      cache && cache_key && vae_ccache_has_data(cache, cache_key);
  int warm = have_cache && vae_warm_broker_enabled();
  /* Broker-side cache path: default (F1012). Shuttle = old GET/PUT warm. */
  int use_broker_cache = !vae_env_truthy("WAN_VAE_WARM_HEADT_SHUTTLE");
  size_t n0 = (size_t)Cin * (size_t)lt * (size_t)lh * (size_t)lw;
  size_t n1 = (size_t)Cout * (size_t)lt * (size_t)lh * (size_t)lw;
  size_t np =
      (size_t)Cin * (size_t)(lt + 2) * (size_t)(lh + 2) * (size_t)(lw + 2);
  size_t nc =
      (size_t)Cin * (size_t)VAE_CACHE_T * (size_t)lh * (size_t)lw * 4;
  const char *bx = "vae_hx";
  const char *bg = "vae_hg";
  const char *bb = "vae_hb";
  const char *brms = "vae_hrms";
  const char *bsil = "vae_hsil";
  const char *bpad = "vae_hpad";
  const char *by = "vae_hy";
  const char *bw = "vae_hW";
  char bcache[96];
  bcache[0] = 0;
  if (use_broker_cache && cache_key)
    vae_broker_cache_name(bcache, sizeof(bcache), cache_key);
  int used_bank = 0;
  if (w_bank_key && w_bank_key[0] && ctx->vae_headt_ready) {
    if (uma_buf_pool_bank_bind(ctx->bufs, w_bank_key, bw) == 0)
      used_bank = 1;
  }
  if (!used_bank) {
    size_t nw = 0;
    const float *W = wan_borrow_tensor_f32(ctx, w_tensor, &nw);
    size_t expect = (size_t)Cout * (size_t)Cin * 3 * 3 * 3;
    if (!W || nw != expect)
      return -1;
    if (uma_buf_pool_ensure_put(ctx->bufs, bw, W, nw * sizeof(float)) != 0)
      return -1;
  }
  if (uma_buf_pool_alloc(ctx->bufs, bx, n0 * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, brms, n0 * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bsil, n0 * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, by, n1 * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bpad, np * 4) != 0 ||
      uma_buf_pool_ensure_put(ctx->bufs, bx, in, n0 * 4) != 0 ||
      uma_buf_pool_ensure_put(ctx->bufs, bg, gamma, (size_t)Cin * 4) != 0)
    return -1;
  if (bias) {
    if (uma_buf_pool_ensure_put(ctx->bufs, bb, bias, (size_t)Cout * 4) != 0)
      return -1;
  }

  double t0 = wan_profile_on() ? wan_profile_now_ms() : 0.0;
  char resp[256];
  size_t got = 0;

  /* F1012: single GRAPH with sticky mid/out (cold t=0 or warm t=ent->T). */
  if (use_broker_cache && bcache[0]) {
    int cache_t = 0;
    vae_cache_ent *ent = cache && cache_key ? vae_ccache_find(cache, cache_key)
                                           : NULL;
    if (ent && ent->data && ent->T > 0 && ent->C == Cin && ent->H == lh &&
        ent->W == lw)
      cache_t = ent->T > VAE_CACHE_T ? VAE_CACHE_T : ent->T;
    if (vae_env_truthy("WAN_VAE_NO_WARM_HEADT") && cache_t > 0)
      return -1;
    /* Seed sticky once; later calls trust broker mid/out (no sil/pad shuttle). */
    if (uma_buf_pool_alloc(ctx->bufs, bcache, nc) != 0)
      return -1;
    if (!ent || !ent->on_broker) {
      float *seed = calloc(nc / 4, sizeof(float));
      if (!seed)
        return -1;
      if (cache_t > 0 && ent && ent->data)
        memcpy(seed, ent->data,
               (size_t)Cin * (size_t)cache_t * (size_t)lh * (size_t)lw *
                   sizeof(float));
      else
        cache_t = 0;
      if (uma_buf_pool_ensure_put(ctx->bufs, bcache, seed, nc) != 0) {
        free(seed);
        return -1;
      }
      free(seed);
    }
    if (wan_graph_vae_headt_cache(ctx, bx, by, bw, bg, bias ? bb : NULL, brms,
                                  bsil, bpad, bcache, cache_t, Cin, Cout, lt,
                                  lh, lw) != 0)
      return -1;
    if (uma_client_buf_get(ctx->uma, by, out, n1 * 4, &got, resp,
                           sizeof(resp)) != 0 ||
        got != n1 * 4)
      return -1;
    if (update_cache && cache && cache_key) {
      /* Bookkeep local T/on_broker only — sticky mid/out holds frames. */
      vae_cache_ent *e = vae_ccache_find(cache, cache_key);
      if (e) {
        int next_t = lt < VAE_CACHE_T ? lt : VAE_CACHE_T;
        if (next_t < VAE_CACHE_T && cache_t > 0) {
          next_t = next_t + 1;
          if (next_t > VAE_CACHE_T)
            next_t = VAE_CACHE_T;
        }
        if (!e->data)
          e->data = calloc((size_t)Cin * (size_t)VAE_CACHE_T * (size_t)lh *
                               (size_t)lw,
                           sizeof(float));
        e->C = Cin;
        e->T = next_t;
        e->H = lh;
        e->W = lw;
        e->on_broker = 1;
      }
    }
    if (wan_profile_on())
      wan_profile_add_ms(cache_t > 0 ? "vae_headt_warm" : "vae_headt_cold",
                         wan_profile_now_ms() - t0);
    static int logged_fc;
    if (!logged_fc) {
      fprintf(stderr,
              "wan-c: VAE HEADT broker feat_cache (F1012 mid/out; opt-in "
              "WAN_VAE_WARM_HEADT=1)\n");
      logged_fc = 1;
    }
    return 0;
  }

  if (!warm) {
    if (wan_graph_vae_headt(ctx, bx, by, bw, bg, bias ? bb : NULL, brms, bsil,
                            bpad, Cin, Cout, lt, lh, lw) != 0)
      return -1;
    if (uma_client_buf_get(ctx->uma, by, out, n1 * 4, &got, resp, sizeof(resp)) !=
            0 ||
        got != n1 * 4)
      return -1;
    if (update_cache && cache && cache_key) {
      float *sil = calloc(n0, sizeof(float));
      size_t gs = 0;
      if (sil &&
          uma_client_buf_get(ctx->uma, bsil, sil, n0 * 4, &gs, resp,
                             sizeof(resp)) == 0 &&
          gs == n0 * 4)
        vae_ccache_poke_from_in(cache, cache_key, sil, Cin, lt, lh, lw);
      free(sil);
    }
    if (wan_profile_on())
      wan_profile_add_ms("vae_headt_cold", wan_profile_now_ms() - t0);
    static int logged;
    if (!logged) {
      fprintf(stderr,
              "wan-c: VAE HEADT on broker (RMS→SILU→PAD→CONV3D; F1001)\n");
      logged = 1;
    }
    return 0;
  }

  /* Legacy warm: RMS+SILU on broker, feat_cache pad on client, CONV3D. */
  {
    char nodes[512];
    int n = snprintf(nodes, sizeof(nodes),
                     "CHANNEL_RMS@CPU! x=%s y=%s w=%s kind=1_%d_%d_%d_%d ; "
                     "SILU@CPU! x=%s y=%s D=%d ; MARK@CPU?",
                     bx, brms, bg, Cin, lt, lh, lw, brms, bsil, (int)n0);
    if (n < 0 || (size_t)n >= sizeof(nodes) ||
        wan_submit_graph(ctx->uma, nodes) != 0)
      return -1;
  }
  float *sil = calloc(n0, sizeof(float));
  if (!sil ||
      uma_client_buf_get(ctx->uma, bsil, sil, n0 * 4, &got, resp, sizeof(resp)) !=
          0 ||
      got != n0 * 4) {
    free(sil);
    return -1;
  }
  int Twork = 0;
  float *work =
      vae_build_causal_work(sil, Cin, lt, lh, lw, cache, cache_key, &Twork);
  if (!work || Twork < 3) {
    free(sil);
    free(work);
    return -1;
  }
  size_t nwork = (size_t)Cin * (size_t)Twork * (size_t)lh * (size_t)lw;
  if (uma_buf_pool_alloc(ctx->bufs, bpad, nwork * 4) != 0 ||
      uma_buf_pool_ensure_put(ctx->bufs, bpad, work, nwork * 4) != 0) {
    free(sil);
    free(work);
    return -1;
  }
  free(work);
  {
    char nodes[640];
    int n;
    if (bias)
      n = snprintf(nodes, sizeof(nodes),
                   "CONV3D@CPU! x=%s y=%s w=%s gate=%s "
                   "kind=1_%d_%d_%d_%d_%d_3_3_3_1_1_1_0_1_1 ; MARK@CPU?",
                   bpad, by, bw, bb, Cin, Twork, lh, lw, Cout);
    else
      n = snprintf(nodes, sizeof(nodes),
                   "CONV3D@CPU! x=%s y=%s w=%s "
                   "kind=1_%d_%d_%d_%d_%d_3_3_3_1_1_1_0_1_1 ; MARK@CPU?",
                   bpad, by, bw, Cin, Twork, lh, lw, Cout);
    if (n < 0 || (size_t)n >= sizeof(nodes) ||
        wan_submit_graph(ctx->uma, nodes) != 0) {
      free(sil);
      return -1;
    }
  }
  got = 0;
  if (uma_client_buf_get(ctx->uma, by, out, n1 * 4, &got, resp, sizeof(resp)) !=
          0 ||
      got != n1 * 4) {
    free(sil);
    return -1;
  }
  if (update_cache && cache && cache_key)
    vae_ccache_poke_from_in(cache, cache_key, sil, Cin, lt, lh, lw);
  free(sil);
  if (wan_profile_on())
    wan_profile_add_ms("vae_headt_warm", wan_profile_now_ms() - t0);
  static int logged_w;
  if (!logged_w) {
    fprintf(stderr,
            "wan-c: VAE HEADT warm on broker (feat_cache pad + CONV3D)\n");
    logged_w = 1;
  }
  return 0;
}

/* Causal Conv3d k=3: left-pad 2 (or cat feat_cache), spatial pad 1. */
static int vae_causal_conv3d_k3(float *out, const float *in, const float *W,
                                const float *bias, int Cin, int Cout, int lt,
                                int lh, int lw, vae_ccache *cache,
                                const char *key) {
  if (!out || !in || !W || lt < 1 || lh < 1 || lw < 1)
    return -1;
  double st0 = VAE_STAGE_T0();
  vae_cache_ent *ent = vae_ccache_find(cache, key);
  int have_prev = ent && ent->data && ent->C == Cin && ent->H == lh &&
                  ent->W == lw && ent->T > 0;
  int prev_t = have_prev ? ent->T : 0;
  if (prev_t > VAE_CACHE_T)
    prev_t = VAE_CACHE_T;

  /* Build next-cache from current input (Python ResidualBlock). */
  int store_t = lt < VAE_CACHE_T ? lt : VAE_CACHE_T;
  float *next = NULL;
  int next_t = store_t;
  if (store_t < VAE_CACHE_T && have_prev) {
    next_t = store_t + 1;
    if (next_t > VAE_CACHE_T)
      next_t = VAE_CACHE_T;
  }
  next = calloc((size_t)Cin * (size_t)next_t * (size_t)lh * (size_t)lw,
                sizeof(float));
  if (!next)
    return -1;
  int dst0 = 0;
  if (store_t < VAE_CACHE_T && have_prev) {
    /* prepend last frame of previous cache */
    for (int c = 0; c < Cin; c++)
      for (int h = 0; h < lh; h++)
        for (int w = 0; w < lw; w++)
          next[((((size_t)c * next_t + 0) * lh + h) * lw + w)] =
              ent->data[((((size_t)c * ent->T + (ent->T - 1)) * lh + h) * lw +
                         w)];
    dst0 = 1;
  }
  int src0 = lt - store_t;
  for (int ti = 0; ti < store_t; ti++)
    for (int c = 0; c < Cin; c++)
      for (int h = 0; h < lh; h++)
        for (int w = 0; w < lw; w++)
          next[((((size_t)c * next_t + (dst0 + ti)) * lh + h) * lw + w)] =
              in[((((size_t)c * lt + (src0 + ti)) * lh + h) * lw + w)];

  /* Work buffer: optional prev frames + input, then zero left-pad to total left=2 */
  int cat_t = prev_t;
  int pad_left = 2 - cat_t;
  if (pad_left < 0)
    pad_left = 0;
  int Twork = pad_left + cat_t + lt;
  float *work =
      calloc((size_t)Cin * (size_t)Twork * (size_t)lh * (size_t)lw, sizeof(float));
  float *mid = calloc((size_t)Cout * (size_t)lt * (size_t)lh * (size_t)lw,
                      sizeof(float));
  if (!work || !mid) {
    free(next);
    free(work);
    free(mid);
    return -1;
  }
  /* zeros already; copy cat then in */
  for (int ti = 0; ti < cat_t; ti++)
    for (int c = 0; c < Cin; c++)
      for (int h = 0; h < lh; h++)
        for (int w = 0; w < lw; w++)
          work[((((size_t)c * Twork + (pad_left + ti)) * lh + h) * lw + w)] =
              ent->data[((((size_t)c * ent->T + ti) * lh + h) * lw + w)];
  for (int ti = 0; ti < lt; ti++)
    for (int c = 0; c < Cin; c++)
      for (int h = 0; h < lh; h++)
        for (int w = 0; w < lw; w++)
          work[((((size_t)c * Twork + (pad_left + cat_t + ti)) * lh + h) * lw +
                w)] = in[((((size_t)c * lt + ti) * lh + h) * lw + w)];

  /* Conv pad_d=0, pad_h=1, pad_w=1 → Dout=Twork-2, need == lt when Twork=lt+2 */
  float *tmp = calloc((size_t)Cout * (size_t)(Twork - 2) * (size_t)lh *
                          (size_t)lw,
                      sizeof(float));
  if (!tmp) {
    free(next);
    free(work);
    free(mid);
    return -1;
  }
  uma_wan_conv3d_f32(tmp, work, W, bias, 1, Cin, Twork, lh, lw, Cout, 3, 3, 3, 1,
                     1, 1, 0, 1, 1);
  int Dout = Twork - 2;
  /* Take last lt frames if Dout > lt (shouldn't), else copy Dout */
  int t0 = Dout - lt;
  if (t0 < 0)
    t0 = 0;
  for (int ti = 0; ti < lt; ti++)
    for (int c = 0; c < Cout; c++)
      for (int h = 0; h < lh; h++)
        for (int w = 0; w < lw; w++)
          out[((((size_t)c * lt + ti) * lh + h) * lw + w)] =
              tmp[((((size_t)c * Dout + (t0 + ti)) * lh + h) * lw + w)];

  if (ent) {
    free(ent->data);
    ent->data = next;
    ent->C = Cin;
    ent->T = next_t;
    ent->H = lh;
    ent->W = lw;
    next = NULL;
  }
  free(next);
  free(work);
  free(mid);
  free(tmp);
  VAE_STAGE_ADD("vae_conv_host", st0);
  return 0;
}

/* Brick 7: same-C residual — dual HEADT + resid ADD on broker (one GRAPH).
 * Cold path: no per-key F1012 sticky caches (avoids BUF table_full); poke host
 * feat_cache from SILU outs after the GRAPH. Warm F1012 stays opt-in. */
static int vae_broker_resblock_fuse(wan_ctx *ctx, float *x, int C, int lt,
                                    int lh, int lw, const float *g0,
                                    const float *b2, const float *g3,
                                    const float *b6, const char *bank2,
                                    const char *n2w, const char *bank6,
                                    const char *n6w, vae_ccache *cache,
                                    const char *k2, const char *k6) {
  if (!vae_headt_enabled(ctx) || !x || !g0 || !g3 || !n2w || !n6w || C < 1 ||
      lt < 1 || lh < 1 || lw < 1)
    return -1;
  if (vae_env_truthy("WAN_VAE_NO_RESID_FUSE"))
    return -1;
  int have2 = cache && k2 && vae_ccache_has_data(cache, k2);
  int have6 = cache && k6 && vae_ccache_has_data(cache, k6);
  /* Dual-fuse is cold-only; warm/reuse → single F1012 HEADT. */
  if (have2 || have6)
    return -1;

  size_t n0 = (size_t)C * (size_t)lt * (size_t)lh * (size_t)lw;
  size_t np =
      (size_t)C * (size_t)(lt + 2) * (size_t)(lh + 2) * (size_t)(lw + 2);
  size_t nc =
      (size_t)C * (size_t)VAE_CACHE_T * (size_t)lh * (size_t)lw * 4;
  size_t expect_w = (size_t)C * (size_t)C * 3 * 3 * 3;
  const char *bx = "vae_rx";
  const char *bmid = "vae_rmid";
  const char *by = "vae_ry";
  const char *bg1 = "vae_rg1";
  const char *bg2 = "vae_rg2";
  const char *bb1 = "vae_rb1";
  const char *bb2 = "vae_rb2";
  const char *bw1 = "vae_rW1";
  const char *bw2 = "vae_rW2";
  const char *brms1 = "vae_rrms1";
  const char *brms2 = "vae_rrms2";
  const char *bsil1 = "vae_rsil1";
  const char *bsil2 = "vae_rsil2";
  const char *bpad1 = "vae_rpad1";
  const char *bpad2 = "vae_rpad2";
  /* Fixed F1012 slots (reuse) — avoid per-key table_full. Cold t=0. */
  int use_fc = !vae_env_truthy("WAN_VAE_WARM_HEADT_SHUTTLE");
  const char *bc1 = use_fc ? "vae_rfc1" : NULL;
  const char *bc2 = use_fc ? "vae_rfc2" : NULL;

  int used1 = 0, used2 = 0;
  if (bank2 && bank2[0] && ctx->vae_headt_ready &&
      uma_buf_pool_bank_bind(ctx->bufs, bank2, bw1) == 0)
    used1 = 1;
  if (!used1) {
    size_t nw = 0;
    const float *W = wan_borrow_tensor_f32(ctx, n2w, &nw);
    if (!W || nw != expect_w ||
        uma_buf_pool_ensure_put(ctx->bufs, bw1, W, nw * sizeof(float)) != 0)
      return -1;
  }
  if (bank6 && bank6[0] && ctx->vae_headt_ready &&
      uma_buf_pool_bank_bind(ctx->bufs, bank6, bw2) == 0)
    used2 = 1;
  if (!used2) {
    size_t nw = 0;
    const float *W = wan_borrow_tensor_f32(ctx, n6w, &nw);
    if (!W || nw != expect_w ||
        uma_buf_pool_ensure_put(ctx->bufs, bw2, W, nw * sizeof(float)) != 0)
      return -1;
  }

  if (uma_buf_pool_alloc(ctx->bufs, bx, n0 * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bmid, n0 * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, by, n0 * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, brms1, n0 * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, brms2, n0 * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bsil1, n0 * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bsil2, n0 * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bpad1, np * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bpad2, np * 4) != 0 ||
      uma_buf_pool_ensure_put(ctx->bufs, bx, x, n0 * 4) != 0 ||
      uma_buf_pool_ensure_put(ctx->bufs, bg1, g0, (size_t)C * 4) != 0 ||
      uma_buf_pool_ensure_put(ctx->bufs, bg2, g3, (size_t)C * 4) != 0)
    return -1;
  if (b2 &&
      uma_buf_pool_ensure_put(ctx->bufs, bb1, b2, (size_t)C * 4) != 0)
    return -1;
  if (b6 &&
      uma_buf_pool_ensure_put(ctx->bufs, bb2, b6, (size_t)C * 4) != 0)
    return -1;
  if (use_fc) {
    float *z1 = calloc(nc / 4, sizeof(float));
    float *z2 = calloc(nc / 4, sizeof(float));
    if (!z1 || !z2 || uma_buf_pool_alloc(ctx->bufs, bc1, nc) != 0 ||
        uma_buf_pool_alloc(ctx->bufs, bc2, nc) != 0 ||
        uma_buf_pool_ensure_put(ctx->bufs, bc1, z1, nc) != 0 ||
        uma_buf_pool_ensure_put(ctx->bufs, bc2, z2, nc) != 0) {
      free(z1);
      free(z2);
      return -1;
    }
    free(z1);
    free(z2);
  }

  double t0 = wan_profile_on() ? wan_profile_now_ms() : 0.0;
  double st0 = VAE_STAGE_T0();
  /* Cold t=0 F1012 (fixed slots) or plain pad; host resid ADD after GET. */
  if (wan_graph_vae_resblock_dual_headt(
          ctx, bx, bmid, by, bw1, bg1, b2 ? bb1 : NULL, brms1, bsil1, bpad1,
          bc1, 0, bw2, bg2, b6 ? bb2 : NULL, brms2, bsil2, bpad2, bc2, 0, C, lt,
          lh, lw, 0) != 0)
    return -1;

  char resp[256];
  size_t got = 0;
  float *y = calloc(n0, sizeof(float));
  if (!y ||
      uma_client_buf_get(ctx->uma, by, y, n0 * 4, &got, resp, sizeof(resp)) !=
          0 ||
      got != n0 * 4) {
    free(y);
    return -1;
  }
  for (size_t i = 0; i < n0; i++)
    x[i] += y[i];
  free(y);

  /* Match single cold F1012 bookkeeping (on_broker + T; sticky holds frames).
   * Do not sil-poke — that diverges from F1012 host-warm later slices. */
  if (cache && k2) {
    vae_cache_ent *e = vae_ccache_find(cache, k2);
    if (e) {
      int next_t = lt < VAE_CACHE_T ? lt : VAE_CACHE_T;
      if (!e->data)
        e->data = calloc((size_t)C * (size_t)VAE_CACHE_T * (size_t)lh *
                             (size_t)lw,
                         sizeof(float));
      e->C = C;
      e->T = next_t;
      e->H = lh;
      e->W = lw;
      e->on_broker = use_fc ? 1 : 0;
    }
  }
  if (cache && k6) {
    vae_cache_ent *e = vae_ccache_find(cache, k6);
    if (e) {
      int next_t = lt < VAE_CACHE_T ? lt : VAE_CACHE_T;
      if (!e->data)
        e->data = calloc((size_t)C * (size_t)VAE_CACHE_T * (size_t)lh *
                             (size_t)lw,
                         sizeof(float));
      e->C = C;
      e->T = next_t;
      e->H = lh;
      e->W = lw;
      e->on_broker = use_fc ? 1 : 0;
    }
  }

  if (wan_profile_on())
    wan_profile_add_ms("vae_headt_cold", wan_profile_now_ms() - t0);
  VAE_STAGE_ADD("vae_resid_fuse", st0);
  static int logged;
  if (!logged) {
    fprintf(stderr,
            "wan-c: VAE residual dual-HEADT fused (cold; "
            "WAN_VAE_NO_RESID_FUSE=1 rollback)\n");
    logged = 1;
  }
  return 0;
}

/* Wan ResidualBlock with causal cache. Same Cin==Cout. */
static int vae_resblock_same(wan_ctx *ctx, float *x, int C, int lt, int lh,
                             int lw, const char *prefix, vae_ccache *cache) {
  char n0[160], n2w[160], n2b[160], n3[160], n6w[160], n6b[160], k2[160],
      k6[160];
  snprintf(n0, sizeof(n0), "%s.residual.0.gamma", prefix);
  snprintf(n2w, sizeof(n2w), "%s.residual.2.weight", prefix);
  snprintf(n2b, sizeof(n2b), "%s.residual.2.bias", prefix);
  snprintf(n3, sizeof(n3), "%s.residual.3.gamma", prefix);
  snprintf(n6w, sizeof(n6w), "%s.residual.6.weight", prefix);
  snprintf(n6b, sizeof(n6b), "%s.residual.6.bias", prefix);
  snprintf(k2, sizeof(k2), "%s.residual.2", prefix);
  snprintf(k6, sizeof(k6), "%s.residual.6", prefix);
  if (!wan_gguf_has(ctx, n2w) || !wan_gguf_has(ctx, n6w))
    return -1;

  size_t n = (size_t)C * (size_t)lt * (size_t)lh * (size_t)lw;
  size_t ng0 = 0, nw2 = 0, nb2 = 0, ng3 = 0, nw6 = 0, nb6 = 0;
  float *g0 = wan_load_tensor_f32(ctx, n0, &ng0);
  float *W2 = wan_load_tensor_f32(ctx, n2w, &nw2);
  float *b2 = wan_load_tensor_f32(ctx, n2b, &nb2);
  float *g3 = wan_load_tensor_f32(ctx, n3, &ng3);
  float *W6 = wan_load_tensor_f32(ctx, n6w, &nw6);
  float *b6 = wan_load_tensor_f32(ctx, n6b, &nb6);
  float *h = calloc(n, sizeof(float));
  float *y = calloc(n, sizeof(float));
  int ok = 0;
  size_t expect_w = (size_t)C * (size_t)C * 3 * 3 * 3;
  if (W2 && W6 && h && y && nw2 == expect_w && nw6 == expect_w) {
    /* Broker residual: cold always; warm needs WAN_VAE_WARM_HEADT=1
     * (F1012 mid/out). Brick 8 default-on miss — keep opt-in. */
    int try_broker = vae_headt_enabled(ctx) && g0 && g3 && ng0 == (size_t)C &&
                     ng3 == (size_t)C &&
                     (vae_warm_broker_enabled() ||
                      (!vae_ccache_has_data(cache, k2) &&
                       !vae_ccache_has_data(cache, k6)));
    if (try_broker) {
      const char *bank2 = NULL;
      const char *bank6 = NULL;
      if (ctx->vae_headt_ready)
        (void)vae_tip_residual_bank(prefix, &bank2, &bank6);
      /* Brick 7: dual HEADT + resid ADD — one GRAPH / one GET. */
      if (vae_broker_resblock_fuse(
              ctx, x, C, lt, lh, lw, g0, (b2 && nb2 == (size_t)C) ? b2 : NULL,
              g3, (b6 && nb6 == (size_t)C) ? b6 : NULL, bank2, n2w, bank6, n6w,
              cache, k2, k6) == 0) {
        ok = 1;
      } else {
      memcpy(h, x, n * sizeof(float));
      int rc2 =
          vae_broker_headt(ctx, y, h, g0, (b2 && nb2 == (size_t)C) ? b2 : NULL,
                           C, C, lt, lh, lw, bank2, n2w, cache, k2, 1);
      if (rc2 == 0) {
        memcpy(h, y, n * sizeof(float));
        int rc6 = vae_broker_headt(
            ctx, y, h, g3, (b6 && nb6 == (size_t)C) ? b6 : NULL, C, C, lt, lh,
            lw, bank6, n6w, cache, k6, 1);
        if (rc6 == 0) {
          for (size_t i = 0; i < n; i++)
            x[i] += y[i];
          ok = 1;
          static int logged_chain;
          if (!logged_chain && bank2) {
            fprintf(stderr,
                    "wan-c: VAE tip residuals on broker "
                    "(middle→up→HEADT BANK)\n");
            logged_chain = 1;
          }
        } else {
          vae_ccache_clear_key(cache, k2);
        }
      }
      }
    }
    if (!ok) {
      memcpy(h, x, n * sizeof(float));
      rms_norm_ncdhw(h, g0, C, lt, lh, lw, 1e-6f);
      silu_inplace(h, n);
      if (vae_causal_conv3d_k3(y, h, W2, (b2 && nb2 == (size_t)C) ? b2 : NULL, C,
                               C, lt, lh, lw, cache, k2) != 0)
        goto done;
      memcpy(h, y, n * sizeof(float));
      rms_norm_ncdhw(h, g3, C, lt, lh, lw, 1e-6f);
      silu_inplace(h, n);
      if (vae_causal_conv3d_k3(y, h, W6, (b6 && nb6 == (size_t)C) ? b6 : NULL, C,
                               C, lt, lh, lw, cache, k6) != 0)
        goto done;
      for (size_t i = 0; i < n; i++)
        x[i] += y[i];
      ok = 1;
    }
  }
done:
  free(g0);
  free(W2);
  free(b2);
  free(g3);
  free(W6);
  free(b6);
  free(h);
  free(y);
  return ok ? 0 : -1;
}

static int nearest_x2_spatial(float **feat, int C, int lt, int *lh, int *lw) {
  int lh2 = (*lh) * 2, lw2 = (*lw) * 2;
  size_t n2 = (size_t)C * (size_t)lt * (size_t)lh2 * (size_t)lw2;
  float *out = calloc(n2, sizeof(float));
  if (!out)
    return -1;
  for (int c = 0; c < C; c++)
    for (int t = 0; t < lt; t++)
      for (int h = 0; h < *lh; h++)
        for (int w = 0; w < *lw; w++) {
          float v =
              (*feat)[((((size_t)c * lt + t) * (*lh) + h) * (*lw) + w)];
          for (int dh = 0; dh < 2; dh++)
            for (int dw = 0; dw < 2; dw++)
              out[((((size_t)c * lt + t) * lh2 + h * 2 + dh) * lw2 + w * 2 +
                   dw)] = v;
        }
  free(*feat);
  *feat = out;
  *lh = lh2;
  *lw = lw2;
  return 0;
}

/* Spatial Conv2d on each time slice: weight [Cout,Cin,3,3], NCDHW in/out. */
static int vae_resample2d(wan_ctx *ctx, float **feat, int *C, int lt, int lh,
                          int lw, const char *wname, const char *bname,
                          int Cout) {
  size_t nw = 0, nb = 0;
  float *W = wan_load_tensor_f32(ctx, wname, &nw);
  float *b = wan_load_tensor_f32(ctx, bname, &nb);
  int Cin = *C;
  size_t expect = (size_t)Cout * (size_t)Cin * 3 * 3;
  if (!W || nw != expect) {
    free(W);
    free(b);
    return -1;
  }
  size_t n_out = (size_t)Cout * (size_t)lt * (size_t)lh * (size_t)lw;
  float *out = calloc(n_out, sizeof(float));
  float *plane_in = calloc((size_t)Cin * (size_t)lh * (size_t)lw, sizeof(float));
  float *plane_out =
      calloc((size_t)Cout * (size_t)lh * (size_t)lw, sizeof(float));
  if (!out || !plane_in || !plane_out) {
    free(W);
    free(b);
    free(out);
    free(plane_in);
    free(plane_out);
    return -1;
  }
  size_t spat = (size_t)lh * (size_t)lw;
  for (int t = 0; t < lt; t++) {
    for (int c = 0; c < Cin; c++)
      for (size_t s = 0; s < spat; s++) {
        int h = (int)(s / (size_t)lw);
        int w = (int)(s % (size_t)lw);
        plane_in[(size_t)c * spat + s] =
            (*feat)[((((size_t)c * lt + t) * lh + h) * lw + w)];
      }
    uma_wan_conv2d_f32(plane_out, plane_in, W, (b && nb == (size_t)Cout) ? b : NULL,
                       1, Cin, lh, lw, Cout, 3, 3, 1, 1);
    for (int c = 0; c < Cout; c++)
      for (size_t s = 0; s < spat; s++) {
        int h = (int)(s / (size_t)lw);
        int w = (int)(s % (size_t)lw);
        out[((((size_t)c * lt + t) * lh + h) * lw + w)] =
            plane_out[(size_t)c * spat + s];
      }
  }
  free(*feat);
  *feat = out;
  *C = Cout;
  free(W);
  free(b);
  free(plane_in);
  free(plane_out);
  return 0;
}

/* F1001: nearest×2 + CONV2D on broker (per time plane). */
static int vae_nearest_resample_broker(wan_ctx *ctx, float **feat, int *C,
                                       int lt, int *lh, int *lw,
                                       const char *wname, const char *bname,
                                       const char *w_bank_key, int Cout) {
  if (!vae_headt_enabled(ctx) || !ctx->caps.nearest || !ctx->caps.conv2d)
    return -1;
  int Cin = *C;
  int H = *lh, W = *lw;
  int H2 = H * 2, W2 = W * 2;
  size_t nw = 0, nb = 0;
  const float *Wb = wan_borrow_tensor_f32(ctx, wname, &nw);
  float *b = wan_load_tensor_f32(ctx, bname, &nb);
  size_t expect = (size_t)Cout * (size_t)Cin * 3 * 3;
  if (!Wb || nw != expect) {
    free(b);
    return -1;
  }
  const char *bw = "vae_rsW";
  const char *bb = "vae_rsb";
  const char *blo = "vae_rslo";
  const char *bhi = "vae_rshi";
  const char *by = "vae_rsy";
  int used_bank = 0;
  if (w_bank_key && w_bank_key[0] && ctx->vae_headt_ready &&
      uma_buf_pool_bank_bind(ctx->bufs, w_bank_key, bw) == 0)
    used_bank = 1;
  if (!used_bank &&
      uma_buf_pool_ensure_put(ctx->bufs, bw, Wb, nw * sizeof(float)) != 0) {
    free(b);
    return -1;
  }
  if (b && nb == (size_t)Cout &&
      uma_buf_pool_ensure_put(ctx->bufs, bb, b, (size_t)Cout * 4) != 0) {
    free(b);
    return -1;
  }
  size_t n_lo = (size_t)Cin * (size_t)H * (size_t)W;
  size_t n_hi = (size_t)Cin * (size_t)H2 * (size_t)W2;
  size_t n_out = (size_t)Cout * (size_t)H2 * (size_t)W2;
  size_t n_vol = (size_t)Cout * (size_t)lt * (size_t)H2 * (size_t)W2;
  float *out = calloc(n_vol, sizeof(float));
  float *plane = calloc(n_lo, sizeof(float));
  float *gotp = calloc(n_out, sizeof(float));
  if (!out || !plane || !gotp) {
    free(b);
    free(out);
    free(plane);
    free(gotp);
    return -1;
  }
  if (uma_buf_pool_alloc(ctx->bufs, blo, n_lo * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bhi, n_hi * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, by, n_out * 4) != 0) {
    free(b);
    free(out);
    free(plane);
    free(gotp);
    return -1;
  }
  char resp[256];
  for (int t = 0; t < lt; t++) {
    for (int c = 0; c < Cin; c++)
      for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
          plane[(((size_t)c * H + h) * W + w)] =
              (*feat)[((((size_t)c * lt + t) * H + h) * W + w)];
    if (uma_buf_pool_ensure_put(ctx->bufs, blo, plane, n_lo * 4) != 0) {
      free(b);
      free(out);
      free(plane);
      free(gotp);
      return -1;
    }
    if (wan_graph_vae_nearest_conv2d(ctx, blo, bhi, by, bw,
                                     (b && nb == (size_t)Cout) ? bb : NULL, Cin,
                                     Cout, H, W) != 0) {
      free(b);
      free(out);
      free(plane);
      free(gotp);
      return -1;
    }
    size_t got = 0;
    if (uma_client_buf_get(ctx->uma, by, gotp, n_out * 4, &got, resp,
                           sizeof(resp)) != 0 ||
        got != n_out * 4) {
      free(b);
      free(out);
      free(plane);
      free(gotp);
      return -1;
    }
    for (int c = 0; c < Cout; c++)
      for (int h = 0; h < H2; h++)
        for (int w = 0; w < W2; w++)
          out[((((size_t)c * lt + t) * H2 + h) * W2 + w)] =
              gotp[(((size_t)c * H2 + h) * W2 + w)];
  }
  free(*feat);
  *feat = out;
  *C = Cout;
  *lh = H2;
  *lw = W2;
  free(b);
  free(plane);
  free(gotp);
  static int logged;
  if (!logged) {
    fprintf(stderr, "wan-c: VAE NEAREST→CONV2D on broker (F1001 resample)\n");
    logged = 1;
  }
  return 0;
}

static int vae_upsample_resample(wan_ctx *ctx, float **feat, int *C, int lt,
                                 int *lh, int *lw, const char *wname,
                                 const char *bname, const char *w_bank_key,
                                 int Cout) {
  double st0 = VAE_STAGE_T0();
  int rc;
  /* F1018: host nearest+BNNS CONV2D default ON (no daemon bump).
   * Rollback WAN_VAE_HOST_RESAMPLE=0 → broker NEAREST→CONV2D. */
  int force_host = 1;
  {
    const char *e = getenv("WAN_VAE_HOST_RESAMPLE");
    if (e && e[0] == '0')
      force_host = 0;
    else if (e && e[0] == '1')
      force_host = 1;
  }
  if (!force_host &&
      vae_nearest_resample_broker(ctx, feat, C, lt, lh, lw, wname, bname,
                                  w_bank_key, Cout) == 0)
    rc = 0;
  else if (nearest_x2_spatial(feat, *C, lt, lh, lw) != 0)
    rc = -1;
  else {
    static int logged_hr;
    if (!logged_hr) {
      fprintf(stderr,
              "wan-c: VAE resample on host (F1018 BNNS CONV2D; "
              "WAN_VAE_HOST_RESAMPLE=0 → broker)\n");
      logged_hr = 1;
    }
    rc = vae_resample2d(ctx, feat, C, lt, *lh, *lw, wname, bname, Cout);
  }
  if (rc == 0)
    VAE_STAGE_ADD("vae_resample", st0);
  return rc;
}

/* ResidualBlock Cin→Cout + 1×1 shortcut (upsamples.4). */
static int vae_resblock_expand(wan_ctx *ctx, float **feat, int *C, int lt,
                               int lh, int lw, int Cout, const char *prefix,
                               vae_ccache *cache) {
  int Cin = *C;
  char n0[160], n2w[160], n2b[160], n3[160], n6w[160], n6b[160], nsw[160],
      nsb[160], k2[160], k6[160];
  snprintf(n0, sizeof(n0), "%s.residual.0.gamma", prefix);
  snprintf(n2w, sizeof(n2w), "%s.residual.2.weight", prefix);
  snprintf(n2b, sizeof(n2b), "%s.residual.2.bias", prefix);
  snprintf(n3, sizeof(n3), "%s.residual.3.gamma", prefix);
  snprintf(n6w, sizeof(n6w), "%s.residual.6.weight", prefix);
  snprintf(n6b, sizeof(n6b), "%s.residual.6.bias", prefix);
  snprintf(nsw, sizeof(nsw), "%s.shortcut.weight", prefix);
  snprintf(nsb, sizeof(nsb), "%s.shortcut.bias", prefix);
  snprintf(k2, sizeof(k2), "%s.residual.2", prefix);
  snprintf(k6, sizeof(k6), "%s.residual.6", prefix);
  if (!wan_gguf_has(ctx, n2w) || !wan_gguf_has(ctx, nsw))
    return -1;

  size_t nin = (size_t)Cin * (size_t)lt * (size_t)lh * (size_t)lw;
  size_t nout = (size_t)Cout * (size_t)lt * (size_t)lh * (size_t)lw;
  size_t nw2 = 0, nb2 = 0, nw6 = 0, nb6 = 0, nws = 0, nbs = 0, ng0 = 0,
         ng3 = 0;
  float *g0 = wan_load_tensor_f32(ctx, n0, &ng0);
  float *W2 = wan_load_tensor_f32(ctx, n2w, &nw2);
  float *b2 = wan_load_tensor_f32(ctx, n2b, &nb2);
  float *g3 = wan_load_tensor_f32(ctx, n3, &ng3);
  float *W6 = wan_load_tensor_f32(ctx, n6w, &nw6);
  float *b6 = wan_load_tensor_f32(ctx, n6b, &nb6);
  float *Ws = wan_load_tensor_f32(ctx, nsw, &nws);
  float *bs = wan_load_tensor_f32(ctx, nsb, &nbs);
  float *h = calloc(nin, sizeof(float));
  float *y = calloc(nout, sizeof(float));
  float *out = calloc(nout, sizeof(float));
  float *h2 = calloc(nout, sizeof(float));
  int ok = 0;
  size_t exp2 = (size_t)Cout * (size_t)Cin * 3 * 3 * 3;
  size_t exp6 = (size_t)Cout * (size_t)Cout * 3 * 3 * 3;
  size_t exps = (size_t)Cout * (size_t)Cin;
  if (W2 && W6 && Ws && h && y && out && h2 && nw2 == exp2 && nw6 == exp6 &&
      nws == exps) {
    int try_broker =
        vae_headt_enabled(ctx) && g0 && g3 && ng0 == (size_t)Cin &&
        ng3 == (size_t)Cout && ctx->caps.conv3d && ctx->caps.residual_add &&
        (vae_warm_broker_enabled() ||
         (!vae_ccache_has_data(cache, k2) && !vae_ccache_has_data(cache, k6)));
    if (try_broker) {
      const char *bank2 = "decoder.upsamples.4.residual.2.weight";
      const char *bank6 = "decoder.upsamples.4.residual.6.weight";
      const char *banks = "decoder.upsamples.4.shortcut.weight";
      const char *bx = "vae_ex";
      const char *bw = "vae_eW";
      const char *bb = "vae_eb";
      const char *by = "vae_ey";
      const char *bout = "vae_eout";
      memcpy(h, *feat, nin * sizeof(float));
      int rc2 = vae_broker_headt(
          ctx, y, h, g0, (b2 && nb2 == (size_t)Cout) ? b2 : NULL, Cin, Cout, lt,
          lh, lw, ctx->vae_headt_ready ? bank2 : NULL, n2w, cache, k2, 1);
      if (rc2 == 0) {
        memcpy(h2, y, nout * sizeof(float));
        int rc6 = vae_broker_headt(
            ctx, y, h2, g3, (b6 && nb6 == (size_t)Cout) ? b6 : NULL, Cout, Cout,
            lt, lh, lw, ctx->vae_headt_ready ? bank6 : NULL, n6w, cache, k6, 1);
        if (rc6 == 0) {
          int used_bank = 0;
          if (ctx->vae_headt_ready &&
              uma_buf_pool_bank_bind(ctx->bufs, banks, bw) == 0)
            used_bank = 1;
          if (!used_bank &&
              uma_buf_pool_ensure_put(ctx->bufs, bw, Ws, nws * sizeof(float)) !=
                  0)
            rc6 = -1;
          if (rc6 == 0 &&
              (uma_buf_pool_alloc(ctx->bufs, bx, nin * 4) != 0 ||
               uma_buf_pool_alloc(ctx->bufs, by, nout * 4) != 0 ||
               uma_buf_pool_alloc(ctx->bufs, bout, nout * 4) != 0 ||
               uma_buf_pool_ensure_put(ctx->bufs, bx, *feat, nin * 4) != 0 ||
               uma_buf_pool_ensure_put(ctx->bufs, by, y, nout * 4) != 0))
            rc6 = -1;
          if (rc6 == 0 && bs && nbs == (size_t)Cout &&
              uma_buf_pool_ensure_put(ctx->bufs, bb, bs, (size_t)Cout * 4) != 0)
            rc6 = -1;
          if (rc6 == 0) {
            char nodes[640];
            int n;
            if (bs && nbs == (size_t)Cout)
              n = snprintf(
                  nodes, sizeof(nodes),
                  "CONV3D@CPU! x=%s y=%s w=%s gate=%s "
                  "kind=1_%d_%d_%d_%d_%d_1_1_1_1_1_1_0_0_0 ; "
                  "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@CPU?",
                  bx, bout, bw, bb, Cin, lt, lh, lw, Cout, by, bout,
                  (int)nout);
            else
              n = snprintf(
                  nodes, sizeof(nodes),
                  "CONV3D@CPU! x=%s y=%s w=%s "
                  "kind=1_%d_%d_%d_%d_%d_1_1_1_1_1_1_0_0_0 ; "
                  "RESIDUAL_ADD@GPU! x=%s y=%s D=%d ; MARK@CPU?",
                  bx, bout, bw, Cin, lt, lh, lw, Cout, by, bout, (int)nout);
            if (n < 0 || (size_t)n >= sizeof(nodes) ||
                wan_submit_graph(ctx->uma, nodes) != 0)
              rc6 = -1;
          }
          if (rc6 == 0) {
            char resp[256];
            size_t got = 0;
            if (uma_client_buf_get(ctx->uma, bout, out, nout * 4, &got, resp,
                                   sizeof(resp)) == 0 &&
                got == nout * 4) {
              free(*feat);
              *feat = out;
              out = NULL;
              *C = Cout;
              ok = 1;
              static int logged;
              if (!logged) {
                fprintf(stderr,
                        "wan-c: VAE ups.4 expand on broker "
                        "(192→384 + shortcut)\n");
                logged = 1;
              }
            } else {
              vae_ccache_clear_key(cache, k2);
              vae_ccache_clear_key(cache, k6);
            }
          } else {
            vae_ccache_clear_key(cache, k2);
            vae_ccache_clear_key(cache, k6);
          }
        } else {
          vae_ccache_clear_key(cache, k2);
        }
      }
    }
    if (!ok) {
      memcpy(h, *feat, nin * sizeof(float));
      rms_norm_ncdhw(h, g0, Cin, lt, lh, lw, 1e-6f);
      silu_inplace(h, nin);
      if (vae_causal_conv3d_k3(y, h, W2, (b2 && nb2 == (size_t)Cout) ? b2 : NULL,
                               Cin, Cout, lt, lh, lw, cache, k2) != 0)
        goto done_exp;
      memcpy(h2, y, nout * sizeof(float));
      rms_norm_ncdhw(h2, g3, Cout, lt, lh, lw, 1e-6f);
      silu_inplace(h2, nout);
      if (vae_causal_conv3d_k3(y, h2, W6, (b6 && nb6 == (size_t)Cout) ? b6 : NULL,
                               Cout, Cout, lt, lh, lw, cache, k6) != 0)
        goto done_exp;
      uma_wan_conv3d_f32(out, *feat, Ws, (bs && nbs == (size_t)Cout) ? bs : NULL,
                         1, Cin, lt, lh, lw, Cout, 1, 1, 1, 1, 1, 1, 0, 0, 0);
      for (size_t i = 0; i < nout; i++)
        out[i] += y[i];
      free(*feat);
      *feat = out;
      out = NULL;
      *C = Cout;
      ok = 1;
    }
  }
done_exp:
  free(g0);
  free(W2);
  free(b2);
  free(g3);
  free(W6);
  free(b6);
  free(Ws);
  free(bs);
  free(h);
  free(y);
  free(h2);
  free(out);
  return ok ? 0 : -1;
}

/* AttentionBlock mid: single-head spatial attn per time slice. */
static int vae_attn_mid_broker(wan_ctx *ctx, float *x, int C, int lt, int lh,
                               int lw, const float *gamma, const float *Wqkv,
                               const float *bias_qkv, const float *Wp,
                               const float *bias_p, size_t nbq, size_t nbp) {
  if (!vae_headt_enabled(ctx) || !ctx->caps.channel_rms || !ctx->caps.conv2d ||
      !ctx->caps.attn_full)
    return -1;
  size_t spat = (size_t)lh * (size_t)lw;
  size_t n_plane = (size_t)C * spat;
  size_t n_qkv = (size_t)(3 * C) * spat;
  size_t n_tok = spat * (size_t)C;
  const char *bx = "vae_ax";
  const char *brms = "vae_arms";
  const char *bg = "vae_ag";
  const char *bwq = "vae_aWq";
  const char *bwp = "vae_aWp";
  const char *bbq = "vae_abq";
  const char *bbp = "vae_abp";
  const char *bqkvb = "vae_aqkv";
  const char *bq = "vae_aq";
  const char *bk = "vae_ak";
  const char *bv = "vae_av";
  const char *bao = "vae_ao";
  const char *bpr = "vae_apr";
  const char *bin = "vae_ain";
  int used_q = 0, used_p = 0;
  if (ctx->vae_headt_ready &&
      uma_buf_pool_bank_bind(ctx->bufs, "decoder.middle.1.to_qkv.weight",
                             bwq) == 0)
    used_q = 1;
  if (ctx->vae_headt_ready &&
      uma_buf_pool_bank_bind(ctx->bufs, "decoder.middle.1.proj.weight", bwp) ==
          0)
    used_p = 1;
  if (!used_q &&
      uma_buf_pool_ensure_put(ctx->bufs, bwq, Wqkv,
                              (size_t)(3 * C) * (size_t)C * 4) != 0)
    return -1;
  if (!used_p &&
      uma_buf_pool_ensure_put(ctx->bufs, bwp, Wp, (size_t)C * (size_t)C * 4) !=
          0)
    return -1;
  {
    float *gbuf = NULL;
    const float *gput = gamma;
    if (!gput) {
      gbuf = calloc((size_t)C, sizeof(float));
      if (!gbuf)
        return -1;
      for (int c = 0; c < C; c++)
        gbuf[c] = 1.f;
      gput = gbuf;
    }
    int rc = uma_buf_pool_ensure_put(ctx->bufs, bg, gput, (size_t)C * 4);
    free(gbuf);
    if (rc != 0)
      return -1;
  }
  if (bias_qkv && nbq == (size_t)(3 * C) &&
      uma_buf_pool_ensure_put(ctx->bufs, bbq, bias_qkv, (size_t)(3 * C) * 4) !=
          0)
    return -1;
  if (bias_p && nbp == (size_t)C &&
      uma_buf_pool_ensure_put(ctx->bufs, bbp, bias_p, (size_t)C * 4) != 0)
    return -1;
  if (uma_buf_pool_alloc(ctx->bufs, bx, n_plane * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, brms, n_plane * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bqkvb, n_qkv * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bq, n_tok * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bk, n_tok * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bv, n_tok * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bao, n_tok * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bin, n_plane * 4) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bpr, n_plane * 4) != 0)
    return -1;

  float *plane = calloc(n_plane, sizeof(float));
  float *qkv = calloc(n_qkv, sizeof(float));
  float *tok = calloc(n_tok, sizeof(float));
  float *ktmp = calloc(n_tok, sizeof(float));
  float *proj = calloc(n_plane, sizeof(float));
  if (!plane || !qkv || !tok || !ktmp || !proj) {
    free(plane);
    free(qkv);
    free(tok);
    free(ktmp);
    free(proj);
    return -1;
  }
  char resp[256];
  char nodes[384];
  for (int t = 0; t < lt; t++) {
    for (int c = 0; c < C; c++)
      for (size_t s = 0; s < spat; s++) {
        int h = (int)(s / (size_t)lw);
        int w = (int)(s % (size_t)lw);
        plane[(size_t)c * spat + s] =
            x[((((size_t)c * lt + t) * lh + h) * lw + w)];
      }
    double ipc0 = VAE_STAGE_T0();
    if (uma_buf_pool_ensure_put(ctx->bufs, bx, plane, n_plane * 4) != 0)
      goto fail;
    VAE_STAGE_ADD("vae_ipc", ipc0);
    int n = snprintf(nodes, sizeof(nodes),
                     "CHANNEL_RMS@CPU! x=%s y=%s w=%s kind=1_%d_1_%d_%d ; "
                     "MARK@CPU?",
                     bx, brms, bg, C, lh, lw);
    if (n < 0 || (size_t)n >= sizeof(nodes) ||
        wan_submit_graph(ctx->uma, nodes) != 0)
      goto fail;
    if (wan_graph_conv2d(ctx, brms, bqkvb, bwq,
                         (bias_qkv && nbq == (size_t)(3 * C)) ? bbq : NULL, 1, C,
                         lh, lw, 3 * C, 1, 1, 1, 0) != 0)
      goto fail;
    size_t got = 0;
    ipc0 = VAE_STAGE_T0();
    if (uma_client_buf_get(ctx->uma, bqkvb, qkv, n_qkv * 4, &got, resp,
                           sizeof(resp)) != 0 ||
        got != n_qkv * 4)
      goto fail;
    VAE_STAGE_ADD("vae_ipc", ipc0);
    for (size_t s = 0; s < spat; s++)
      for (int c = 0; c < C; c++) {
        tok[s * (size_t)C + (size_t)c] = qkv[(size_t)c * spat + s];
        ktmp[s * (size_t)C + (size_t)c] =
            qkv[((size_t)C + (size_t)c) * spat + s];
      }
    ipc0 = VAE_STAGE_T0();
    if (uma_buf_pool_ensure_put(ctx->bufs, bq, tok, n_tok * 4) != 0 ||
        uma_buf_pool_ensure_put(ctx->bufs, bk, ktmp, n_tok * 4) != 0)
      goto fail;
    VAE_STAGE_ADD("vae_ipc", ipc0);
    for (size_t s = 0; s < spat; s++)
      for (int c = 0; c < C; c++)
        tok[s * (size_t)C + (size_t)c] =
            qkv[((size_t)(2 * C) + (size_t)c) * spat + s];
    ipc0 = VAE_STAGE_T0();
    if (uma_buf_pool_ensure_put(ctx->bufs, bv, tok, n_tok * 4) != 0)
      goto fail;
    VAE_STAGE_ADD("vae_ipc", ipc0);
    if (wan_graph_attn_full(ctx, bq, bk, bv, bao, (int)spat, (int)spat, 1, 1,
                            C) != 0)
      goto fail;
    ipc0 = VAE_STAGE_T0();
    if (uma_client_buf_get(ctx->uma, bao, tok, n_tok * 4, &got, resp,
                           sizeof(resp)) != 0 ||
        got != n_tok * 4)
      goto fail;
    VAE_STAGE_ADD("vae_ipc", ipc0);
    for (size_t s = 0; s < spat; s++)
      for (int c = 0; c < C; c++)
        plane[(size_t)c * spat + s] = tok[s * (size_t)C + (size_t)c];
    ipc0 = VAE_STAGE_T0();
    if (uma_buf_pool_ensure_put(ctx->bufs, bin, plane, n_plane * 4) != 0)
      goto fail;
    VAE_STAGE_ADD("vae_ipc", ipc0);
    if (wan_graph_conv2d(ctx, bin, bpr, bwp,
                         (bias_p && nbp == (size_t)C) ? bbp : NULL, 1, C, lh, lw,
                         C, 1, 1, 1, 0) != 0)
      goto fail;
    ipc0 = VAE_STAGE_T0();
    if (uma_client_buf_get(ctx->uma, bpr, proj, n_plane * 4, &got, resp,
                           sizeof(resp)) != 0 ||
        got != n_plane * 4)
      goto fail;
    VAE_STAGE_ADD("vae_ipc", ipc0);
    for (int c = 0; c < C; c++)
      for (size_t s = 0; s < spat; s++) {
        int h = (int)(s / (size_t)lw);
        int w = (int)(s % (size_t)lw);
        x[((((size_t)c * lt + t) * lh + h) * lw + w)] +=
            proj[(size_t)c * spat + s];
      }
  }
  free(plane);
  free(qkv);
  free(tok);
  free(ktmp);
  free(proj);
  static int logged;
  if (!logged) {
    fprintf(stderr,
            "wan-c: VAE middle.1 attn on broker "
            "(RMS→CONV2D→ATTN→proj; F0964)\n");
    logged = 1;
  }
  return 0;
fail:
  free(plane);
  free(qkv);
  free(tok);
  free(ktmp);
  free(proj);
  return -1;
}

static int vae_attn_mid(wan_ctx *ctx, float *x, int C, int lt, int lh, int lw) {
  if (!wan_gguf_has(ctx, "vae.decoder.middle.1.to_qkv.weight"))
    return -1;
  double st0 = VAE_STAGE_T0();
  size_t ng = 0, nqkv = 0, nbq = 0, nproj = 0, nbp = 0;
  float *gamma =
      wan_load_tensor_f32(ctx, "vae.decoder.middle.1.norm.gamma", &ng);
  float *Wqkv =
      wan_load_tensor_f32(ctx, "vae.decoder.middle.1.to_qkv.weight", &nqkv);
  float *bqkv =
      wan_load_tensor_f32(ctx, "vae.decoder.middle.1.to_qkv.bias", &nbq);
  float *Wp =
      wan_load_tensor_f32(ctx, "vae.decoder.middle.1.proj.weight", &nproj);
  float *bp = wan_load_tensor_f32(ctx, "vae.decoder.middle.1.proj.bias", &nbp);
  size_t spat = (size_t)lh * (size_t)lw;
  size_t expect_qkv = (size_t)(3 * C) * (size_t)C;
  size_t expect_p = (size_t)C * (size_t)C;
  if (!Wqkv || !Wp || nqkv != expect_qkv || nproj != expect_p) {
    free(gamma);
    free(Wqkv);
    free(bqkv);
    free(Wp);
    free(bp);
    return -1;
  }
  if (vae_attn_mid_broker(ctx, x, C, lt, lh, lw, gamma, Wqkv, bqkv, Wp, bp, nbq,
                          nbp) == 0) {
    free(gamma);
    free(Wqkv);
    free(bqkv);
    free(Wp);
    free(bp);
    VAE_STAGE_ADD("vae_attn", st0);
    return 0;
  }
  float *plane = calloc((size_t)C * spat, sizeof(float));
  float *qkv = calloc((size_t)(3 * C) * spat, sizeof(float));
  float *proj = calloc((size_t)C * spat, sizeof(float));
  float *attn = calloc(spat * spat, sizeof(float));
  if (!plane || !qkv || !proj || !attn) {
    free(gamma);
    free(Wqkv);
    free(bqkv);
    free(Wp);
    free(bp);
    free(plane);
    free(qkv);
    free(proj);
    free(attn);
    return -1;
  }
  float scale = 1.f / sqrtf((float)C);
  for (int t = 0; t < lt; t++) {
    for (int c = 0; c < C; c++)
      for (size_t s = 0; s < spat; s++) {
        int h = (int)(s / (size_t)lw);
        int w = (int)(s % (size_t)lw);
        plane[(size_t)c * spat + s] =
            x[((((size_t)c * lt + t) * lh + h) * lw + w)];
      }
    {
      float scale_rms = sqrtf((float)C);
      for (size_t s = 0; s < spat; s++) {
        float n2 = 0.f;
        for (int c = 0; c < C; c++) {
          float v = plane[(size_t)c * spat + s];
          n2 += v * v;
        }
        float nrm = sqrtf(n2);
        if (nrm < 1e-6f)
          nrm = 1e-6f;
        float inv = scale_rms / nrm;
        for (int c = 0; c < C; c++) {
          float g = gamma ? gamma[c] : 1.f;
          plane[(size_t)c * spat + s] *= inv * g;
        }
      }
    }
    uma_wan_conv2d_f32(qkv, plane, Wqkv, (bqkv && nbq == (size_t)(3 * C)) ? bqkv
                                                                          : NULL,
                       1, C, lh, lw, 3 * C, 1, 1, 1, 0);
    const float *q = qkv;
    const float *k = qkv + (size_t)C * spat;
    const float *v = qkv + (size_t)(2 * C) * spat;
    memset(plane, 0, (size_t)C * spat * sizeof(float));
#if WAN_VAE_ACCEL
    /* scores = scale * Q^T K  (Q,K: C×S row-major); then softmax rows; out = V·A^T */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)spat, (int)spat,
                C, scale, q, (int)spat, k, (int)spat, 0.f, attn, (int)spat);
    for (size_t i = 0; i < spat; i++) {
      float maxv = -1e30f;
      float *row = attn + i * spat;
      for (size_t j = 0; j < spat; j++)
        if (row[j] > maxv)
          maxv = row[j];
      float sum = 0.f;
      for (size_t j = 0; j < spat; j++) {
        row[j] = expf(row[j] - maxv);
        sum += row[j];
      }
      float inv = sum > 0.f ? 1.f / sum : 0.f;
      for (size_t j = 0; j < spat; j++)
        row[j] *= inv;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, C, (int)spat,
                (int)spat, 1.f, v, (int)spat, attn, (int)spat, 0.f, plane,
                (int)spat);
#else
    for (size_t i = 0; i < spat; i++) {
      float maxv = -1e30f;
      for (size_t j = 0; j < spat; j++) {
        float ssum = 0.f;
        for (int c = 0; c < C; c++)
          ssum += q[(size_t)c * spat + i] * k[(size_t)c * spat + j];
        ssum *= scale;
        attn[i * spat + j] = ssum;
        if (ssum > maxv)
          maxv = ssum;
      }
      float sum = 0.f;
      for (size_t j = 0; j < spat; j++) {
        float e = expf(attn[i * spat + j] - maxv);
        attn[i * spat + j] = e;
        sum += e;
      }
      float inv = sum > 0.f ? 1.f / sum : 0.f;
      for (size_t j = 0; j < spat; j++)
        attn[i * spat + j] *= inv;
      for (size_t j = 0; j < spat; j++) {
        float a = attn[i * spat + j];
        for (int c = 0; c < C; c++)
          plane[(size_t)c * spat + i] += a * v[(size_t)c * spat + j];
      }
    }
#endif
    uma_wan_conv2d_f32(proj, plane, Wp, (bp && nbp == (size_t)C) ? bp : NULL, 1,
                       C, lh, lw, C, 1, 1, 1, 0);
    for (int c = 0; c < C; c++)
      for (size_t s = 0; s < spat; s++) {
        int h = (int)(s / (size_t)lw);
        int w = (int)(s % (size_t)lw);
        x[((((size_t)c * lt + t) * lh + h) * lw + w)] +=
            proj[(size_t)c * spat + s];
      }
  }
  free(gamma);
  free(Wqkv);
  free(bqkv);
  free(Wp);
  free(bp);
  free(plane);
  free(qkv);
  free(proj);
  free(attn);
  VAE_STAGE_ADD("vae_attn", st0);
  return 0;
}

/*
 * Wan upsample3d time_conv + T×2 reshape with feat_cache:
 *   1st hit → mark Rep, skip (spatial upsample only)
 *   2nd hit (Rep) → time_conv with zero pad, store input cache
 *   later → time_conv with cached frames
 * Returns 1 if T doubled, 0 if skipped (Rep), -1 on error.
 */
static int vae_time_conv_double(wan_ctx *ctx, float **feat, int *lt, int C,
                                int lh, int lw, const char *wname,
                                const char *bname, vae_ccache *cache,
                                const char *key) {
  vae_cache_ent *ent = vae_ccache_find(cache, key);
  if (!ent)
    return -1;

  /* First visit: 'Rep' — skip temporal expand. */
  if (!ent->rep && !ent->data) {
    ent->rep = 1;
    return 0;
  }

  double st0 = VAE_STAGE_T0();
  size_t nw = 0, nb = 0;
  float *W = wan_load_tensor_f32(ctx, wname, &nw);
  float *b = wan_load_tensor_f32(ctx, bname, &nb);
  size_t expect = (size_t)(2 * C) * (size_t)C * 3 * 1 * 1;
  if (!W || nw != expect) {
    free(W);
    free(b);
    return -1;
  }

  int Tin = *lt;
  int use_cache = ent->data && ent->C == C && ent->H == lh && ent->W == lw &&
                  ent->T > 0 && !ent->rep;
  int prev_t = use_cache ? ent->T : 0;
  if (prev_t > VAE_CACHE_T)
    prev_t = VAE_CACHE_T;
  int pad_left = 2 - prev_t;
  if (pad_left < 0)
    pad_left = 0;
  int Twork = pad_left + prev_t + Tin;
  /* Conv kT=3 pad0 → Dout = Twork - 2; need Dout == Tin when Twork == Tin+2 */
  int Dout = Twork - 2;
  if (Dout < 1) {
    free(W);
    free(b);
    return -1;
  }

  size_t n_work = (size_t)C * (size_t)Twork * (size_t)lh * (size_t)lw;
  size_t n_mid = (size_t)(2 * C) * (size_t)Dout * (size_t)lh * (size_t)lw;
  size_t n_out = (size_t)C * (size_t)(Tin * 2) * (size_t)lh * (size_t)lw;
  float *work = calloc(n_work, sizeof(float));
  float *mid = calloc(n_mid, sizeof(float));
  float *out = calloc(n_out, sizeof(float));
  if (!work || !mid || !out) {
    free(W);
    free(b);
    free(work);
    free(mid);
    free(out);
    return -1;
  }

  if (prev_t > 0) {
    for (int ti = 0; ti < prev_t; ti++)
      for (int c = 0; c < C; c++)
        for (int h = 0; h < lh; h++)
          for (int w = 0; w < lw; w++)
            work[((((size_t)c * Twork + (pad_left + ti)) * lh + h) * lw + w)] =
                ent->data[((((size_t)c * ent->T + ti) * lh + h) * lw + w)];
  }
  for (int ti = 0; ti < Tin; ti++)
    for (int c = 0; c < C; c++)
      for (int h = 0; h < lh; h++)
        for (int w = 0; w < lw; w++)
          work[((((size_t)c * Twork + (pad_left + prev_t + ti)) * lh + h) * lw +
                w)] = (*feat)[((((size_t)c * Tin + ti) * lh + h) * lw + w)];

  uma_wan_conv3d_f32(mid, work, W, (b && nb == (size_t)(2 * C)) ? b : NULL, 1, C,
                     Twork, lh, lw, 2 * C, 3, 1, 1, 1, 1, 1, 0, 0, 0);

  /* Take last Tin frames of Dout if needed, then interleave 2C → C×(T*2). */
  int t0 = Dout - Tin;
  if (t0 < 0)
    t0 = 0;
  for (int t = 0; t < Tin; t++)
    for (int c = 0; c < C; c++)
      for (int h = 0; h < lh; h++)
        for (int w = 0; w < lw; w++) {
          float g0 =
              mid[((((size_t)c * Dout + (t0 + t)) * lh + h) * lw + w)];
          float g1 = mid[(((((size_t)C + c) * Dout + (t0 + t)) * lh + h) * lw +
                          w)];
          out[((((size_t)c * (Tin * 2) + (t * 2)) * lh + h) * lw + w)] = g0;
          out[((((size_t)c * (Tin * 2) + (t * 2 + 1)) * lh + h) * lw + w)] =
              g1;
        }

  /* Update feat_cache from current input (Wan Resample). */
  {
    int store_t = Tin < VAE_CACHE_T ? Tin : VAE_CACHE_T;
    int next_t = store_t;
    int dst0 = 0;
    if (store_t < VAE_CACHE_T && use_cache && ent->data && ent->T > 0) {
      next_t = store_t + 1;
      if (next_t > VAE_CACHE_T)
        next_t = VAE_CACHE_T;
      dst0 = 1;
    }
    float *next = calloc((size_t)C * (size_t)next_t * (size_t)lh * (size_t)lw,
                         sizeof(float));
    if (next) {
      if (dst0) {
        for (int c = 0; c < C; c++)
          for (int h = 0; h < lh; h++)
            for (int w = 0; w < lw; w++)
              next[((((size_t)c * next_t + 0) * lh + h) * lw + w)] =
                  ent->data[((((size_t)c * ent->T + (ent->T - 1)) * lh + h) *
                                 lw +
                             w)];
      }
      int src0 = Tin - store_t;
      for (int ti = 0; ti < store_t; ti++)
        for (int c = 0; c < C; c++)
          for (int h = 0; h < lh; h++)
            for (int w = 0; w < lw; w++)
              next[((((size_t)c * next_t + (dst0 + ti)) * lh + h) * lw + w)] =
                  (*feat)[((((size_t)c * Tin + (src0 + ti)) * lh + h) * lw +
                           w)];
      free(ent->data);
      ent->data = next;
      ent->C = C;
      ent->T = next_t;
      ent->H = lh;
      ent->W = lw;
      ent->rep = 0;
    }
  }

  free(*feat);
  *feat = out;
  *lt = Tin * 2;
  free(W);
  free(b);
  free(work);
  free(mid);
  VAE_STAGE_ADD("vae_tconv", st0);
  return 1;
}

/*
 * Decode one latent time slice (Wan decode loops T with feat_cache).
 * upsample3d time_conv uses per-stage Rep/cache inside vae_time_conv_double.
 */
static int vae_tip_one_slice(wan_ctx *ctx, const float *slice16, int lh,
                             int lw, float *feat3, int *out_lt, int *out_lh,
                             int *out_lw, int *nres, int *nattn, int *nre,
                             int *ntc, vae_ccache *cache) {
  size_t nw1 = 0, nb1 = 0, nwh = 0, nbh = 0;
  float *Wc = wan_load_tensor_f32(ctx, "vae.decoder.conv1.weight", &nw1);
  float *bc = wan_load_tensor_f32(ctx, "vae.decoder.conv1.bias", &nb1);
  float *Wh = wan_load_tensor_f32(ctx, "vae.decoder.head.2.weight", &nwh);
  float *bh = wan_load_tensor_f32(ctx, "vae.decoder.head.2.bias", &nbh);
  int lt = 1;
  int flh = lh, flw = lw;
  int ch = 384;
  float *feat =
      calloc((size_t)384 * (size_t)lt * (size_t)lh * (size_t)lw, sizeof(float));
  int ok = 0;
  if (!Wc || !Wh || !feat || nw1 != (size_t)384 * 16 * 3 * 3 * 3 ||
      nwh != (size_t)3 * 96 * 3 * 3 * 3) {
    free(Wc);
    free(bc);
    free(Wh);
    free(bh);
    free(feat);
    return -1;
  }
  if (vae_causal_conv3d_k3(feat, slice16, Wc, (bc && nb1 == 384) ? bc : NULL, 16,
                            384, lt, lh, lw, cache, "vae.decoder.conv1") != 0) {
    free(Wc);
    free(bc);
    free(Wh);
    free(bh);
    free(feat);
    return -1;
  }

  if (vae_resblock_same(ctx, feat, 384, lt, flh, flw,
                        "vae.decoder.middle.0", cache) == 0)
    (*nres)++;
  if (vae_attn_mid(ctx, feat, 384, lt, flh, flw) == 0)
    (*nattn)++;
  if (vae_resblock_same(ctx, feat, 384, lt, flh, flw,
                        "vae.decoder.middle.2", cache) == 0)
    (*nres)++;
  if (vae_resblock_same(ctx, feat, 384, lt, flh, flw,
                        "vae.decoder.upsamples.0", cache) == 0)
    (*nres)++;
  if (vae_resblock_same(ctx, feat, 384, lt, flh, flw,
                        "vae.decoder.upsamples.1", cache) == 0)
    (*nres)++;
  if (vae_resblock_same(ctx, feat, 384, lt, flh, flw,
                        "vae.decoder.upsamples.2", cache) == 0)
    (*nres)++;

  /* upsample3d #1: time_conv (Rep/cache) then spatial */
  {
    int tc = vae_time_conv_double(
        ctx, &feat, &lt, ch, flh, flw,
        "vae.decoder.upsamples.3.time_conv.weight",
        "vae.decoder.upsamples.3.time_conv.bias", cache,
        "vae.decoder.upsamples.3.time_conv");
    if (tc < 0) {
      free(Wc);
      free(bc);
      free(Wh);
      free(bh);
      free(feat);
      return -1;
    }
    if (tc > 0)
      (*ntc)++;
  }
  if (vae_upsample_resample(
          ctx, &feat, &ch, lt, &flh, &flw,
          "vae.decoder.upsamples.3.resample.1.weight",
          "vae.decoder.upsamples.3.resample.1.bias",
          "decoder.upsamples.3.resample.1.weight", 192) != 0) {
    free(Wc);
    free(bc);
    free(Wh);
    free(bh);
    free(feat);
    return -1;
  }
  (*nre)++;

  if (ch == 192 &&
      vae_resblock_expand(ctx, &feat, &ch, lt, flh, flw, 384,
                          "vae.decoder.upsamples.4", cache) == 0)
    (*nres)++;
  if (ch == 384) {
    if (vae_resblock_same(ctx, feat, 384, lt, flh, flw,
                          "vae.decoder.upsamples.5", cache) == 0)
      (*nres)++;
    if (vae_resblock_same(ctx, feat, 384, lt, flh, flw,
                          "vae.decoder.upsamples.6", cache) == 0)
      (*nres)++;
    {
      int tc = vae_time_conv_double(
          ctx, &feat, &lt, ch, flh, flw,
          "vae.decoder.upsamples.7.time_conv.weight",
          "vae.decoder.upsamples.7.time_conv.bias", cache,
          "vae.decoder.upsamples.7.time_conv");
      if (tc < 0) {
        free(Wc);
        free(bc);
        free(Wh);
        free(bh);
        free(feat);
        return -1;
      }
      if (tc > 0)
        (*ntc)++;
    }
    if (vae_upsample_resample(
            ctx, &feat, &ch, lt, &flh, &flw,
            "vae.decoder.upsamples.7.resample.1.weight",
            "vae.decoder.upsamples.7.resample.1.bias",
            "decoder.upsamples.7.resample.1.weight", 192) != 0) {
      free(Wc);
      free(bc);
      free(Wh);
      free(bh);
      free(feat);
      return -1;
    }
    (*nre)++;
  }

  if (ch == 192) {
    if (vae_resblock_same(ctx, feat, 192, lt, flh, flw,
                          "vae.decoder.upsamples.8", cache) == 0)
      (*nres)++;
    if (vae_resblock_same(ctx, feat, 192, lt, flh, flw,
                          "vae.decoder.upsamples.9", cache) == 0)
      (*nres)++;
    if (vae_resblock_same(ctx, feat, 192, lt, flh, flw,
                          "vae.decoder.upsamples.10", cache) == 0)
      (*nres)++;
    if (vae_upsample_resample(
            ctx, &feat, &ch, lt, &flh, &flw,
            "vae.decoder.upsamples.11.resample.1.weight",
            "vae.decoder.upsamples.11.resample.1.bias",
            "decoder.upsamples.11.resample.1.weight", 96) != 0) {
      free(Wc);
      free(bc);
      free(Wh);
      free(bh);
      free(feat);
      return -1;
    }
    (*nre)++;
  }

  if (ch == 96) {
    if (vae_resblock_same(ctx, feat, 96, lt, flh, flw,
                          "vae.decoder.upsamples.12", cache) == 0)
      (*nres)++;
    if (vae_resblock_same(ctx, feat, 96, lt, flh, flw,
                          "vae.decoder.upsamples.13", cache) == 0)
      (*nres)++;
    if (vae_resblock_same(ctx, feat, 96, lt, flh, flw,
                          "vae.decoder.upsamples.14", cache) == 0)
      (*nres)++;
  }

  size_t spat = (size_t)lt * (size_t)flh * (size_t)flw;
  float *tmp96 = (ch == 96) ? feat : NULL;
  float *owned96 = NULL;
  if (!tmp96) {
    owned96 = calloc((size_t)96 * spat, sizeof(float));
    tmp96 = owned96;
    if (tmp96) {
      int groups = ch / 96;
      if (groups < 1)
        groups = 1;
      for (int oc = 0; oc < 96; oc++)
        for (size_t s = 0; s < spat; s++) {
          float sum = 0.f;
          for (int g = 0; g < groups; g++)
            sum += feat[((size_t)(oc * groups + g) * spat) + s];
          tmp96[(size_t)oc * spat + s] = sum / (float)groups;
        }
    }
  }
  float *tmp3 = calloc((size_t)3 * spat, sizeof(float));
  if (tmp96 && tmp3) {
    size_t ng = 0;
    float *hg = wan_load_tensor_f32(ctx, "vae.decoder.head.0.gamma", &ng);
    int head_ok = 0;
    if (hg && ng == 96) {
      /* Keep a pre-activation copy for broker (RMS+SILU inside HEADT). */
      float *raw96 = malloc((size_t)96 * spat * sizeof(float));
      if (raw96) {
        memcpy(raw96, tmp96, (size_t)96 * spat * sizeof(float));
        if (vae_broker_headt(ctx, tmp3, raw96, hg,
                             (bh && nbh == 3) ? bh : NULL, 96, 3, lt, flh, flw,
                             "decoder.head.2.weight",
                             "vae.decoder.head.2.weight", cache,
                             "vae.decoder.head.2", 1) == 0)
          head_ok = 1;
        free(raw96);
      }
    }
    if (!head_ok) {
      rms_norm_ncdhw(tmp96, hg, 96, lt, flh, flw, 1e-6f);
      silu_inplace(tmp96, (size_t)96 * spat);
      if (vae_causal_conv3d_k3(tmp3, tmp96, Wh, (bh && nbh == 3) ? bh : NULL, 96,
                                3, lt, flh, flw, cache, "vae.decoder.head.2") !=
          0) {
        free(hg);
        free(owned96);
        free(tmp3);
        free(Wc);
        free(bc);
        free(Wh);
        free(bh);
        free(feat);
        return -1;
      }
    }
    free(hg);
    for (size_t s = 0; s < spat; s++)
      for (int c = 0; c < 3; c++)
        feat3[s * 3 + (size_t)c] = tmp3[(size_t)c * spat + s];
    *out_lt = lt;
    *out_lh = flh;
    *out_lw = flw;
    ok = 1;
  }
  free(owned96);
  free(tmp3);
  free(Wc);
  free(bc);
  free(Wh);
  free(bh);
  free(feat);
  return ok ? 0 : -1;
}

/*
 * Real-weight VAE tip: Wan-style per-latent-frame decode.
 * First frame skips time_conv (Rep); later frames apply time_conv (T×4).
 */
static int vae_real_head_tip(wan_ctx *ctx, const float *feat_nchw, int C,
                             int lt, int lh, int lw, float *feat3, int *out_lt,
                             int *out_lh, int *out_lw) {
  if (!ctx || !feat_nchw || !feat3 || C != 16 || lt < 1)
    return -1;
  if (!wan_gguf_has(ctx, "vae.decoder.conv1.weight") ||
      !wan_gguf_has(ctx, "vae.decoder.head.2.weight"))
    return -1;

  (void)vae_headt_bank_all(ctx);

  size_t nw2 = 0, nb2 = 0;
  float *W2 = wan_load_tensor_f32(ctx, "vae.conv2.weight", &nw2);
  float *b2 = wan_load_tensor_f32(ctx, "vae.conv2.bias", &nb2);
  size_t n_slice = (size_t)16 * 1 * (size_t)lh * (size_t)lw;
  float *slice = calloc(n_slice, sizeof(float));
  /* max out frames: 1 + (lt-1)*4 */
  int max_ot = 1 + (lt > 1 ? (lt - 1) * 4 : 0);
  size_t max_spat = (size_t)max_ot * (size_t)(lh * 8) * (size_t)(lw * 8);
  float *chunk = calloc(max_spat * 3, sizeof(float));
  if (!slice || !chunk) {
    free(W2);
    free(b2);
    free(slice);
    free(chunk);
    return -1;
  }

  int nres = 0, nattn = 0, nre = 0, ntc = 0;
  int written = 0;
  int olh = lh * 8, olw = lw * 8;
  int ok_any = 0;
  vae_ccache cache;
  memset(&cache, 0, sizeof(cache));
  for (int ti = 0; ti < lt; ti++) {
    /* conv2 per slice then tip */
    float *raw = calloc(n_slice, sizeof(float));
    if (!raw)
      break;
    for (int c = 0; c < 16; c++)
      for (int h = 0; h < lh; h++)
        for (int w = 0; w < lw; w++)
          raw[(((size_t)c * lh + h) * lw + w)] =
              feat_nchw[((((size_t)c * lt + ti) * lh + h) * lw + w)];
    if (W2 && nw2 == (size_t)16 * 16 * 1 * 1 * 1)
      uma_wan_conv3d_f32(slice, raw, W2, (b2 && nb2 == 16) ? b2 : NULL, 1, 16,
                         1, lh, lw, 16, 1, 1, 1, 1, 1, 1, 0, 0, 0);
    else
      memcpy(slice, raw, n_slice * sizeof(float));
    free(raw);

    int cot = 0, clh = 0, clw = 0;
    if (vae_tip_one_slice(ctx, slice, lh, lw, chunk, &cot, &clh, &clw, &nres,
                          &nattn, &nre, &ntc, &cache) != 0)
      break;
    fprintf(stderr, "wan-c: VAE tip slice %d/%d out_t=%d grid=%dx%d\n", ti + 1,
            lt, cot, clh, clw);
    fflush(stderr);
    size_t cspat = (size_t)cot * (size_t)clh * (size_t)clw;
    memcpy(feat3 + (size_t)written * (size_t)clh * (size_t)clw * 3, chunk,
           cspat * 3 * sizeof(float));
    written += cot;
    olh = clh;
    olw = clw;
    ok_any = 1;
  }
  vae_ccache_clear(&cache);
  free(W2);
  free(b2);
  free(slice);
  free(chunk);
  if (!ok_any)
    return -1;
  if (out_lt)
    *out_lt = written;
  if (out_lh)
    *out_lh = olh;
  if (out_lw)
    *out_lw = olw;
  static int logged;
  if (!logged) {
    fprintf(stderr,
            "wan-c: VAE tip frame-loop res×%d attn×%d resample×%d tconv×%d "
            "causal_cache+Rep out_t=%d grid=%dx%d +head\n",
            nres, nattn, nre, ntc, written, olh, olw);
    logged = 1;
    fflush(stderr);
  }
  return 0;
}

static void upsample_rgb(const float *feat, int lt, int lh, int lw, int frames,
                         int height, int width, float *rgb) {
  for (int f = 0; f < frames; f++) {
    int st = (int)((int64_t)f * lt / frames);
    if (st >= lt)
      st = lt - 1;
    for (int y = 0; y < height; y++) {
      int sy = (int)((int64_t)y * lh / height);
      if (sy >= lh)
        sy = lh - 1;
      for (int x = 0; x < width; x++) {
        int sx = (int)((int64_t)x * lw / width);
        if (sx >= lw)
          sx = lw - 1;
        size_t si =
            ((size_t)st * (size_t)lh + (size_t)sy) * (size_t)lw + (size_t)sx;
        const float *src = feat + si * 3;
        float *dst =
            rgb + (((size_t)f * (size_t)height + (size_t)y) * (size_t)width +
                   (size_t)x) *
                      3;
        for (int c = 0; c < 3; c++) {
          float v = src[c] * 0.5f + 0.5f;
          if (v < 0.0f)
            v = 0.0f;
          if (v > 1.0f)
            v = 1.0f;
          dst[c] = v * 255.0f;
        }
      }
    }
  }
}

static void nchw_to_rgb3(const float *nchw, int C, size_t spatial, float *feat3,
                         const float *w1x1) {
  float *tmp = calloc(3 * spatial, sizeof(float));
  if (!tmp)
    return;
  uma_wan_conv2d_f32(tmp, nchw, w1x1, NULL, 1, C, 1, (int)spatial, 3, 1, 1, 1,
                     0);
  for (size_t s = 0; s < spatial; s++) {
    for (int ch = 0; ch < 3; ch++)
      feat3[s * 3 + (size_t)ch] = tmp[(size_t)ch * spatial + s];
  }
  free(tmp);
}

static int vae_broker_recipe(wan_ctx *ctx, const float *latent, size_t latent_n,
                             float *feat_nchw, int C, int lt, int lh, int lw) {
  /* F0782-D / F0784: GN → SILU → C2D → residual → UP3 when caps allow. */
  size_t spatial = (size_t)lt * (size_t)lh * (size_t)lw;
  size_t nbytes = latent_n * sizeof(float);
  size_t wne = (size_t)C * (size_t)C;
  size_t wbytes = wne * sizeof(float);
  int Dflat = (int)latent_n;

  float *eye = calloc(wne, sizeof(float));
  float *bias = calloc((size_t)C, sizeof(float));
  float *ones = calloc(latent_n, sizeof(float));
  if (!eye || !bias || !ones) {
    free(eye);
    free(bias);
    free(ones);
    return -1;
  }
  wan_fill_eye_nt(eye, C, C);
  for (size_t i = 0; i < latent_n; i++)
    ones[i] = 1.f;

  const char *bx = "x_vae_x";
  const char *bt1 = "x_vae_t1";
  const char *bt2 = "x_vae_t2";
  const char *bsilu = "x_vae_silu";
  const char *by = "x_vae_y";
  const char *bw = "x_vae_cw";
  const char *bb = "x_vae_cb";
  const char *bones = "x_vae_ones";

  if (uma_buf_pool_alloc(ctx->bufs, bx, nbytes) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bt1, nbytes) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bt2, nbytes) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, by, nbytes) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bw, wbytes) != 0 ||
      uma_buf_pool_alloc(ctx->bufs, bb, (size_t)C * 4) != 0 ||
      uma_buf_pool_put(ctx->bufs, bx, latent, nbytes) != 0 ||
      uma_buf_pool_put(ctx->bufs, bw, eye, wbytes) != 0 ||
      uma_buf_pool_put(ctx->bufs, bb, bias, (size_t)C * 4) != 0) {
    free(eye);
    free(bias);
    free(ones);
    return -1;
  }
  free(eye);
  free(bias);

  int G = 8;
  if (C % G != 0)
    G = (C % 4 == 0) ? 4 : 1;

  if (wan_graph_groupnorm(ctx, bx, bt1, 1, C, (int)spatial, G) != 0) {
    free(ones);
    return -1;
  }

  const char *conv_in = bt1;
  if (ctx->caps.silu_mul) {
    if (uma_buf_pool_alloc(ctx->bufs, bsilu, nbytes) != 0 ||
        uma_buf_pool_alloc(ctx->bufs, bones, nbytes) != 0 ||
        uma_buf_pool_put(ctx->bufs, bones, ones, nbytes) != 0 ||
        wan_graph_silu_mul(ctx, bt1, bones, bsilu, Dflat) != 0) {
      free(ones);
      return -1;
    }
    conv_in = bsilu;
  }
  free(ones);

  if (wan_graph_conv2d(ctx, conv_in, bt2, bw, bb, 1, C, lt, lh * lw, C, 1, 1, 1,
                       0) != 0)
    return -1;

  const char *up_in = bt2;
  if (ctx->caps.residual_add && ctx->caps.silu_mul) {
    /* Seed residual with pre-conv (silu), then add conv out in-place. */
    if (wan_graph_copy(ctx, by, conv_in, Dflat) != 0 ||
        wan_graph_residual_add(ctx, by, bt2, Dflat) != 0)
      return -1;
    up_in = by;
  }

  const char *up_out = (up_in == by) ? bt1 : by;
  if (wan_graph_unpatchify3d(ctx, up_in, up_out, 1, C, lt, lh, lw, 1, 1, 1) != 0)
    return -1;

  char resp[512];
  size_t got = 0;
  if (uma_client_buf_get(ctx->uma, up_out, feat_nchw, nbytes, &got, resp,
                         sizeof(resp)) != 0 ||
      got != nbytes) {
    fprintf(stderr, "wan-c: VAE recipe BUF_GET failed: %.160s\n", resp);
    return -1;
  }

  /* F0801/F0889: 3× CT2D spatial upsample (N=lt), then nearest back to grid.
   * F0826: optional TOK3↔NCDHW3 rematch tip on CTHW volume before permute. */
  if (ctx->caps.ct2d && lh >= 2 && lw >= 2) {
    int h1 = (lh - 1) * 2 + 2; /* KH=2,stride=2,pad=0 */
    int w1 = (lw - 1) * 2 + 2;
    int h2 = (h1 - 1) * 2 + 2;
    int w2 = (w1 - 1) * 2 + 2;
    int h3 = (h2 - 1) * 2 + 2;
    int w3 = (w2 - 1) * 2 + 2;
    size_t n0 = (size_t)lt * (size_t)C * (size_t)lh * (size_t)lw;
    size_t n1 = (size_t)lt * (size_t)C * (size_t)h1 * (size_t)w1;
    size_t n2 = (size_t)lt * (size_t)C * (size_t)h2 * (size_t)w2;
    size_t n3 = (size_t)lt * (size_t)C * (size_t)h3 * (size_t)w3;
    size_t wct = (size_t)C * (size_t)C * 4;
    float *perm = calloc(n0, sizeof(float));
    float *wtr = calloc(wct, sizeof(float));
    float *big = calloc(n3, sizeof(float));
    if (!perm || !wtr || !big) {
      free(perm);
      free(wtr);
      free(big);
      return 0; /* keep UP3 result */
    }

    if (ctx->caps.tok3 && ctx->caps.ncdhw3 &&
        (size_t)C * (size_t)lt * (size_t)lh * (size_t)lw == n0) {
      char kind[64];
      const char *bv = "x_vae_vol";
      const char *bt = "x_vae_tok";
      snprintf(kind, sizeof(kind), "1_%d_%d_%d_%d", C, lt, lh, lw);
      if (uma_buf_pool_alloc(ctx->bufs, bv, n0 * 4) == 0 &&
          uma_buf_pool_alloc(ctx->bufs, bt, n0 * 4) == 0 &&
          uma_buf_pool_put(ctx->bufs, bv, feat_nchw, n0 * 4) == 0 &&
          wan_graph_tok3(ctx, bv, bt, kind) == 0 &&
          wan_graph_ncdhw3(ctx, bt, bv, kind) == 0) {
        size_t g2 = 0;
        if (uma_client_buf_get(ctx->uma, bv, feat_nchw, n0 * 4, &g2, resp,
                               sizeof(resp)) == 0 &&
            g2 == n0 * 4) {
          /* rematch tip OK — feat restored to CTHW */
        }
      }
    }

    /* C,T,H,W → T,C,H,W for CT2D N=lt */
    for (int t = 0; t < lt; t++)
      for (int c = 0; c < C; c++)
        for (int h = 0; h < lh; h++)
          for (int w = 0; w < lw; w++) {
            size_t src =
                ((((size_t)c * lt + t) * lh + h) * lw + w);
            size_t dst =
                ((((size_t)t * C + c) * lh + h) * lw + w);
            perm[dst] = feat_nchw[src];
          }
    for (int ic = 0; ic < C; ic++)
      for (int oc = 0; oc < C; oc++)
        if (ic == oc)
          wtr[((((size_t)ic * C + oc) * 2) + 0) * 2 + 0] = 1.f;

    const char *bu0 = "x_vae_u0";
    const char *bu1 = "x_vae_u1";
    const char *bu2 = "x_vae_u2";
    const char *bu3 = "x_vae_u3";
    const char *bwct = "x_vae_wct";
    if (uma_buf_pool_alloc(ctx->bufs, bu0, n0 * 4) == 0 &&
        uma_buf_pool_alloc(ctx->bufs, bu1, n1 * 4) == 0 &&
        uma_buf_pool_alloc(ctx->bufs, bu2, n2 * 4) == 0 &&
        uma_buf_pool_alloc(ctx->bufs, bu3, n3 * 4) == 0 &&
        uma_buf_pool_put_weight(ctx->bufs, bwct, "vae.Wct", wtr, wct * 4) ==
            0 &&
        uma_buf_pool_put(ctx->bufs, bu0, perm, n0 * 4) == 0 &&
        wan_graph_ct2d(ctx, bu0, bu1, bwct, NULL, lt, C, lh, lw, C, 2, 2, 2, 0,
                       0) == 0 &&
        wan_graph_ct2d(ctx, bu1, bu2, bwct, NULL, lt, C, h1, w1, C, 2, 2, 2, 0,
                       0) == 0 &&
        wan_graph_ct2d(ctx, bu2, bu3, bwct, NULL, lt, C, h2, w2, C, 2, 2, 2, 0,
                       0) == 0 &&
        uma_client_buf_get(ctx->uma, bu3, big, n3 * 4, &got, resp,
                           sizeof(resp)) == 0 &&
        got == n3 * 4) {
      /* Nearest back to C,T,lh,lw */
      for (int t = 0; t < lt; t++)
        for (int c = 0; c < C; c++)
          for (int h = 0; h < lh; h++)
            for (int w = 0; w < lw; w++) {
              int hs = h * h3 / lh;
              int ws = w * w3 / lw;
              if (hs >= h3)
                hs = h3 - 1;
              if (ws >= w3)
                ws = w3 - 1;
              size_t src =
                  ((((size_t)t * C + c) * h3 + hs) * w3 + ws);
              size_t dst =
                  ((((size_t)c * lt + t) * lh + h) * lw + w);
              feat_nchw[dst] = big[src];
            }
    }
    free(perm);
    free(wtr);
    free(big);
  }

  /* F0905: true-3D CT3D upsample tip (spatial ×2), nearest back to grid. */
  if (ctx->caps.ct3d && lh >= 2 && lw >= 2) {
    int dout = lt; /* KD=1, sd=1 */
    int hout = (lh - 1) * 2 + 2;
    int wout = (lw - 1) * 2 + 2;
    size_t n_in = (size_t)C * (size_t)lt * (size_t)lh * (size_t)lw;
    size_t n_out = (size_t)C * (size_t)dout * (size_t)hout * (size_t)wout;
    size_t nw = (size_t)C * (size_t)C * 1 * 2 * 2;
    float *w3 = calloc(nw, sizeof(float));
    float *big3 = calloc(n_out, sizeof(float));
    if (w3 && big3) {
      for (int ic = 0; ic < C; ic++)
        for (int oc = 0; oc < C; oc++)
          if (ic == oc)
            w3[(((((size_t)ic * C + oc) * 1 + 0) * 2 + 0) * 2 + 0)] = 1.f;
      const char *bx3 = "x_vae_c3in";
      const char *by3 = "x_vae_c3out";
      const char *bw3 = "x_vae_Wct3";
      char kind[128];
      snprintf(kind, sizeof(kind),
               "1_%d_%d_%d_%d_%d_1_2_2_1_2_2_0_0_0_0_0_0", C, lt, lh, lw, C);
      if (uma_buf_pool_alloc(ctx->bufs, bx3, n_in * 4) == 0 &&
          uma_buf_pool_alloc(ctx->bufs, by3, n_out * 4) == 0 &&
          uma_buf_pool_put_weight(ctx->bufs, bw3, "vae.Wct3", w3, nw * 4) ==
              0 &&
          uma_buf_pool_put(ctx->bufs, bx3, feat_nchw, n_in * 4) == 0 &&
          wan_graph_ct3d(ctx, bx3, by3, bw3, kind) == 0 &&
          uma_client_buf_get(ctx->uma, by3, big3, n_out * 4, &got, resp,
                             sizeof(resp)) == 0 &&
          got == n_out * 4) {
        static int logged_ct3d;
        if (!logged_ct3d) {
          fprintf(stderr, "wan-c: VAE CT3D tip ok grid=%dx%dx%d (F0905)\n", lt,
                  lh, lw);
          logged_ct3d = 1;
        }
        for (int c = 0; c < C; c++)
          for (int t = 0; t < lt; t++)
            for (int h = 0; h < lh; h++)
              for (int w = 0; w < lw; w++) {
                int hs = h * hout / lh;
                int ws = w * wout / lw;
                if (hs >= hout)
                  hs = hout - 1;
                if (ws >= wout)
                  ws = wout - 1;
                size_t src =
                    ((((size_t)c * dout + t) * hout + hs) * wout + ws);
                size_t dst =
                    ((((size_t)c * lt + t) * lh + h) * lw + w);
                feat_nchw[dst] = big3[src];
              }
      }
    }
    free(w3);
    free(big3);
  }
  return 0;
}

static int vae_can_broker_recipe(const wan_ctx *ctx) {
  if (!ctx)
    return 0;
  if (ctx->caps.prefer_ext && ctx->caps.ext_ready)
    return 1;
  if (ctx->caps.group_norm && ctx->caps.ct2d)
    return 1;
  return ctx->caps.group_norm && ctx->caps.conv2d && ctx->caps.unpatchify;
}

static int vae_core(wan_ctx *ctx, const float *latent, size_t latent_n,
                    float *rgb, size_t rgb_n, int width, int height, int frames,
                    int use_broker) {
  const wan_model_config *c = &ctx->cfg;
  int C = c->z_channels;
  int lt = (frames - 1) / c->vae_stride_t + 1;
  int lh = height / c->vae_stride_h;
  int lw = width / c->vae_stride_w;
  size_t expect = (size_t)C * (size_t)lt * (size_t)lh * (size_t)lw;
  size_t rgb_expect = (size_t)width * (size_t)height * (size_t)frames * 3;

  if (latent_n != expect || rgb_n < rgb_expect || lh < 1 || lw < 1) {
    fprintf(stderr,
            "wan-c: VAE size mismatch latent=%zu expect=%zu rgb=%zu/%zu "
            "grid=%dx%dx%d\n",
            latent_n, expect, rgb_n, rgb_expect, lt, lh, lw);
    return -1;
  }

  /* WanVAE scale: decode(z) with z = latent * std + mean (per channel). */
  static const float k_vae_mean[16] = {
      -0.7571f, -0.7089f, -0.9113f, 0.1075f,  -0.1745f, 0.9653f,
      -0.1517f, 1.5508f,  0.4134f,  -0.0715f, 0.5517f,  -0.3632f,
      -0.1922f, -0.9497f, 0.2503f,  -0.2921f};
  static const float k_vae_std[16] = {
      2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
      3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};

  size_t spatial = (size_t)lt * (size_t)lh * (size_t)lw;
  int max_ot = 1 + (lt > 1 ? (lt - 1) * 4 : 0);
  size_t feat3_spat = (size_t)max_ot * (size_t)(lh * 8) * (size_t)(lw * 8);
  float *z = calloc(latent_n, sizeof(float));
  float *normed = calloc(latent_n, sizeof(float));
  float *feat3 = calloc(feat3_spat * 3, sizeof(float));
  float *w1x1 = calloc((size_t)3 * (size_t)C, sizeof(float));
  if (!z || !normed || !feat3 || !w1x1) {
    free(z);
    free(normed);
    free(feat3);
    free(w1x1);
    return -1;
  }

  if (C == 16) {
    for (int ch = 0; ch < 16; ch++) {
      size_t off = (size_t)ch * spatial;
      for (size_t s = 0; s < spatial; s++)
        z[off + s] = latent[off + s] * k_vae_std[ch] + k_vae_mean[ch];
    }
    static int logged_scale;
    if (!logged_scale) {
      fprintf(stderr, "wan-c: VAE latent denorm (mean/std ×16 channels)\n");
      logged_scale = 1;
      fflush(stderr);
    }
  } else {
    memcpy(z, latent, latent_n * sizeof(float));
  }

  memset(w1x1, 0, (size_t)3 * (size_t)C * sizeof(float));
  int d = C < 3 ? C : 3;
  for (int i = 0; i < d; i++)
    w1x1[i * C + i] = 1.0f;

  /* Prefer real-weight tip on denormed latents. */
  int olt = lt, olh = lh, olw = lw;
  fprintf(stderr, "wan-c: VAE tip begin grid=%dx%dx%d\n", lt, lh, lw);
  fflush(stderr);
  if (vae_real_head_tip(ctx, z, C, lt, lh, lw, feat3, &olt, &olh, &olw) == 0) {
    upsample_rgb(feat3, olt, olh, olw, frames, height, width, rgb);
    free(z);
    free(normed);
    free(feat3);
    free(w1x1);
    return 0;
  }

  /* Scaffold fallback: optional broker recipe / groupnorm on denormed z. */
  int used_recipe = 0;
  if (use_broker && vae_can_broker_recipe(ctx)) {
    if (vae_broker_recipe(ctx, z, latent_n, normed, C, lt, lh, lw) == 0)
      used_recipe = 1;
    else
      fprintf(stderr, "wan-c: VAE broker recipe failed — host fallback\n");
  }

  if (!used_recipe) {
    int G = 8;
    if (C % G != 0)
      G = (C % 4 == 0) ? 4 : 1;
    if (use_broker && ctx->caps.group_norm) {
      char resp[512];
      size_t got = 0;
      const char *bx = "x_vae_x";
      const char *by = "x_vae_y";
      size_t nbytes = latent_n * sizeof(float);
      if (uma_buf_pool_alloc(ctx->bufs, bx, nbytes) == 0 &&
          uma_buf_pool_alloc(ctx->bufs, by, nbytes) == 0 &&
          uma_buf_pool_put(ctx->bufs, bx, z, nbytes) == 0 &&
          uma_buf_pool_put(ctx->bufs, by, normed, nbytes) == 0 &&
          wan_graph_groupnorm(ctx, bx, by, 1, C, (int)spatial, G) == 0 &&
          uma_client_buf_get(ctx->uma, by, normed, nbytes, &got, resp,
                             sizeof(resp)) == 0 &&
          got == nbytes) {
        /* ok */
      } else {
        uma_wan_groupnorm_f32(normed, z, NULL, NULL, 1, C, (int)spatial, G,
                              1e-6f);
      }
    } else {
      uma_wan_groupnorm_f32(normed, z, NULL, NULL, 1, C, (int)spatial, G,
                            1e-6f);
    }
  }

  olt = lt;
  olh = lh;
  olw = lw;
  nchw_to_rgb3(normed, C, spatial, feat3, w1x1);
  upsample_rgb(feat3, olt, olh, olw, frames, height, width, rgb);

  free(z);
  free(normed);
  free(feat3);
  free(w1x1);
  return 0;
}

int wan_vae_decode(wan_ctx *ctx, const float *latent, size_t latent_n,
                   float *rgb, size_t rgb_n, int width, int height,
                   int frames) {
  if (!ctx || !latent || !rgb || width < 1 || height < 1 || frames < 1)
    return -1;
  if (wan_env_local() || ctx->local_mode)
    return vae_core(ctx, latent, latent_n, rgb, rgb_n, width, height, frames,
                    0);
  if (!ctx->uma || !ctx->bufs) {
    fprintf(stderr, "wan-c: VAE needs UMA client (or UMA_WAN_LOCAL=1)\n");
    return -1;
  }
  return vae_core(ctx, latent, latent_n, rgb, rgb_n, width, height, frames, 1);
}
