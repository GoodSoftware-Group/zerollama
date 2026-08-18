#include "h3_dit_pack.h"

#include "h3_adaln_host.h"
#include "h3_audio_vae_host.h"
#include "h3_dit_forward.h"
#include "h3_dit_host.h"
#include "h3_video_vae_host.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void h3_dit_seq_plan_free(h3_dit_seq_plan *plan) {
  if (!plan)
    return;
  free(plan->tags);
  free(plan->position_ids);
  free(plan->text_index);
  free(plan->audio_index);
  free(plan->video_index);
  memset(plan, 0, sizeof(*plan));
}

static int tag_for_kind(h3_segment_kind kind) {
  switch (kind) {
  case H3_SEG_TEXT:
    return H3_ADALN_TAG_TEXT;
  case H3_SEG_AUDIO:
  case H3_SEG_REF_AUDIO:
    return H3_ADALN_TAG_AUDIO;
  default:
    return H3_ADALN_TAG_VIDEO;
  }
}

int h3_dit_seq_plan_from_layout(const h3_layout *layout, h3_dit_seq_plan *plan) {
  if (!layout || !plan || layout->seq_len < 1)
    return -1;
  memset(plan, 0, sizeof(*plan));
  const int seq = (int)layout->seq_len;
  plan->seq = seq;
  plan->tags = (int *)malloc((size_t)seq * sizeof(int));
  plan->position_ids = (float *)malloc((size_t)seq * 3 * sizeof(float));
  if (!plan->tags || !plan->position_ids) {
    h3_dit_seq_plan_free(plan);
    return -1;
  }
  for (int s = 0; s < seq; s++) {
    plan->tags[s] = H3_ADALN_TAG_PAD;
    plan->position_ids[(size_t)s * 3 + 0] = (float)layout->positions[s].t;
    plan->position_ids[(size_t)s * 3 + 1] = (float)layout->positions[s].h;
    plan->position_ids[(size_t)s * 3 + 2] = (float)layout->positions[s].w;
  }
  int nt = 0, na = 0, nv = 0;
  for (size_t i = 0; i < layout->segment_count; i++) {
    const h3_segment *seg = &layout->segments[i];
    int tag = tag_for_kind(seg->kind);
    for (size_t s = seg->start; s < seg->stop; s++)
      plan->tags[s] = tag;
    int n = (int)(seg->stop - seg->start);
    if (seg->kind == H3_SEG_TEXT)
      nt += n;
    else if (seg->kind == H3_SEG_AUDIO)
      na += n;
    else if (seg->kind == H3_SEG_VIDEO)
      nv += n;
  }
  plan->nt = nt;
  plan->na = na;
  plan->nv = nv;
  if (nt < 1 || na < 1 || nv < 1) {
    h3_dit_seq_plan_free(plan);
    return -1;
  }
  plan->text_index = (int *)malloc((size_t)nt * sizeof(int));
  plan->audio_index = (int *)malloc((size_t)na * sizeof(int));
  plan->video_index = (int *)malloc((size_t)nv * sizeof(int));
  if (!plan->text_index || !plan->audio_index || !plan->video_index) {
    h3_dit_seq_plan_free(plan);
    return -1;
  }
  int it = 0, ia = 0, iv = 0;
  for (size_t i = 0; i < layout->segment_count; i++) {
    const h3_segment *seg = &layout->segments[i];
    for (size_t s = seg->start; s < seg->stop; s++) {
      if (seg->kind == H3_SEG_TEXT)
        plan->text_index[it++] = (int)s;
      else if (seg->kind == H3_SEG_AUDIO)
        plan->audio_index[ia++] = (int)s;
      else if (seg->kind == H3_SEG_VIDEO)
        plan->video_index[iv++] = (int)s;
    }
  }
  return 0;
}

int h3_dit_seq_plan_apply_text_tags(h3_dit_seq_plan *plan, const int *tags,
                                    int nt) {
  if (!plan || !tags || nt < 1)
    return 0;
  if (nt != plan->nt || !plan->text_index || !plan->tags)
    return -1;
  for (int i = 0; i < nt; i++) {
    int t = tags[i];
    if (t < 0 || t > 2)
      t = H3_ADALN_TAG_TEXT;
    int s = plan->text_index[i];
    if (s < 0 || s >= plan->seq)
      return -1;
    plan->tags[s] = t;
  }
  return 0;
}

int h3_dit_t2va_geom_build(int pixel_w, int pixel_h, int frames,
                           h3_dit_t2va_geom *geom) {
  if (!geom || pixel_w < 32 || pixel_h < 32 || frames < 5)
    return -1;
  memset(geom, 0, sizeof(*geom));
  int aw = pixel_w, ah = pixel_h;
  int aligned = (pixel_w % 32) == 0 && (pixel_h % 32) == 0 &&
                (size_t)pixel_w * (size_t)pixel_h <= (size_t)H3_MAX_PIXELS;
  /* Keep 32-aligned lab sizes (256² used to snap to 768 via adapt_canvas). */
  if (!aligned) {
    if (pixel_w >= 256 || pixel_h >= 256) {
      if (!h3_adapt_canvas(pixel_w, pixel_h, &aw, &ah))
        return -1;
    } else
      return -1;
  }
  h3_temporal_shape ts = h3_temporal(frames);
  int lw = 0, lh = 0;
  h3_latent_canvas(aw, ah, &lw, &lh);
  if (lw < 2 || lh < 2 || (lw % 2) || (lh % 2) || ts.video_t < 1 ||
      ts.audio_t < 1)
    return -1;
  geom->pixel_w = aw;
  geom->pixel_h = ah;
  geom->frames = ts.frame_count;
  geom->latent_w = lw;
  geom->latent_h = lh;
  geom->latent_t = ts.video_t;
  geom->audio_t = ts.audio_t;
  geom->nv = (ts.video_t / H3_DIT_PATCH_T) * (lh / H3_DIT_PATCH_H) *
             (lw / H3_DIT_PATCH_W);
  geom->na = 2 * ts.audio_t;
  geom->video_n = (size_t)H3_VIDEO_VAE_LATENT_CHANNELS * (size_t)ts.video_t *
                  (size_t)lh * (size_t)lw;
  geom->audio_n = 2ull * (size_t)H3_AUDIO_VAE_LATENT_CHANNELS * (size_t)ts.audio_t;
  return 0;
}

static int load_f32_file(const char *path, float *dst, size_t n) {
  if (!path || !path[0] || !dst || n < 1)
    return 0;
  FILE *f = fopen(path, "rb");
  if (!f)
    return 0;
  size_t r = fread(dst, sizeof(float), n, f);
  fclose(f);
  return r == n;
}

int h3_dit_t2va(const h3_st_store *store, const float *text, int nt,
                const int *text_tags, int steps, int n_layers,
                int reuse_interval, int adaln_t_sigma, uint64_t seed,
                const h3_dit_t2va_geom *geom, float *video_cthw,
                float *audio_2ct, char *error, size_t error_size) {
  if (error && error_size)
    error[0] = 0;
  if (!store || !text || nt < 1 || steps < 1 || !geom || !video_cthw ||
      !audio_2ct)
    return -1;
  h3_layout_spec spec = {nt,
                         geom->latent_t,
                         geom->latent_h,
                         geom->latent_w,
                         geom->audio_t,
                         geom->frames,
                         NULL,
                         0,
                         NULL,
                         0};
  h3_layout layout;
  char lerr[256];
  if (!h3_layout_build(&spec, &layout, lerr, sizeof(lerr))) {
    if (error && error_size)
      snprintf(error, error_size, "layout: %s", lerr);
    return -1;
  }
  h3_dit_seq_plan plan;
  if (h3_dit_seq_plan_from_layout(&layout, &plan) != 0 || plan.nt != nt ||
      plan.nv != geom->nv || plan.na != geom->na ||
      h3_dit_seq_plan_apply_text_tags(&plan, text_tags, nt) != 0) {
    h3_layout_free(&layout);
    h3_dit_seq_plan_free(&plan);
    return -1;
  }
  const int C = H3_VIDEO_VAE_LATENT_CHANNELS;
  const int Ac = H3_AUDIO_VAE_LATENT_CHANNELS;
  const int F = geom->latent_t, H = geom->latent_h, W = geom->latent_w,
            AT = geom->audio_t;
  float *vlat = (float *)malloc(geom->video_n * sizeof(float));
  float *vrows = (float *)malloc((size_t)plan.nv * (size_t)H3_DIT_VIDEO_PATCH_DIM *
                                 sizeof(float));
  float *alat = (float *)malloc(geom->audio_n * sizeof(float));
  float *arows = (float *)malloc((size_t)plan.na * (size_t)Ac * sizeof(float));
  int rc = 0;
  if (!vlat || !vrows || !alat || !arows)
    rc = -1;
  if (rc == 0) {
    const char *vl = getenv("H3_VIDEO_LATENT");
    const char *al = getenv("H3_AUDIO_LATENT");
    if (vl && vl[0] && al && al[0]) {
      if (!load_f32_file(vl, vlat, geom->video_n) ||
          !load_f32_file(al, alat, geom->audio_n)) {
        if (error && error_size)
          snprintf(error, error_size, "H3_VIDEO_LATENT/H3_AUDIO_LATENT read");
        rc = -1;
      } else {
        fprintf(stderr, "video-c: loaded latents %s + %s\n", vl, al);
      }
    } else {
      h3_rng rng;
      h3_rng_seed(&rng, seed);
      h3_rng_fill_normal(&rng, vlat, geom->video_n);
      h3_rng_fill_normal(&rng, alat, geom->audio_n);
    }
    if (rc == 0)
      rc = h3_dit_patchify_video(vlat, 1, C, F, H, W, H3_DIT_PATCH_T,
                                 H3_DIT_PATCH_H, H3_DIT_PATCH_W, vrows);
  }
  if (rc == 0)
    rc = h3_dit_pack_audio(alat, Ac, AT, arows);
  if (rc == 0)
    rc = h3_dit_denoise(store, vrows, plan.nv, arows, plan.na, text, nt,
                        plan.video_index, plan.audio_index, plan.text_index,
                        plan.tags, plan.position_ids, plan.seq, steps, n_layers,
                        reuse_interval, adaln_t_sigma, error, error_size);
  if (rc == 0)
    rc = h3_dit_unpatchify_video(vrows, 1, C, F, H, W, H3_DIT_PATCH_T,
                                 H3_DIT_PATCH_H, H3_DIT_PATCH_W, vlat);
  if (rc == 0)
    rc = h3_dit_unpack_audio(arows, Ac, AT, alat);
  if (rc == 0) {
    memcpy(video_cthw, vlat, geom->video_n * sizeof(float));
    memcpy(audio_2ct, alat, geom->audio_n * sizeof(float));
  }
  if (rc != 0 && error && error_size && !error[0])
    snprintf(error, error_size, "h3_dit_t2va failed");
  free(vlat);
  free(vrows);
  free(alat);
  free(arows);
  h3_dit_seq_plan_free(&plan);
  h3_layout_free(&layout);
  return rc;
}

static int latent_hw_map(const float *z, int C, int T, int H, int W, float *map) {
  if (!z || !map || C < 1 || T < 1 || H < 1 || W < 1)
    return -1;
  size_t hw = (size_t)H * (size_t)W;
  for (size_t i = 0; i < hw; i++)
    map[i] = 0.f;
  for (int c = 0; c < C; c++) {
    const float *plane = z + ((size_t)c * (size_t)T + 0) * hw;
    for (size_t i = 0; i < hw; i++)
      map[i] += plane[i];
  }
  float inv = 1.f / (float)C;
  for (size_t i = 0; i < hw; i++)
    map[i] *= inv;
  return 0;
}

void h3_dit_log_latent_spatial(const float *z, int C, int T, int H, int W) {
  if (!z || C < 1 || T < 1 || H < 1 || W < 1)
    return;
  size_t hw = (size_t)H * (size_t)W;
  float *map = (float *)malloc(hw * sizeof(float));
  if (!map || latent_hw_map(z, C, T, H, W, map) != 0) {
    free(map);
    return;
  }
  double mean = 0, var = 0, ac1 = 0, acn = 0;
  for (size_t i = 0; i < hw; i++)
    mean += map[i];
  mean /= (double)hw;
  for (size_t i = 0; i < hw; i++) {
    double d = map[i] - mean;
    var += d * d;
  }
  var /= (double)hw;
  int n1 = 0;
  for (int y = 0; y < H; y++) {
    for (int x = 0; x + 1 < W; x++) {
      double a = map[(size_t)y * (size_t)W + (size_t)x] - mean;
      double b = map[(size_t)y * (size_t)W + (size_t)x + 1] - mean;
      ac1 += a * b;
      n1++;
    }
  }
  if (n1)
    acn = (var > 1e-20) ? (ac1 / (double)n1) / var : 0.0;
  double ch_std_sum = 0;
  int t0 = 0;
  for (int c = 0; c < C; c++) {
    const float *plane = z + ((size_t)c * (size_t)T + (size_t)t0) * hw;
    double cm = 0, cv = 0;
    for (size_t i = 0; i < hw; i++)
      cm += plane[i];
    cm /= (double)hw;
    for (size_t i = 0; i < hw; i++) {
      double d = plane[i] - cm;
      cv += d * d;
    }
    ch_std_sum += sqrt(cv / (double)hw);
  }
  fprintf(stderr,
          "video-c: latent spatial %dx%d (t=0 mean_C) std=%.4g ac1=%.3f "
          "per-ch_std=%.4g\n",
          W, H, sqrt(var), acn, ch_std_sum / (double)C);
  fflush(stderr);
  if (H * W <= 16) {
    fprintf(stderr, "video-c: latent mean_C map");
    for (size_t i = 0; i < hw; i++)
      fprintf(stderr, " %.3f", map[i]);
    fprintf(stderr, "\n");
    fflush(stderr);
  }
  free(map);
}

int h3_dit_write_latent_pgm(const float *z, int C, int T, int H, int W,
                            const char *path) {
  if (!z || !path || C < 1 || T < 1 || H < 1 || W < 1)
    return -1;
  size_t hw = (size_t)H * (size_t)W;
  float *map = (float *)malloc(hw * sizeof(float));
  if (!map || latent_hw_map(z, C, T, H, W, map) != 0) {
    free(map);
    return -1;
  }
  float lo = map[0], hi = map[0];
  for (size_t i = 1; i < hw; i++) {
    if (map[i] < lo)
      lo = map[i];
    if (map[i] > hi)
      hi = map[i];
  }
  float span = hi - lo;
  if (span < 1e-8f)
    span = 1.f;
  FILE *fp = fopen(path, "wb");
  if (!fp) {
    free(map);
    return -1;
  }
  fprintf(fp, "P5\n%d %d\n255\n", W, H);
  for (size_t i = 0; i < hw; i++) {
    int v = (int)((map[i] - lo) / span * 255.f + 0.5f);
    if (v < 0)
      v = 0;
    if (v > 255)
      v = 255;
    fputc(v, fp);
  }
  fclose(fp);
  free(map);
  fprintf(stderr, "video-c: wrote latent preview %s (%dx%d)\n", path, W, H);
  return 0;
}

int h3_dit_tiny_t2va(const h3_st_store *store, const float *text, int nt,
                     int steps, int n_layers, int adaln_t_sigma, uint64_t seed,
                     float *video_cthw, float *audio_2ct, char *error,
                     size_t error_size) {
  h3_dit_t2va_geom geom;
  if (h3_dit_t2va_geom_build(H3_DIT_TINY_PIXEL, H3_DIT_TINY_PIXEL,
                             H3_DIT_TINY_FRAMES, &geom) != 0)
    return -1;
  return h3_dit_t2va(store, text, nt, NULL, steps, n_layers, 1, adaln_t_sigma,
                     seed, &geom, video_cthw, audio_2ct, error, error_size);
}
