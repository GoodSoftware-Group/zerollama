#include "h3_dit_forward.h"

#include "h3_adaln_host.h"
#include "h3_dit_block.h"
#include "h3_dit_host.h"
#include "h3_host.h"
#include "h3_prof.h"
#include "h3_reuse.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int env_on(const char *name) {
  const char *e = getenv(name);
  return e && e[0] && e[0] != '0';
}

static int dit_progress(void) {
  return env_on("H3_DIT_DEBUG") || env_on("WAN_PROFILE");
}

/* Comfy keeps the packed stream in model dtype (bf16) from patch_proj onward. */
static void pack_cast_bf16(float *x, size_t n) {
  const char *e = getenv("H3_DIT_BF16_ACT");
  if (!e || !e[0] || e[0] == '0' || !x)
    return;
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

static void log_vid_div(const char *tag, const float *packed, const int *vidx,
                        int nv, int dim) {
  if (!packed || !vidx || nv < 2 || dim < 1)
    return;
  int n = nv < 8 ? nv : 8;
  double cos_sum = 0;
  int np = 0;
  for (int i = 0; i < n; i++) {
    const float *a = packed + (size_t)vidx[i] * (size_t)dim;
    double na = 0;
    for (int d = 0; d < dim; d++)
      na += (double)a[d] * a[d];
    na = sqrt(na);
    for (int j = i + 1; j < n; j++) {
      const float *b = packed + (size_t)vidx[j] * (size_t)dim;
      double nb = 0, dot = 0;
      for (int d = 0; d < dim; d++) {
        dot += (double)a[d] * b[d];
        nb += (double)b[d] * b[d];
      }
      nb = sqrt(nb);
      if (na > 1e-12 && nb > 1e-12)
        cos_sum += dot / (na * nb);
      np++;
    }
  }
  const float *r0 = packed + (size_t)vidx[0] * (size_t)dim;
  const float *r1 = packed + (size_t)vidx[1] * (size_t)dim;
  double dlt = 0;
  for (int d = 0; d < dim; d++) {
    double x = (double)r0[d] - r1[d];
    dlt += x * x;
  }
  fprintf(stderr, "video-c: vid-div %s n=%d mean_cos=%.4f row0_vs_1_rms=%.4g\n",
          tag, n, np ? cos_sum / (double)np : 0.0, sqrt(dlt / (double)dim));
  fflush(stderr);
}

static int scatter(float *packed, int seq, int dim, const float *src, int n,
                   const int *index) {
  for (int i = 0; i < n; i++) {
    int p = index[i];
    if (p < 0 || p >= seq)
      return -1;
    memcpy(packed + (size_t)p * (size_t)dim, src + (size_t)i * (size_t)dim,
           (size_t)dim * sizeof(float));
  }
  return 0;
}

static int gather(float *dst, int dim, const float *packed, int n,
                  const int *index, int seq) {
  for (int i = 0; i < n; i++) {
    int p = index[i];
    if (p < 0 || p >= seq)
      return -1;
    memcpy(dst + (size_t)i * (size_t)dim, packed + (size_t)p * (size_t)dim,
           (size_t)dim * sizeof(float));
  }
  return 0;
}

int h3_dit_forward(const h3_st_store *store, const float *video, int nv,
                   const float *audio, int na, const float *text, int nt,
                   const int *video_index, const int *audio_index,
                   const int *text_index, const int *tags,
                   const float *position_ids, int seq, float timestep,
                   const float *row_t, int n_layers, float *video_out,
                   float *audio_out, char *error, size_t error_size) {
  if (error && error_size)
    error[0] = 0;
  if (!store || !tags || !position_ids || seq < 1 || n_layers < 0 ||
      n_layers > H3_DIT_NUM_LAYERS)
    return -1;
  if (nv < 1 || na < 1 || nt < 1 || !video || !audio || !text || !video_index ||
      !audio_index || !text_index || !video_out || !audio_out)
    return -1;

  {
    const st_tensor_t *t = h3_st_store_find(store, "adaln_t_table", NULL);
    if (!t || t->ndim != 2) {
      if (error && error_size)
        snprintf(error, error_size, "missing adaln_t_table");
      return -1;
    }
    if (t->shape[0] != H3_ADALN_TABLE_GRID ||
        t->shape[1] != H3_ADALN_TABLE_RANK) {
      if (error && error_size)
        snprintf(error, error_size,
                 "adaln_t_table is [%lld,%lld]; video-c needs [%d,%d] "
                 "(pruned int8 ConvRot). Comfy pruned_bf16 is [1025,8] — do "
                 "not set H3_DIT_ST to that file",
                 (long long)t->shape[0], (long long)t->shape[1],
                 H3_ADALN_TABLE_GRID, H3_ADALN_TABLE_RANK);
      return -1;
    }
  }

  const int H = H3_DIT_HIDDEN_SIZE;
  const int Vp = H3_DIT_VIDEO_PATCH_DIM;
  const int Ap = 32;
  const int Td = H3_DIT_TEXT_DIM;
  const int grid = H3_ADALN_TABLE_GRID;
  const int rank = H3_ADALN_TABLE_RANK;

  float *packed = (float *)calloc((size_t)seq * (size_t)H, sizeof(float));
  float *tmp = (float *)malloc((size_t)seq * (size_t)H * sizeof(float));
  float *text_h = (float *)malloc((size_t)nt * (size_t)H * sizeof(float));
  float *vid_h = (float *)malloc((size_t)nv * (size_t)H * sizeof(float));
  float *aud_h = (float *)malloc((size_t)na * (size_t)H * sizeof(float));
  if (!packed || !tmp || !text_h || !vid_h || !aud_h) {
    free(packed);
    free(tmp);
    free(text_h);
    free(vid_h);
    free(aud_h);
    return -1;
  }

  double t_ref = h3_prof_now_ms ? h3_prof_now_ms() : 0;
  int rc = h3_dit_linear_named(store, "condition_proj.weight",
                               "condition_proj.bias", text, nt, Td, H, text_h,
                               error, error_size);
  for (int r = 0; r < H3_DIT_TOKEN_REFINER_LAYERS && rc == 0; r++) {
    char pfx[64];
    snprintf(pfx, sizeof(pfx), "token_refiner.blocks.%d.", r);
    rc = h3_dit_plain_block_forward(store, pfx, text_h, nt, tmp, error,
                                    error_size);
    if (rc == 0)
      memcpy(text_h, tmp, (size_t)nt * (size_t)H * sizeof(float));
  }
  if (rc == 0) {
float *nw = (float *)malloc((size_t)H * sizeof(float));
    if (!nw)
      rc = -1;
    else {
      rc = h3_st_store_load_f32(store, "token_refiner.final_norm.weight", nw,
                                (size_t)H, error, error_size);
      if (rc == 0)
        rc = h3_dit_rmsnorm(text_h, nt, H, H3_DIT_NORM_EPS, nw, text_h);
      free(nw);
    }
  }
  if (rc == 0)
    rc = h3_dit_linear_named(store, "video_patch_proj.weight",
                             "video_patch_proj.bias", video, nv, Vp, H, vid_h,
                             error, error_size);
  if (rc == 0)
    rc = h3_dit_linear_named(store, "audio_patch_proj.weight",
                             "audio_patch_proj.bias", audio, na, Ap, H, aud_h,
                             error, error_size);
  if (rc == 0)
    rc = scatter(packed, seq, H, text_h, nt, text_index);
  if (rc == 0)
    rc = scatter(packed, seq, H, vid_h, nv, video_index);
  if (rc == 0)
    rc = scatter(packed, seq, H, aud_h, na, audio_index);
  if (rc == 0) {
    pack_cast_bf16(packed, (size_t)seq * (size_t)H);
    log_vid_div("embed", packed, video_index, nv, H);
    const char *ed = getenv("H3_DUMP_EMBED");
    if (ed && ed[0] && ed[0] != '0') {
      const char *dir = (ed[0] == '1' && !ed[1]) ? "/tmp/h3_embed" : ed;
      char cmd[320], path[768];
      snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", dir);
      if (system(cmd) == 0) {
        FILE *f;
        snprintf(path, sizeof(path), "%s/packed.bin", dir);
        f = fopen(path, "wb");
        if (f) {
          fwrite(packed, sizeof(float), (size_t)seq * (size_t)H, f);
          fclose(f);
        }
        snprintf(path, sizeof(path), "%s/pos.bin", dir);
        f = fopen(path, "wb");
        if (f) {
          fwrite(position_ids, sizeof(float), (size_t)seq * 3, f);
          fclose(f);
        }
        snprintf(path, sizeof(path), "%s/meta.txt", dir);
        f = fopen(path, "w");
        if (f) {
          fprintf(f, "seq=%d H=%d nv=%d na=%d nt=%d\n", seq, H, nv, na, nt);
          fclose(f);
        }
        fprintf(stderr, "video-c: dumped embed packed/pos to %s seq=%d\n", dir,
                seq);
      }
    }
  }
  if (rc == 0 && position_ids && env_on("H3_DIT_DEBUG")) {
    int shown[3] = {0, 0, 0};
    fprintf(stderr, "video-c: rope pos");
    for (int s = 0; s < seq; s++) {
      int tag = tags[s];
      int slot = tag == H3_ADALN_TAG_TEXT ? 0
                 : tag == H3_ADALN_TAG_AUDIO ? 1
                 : tag == H3_ADALN_TAG_VIDEO ? 2
                                             : -1;
      if (slot < 0 || shown[slot] >= 2)
        continue;
      shown[slot]++;
      const float *p = position_ids + (size_t)s * 3;
      fprintf(stderr, " %s[%d]=(%.3g,%.3g,%.3g)",
              slot == 0 ? "txt" : slot == 1 ? "aud" : "vid", s, p[0], p[1],
              p[2]);
    }
    fprintf(stderr, "\n");
    fflush(stderr);
  }
  free(text_h);
  free(vid_h);
  free(aud_h);
  text_h = vid_h = aud_h = NULL;

  float *table = (float *)malloc((size_t)grid * (size_t)rank * sizeof(float));
  if (!table)
    rc = -1;
  if (rc == 0)
    rc = h3_st_store_load_f32(store, "adaln_t_table", table,
                              (size_t)grid * (size_t)rank, error, error_size);
  if (t_ref > 0 && h3_prof_add_ms)
    h3_prof_add_ms("h3_dit_fwd_refiner", h3_prof_now_ms() - t_ref);
  double t_blk = h3_prof_now_ms ? h3_prof_now_ms() : 0;
  for (int b = 0; b < n_layers && rc == 0; b++) {
    double t_l = h3_prof_now_ms ? h3_prof_now_ms() : 0;
    if (n_layers > 1 && dit_progress()) {
      fprintf(stderr, "video-c: dit layer %d/%d seq=%d\n", b + 1, n_layers, seq);
      fflush(stderr);
    }
    rc = h3_dit_block_forward(store, b, packed, seq, tags, timestep, row_t,
                              position_ids, table, grid, rank, tmp, error,
                              error_size);
    if (t_l > 0 && getenv("H3_BLK_MS")) {
      fprintf(stderr, "video-c: layer %d took %.1f ms\n", b,
              h3_prof_now_ms() - t_l);
      fflush(stderr);
    }
    if (rc == 0) {
      memcpy(packed, tmp, (size_t)seq * (size_t)H * sizeof(float));
      {
        const char *ctr = getenv("H3_RES_CENTER");
        if (ctr && ctr[0] && ctr[0] != '0') {
          /* Lab. `1` = mean over all rows; `v` = mean over video rows only. */
          int video_only = (ctr[0] == 'v' || ctr[0] == 'V');
          int nuse = video_only ? nv : seq;
          const int *ix = video_only ? video_index : NULL;
          if (nuse > 1) {
            for (int d = 0; d < H; d++) {
              double m = 0;
              for (int i = 0; i < nuse; i++) {
                int s = ix ? ix[i] : i;
                m += packed[(size_t)s * (size_t)H + (size_t)d];
              }
              float mf = (float)(m / (double)nuse);
              if (ix) {
                for (int i = 0; i < nuse; i++)
                  packed[(size_t)ix[i] * (size_t)H + (size_t)d] -= mf;
              } else {
                for (int s = 0; s < seq; s++)
                  packed[(size_t)s * (size_t)H + (size_t)d] -= mf;
              }
            }
          }
        }
      }
      if (b == 0 || b + 1 == n_layers)
        log_vid_div(b == 0 ? "after_L0" : "after_Llast", packed, video_index, nv,
                    H);
      if (rc == 0 && b + 1 == n_layers) {
        const char *dump = getenv("H3_DUMP_LAST");
        if (dump && dump[0] && dump[0] != '0') {
          const char *dir =
              (dump[0] == '1' && dump[1] == 0) ? "/tmp/h3_last" : dump;
          char cmd[320];
          snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", dir);
          if (system(cmd) == 0) {
            char path[768];
            snprintf(path, sizeof(path), "%s/meta.txt", dir);
            FILE *f = fopen(path, "w");
            if (f) {
              fprintf(f, "seq=%d H=%d nv=%d n_layers=%d\n", seq, H, nv,
                      n_layers);
              fprintf(f, "vidx");
              for (int i = 0; i < nv; i++)
                fprintf(f, " %d", video_index[i]);
              fprintf(f, "\n");
              fclose(f);
            }
            snprintf(path, sizeof(path), "%s/vid.bin", dir);
            f = fopen(path, "wb");
            if (f) {
              for (int i = 0; i < nv; i++)
                fwrite(packed + (size_t)video_index[i] * (size_t)H,
                       sizeof(float), (size_t)H, f);
              fclose(f);
            }
            fprintf(stderr, "video-c: dumped last-layer video rows to %s\n",
                    dir);
          }
        }
      }
      if (getenv("H3_DIT_DEBUG") && getenv("H3_DIT_DEBUG")[0] &&
          getenv("H3_DIT_DEBUG")[0] != '0') {
        double acc = 0;
        size_t n = (size_t)seq * (size_t)H;
        for (size_t i = 0; i < n; i++)
          acc += (double)packed[i] * (double)packed[i];
        fprintf(stderr, "video-c: dit-debug after layer %d x_rms=%.4g\n", b,
                sqrt(acc / (double)n));
      }
    }
  }

  if (t_blk > 0 && h3_prof_add_ms)
    h3_prof_add_ms("h3_dit_fwd_blocks", h3_prof_now_ms() - t_blk);
  double t_fin = h3_prof_now_ms ? h3_prof_now_ms() : 0;

  /* Final AdaLN: per-row t → shift||scale [T, 2H]. Comfy unique_t is sorted. */
  float uniq[8];
  int nuniq = 0;
  int *tslots = (int *)malloc((size_t)seq * sizeof(int));
  if (!tslots)
    rc = -1;
  if (rc == 0) {
    nuniq = h3_adaln_collect_timesteps(row_t, timestep, seq, uniq, 8, tslots);
    if (nuniq < 1)
      rc = -1;
  }
  float *emb = NULL;
  float *fin = NULL;
  float *nw = (float *)malloc((size_t)H * sizeof(float));
  int st_slot = 0;
  double st_scale_rms = 0.0, st_shift_rms = 0.0;
  if (rc == 0 && nuniq > 0) {
    emb = (float *)malloc((size_t)nuniq * (size_t)rank * sizeof(float));
    fin = (float *)malloc((size_t)nuniq * (size_t)H3_ADALN_FINAL_OUT_FEATURES *
                          sizeof(float));
    if (!emb || !fin || !nw)
      rc = -1;
    for (int u = 0; u < nuniq && rc == 0; u++)
      rc = h3_adaln_table_embed(table, grid, rank, uniq[u],
                                emb + (size_t)u * (size_t)rank);
    if (rc == 0) {
      const char *silu = getenv("H3_ADALN_SILU");
      if (silu && silu[0] && silu[0] != '0')
        h3_dit_silu(emb, emb, (size_t)nuniq * (size_t)rank);
    }
    if (rc == 0)
      rc = h3_dit_linear_named(store, "final_layer.adaln_proj.linear.weight",
                               "final_layer.adaln_proj.linear.bias", emb, nuniq,
                               rank, H3_ADALN_FINAL_OUT_FEATURES, fin, error,
                               error_size);
  }
  if (rc == 0)
    rc = h3_st_store_load_f32(store, "final_layer.norm.weight", nw, (size_t)H,
                              error, error_size);
  const char *stage_dump = getenv("H3_DUMP_STAGES");
  float *st_raw = NULL;
  if (rc == 0 && stage_dump && stage_dump[0] && stage_dump[0] != '0') {
    st_raw = (float *)malloc((size_t)seq * (size_t)H * sizeof(float));
    if (st_raw)
      memcpy(st_raw, packed, (size_t)seq * (size_t)H * sizeof(float));
  }
  if (rc == 0)
    rc = h3_dit_rmsnorm(packed, seq, H, H3_DIT_NORM_EPS, nw, tmp);
  if (rc == 0) {
    const char *dump = getenv("H3_DUMP_LAST");
    if (dump && dump[0] && dump[0] != '0') {
      const char *dir =
          (dump[0] == '1' && dump[1] == 0) ? "/tmp/h3_last" : dump;
      char path[768];
      snprintf(path, sizeof(path), "%s/vid_norm.bin", dir);
      FILE *f = fopen(path, "wb");
      if (f) {
        for (int i = 0; i < nv; i++)
          fwrite(tmp + (size_t)video_index[i] * (size_t)H, sizeof(float),
                 (size_t)H, f);
        fclose(f);
      }
    }
  }
  if (rc == 0) {
    float *shift = (float *)malloc((size_t)nuniq * (size_t)H * sizeof(float));
    float *scale = (float *)malloc((size_t)nuniq * (size_t)H * sizeof(float));
    if (!shift || !scale)
      rc = -1;
    if (rc == 0)
      rc = h3_adaln_split_final(fin, nuniq, H, shift, scale);
    {
      const char *skip = getenv("H3_SKIP_FINAL_ADALN");
      if (rc == 0 && skip && skip[0] && skip[0] != '0')
        memcpy(packed, tmp, (size_t)seq * (size_t)H * sizeof(float));
      else if (rc == 0)
        rc = h3_dit_modulate_indexed(tmp, shift, scale, tslots, seq, H, packed);
    }
    if (rc == 0)
      log_vid_div("after_final_adaln", packed, video_index, nv, H);
    st_slot = tslots ? tslots[audio_index[0]] : 0;
    st_scale_rms = 0.0;
    st_shift_rms = 0.0;
    {
      const float *scp = scale + (size_t)st_slot * (size_t)H;
      const float *shp = shift + (size_t)st_slot * (size_t)H;
      for (int j = 0; j < H; j++) {
        st_scale_rms += (double)scp[j] * (double)scp[j];
        st_shift_rms += (double)shp[j] * (double)shp[j];
      }
      st_scale_rms = sqrt(st_scale_rms / (double)H);
      st_shift_rms = sqrt(st_shift_rms / (double)H);
    }
    free(shift);
    free(scale);
  }
  free(nw);
  free(fin);
  free(emb);
  free(tslots);
  free(table);

  float *vhead = (float *)malloc((size_t)seq * (size_t)Vp * sizeof(float));
  float *ahead = (float *)malloc((size_t)seq * (size_t)Ap * sizeof(float));
  if (!vhead || !ahead)
    rc = -1;
  if (rc == 0)
    rc = h3_dit_linear_named(store, "final_layer.video_out.weight",
                             "final_layer.video_out.bias", packed, seq, H, Vp,
                             vhead, error, error_size);
  if (rc == 0)
    rc = h3_dit_linear_named(store, "final_layer.audio_out.weight",
                             "final_layer.audio_out.bias", packed, seq, H, Ap,
                             ahead, error, error_size);
  if (rc == 0)
    rc = gather(video_out, Vp, vhead, nv, video_index, seq);
  if (rc == 0)
    log_vid_div("video_out", vhead, video_index, nv, Vp);
  if (rc == 0)
    rc = gather(audio_out, Ap, ahead, na, audio_index, seq);

  if (rc == 0 && stage_dump && stage_dump[0] && stage_dump[0] != '0') {
    double sr = 0.0, nr = 0.0, hr = 0.0, vr = 0.0;
    long cnt = 0;
    for (int i = 0; i < na; i++) {
      const float *p = st_raw ? st_raw + (size_t)audio_index[i] * (size_t)H : NULL;
      const float *n = tmp + (size_t)audio_index[i] * (size_t)H;
      const float *hh = packed + (size_t)audio_index[i] * (size_t)H;
      const float *v = ahead + (size_t)audio_index[i] * (size_t)Ap;
      for (int j = 0; j < H; j++) {
        if (p) sr += (double)p[j] * (double)p[j];
        nr += (double)n[j] * (double)n[j];
        hr += (double)hh[j] * (double)hh[j];
      }
      for (int j = 0; j < Ap; j++)
        vr += (double)v[j] * (double)v[j];
      cnt += H;
    }
    int slot = st_slot;
    fprintf(stderr,
            "video-c STAGES: nuniq=%d slot=%d h_audio_rms=%.6g "
            "norm_audio_rms=%.6g scale_rms=%.6g shift_rms=%.6g "
            "ha_rms=%.6g vel_audio_rms=%.6g\n",
            nuniq, slot, sqrt(sr / (double)cnt), sqrt(nr / (double)cnt),
            st_scale_rms, st_shift_rms,
            sqrt(hr / (double)cnt), sqrt(vr / (double)(cnt / H * Ap)));
  }
  if (st_raw)
    free(st_raw);

  if (rc != 0 && error && error_size && !error[0])
    snprintf(error, error_size, "h3_dit_forward failed");
  free(vhead);
  free(ahead);
  free(packed);
  free(tmp);
  if (t_fin > 0 && h3_prof_add_ms)
    h3_prof_add_ms("h3_dit_fwd_final", h3_prof_now_ms() - t_fin);
  return rc;
}

static void fill_row_t(float *row_t, const int *tags, int seq, float tv,
                       float ta) {
  for (int s = 0; s < seq; s++)
    row_t[s] = (tags[s] == H3_ADALN_TAG_AUDIO) ? ta : tv;
}

int h3_dit_denoise(const h3_st_store *store, float *video, int nv, float *audio,
                   int na, const float *text, int nt, const int *video_index,
                   const int *audio_index, const int *text_index,
                   const int *tags, const float *position_ids, int seq,
                   int steps, int n_layers, int reuse_interval,
                   int adaln_t_sigma, char *error, size_t error_size) {
  if (!store || !video || !audio || steps < 1)
    return -1;
  if (reuse_interval < 1)
    reuse_interval = 1;
  uint8_t selected[H3_MAX_STEPS];
  int n_eval = h3_dit_reuse_schedule(steps, reuse_interval, selected,
                                     sizeof(selected));
  if (n_eval < 0)
    return -1;
  h3_sigma_schedule sched;
  if (steps >= 2) {
    if (!h3_serving_schedule_build(steps, &sched))
      return -1;
  } else if (!h3_schedule_build(steps, &sched)) {
    return -1;
  }
  const int Vp = H3_DIT_VIDEO_PATCH_DIM;
  const int Ap = 32;
  size_t vn = (size_t)nv * (size_t)Vp;
  size_t an = (size_t)na * (size_t)Ap;
  float *vpred = (float *)malloc(vn * sizeof(float));
  float *apred = (float *)malloc(an * sizeof(float));
  float *vpred_prev = NULL;
  float *apred_prev = NULL;
  float *row_t = (float *)malloc((size_t)seq * sizeof(float));
  int use_res = 0;
  {
    const char *sm = getenv("H3_SAMPLER");
    use_res = sm && strcmp(sm, "res_multistep") == 0;
  }
  float *vden = NULL, *vden_old = NULL, *vscratch = NULL;
  float *aden = NULL, *aden_old = NULL, *ascratch = NULL;
  if (reuse_interval > 1) {
    vpred_prev = (float *)malloc(vn * sizeof(float));
    apred_prev = (float *)malloc(an * sizeof(float));
  }
  if (use_res) {
    vden = (float *)malloc(vn * sizeof(float));
    vden_old = (float *)malloc(vn * sizeof(float));
    vscratch = (float *)malloc(vn * sizeof(float));
    aden = (float *)malloc(an * sizeof(float));
    aden_old = (float *)malloc(an * sizeof(float));
    ascratch = (float *)malloc(an * sizeof(float));
  }
  if (!vpred || !apred || !row_t ||
      (reuse_interval > 1 && (!vpred_prev || !apred_prev)) ||
      (use_res && (!vden || !vden_old || !vscratch || !aden || !aden_old ||
                   !ascratch))) {
    free(vpred);
    free(apred);
    free(vpred_prev);
    free(apred_prev);
    free(row_t);
    free(vden);
    free(vden_old);
    free(vscratch);
    free(aden);
    free(aden_old);
    free(ascratch);
    return -1;
  }
  int rc = 0;
  int last_evaluated = -1;
  int previous_evaluated = -1;
  clock_t t0 = clock();
  fprintf(stderr,
          "video-c: dit schedule %s evals=%d/%d reuse=%d sampler=%s sigma0=%.4g "
          "sigma_last=%.4g\n",
          steps >= 2 ? "linspace" : "antirez-1000", n_eval, steps,
          reuse_interval, use_res ? "res_multistep" : "euler", sched.video[0],
          sched.video[steps]);
  fflush(stderr);
  int use_sigma = adaln_t_sigma > 0;
  if (adaln_t_sigma < 0) {
    const char *use_s = getenv("H3_ADALN_T_SIGMA");
    use_sigma = use_s && use_s[0] && use_s[0] != '0';
  }
  for (int i = 0; i < steps && rc == 0; i++) {
    int evaluate = selected[i];
    fprintf(stderr, "video-c: dit step %d/%d %s layers=%d seq=%d nv=%d na=%d\n",
            i + 1, steps, evaluate ? "eval" : "reuse", n_layers, seq, nv, na);
    fflush(stderr);
    float sv = sched.video[i];
    float sa = sched.audio[i];
    /* Default: data-time t=1-σ (MLX / distilled Euler). t=σ indexes the table by σ. */
    float tv = use_sigma ? sv : 1.f - sv;
    float ta = use_sigma ? sa : 1.f - sa;
    fill_row_t(row_t, tags, seq, tv, ta);
    if (evaluate) {
      if (last_evaluated >= 0 && reuse_interval > 1) {
        memcpy(vpred_prev, vpred, vn * sizeof(float));
        memcpy(apred_prev, apred, an * sizeof(float));
        previous_evaluated = last_evaluated;
      }
      double t_eval = h3_prof_now_ms ? h3_prof_now_ms() : 0;
      rc = h3_dit_forward(store, video, nv, audio, na, text, nt, video_index,
                          audio_index, text_index, tags, position_ids, seq, tv,
                          row_t, n_layers, vpred, apred, error, error_size);
      if (t_eval > 0 && h3_prof_add_ms)
        h3_prof_add_ms(last_evaluated < 0 ? "h3_dit_eval1" : "h3_dit_eval2",
                       h3_prof_now_ms() - t_eval);
      if (rc == 0)
        last_evaluated = i;
    } else if (last_evaluated >= 0) {
      h3_dit_extrapolate_velocity(vpred, vpred, vpred_prev, vn, sv,
                                  sched.video[last_evaluated],
                                  previous_evaluated >= 0
                                      ? sched.video[previous_evaluated]
                                      : 0.f,
                                  previous_evaluated >= 0);
      h3_dit_extrapolate_velocity(apred, apred, apred_prev, an, sa,
                                  sched.audio[last_evaluated],
                                  previous_evaluated >= 0
                                      ? sched.audio[previous_evaluated]
                                      : 0.f,
                                  previous_evaluated >= 0);
    } else {
      rc = -1;
    }
    if (rc == 0) {
      double vs = 0, xs = 0;
      for (size_t j = 0; j < vn; j++) {
        vs += (double)vpred[j] * (double)vpred[j];
        xs += (double)video[j] * (double)video[j];
      }
      double vel = sqrt(vs / (double)vn);
      fprintf(stderr,
              "video-c: dit step %d/%d vel_rms=%.4g latent_rms=%.4g sigma=%.4g\n",
              i + 1, steps, vel, sqrt(xs / (double)vn), sv);
      {
        /* Lab. Target velocity RMS (VAE-encoded orange ≈ 1.3). Default off. */
        const char *vr = getenv("H3_VEL_RMS");
        if (vr && vr[0] && vel > 1e-8) {
          float want = strtof(vr, NULL);
          if (want > 0.f) {
            float s = want / (float)vel;
            for (size_t j = 0; j < vn; j++)
              vpred[j] *= s;
            double as = 0;
            for (size_t j = 0; j < an; j++)
              as += (double)apred[j] * (double)apred[j];
            double ar = sqrt(as / (double)an);
            if (ar > 1e-8) {
              float sa = want / (float)ar;
              for (size_t j = 0; j < an; j++)
                apred[j] *= sa;
            }
            fprintf(stderr, "video-c: H3_VEL_RMS=%.4g scale_v=%.4g\n", want, s);
          }
        }
      }
      if (use_res) {
        h3_const_denoised_from_host_velocity(vden, video, vpred, vn, sv);
        h3_const_denoised_from_host_velocity(aden, audio, apred, an, sa);
        if (!h3_res_step(vscratch, video, vden, i > 0 ? vden_old : NULL, vn,
                         sched.video, i, steps) ||
            !h3_res_step(ascratch, audio, aden, i > 0 ? aden_old : NULL, an,
                         sched.audio, i, steps))
          rc = -1;
        else {
          memcpy(video, vscratch, vn * sizeof(float));
          memcpy(audio, ascratch, an * sizeof(float));
          memcpy(vden_old, vden, vn * sizeof(float));
          memcpy(aden_old, aden, an * sizeof(float));
        }
      } else {
        if (!h3_euler_velocity_step(video, vpred, vn, sv, sched.video[i + 1]))
          rc = -1;
        if (!h3_euler_velocity_step(audio, apred, an, sa, sched.audio[i + 1]))
          rc = -1;
      }
    }
    double sec = (double)(clock() - t0) / (double)CLOCKS_PER_SEC;
    fprintf(stderr, "video-c: dit step %d/%d done (%.1fs elapsed)\n", i + 1,
            steps, sec);
    fflush(stderr);
  }
  free(vpred);
  free(apred);
  free(vpred_prev);
  free(apred_prev);
  free(row_t);
  free(vden);
  free(vden_old);
  free(vscratch);
  free(aden);
  free(aden_old);
  free(ascratch);
  return rc;
}
