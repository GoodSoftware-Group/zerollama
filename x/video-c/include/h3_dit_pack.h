/* Map h3_layout segments onto packed DiT indices / AdaLN tags. */
#ifndef H3_DIT_PACK_H
#define H3_DIT_PACK_H

#include "h3_host.h"
#include "h3_st_store.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int seq;
  int nt;
  int na;
  int nv;
  int *tags;
  float *position_ids; /* [seq, 3] t,h,w */
  int *text_index;
  int *audio_index;
  int *video_index;
} h3_dit_seq_plan;

void h3_dit_seq_plan_free(h3_dit_seq_plan *plan);

/* TEXT → text; AUDIO (not ref) → audio; VIDEO (not cond/ref) → video. */
int h3_dit_seq_plan_from_layout(const h3_layout *layout, h3_dit_seq_plan *plan);
/* Overlay MiniMax text-span AdaLN tags (0/1/2). NULL is a no-op. */
int h3_dit_seq_plan_apply_text_tags(h3_dit_seq_plan *plan, const int *tags,
                                    int nt);

enum {
  H3_DIT_TINY_FRAMES = 5,
  H3_DIT_TINY_PIXEL = 32,
  H3_DIT_TINY_LATENT_T = 2,
  H3_DIT_TINY_LATENT_HW = 2,
  H3_DIT_TINY_AUDIO_T = 8,
  H3_DIT_CANVAS_768 = 768
};

typedef struct {
  int pixel_w;
  int pixel_h;
  int frames;
  int latent_w;
  int latent_h;
  int latent_t;
  int audio_t;
  int nv;
  int na;
  size_t video_n; /* C * T * H * W */
  size_t audio_n; /* 2 * 32 * audio_t */
} h3_dit_t2va_geom;

int h3_dit_t2va_geom_build(int pixel_w, int pixel_h, int frames,
                           h3_dit_t2va_geom *geom);

/*
 * T2VA Euler on `geom` canvas. text [nt, 5120]. text_tags NULL = all text.
 * video_cthw [24, T, H, W], audio_2ct [2, 32, audio_t].
 */
int h3_dit_t2va(const h3_st_store *store, const float *text, int nt,
                const int *text_tags, int steps, int n_layers,
                int reuse_interval, int adaln_t_sigma, uint64_t seed,
                const h3_dit_t2va_geom *geom, float *video_cthw,
                float *audio_2ct, char *error, size_t error_size);

/* Tiny T2VA: 5 frames, 32² canvas, latent 2×2×2, audio T=8. */
int h3_dit_tiny_t2va(const h3_st_store *store, const float *text, int nt,
                     int steps, int n_layers, int adaln_t_sigma, uint64_t seed,
                     float *video_cthw, float *audio_2ct, char *error,
                     size_t error_size);

/* Channel-major [C,T,H,W]: stderr spatial std / lag-1 autocorrelation (T=0 map). */
void h3_dit_log_latent_spatial(const float *z, int C, int T, int H, int W);
/* 8-bit PGM of |mean_C| at t=0, HxW. */
int h3_dit_write_latent_pgm(const float *z, int C, int T, int H, int W,
                            const char *path);

#ifdef __cplusplus
}
#endif

#endif /* H3_DIT_PACK_H */
