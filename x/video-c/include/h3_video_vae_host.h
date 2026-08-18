/* MiniMax-H3 video VAE geometry + ViT decoder kernels (MLX / antirez rematch). */
#ifndef H3_VIDEO_VAE_HOST_H
#define H3_VIDEO_VAE_HOST_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  H3_VIDEO_VAE_IN_CHANNELS = 3,
  H3_VIDEO_VAE_LATENT_CHANNELS = 24,
  H3_VIDEO_VAE_LEVELS = 6,
  H3_VIDEO_VAE_LAYERS_PER_BLOCK = 2,
  H3_VIDEO_VAE_NORM_GROUPS = 32,
  H3_VIDEO_VAE_DECODER_LAYERS = 36,
  H3_VIDEO_VAE_DECODER_HEADS = 32,
  H3_VIDEO_VAE_DECODER_HEAD_DIM = 64,
  H3_VIDEO_VAE_DECODER_REGISTERS = 4,
  H3_VIDEO_VAE_DECODER_FFN_MULT = 4,
  H3_VIDEO_VAE_CLIP_LENGTH = 17,
  H3_VIDEO_VAE_TOKEN_DROP = 3,
  H3_VIDEO_VAE_MOMENT_CHANNELS = 48,
  H3_VIDEO_VAE_TILE_PIXELS = 256,
  H3_VIDEO_VAE_TILE_OVERLAP_MIN = 64,
  H3_VIDEO_VAE_CHUNK_LATENT_T = 7,
  H3_VIDEO_VAE_DECODER_SUFFIX = 5, /* 4 registers + 1 zero token */
  H3_VIDEO_VAE_ROPE_HALF = 24,
  H3_VIDEO_VAE_FFN = 8192,
  H3_VIDEO_VAE_FRAME_OFFSET = 3,
  H3_VIDEO_VAE_FIRST_CHUNK_FRAMES = 22
};

#define H3_VIDEO_VAE_DECODER_DIM                                               \
  (H3_VIDEO_VAE_DECODER_HEADS * H3_VIDEO_VAE_DECODER_HEAD_DIM) /* 2048 */
#define H3_VIDEO_VAE_DECODER_ROPE_THETA 100.0f
#define H3_VIDEO_VAE_OUTPUT_PATCH                                              \
  (3 * 4 * 16 * 16) /* 3072 = C * pt * ph * pw */

extern const int h3_video_vae_block_out_channels[H3_VIDEO_VAE_LEVELS];
extern const int h3_video_vae_spatial_downsample[H3_VIDEO_VAE_LEVELS];
extern const int h3_video_vae_temporal_downsample[H3_VIDEO_VAE_LEVELS];

int h3_video_vae_spatial_ratio(void);  /* 16 */
int h3_video_vae_temporal_ratio(void); /* 4 */
int h3_video_vae_decoder_dim(void);
/* ceil(clip_length / temporal) — encoder tokens per clip before drop. */
int h3_video_vae_tokens_chunk_size(void);
int h3_video_vae_latent_hw(int pixels);
/* T=2 → 5 frames; T=7 → 22 (first chunk). Else -1. */
int h3_video_vae_first_chunk_frames(int latent_time);
/* T=2 → 5; T>=7 and (T-2)%5==0 → chunks*17+5. Else -1. */
int h3_video_vae_output_frames(int latent_time);
/* Blend `overlap_frames` HWC frames: current[f] = overlap*(1-α)+current*α, α=f/N. */
int h3_video_vae_temporal_overlap_blend_f32(float *current, const float *overlap,
                                           int overlap_frames,
                                           size_t frame_elements);

typedef struct {
  int count;
  int length; /* pixel length of one tile */
  int *starts;
  int *overlaps; /* count-1 entries when count>1 */
} h3_video_vae_tile_axis;

void h3_video_vae_tile_axis_free(h3_video_vae_tile_axis *axis);
int h3_video_vae_tile_overlap(int tile_pixels); /* tile/4, ≥16, multiple of 16 */
int h3_video_vae_tile_count_for_extent(int extent, int tile_pixels);
int h3_video_vae_configured_tile_pixels(int pixel_height, int pixel_width);
int h3_video_vae_tile_axis_build(int extent, int tile_pixels,
                                 h3_video_vae_tile_axis *axis);
/* Channel-major [C,T,H,W] crop. starts are latent (not pixel) indices. */
int h3_video_vae_extract_latent_f32(const float *latent, int full_t, int full_h,
                                    int full_w, int start_t, int start_y,
                                    int start_x, int tile_t, int tile_h,
                                    int tile_w, float *tile);
/* Channel-major RGB [3,T,H,W] pixel crop. */
int h3_video_vae_extract_rgb_f32(const float *pixels, int frames, int full_h,
                                 int full_w, int start_y, int start_x,
                                 int tile_h, int tile_w, float *tile);
/* HWC tiles[ty * nx + tx] each [frames, tile_h, tile_w, 3] → full canvas. */
int h3_video_vae_stitch_tiles_f32(float **tiles, const h3_video_vae_tile_axis *y,
                                  const h3_video_vae_tile_axis *x, int frames,
                                  float *rgb);
/* Pixel-space tile axes; CTHW latent tiles (spatial = pixels/16). */
int h3_video_vae_stitch_latent_tiles_f32(float **tiles,
                                         const h3_video_vae_tile_axis *y,
                                         const h3_video_vae_tile_axis *x,
                                         int channels, int time, float *latent);

/* PyTorch reflect (no edge repeat): -1 → 1, n → n-2. */
int h3_video_vae_reflect_coord(int coordinate, int length);

/*
 * Causal pad: zero frames prepended on D; spatial reflect.
 * in/out NDHWC, batch=1 implied in pointers (or batch folded into D).
 */
int h3_video_vae_pad_ndhwc_f32(float *dst, const float *src, int batch, int depth,
                               int height, int width, int channels,
                               int depth_front, int height_before,
                               int height_after, int width_before,
                               int width_after);

/* Conv3d NDHWC × OIDHW, no padding (apply pad first). bias may be NULL. */
int h3_video_vae_conv3d_f32(float *dst, const float *src, const float *weight,
                            const float *bias, int batch, int depth, int height,
                            int width, int in_ch, int out_ch, int kd, int kh,
                            int kw, int stride_t, int stride_h, int stride_w);

/* T-isolated group norm + SiLU. NDHWC, groups divide channels. */
int h3_video_vae_group_norm_silu_f32(float *dst, const float *src,
                                     const float *weight, const float *bias,
                                     int batch, int depth, int height, int width,
                                     int channels, int groups, float eps);

/* ViT decoder kernels (antirez rematch). Q/K/V layout [seq, heads, dim]. */
int h3_video_vae_qkv_rope_f32(float *query, float *key, float *value,
                              const float *qkv, const float *rope_cos,
                              const float *rope_sin, int seq, int heads,
                              int head_dim, int rope_half, float eps);
int h3_video_vae_sdpa_f32(float *out, const float *q, const float *k,
                          const float *v, int seq, int heads, int head_dim,
                          float scale);
int h3_video_vae_swiglu_f32(float *dst, const float *fused, int rows, int width);
int h3_video_vae_scale_add_f32(float *dst, const float *residual,
                               const float *branch, const float *scale, int rows,
                               int dim);

#ifdef __cplusplus
}
#endif

#endif /* H3_VIDEO_VAE_HOST_H */
