/* Host-CPU MiniMax-H3 video VAE encoder (causal 3D CNN). */
#ifndef H3_VIDEO_VAE_ENCODE_H
#define H3_VIDEO_VAE_ENCODE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int channels; /* 24 */
  int time;
  int height;
  int width;
  /* Channel-major F32 [C,T,H,W], normalized posterior means. */
  float *values;
} h3_video_latent_host;

void h3_video_latent_host_free(h3_video_latent_host *z);

/*
 * Encode channel-major RGB [3,T,H,W] in [0,1]. H and W multiples of 16,
 * in [32, 512]. Canvas larger than the tile size (default 256, or
 * H3_VAE_TILE_PIXELS) uses overlapping spatial tiles (MLX _encode_clip).
 * Frames 1–17.
 */
int h3_video_vae_encode_host(const char *weight_directory, const float *pixels,
                             int frames, int height, int width,
                             h3_video_latent_host *output, char *error,
                             size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* H3_VIDEO_VAE_ENCODE_H */
