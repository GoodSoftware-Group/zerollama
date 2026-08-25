/* Host-CPU MiniMax-H3 video VAE ViT decoder (antirez first-chunk rematch). */
#ifndef H3_VIDEO_VAE_DECODE_H
#define H3_VIDEO_VAE_DECODE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int frames;
  int height;
  int width;
  /* Packed RGB, frame-major HWC in [0,1]. */
  float *rgb;
} h3_video_frames_host;

void h3_video_frames_host_free(h3_video_frames_host *frames);

/*
 * Decode channel-major posterior [C,T,H,W]. T==2 → 5 frames; T>=7 and
 * (T-2)%5==0 → chunks×17+5 with spatial tiles when canvas > 256 px.
 * Host ViT tiles larger than 4×4 latent need H3_VAE_ALLOW_LARGE=1.
 */
int h3_video_vae_decode_host(const char *weight_directory,
                             const float *normalized_latent, int latent_time,
                             int latent_height, int latent_width,
                             h3_video_frames_host *output, char *error,
                             size_t error_size);

/* Repeat last time index: channel-major [C,src_t,H,W] → [C,dst_t,H,W]. */
int h3_video_vae_repeat_last_time(const float *src, int channels, int src_t,
                                  int height, int width, int dst_t, float *dst);

/* Write one HWC frame as binary PPM (P6). */
int h3_video_frames_write_ppm(const h3_video_frames_host *frames, int index,
                              const char *path, char *error, size_t error_size);
/* Read a binary P6 PPM into one HWC frame in [0,1]. */
int h3_video_frames_read_ppm(const char *path, h3_video_frames_host *frames,
                             char *error, size_t error_size);
/* Pack frames as channel-major RGB [3,T,H,W]; repeats frame 0 if T > frames. */
int h3_video_frames_to_cthw(const h3_video_frames_host *frames, int time,
                            float *cthw);

#ifdef __cplusplus
}
#endif

#endif /* H3_VIDEO_VAE_DECODE_H */
