#ifndef MUSIC_DAV_H
#define MUSIC_DAV_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int channels;
  int samples;
  int sample_rate;
  float *pcm; /* [channels * samples], channel-major */
} music3_wav_host;

void music3_wav_host_free(music3_wav_host *w);
int music3_wav_write(const music3_wav_host *w, const char *path, char *error,
                     size_t error_size);

/* Snake1d: x + sin(a x)^2 / (a + 1e-9). Layout [batch, channels, length] CHW. */
int music3_snake1d_f32(float *dst, const float *input, const float *alpha,
                       int batch, int channels, int length);

/*
 * Synthetic DAV: zeros latent [128, T] -> stereo PCM at 44.1 kHz, length T*512.
 * Why not call H3 AudioVAE: hop 800 / BigVGAN vs Music hop 512 DAC at 44.1 kHz.
 * Real dav.pth decode is skipped unless MUSIC3_DAV_SAFETENSORS is set later.
 */
int music3_dav_synthetic_decode(int latent_t, music3_wav_host *out, char *error,
                                size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
