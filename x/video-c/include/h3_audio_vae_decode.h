/* Host-CPU MiniMax-H3 AudioVAE BigVGAN decode (antirez h3_audio_vae rematch). */
#ifndef H3_AUDIO_VAE_DECODE_H
#define H3_AUDIO_VAE_DECODE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int channels;    /* 2 */
  int samples;     /* latent_t * hop */
  int sample_rate; /* 32000 */
  /* Channel-major F32 [channels, samples], clipped to [-1,1]. */
  float *pcm;
} h3_audio_waveform_host;

void h3_audio_waveform_host_free(h3_audio_waveform_host *w);

typedef struct {
  int channels; /* 32 */
  int stereo;   /* 2 */
  int length;   /* T = pad(samples)/hop */
  /* Channel-major F32 [32,2,T], normalized posterior means. */
  float *values;
} h3_audio_latent_host;

void h3_audio_latent_host_free(h3_audio_latent_host *z);

/*
 * Decode normalized latents [32,2,T] (channel × stereo × time) from
 * FL2VA/audio_vae (model.safetensors + config.json latents_mean/std).
 * Returns 1 on success, 0 on failure (error string when provided).
 */
int h3_audio_vae_decode_host(const char *weight_directory,
                             const float *normalized_latent, int latent_length,
                             h3_audio_waveform_host *output, char *error,
                             size_t error_size);

/*
 * Encode channel-major 32 kHz stereo PCM [2, samples], zero-padded to hop=800.
 * Returns normalized latents [32,2,T].
 */
int h3_audio_vae_encode_host(const char *weight_directory, const float *pcm,
                             int samples, h3_audio_latent_host *output,
                             char *error, size_t error_size);

/* Deterministic unit-scale noise used by CLI/tests (same LCG as CUDA smoke). */
void h3_audio_vae_fill_unit_latent(float *latent, size_t n);

/* Write stereo WAV (s16le, interleaved). Channel-major float pcm. Returns 1 ok. */
int h3_audio_waveform_write_wav(const h3_audio_waveform_host *w, const char *path,
                                char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* H3_AUDIO_VAE_DECODE_H */
