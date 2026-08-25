#include "music_dav.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void music3_wav_host_free(music3_wav_host *w) {
  if (!w)
    return;
  free(w->pcm);
  memset(w, 0, sizeof(*w));
}

int music3_snake1d_f32(float *dst, const float *input, const float *alpha,
                       int batch, int channels, int length) {
  if (!dst || !input || !alpha || batch < 1 || channels < 1 || length < 1)
    return -1;
  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channels; c++) {
      float a = alpha[c];
      float inv = 1.f / (a + 1e-9f);
      for (int t = 0; t < length; t++) {
        float x = input[((b * channels) + c) * length + t];
        float s = sinf(a * x);
        dst[((b * channels) + c) * length + t] = x + inv * s * s;
      }
    }
  }
  return 0;
}

int music3_dav_synthetic_decode(int latent_t, music3_wav_host *out, char *error,
                                size_t error_size) {
  if (!out || latent_t < 1) {
    if (error && error_size)
      snprintf(error, error_size, "invalid synthetic DAV args");
    return 0;
  }
  int samples = latent_t * 512;
  float *pcm = (float *)calloc((size_t)2 * (size_t)samples, sizeof(float));
  if (!pcm) {
    if (error && error_size)
      snprintf(error, error_size, "oom");
    return 0;
  }
  out->channels = 2;
  out->samples = samples;
  out->sample_rate = 44100;
  out->pcm = pcm;
  return 1;
}

static int wr4(FILE *f, const char *s) { return fwrite(s, 1, 4, f) == 4 ? 0 : -1; }

static int wr32(FILE *f, uint32_t v) {
  unsigned char b[4] = {(unsigned char)(v), (unsigned char)(v >> 8),
                        (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
  return fwrite(b, 1, 4, f) == 4 ? 0 : -1;
}

static int wr16(FILE *f, uint16_t v) {
  unsigned char b[2] = {(unsigned char)(v), (unsigned char)(v >> 8)};
  return fwrite(b, 1, 2, f) == 2 ? 0 : -1;
}

int music3_wav_write(const music3_wav_host *w, const char *path, char *error,
                     size_t error_size) {
  if (!w || !w->pcm || !path || w->channels != 2) {
    if (error && error_size)
      snprintf(error, error_size, "invalid wav");
    return 0;
  }
  FILE *f = fopen(path, "wb");
  if (!f) {
    if (error && error_size)
      snprintf(error, error_size, "open %s", path);
    return 0;
  }
  uint32_t data_bytes = (uint32_t)w->samples * 2u * 2u; /* s16 stereo */
  uint32_t riff = 36u + data_bytes;
  int ok = wr4(f, "RIFF") == 0 && wr32(f, riff) == 0 && wr4(f, "WAVE") == 0 &&
           wr4(f, "fmt ") == 0 && wr32(f, 16) == 0 && wr16(f, 1) == 0 &&
           wr16(f, 2) == 0 && wr32(f, (uint32_t)w->sample_rate) == 0 &&
           wr32(f, (uint32_t)w->sample_rate * 4u) == 0 && wr16(f, 4) == 0 &&
           wr16(f, 16) == 0 && wr4(f, "data") == 0 && wr32(f, data_bytes) == 0;
  if (!ok) {
    fclose(f);
    if (error && error_size)
      snprintf(error, error_size, "wav header");
    return 0;
  }
  for (int t = 0; t < w->samples; t++) {
    for (int c = 0; c < 2; c++) {
      float x = w->pcm[c * w->samples + t];
      if (x > 1.f)
        x = 1.f;
      if (x < -1.f)
        x = -1.f;
      int16_t s = (int16_t)lrintf(x * 32767.f);
      unsigned char b[2] = {(unsigned char)s, (unsigned char)((uint16_t)s >> 8)};
      if (fwrite(b, 1, 2, f) != 2) {
        fclose(f);
        if (error && error_size)
          snprintf(error, error_size, "wav data");
        return 0;
      }
    }
  }
  fclose(f);
  return 1;
}
