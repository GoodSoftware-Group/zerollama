/* Real-weight AudioVAE host decode smoke (shape + finite + non-zero PCM). */
#include "h3_audio_vae_decode.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { LATENT_CHANNELS = 32, STEREO = 2, HOP = 800 };

int main(int argc, char **argv) {
  char default_path[768];
  const char *home = getenv("HOME");
  if (home)
    snprintf(default_path, sizeof(default_path),
             "%s/.zerollama/models/MiniMax-H3/FL2VA/audio_vae", home);
  else
    default_path[0] = '\0';
  const char *weights = argc > 1 ? argv[1] : default_path;
  int T = argc > 2 ? atoi(argv[2]) : 4;
  if (T < 1) {
    fprintf(stderr, "latent_length must be >= 1\n");
    return 2;
  }
  {
    char st_path[900];
    snprintf(st_path, sizeof(st_path), "%s/model.safetensors",
             weights[0] ? weights : ".");
    if (!weights[0] || access(st_path, R_OK) != 0) {
      printf("test_h3_audio_vae_decode SKIP (no weights at %s)\n",
             weights[0] ? weights : "(unset HOME)");
      return 0;
    }
  }

  size_t n = (size_t)LATENT_CHANNELS * STEREO * (size_t)T;
  float *latent = (float *)calloc(n, sizeof(float));
  if (!latent)
    return 1;
  h3_audio_vae_fill_unit_latent(latent, n);

  char error[1024];
  h3_audio_waveform_host got;
  memset(&got, 0, sizeof(got));
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  if (!h3_audio_vae_decode_host(weights, latent, T, &got, error,
                                sizeof(error))) {
    fprintf(stderr, "FAIL decode: %s\n", error);
    free(latent);
    return 1;
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double sec =
      (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

  int want = T * HOP;
  if (got.channels != STEREO || got.samples != want ||
      got.sample_rate != 32000) {
    fprintf(stderr, "FAIL shape ch=%d samples=%d rate=%d want %d/%d/32000\n",
            got.channels, got.samples, got.sample_rate, STEREO, want);
    h3_audio_waveform_host_free(&got);
    free(latent);
    return 1;
  }

  size_t pcm_n = (size_t)got.channels * (size_t)got.samples;
  double sum = 0.0, square = 0.0, absmax = 0.0;
  for (size_t i = 0; i < pcm_n; i++) {
    if (!isfinite(got.pcm[i])) {
      fprintf(stderr, "FAIL non-finite pcm[%zu]\n", i);
      h3_audio_waveform_host_free(&got);
      free(latent);
      return 1;
    }
    double v = (double)got.pcm[i];
    sum += v;
    square += v * v;
    if (fabs(v) > absmax)
      absmax = fabs(v);
  }
  double mean = sum / (double)pcm_n;
  double rms = sqrt(square / (double)pcm_n);
  printf("AudioVAE host decode T=%d -> %d samples @ %d Hz\n", T, got.samples,
         got.sample_rate);
  printf("pcm mean=%.6g rms=%.6g absmax=%.6g wall=%.3fs\n", mean, rms, absmax,
         sec);
  if (rms < 1e-8) {
    fprintf(stderr, "FAIL near-zero PCM rms=%g\n", rms);
    h3_audio_waveform_host_free(&got);
    free(latent);
    return 1;
  }
  /* Host-decoder regression (T=4 unit latent, MiniMaxAI FL2VA/audio_vae). */
  if (T == 4) {
    if (fabs(mean + 0.00185172) > 2e-5 || fabs(rms - 0.0664794) > 2e-4 ||
        fabs(absmax - 0.370082) > 2e-3) {
      fprintf(stderr, "FAIL pcm regression mean=%g rms=%g absmax=%g\n", mean,
              rms, absmax);
      h3_audio_waveform_host_free(&got);
      free(latent);
      return 1;
    }
  }

  h3_audio_waveform_host_free(&got);
  free(latent);

  /* Encode 800 samples (T=1) then decode; finite non-zero. */
  {
    int samples = HOP;
    float *pcm = (float *)calloc((size_t)STEREO * (size_t)samples, sizeof(float));
    if (!pcm)
      return 1;
    h3_audio_vae_fill_unit_latent(pcm, (size_t)STEREO * (size_t)samples);
    h3_audio_latent_host z;
    memset(&z, 0, sizeof(z));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (!h3_audio_vae_encode_host(weights, pcm, samples, &z, error,
                                  sizeof(error))) {
      fprintf(stderr, "FAIL encode: %s\n", error);
      free(pcm);
      return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double esec =
        (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (z.channels != LATENT_CHANNELS || z.stereo != STEREO || z.length != 1) {
      fprintf(stderr, "FAIL encode shape %d %d %d\n", z.channels, z.stereo,
              z.length);
      h3_audio_latent_host_free(&z);
      free(pcm);
      return 1;
    }
    size_t zn = (size_t)z.channels * z.stereo * z.length;
    double zsum = 0.0, zsq = 0.0;
    for (size_t i = 0; i < zn; i++) {
      if (!isfinite(z.values[i])) {
        fprintf(stderr, "FAIL non-finite latent[%zu]\n", i);
        h3_audio_latent_host_free(&z);
        free(pcm);
        return 1;
      }
      zsum += z.values[i];
      zsq += (double)z.values[i] * z.values[i];
    }
    double zrms = sqrt(zsq / (double)zn);
    printf("AudioVAE host encode 800 samples -> T=%d mean=%.6g rms=%.6g wall=%.3fs\n",
           z.length, zsum / (double)zn, zrms, esec);
    if (zrms < 1e-8) {
      fprintf(stderr, "FAIL near-zero latent rms\n");
      h3_audio_latent_host_free(&z);
      free(pcm);
      return 1;
    }
    h3_audio_waveform_host round;
    memset(&round, 0, sizeof(round));
    if (!h3_audio_vae_decode_host(weights, z.values, z.length, &round, error,
                                  sizeof(error))) {
      fprintf(stderr, "FAIL encode→decode: %s\n", error);
      h3_audio_latent_host_free(&z);
      free(pcm);
      return 1;
    }
    if (round.samples != HOP) {
      fprintf(stderr, "FAIL roundtrip samples %d\n", round.samples);
      h3_audio_waveform_host_free(&round);
      h3_audio_latent_host_free(&z);
      free(pcm);
      return 1;
    }
    h3_audio_waveform_host_free(&round);
    h3_audio_latent_host_free(&z);
    free(pcm);
  }

  puts("test_h3_audio_vae_decode OK");
  return 0;
}
