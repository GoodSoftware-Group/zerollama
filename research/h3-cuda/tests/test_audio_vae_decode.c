/* End-to-end AudioVAE decode smoke: portable h3_audio_vae.c + CUDA h3_gpu.
 * Uses real MiniMax-H3 FL2VA/audio_vae weights (fp32 pack) + config.json.
 * No MLX golden fixture required — checks shape + finite PCM. */
#include "h3_audio_vae.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
  LATENT_CHANNELS = 32,
  STEREO = 2,
  HOP_LENGTH = 800,
  SAMPLE_RATE = 32000
};

static void progress(int completed, int total, void *opaque) {
  (void)opaque;
  fprintf(stderr, "AudioVAE stage: %d/%d\n", completed, total);
}

int main(int argc, char **argv) {
  const char *weights =
      argc > 1 ? argv[1] : "/tmp/h3c-research/weights/FL2VA/audio_vae";
  int latent_length = argc > 2 ? atoi(argv[2]) : 4;
  if (latent_length < 1) {
    fprintf(stderr, "latent_length must be >= 1\n");
    return 2;
  }

  size_t latent_count = (size_t)LATENT_CHANNELS * STEREO * (size_t)latent_length;
  float *latent = calloc(latent_count, sizeof(float));
  if (!latent) {
    fprintf(stderr, "oom latent\n");
    return 1;
  }
  /* Deterministic unit-scale noise in normalized latent space. */
  for (size_t i = 0; i < latent_count; i++) {
    latent[i] = ((float)((i * 1103515245u + 12345u) % 1000) / 1000.0f) - 0.5f;
  }

  char error[1024];
  h3_audio_waveform got;
  memset(&got, 0, sizeof(got));
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  if (!h3_audio_vae_decode(weights, "cuda", latent, latent_length, progress,
                           NULL, &got, error, sizeof(error))) {
    fprintf(stderr, "FAIL decode: %s\n", error);
    free(latent);
    return 1;
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

  int want_samples = latent_length * HOP_LENGTH;
  if (got.channels != STEREO || got.samples != want_samples ||
      got.sample_rate != SAMPLE_RATE) {
    fprintf(stderr,
            "FAIL shape: channels=%d samples=%d rate=%d (want %d/%d/%d)\n",
            got.channels, got.samples, got.sample_rate, STEREO, want_samples,
            SAMPLE_RATE);
    h3_audio_waveform_free(&got);
    free(latent);
    return 1;
  }

  size_t n = (size_t)got.channels * (size_t)got.samples;
  double sum = 0.0, square = 0.0, absmax = 0.0;
  for (size_t i = 0; i < n; i++) {
    if (!isfinite(got.pcm[i])) {
      fprintf(stderr, "FAIL non-finite pcm[%zu]\n", i);
      h3_audio_waveform_free(&got);
      free(latent);
      return 1;
    }
    double v = (double)got.pcm[i];
    sum += v;
    square += v * v;
    if (fabs(v) > absmax) absmax = fabs(v);
  }
  double mean = sum / (double)n;
  double rms = sqrt(square / (double)n);

  printf("AudioVAE decode T=%d -> %d samples @ %d Hz\n", latent_length,
         got.samples, got.sample_rate);
  printf("pcm mean=%.6g rms=%.6g absmax=%.6g wall=%.3fs\n", mean, rms, absmax,
         sec);
  printf("gpu allocated=%.3f GiB gpu_seconds=%.3f submissions=%llu\n",
         (double)got.gpu_stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0),
         got.gpu_stats.gpu_seconds,
         (unsigned long long)got.gpu_stats.submissions);

  /* Sanity: decoder should not collapse to all zeros on non-zero latent. */
  if (rms < 1e-8) {
    fprintf(stderr, "FAIL near-zero PCM (rms=%g)\n", rms);
    h3_audio_waveform_free(&got);
    free(latent);
    return 1;
  }

  h3_audio_waveform_free(&got);
  free(latent);
  puts("ok: CUDA AudioVAE decode shape+finite");
  return 0;
}
