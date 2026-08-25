/* Real-weight video VAE host encode smoke (32x32x1). */
#include "h3_audio_vae_decode.h"
#include "h3_dit_host.h"
#include "h3_video_vae_encode.h"
#include "h3_video_vae_host.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
  char default_path[768];
  const char *home = getenv("HOME");
  if (home)
    snprintf(default_path, sizeof(default_path),
             "%s/.zerollama/models/MiniMax-H3/FL2VA/video_vae/source", home);
  else
    default_path[0] = '\0';
  const char *weights = argc > 1 ? argv[1] : default_path;
  {
    char st_path[900];
    snprintf(st_path, sizeof(st_path), "%s/model.safetensors",
             weights[0] ? weights : ".");
    if (!weights[0] || access(st_path, R_OK) != 0) {
      printf("test_h3_video_vae_encode SKIP (no weights at %s)\n",
             weights[0] ? weights : "(unset HOME)");
      return 0;
    }
  }

  const int F = 1, H = 32, W = 32;
  size_t n = (size_t)3 * F * H * W;
  float *pix = (float *)calloc(n, sizeof(float));
  if (!pix)
    return 1;
  h3_audio_vae_fill_unit_latent(pix, n);
  for (size_t i = 0; i < n; i++) {
    float v = pix[i] * 0.25f + 0.5f;
    if (v < 0.f)
      v = 0.f;
    if (v > 1.f)
      v = 1.f;
    pix[i] = v;
  }

  char error[1024];
  h3_video_latent_host z;
  memset(&z, 0, sizeof(z));
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  if (!h3_video_vae_encode_host(weights, pix, F, H, W, &z, error,
                                sizeof(error))) {
    fprintf(stderr, "FAIL encode: %s\n", error);
    free(pix);
    return 1;
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double sec =
      (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

  int want_hw = H / h3_video_vae_spatial_ratio();
  if (z.channels != H3_VIDEO_VAE_LATENT_CHANNELS || z.height != want_hw ||
      z.width != want_hw || z.time < 1) {
    fprintf(stderr, "FAIL shape C=%d T=%d H=%d W=%d\n", z.channels, z.time,
            z.height, z.width);
    h3_video_latent_host_free(&z);
    free(pix);
    return 1;
  }

  size_t zn = (size_t)z.channels * z.time * z.height * z.width;
  double sum = 0.0, square = 0.0;
  for (size_t i = 0; i < zn; i++) {
    if (!isfinite(z.values[i])) {
      fprintf(stderr, "FAIL non-finite latent[%zu]\n", i);
      h3_video_latent_host_free(&z);
      free(pix);
      return 1;
    }
    sum += z.values[i];
    square += (double)z.values[i] * z.values[i];
  }
  double rms = sqrt(square / (double)zn);
  printf("VideoVAE host encode %dx%dx%d -> C=%d T=%d %dx%d mean=%.6g rms=%.6g wall=%.3fs\n",
         F, H, W, z.channels, z.time, z.height, z.width, sum / (double)zn, rms,
         sec);
  if (rms < 1e-8) {
    fprintf(stderr, "FAIL near-zero latent\n");
    h3_video_latent_host_free(&z);
    free(pix);
    return 1;
  }
  h3_video_latent_host_free(&z);
  free(pix);

  {
    setenv("H3_VAE_TILE_PIXELS", "32", 1);
    const int FH = 48, FW = 32;
    size_t n2 = (size_t)3 * F * FH * FW;
    float *pix2 = (float *)calloc(n2, sizeof(float));
    if (!pix2)
      return 1;
    h3_audio_vae_fill_unit_latent(pix2, n2);
    for (size_t i = 0; i < n2; i++) {
      float v = pix2[i] * 0.25f + 0.5f;
      if (v < 0.f)
        v = 0.f;
      if (v > 1.f)
        v = 1.f;
      pix2[i] = v;
    }
    h3_video_latent_host z2;
    memset(&z2, 0, sizeof(z2));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (!h3_video_vae_encode_host(weights, pix2, F, FH, FW, &z2, error,
                                  sizeof(error))) {
      fprintf(stderr, "FAIL tiled encode: %s\n", error);
      free(pix2);
      return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (z2.channels != H3_VIDEO_VAE_LATENT_CHANNELS || z2.height != FH / 16 ||
        z2.width != FW / 16 || z2.time < 1) {
      fprintf(stderr, "FAIL tiled shape C=%d T=%d H=%d W=%d\n", z2.channels,
              z2.time, z2.height, z2.width);
      h3_video_latent_host_free(&z2);
      free(pix2);
      return 1;
    }
    size_t zn2 = (size_t)z2.channels * z2.time * z2.height * z2.width;
    double sum2 = 0.0, square2 = 0.0;
    for (size_t i = 0; i < zn2; i++) {
      if (!isfinite(z2.values[i])) {
        fprintf(stderr, "FAIL tiled non-finite\n");
        h3_video_latent_host_free(&z2);
        free(pix2);
        return 1;
      }
      sum2 += z2.values[i];
      square2 += (double)z2.values[i] * z2.values[i];
    }
    printf("VideoVAE tiled encode %dx%dx%d -> C=%d T=%d %dx%d mean=%.6g rms=%.6g wall=%.3fs\n",
           F, FH, FW, z2.channels, z2.time, z2.height, z2.width,
           sum2 / (double)zn2, sqrt(square2 / (double)zn2), sec);
    h3_video_latent_host_free(&z2);
    free(pix2);
    unsetenv("H3_VAE_TILE_PIXELS");
  }

  puts("test_h3_video_vae_encode OK");
  return 0;
}
