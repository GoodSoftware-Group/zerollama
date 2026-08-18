/* Real-weight video VAE ViT decode: T=2, T=7 first-chunk, encode→decode. */
#include "h3_audio_vae_decode.h"
#include "h3_video_vae_decode.h"
#include "h3_video_vae_encode.h"
#include "h3_video_vae_host.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int check_frames(const h3_video_frames_host *frames, int want_f, int want_hw,
                        const char *label) {
  if (frames->frames != want_f || frames->height != want_hw ||
      frames->width != want_hw || !frames->rgb) {
    fprintf(stderr, "FAIL %s shape F=%d H=%d W=%d\n", label, frames->frames,
            frames->height, frames->width);
    return 0;
  }
  size_t pn = (size_t)frames->frames * frames->height * frames->width * 3;
  double sum = 0.0, square = 0.0;
  for (size_t i = 0; i < pn; i++) {
    if (!isfinite(frames->rgb[i]) || frames->rgb[i] < 0.f ||
        frames->rgb[i] > 1.f) {
      fprintf(stderr, "FAIL %s rgb[%zu]=%g\n", label, i, frames->rgb[i]);
      return 0;
    }
    sum += frames->rgb[i];
    square += (double)frames->rgb[i] * frames->rgb[i];
  }
  double rms = sqrt(square / (double)pn);
  printf("%s %dx%dx%d mean=%.6g rms=%.6g\n", label, frames->frames,
         frames->height, frames->width, sum / (double)pn, rms);
  if (rms < 1e-4) {
    fprintf(stderr, "FAIL %s near-zero frames\n", label);
    return 0;
  }
  return 1;
}

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
      printf("test_h3_video_vae_decode SKIP (no weights at %s)\n",
             weights[0] ? weights : "(unset HOME)");
      return 0;
    }
  }

  const int C = H3_VIDEO_VAE_LATENT_CHANNELS, H = 2, W = 2;
  char error[1024];

  {
    const int T = 2;
    size_t n = (size_t)C * T * H * W;
    float *z = (float *)calloc(n, sizeof(float));
    if (!z)
      return 1;
    h3_audio_vae_fill_unit_latent(z, n);
    h3_video_frames_host frames;
    memset(&frames, 0, sizeof(frames));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (!h3_video_vae_decode_host(weights, z, T, H, W, &frames, error,
                                  sizeof(error))) {
      fprintf(stderr, "FAIL decode T=2: %s\n", error);
      free(z);
      return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (!check_frames(&frames, 5, 32, "VideoVAE host decode T=2 2x2")) {
      h3_video_frames_host_free(&frames);
      free(z);
      return 1;
    }
    printf("  wall=%.3fs\n", sec);
    h3_video_frames_host_free(&frames);
    free(z);
  }

  {
    const int T = 7;
    size_t n = (size_t)C * T * H * W;
    float *z = (float *)calloc(n, sizeof(float));
    if (!z)
      return 1;
    h3_audio_vae_fill_unit_latent(z, n);
    h3_video_frames_host frames;
    memset(&frames, 0, sizeof(frames));
    if (!h3_video_vae_decode_host(weights, z, T, H, W, &frames, error,
                                  sizeof(error))) {
      fprintf(stderr, "FAIL decode T=7: %s\n", error);
      free(z);
      return 1;
    }
    if (!check_frames(&frames, 22, 32, "VideoVAE host decode T=7 2x2")) {
      h3_video_frames_host_free(&frames);
      free(z);
      return 1;
    }
    h3_video_frames_host_free(&frames);
    free(z);
  }

  {
    const int T = 12;
    size_t n = (size_t)C * T * H * W;
    float *z = (float *)calloc(n, sizeof(float));
    if (!z)
      return 1;
    h3_audio_vae_fill_unit_latent(z, n);
    h3_video_frames_host frames;
    memset(&frames, 0, sizeof(frames));
    if (!h3_video_vae_decode_host(weights, z, T, H, W, &frames, error,
                                  sizeof(error))) {
      fprintf(stderr, "FAIL decode T=12: %s\n", error);
      free(z);
      return 1;
    }
    if (!check_frames(&frames, 39, 32, "VideoVAE host decode T=12 2x2")) {
      h3_video_frames_host_free(&frames);
      free(z);
      return 1;
    }
    h3_video_frames_host_free(&frames);
    free(z);
  }

  {
    const int F = 1, PH = 32, PW = 32;
    size_t n = (size_t)3 * F * PH * PW;
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
    h3_video_latent_host z;
    memset(&z, 0, sizeof(z));
    if (!h3_video_vae_encode_host(weights, pix, F, PH, PW, &z, error,
                                  sizeof(error))) {
      fprintf(stderr, "FAIL roundtrip encode: %s\n", error);
      free(pix);
      return 1;
    }
    if (z.channels != C || z.height != H || z.width != W || z.time < 1) {
      fprintf(stderr, "FAIL roundtrip latent C=%d T=%d %dx%d\n", z.channels,
              z.time, z.height, z.width);
      h3_video_latent_host_free(&z);
      free(pix);
      return 1;
    }
    float *padded = (float *)calloc((size_t)C * 2 * H * W, sizeof(float));
    if (!padded || h3_video_vae_repeat_last_time(z.values, C, z.time, H, W, 2,
                                                 padded) != 0) {
      fprintf(stderr, "FAIL pad latent T=%d→2\n", z.time);
      free(padded);
      h3_video_latent_host_free(&z);
      free(pix);
      return 1;
    }
    h3_video_frames_host frames;
    memset(&frames, 0, sizeof(frames));
    if (!h3_video_vae_decode_host(weights, padded, 2, H, W, &frames, error,
                                  sizeof(error))) {
      fprintf(stderr, "FAIL roundtrip decode: %s\n", error);
      free(padded);
      h3_video_latent_host_free(&z);
      free(pix);
      return 1;
    }
    if (!check_frames(&frames, 5, 32, "VideoVAE encode→pad T=2→decode")) {
      h3_video_frames_host_free(&frames);
      free(padded);
      h3_video_latent_host_free(&z);
      free(pix);
      return 1;
    }
    h3_video_frames_host_free(&frames);
    free(padded);
    h3_video_latent_host_free(&z);
    free(pix);
  }

  {
    h3_video_frames_host one;
    memset(&one, 0, sizeof(one));
    one.frames = 1;
    one.height = 4;
    one.width = 4;
    one.rgb = (float *)malloc(48 * sizeof(float));
    if (!one.rgb)
      return 1;
    for (int i = 0; i < 48; i++)
      one.rgb[i] = (float)(i % 16) / 15.f;
    const char *tmp = "tests/tmp_vae_frame.ppm";
    if (!h3_video_frames_write_ppm(&one, 0, tmp, error, sizeof(error))) {
      fprintf(stderr, "FAIL ppm write: %s\n", error);
      free(one.rgb);
      return 1;
    }
    h3_video_frames_host got;
    memset(&got, 0, sizeof(got));
    if (!h3_video_frames_read_ppm(tmp, &got, error, sizeof(error))) {
      fprintf(stderr, "FAIL ppm read: %s\n", error);
      free(one.rgb);
      return 1;
    }
    if (got.width != 4 || got.height != 4 || got.frames != 1) {
      fprintf(stderr, "FAIL ppm shape\n");
      h3_video_frames_host_free(&got);
      free(one.rgb);
      return 1;
    }
    for (int i = 0; i < 48; i++) {
      float want = (float)((unsigned char)(one.rgb[i] * 255.f + 0.5f)) / 255.f;
      if (fabsf(got.rgb[i] - want) > 1e-5f) {
        fprintf(stderr, "FAIL ppm roundtrip [%d]\n", i);
        h3_video_frames_host_free(&got);
        free(one.rgb);
        return 1;
      }
    }
    float cthw[3 * 2 * 4 * 4];
    if (h3_video_frames_to_cthw(&got, 2, cthw) != 0) {
      fprintf(stderr, "FAIL cthw pack\n");
      h3_video_frames_host_free(&got);
      free(one.rgb);
      return 1;
    }
    if (fabsf(cthw[0] - got.rgb[0]) > 1e-6f ||
        fabsf(cthw[16] - got.rgb[0]) > 1e-6f) {
      fprintf(stderr, "FAIL cthw repeat\n");
      h3_video_frames_host_free(&got);
      free(one.rgb);
      return 1;
    }
    remove(tmp);
    h3_video_frames_host_free(&got);
    free(one.rgb);
    puts("PPM read/write + CTHW pack OK");
  }

  puts("test_h3_video_vae_decode OK");
  return 0;
}
