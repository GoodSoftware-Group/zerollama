#include "h3_adaln_host.h"
#include "h3_dit_forward.h"
#include "h3_dit_pack.h"
#include "h3_dit_host.h"
#include "h3_host.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int test_pack_forward(void) {
  if (!getenv("H3_CONVROT_PACK") || strcmp(getenv("H3_CONVROT_PACK"), "1") != 0)
    return 0;
  char path[768];
  const char *home = getenv("HOME");
  if (!home)
    return 0;
  snprintf(path, sizeof(path),
           "%s/.zerollama/third_party/h3/dit/"
           "MiniMax-H3-FL2VA-pruned_int8_convrot.safetensors",
           home);
  if (access(path, R_OK) != 0) {
    printf("test_h3_dit_forward pack SKIP\n");
    return 0;
  }
  int n_layers = 1;
  const char *le = getenv("H3_DIT_LAYERS");
  if (le && le[0])
    n_layers = atoi(le);
  if (n_layers < 0)
    n_layers = 0;
  if (n_layers > H3_DIT_NUM_LAYERS)
    n_layers = H3_DIT_NUM_LAYERS;

  char err[256];
  h3_st_store *st = h3_st_store_open(path, err, sizeof(err));
  CHECK(st);

  const int B = 1, C = 24, F = 1, Ht = 2, W = 2;
  const int nv = 1, na = 2, nt = 1, seq = 4;
  float lat[24 * 1 * 2 * 2];
  for (int i = 0; i < 96; i++)
    lat[i] = ((i % 13) - 6) * 0.01f;
  float video[96];
  CHECK(h3_dit_patchify_video(lat, B, C, F, Ht, W, 1, 2, 2, video) == 0);
  float audio_lat[2 * 32 * 1];
  for (int i = 0; i < 64; i++)
    audio_lat[i] = ((i % 7) - 3) * 0.02f;
  float audio[64];
  CHECK(h3_dit_pack_audio(audio_lat, 32, 1, audio) == 0);
  float text[H3_DIT_TEXT_DIM];
  for (int i = 0; i < H3_DIT_TEXT_DIM; i++)
    text[i] = sinf((float)i * 0.01f) * 0.05f;

  int tidx[1] = {0};
  int aidx[2] = {1, 2};
  int vidx[1] = {3};
  int tags[4] = {H3_ADALN_TAG_TEXT, H3_ADALN_TAG_AUDIO, H3_ADALN_TAG_AUDIO,
                 H3_ADALN_TAG_VIDEO};
  float pos[12] = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
  float vout[96], aout[64];
  if (h3_dit_forward(st, video, nv, audio, na, text, nt, vidx, aidx, tidx, tags,
                     pos, seq, 0.5f, NULL, n_layers, vout, aout, err,
                     sizeof(err)) != 0) {
    fprintf(stderr, "FAIL forward: %s\n", err);
    return 1;
  }
  int finite = 1, vnz = 0, anz = 0;
  double vsq = 0, asq = 0;
  for (int i = 0; i < 96; i++) {
    if (!isfinite(vout[i]))
      finite = 0;
    if (fabsf(vout[i]) > 1e-8f)
      vnz = 1;
    vsq += (double)vout[i] * vout[i];
  }
  for (int i = 0; i < 64; i++) {
    if (!isfinite(aout[i]))
      finite = 0;
    if (fabsf(aout[i]) > 1e-8f)
      anz = 1;
    asq += (double)aout[i] * aout[i];
  }
  CHECK(finite && vnz && anz);
  float back[96];
  CHECK(h3_dit_unpatchify_video(vout, B, C, F, Ht, W, 1, 2, 2, back) == 0);
  printf("test_h3_dit_forward pack OK layers=%d v_rms=%.6g a_rms=%.6g\n",
         n_layers, sqrt(vsq / 96.0), sqrt(asq / 64.0));
  h3_st_store_free(st);
  return 0;
}

static int test_euler(void) {
  float x[2] = {1.f, 0.f};
  float v[2] = {1.f, -2.f};
  CHECK(h3_euler_velocity_step(x, v, 2, 1.f, 0.5f));
  CHECK(fabsf(x[0] - 1.5f) < 1e-6f);
  CHECK(fabsf(x[1] + 1.f) < 1e-6f);
  return 0;
}

static int test_pack_denoise(void) {
  if (!getenv("H3_CONVROT_PACK") || strcmp(getenv("H3_CONVROT_PACK"), "1") != 0)
    return 0;
  char path[768];
  const char *home = getenv("HOME");
  if (!home)
    return 0;
  snprintf(path, sizeof(path),
           "%s/.zerollama/third_party/h3/dit/"
           "MiniMax-H3-FL2VA-pruned_int8_convrot.safetensors",
           home);
  if (access(path, R_OK) != 0)
    return 0;
  int n_layers = 1;
  const char *le = getenv("H3_DIT_LAYERS");
  if (le && le[0])
    n_layers = atoi(le);
  char err[256];
  h3_st_store *st = h3_st_store_open(path, err, sizeof(err));
  CHECK(st);
  float video[96], audio[64], text[H3_DIT_TEXT_DIM];
  h3_rng rng;
  h3_rng_seed(&rng, 1);
  h3_rng_fill_normal(&rng, video, 96);
  h3_rng_fill_normal(&rng, audio, 64);
  for (int i = 0; i < H3_DIT_TEXT_DIM; i++)
    text[i] = 0.01f * sinf((float)i * 0.02f);
  int tidx[1] = {0};
  int aidx[2] = {1, 2};
  int vidx[1] = {3};
  int tags[4] = {H3_ADALN_TAG_TEXT, H3_ADALN_TAG_AUDIO, H3_ADALN_TAG_AUDIO,
                 H3_ADALN_TAG_VIDEO};
  float pos[12] = {0};
  if (h3_dit_denoise(st, video, 1, audio, 2, text, 1, vidx, aidx, tidx, tags,
                     pos, 4, 2, n_layers, 1, -1, err, sizeof(err)) != 0) {
    fprintf(stderr, "FAIL denoise: %s\n", err);
    return 1;
  }
  int finite = 1;
  double vsq = 0;
  for (int i = 0; i < 96; i++) {
    if (!isfinite(video[i]))
      finite = 0;
    vsq += (double)video[i] * video[i];
  }
  CHECK(finite);
  printf("test_h3_dit_forward denoise OK steps=2 layers=%d v_rms=%.6g\n",
         n_layers, sqrt(vsq / 96.0));
  h3_st_store_free(st);
  return 0;
}

static int test_tiny_t2va(void) {
  if (!getenv("H3_CONVROT_PACK") || strcmp(getenv("H3_CONVROT_PACK"), "1") != 0)
    return 0;
  char path[768];
  const char *home = getenv("HOME");
  if (!home)
    return 0;
  snprintf(path, sizeof(path),
           "%s/.zerollama/third_party/h3/dit/"
           "MiniMax-H3-FL2VA-pruned_int8_convrot.safetensors",
           home);
  if (access(path, R_OK) != 0)
    return 0;
  char err[256];
  h3_st_store *st = h3_st_store_open(path, err, sizeof(err));
  CHECK(st);
  const int nt = 12;
  float *text = (float *)malloc((size_t)nt * (size_t)H3_DIT_TEXT_DIM *
                                sizeof(float));
  CHECK(text);
  for (int i = 0; i < nt * H3_DIT_TEXT_DIM; i++)
    text[i] = 0.01f * sinf((float)i * 0.02f);
  float video[24 * 2 * 2 * 2];
  float audio[2 * 32 * 8];
  if (h3_dit_tiny_t2va(st, text, nt, 2, 1, -1, 1, video, audio, err,
                       sizeof(err)) != 0) {
    fprintf(stderr, "FAIL tiny t2va: %s\n", err);
    free(text);
    h3_st_store_free(st);
    return 1;
  }
  int finite = 1;
  double vsq = 0;
  for (int i = 0; i < 24 * 2 * 2 * 2; i++) {
    if (!isfinite(video[i]))
      finite = 0;
    vsq += (double)video[i] * video[i];
  }
  CHECK(finite);
  printf("test_h3_dit_forward tiny-t2va OK seq=30 v_rms=%.6g\n",
         sqrt(vsq / (24.0 * 8.0)));
  free(text);
  h3_st_store_free(st);
  return 0;
}

int main(void) {
  if (test_euler() || test_pack_forward() || test_pack_denoise() ||
      test_tiny_t2va())
    return 1;
  printf("test_h3_dit_forward OK\n");
  return 0;
}
