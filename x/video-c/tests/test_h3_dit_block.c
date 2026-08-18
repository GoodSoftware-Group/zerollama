#include "h3_adaln_host.h"
#include "h3_dit_block.h"
#include "h3_dit_host.h"

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

static int test_table_lerp(void) {
  float table[4] = {0.f, 10.f, 20.f, 30.f}; /* grid=4 rank=1 */
  float y;
  CHECK(h3_adaln_table_embed(table, 4, 1, 0.f, &y) == 0 && y == 0.f);
  CHECK(h3_adaln_table_embed(table, 4, 1, 1.f, &y) == 0 && y == 30.f);
  CHECK(h3_adaln_table_embed(table, 4, 1, 0.5f, &y) == 0);
  CHECK(fabsf(y - 15.f) < 1e-5f);
  return 0;
}

static int test_sdpa_uniform(void) {
  /* one head, seq=2, dim=2: Q=K=V = identity-ish */
  float q[4] = {1.f, 0.f, 0.f, 1.f};
  float out[4];
  CHECK(h3_dit_sdpa_f32(out, q, q, q, 2, 1, 2, 1.f) == 0);
  CHECK(isfinite(out[0]) && isfinite(out[3]));
  return 0;
}

static int test_pack_block0(void) {
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
    printf("test_h3_dit_block pack SKIP\n");
    return 0;
  }
  char err[256];
  h3_st_store *st = h3_st_store_open(path, err, sizeof(err));
  CHECK(st);

  const int seq = 2;
  const int H = H3_DIT_HIDDEN_SIZE;
  float *x = (float *)calloc((size_t)seq * H, sizeof(float));
  float *y = (float *)calloc((size_t)seq * H, sizeof(float));
  CHECK(x && y);
  for (int i = 0; i < seq * H; i++)
    x[i] = sinf((float)(i + 3) * 0.001f) * 0.1f;
  int tags[2] = {H3_ADALN_TAG_VIDEO, H3_ADALN_TAG_TEXT};
  float pos[6] = {0, 0, 0, 0, 0, 1};
  if (h3_dit_block_forward(st, 0, x, seq, tags, 0.5f, NULL, pos, NULL, 0, 0, y,
                           err, sizeof(err)) != 0) {
    fprintf(stderr, "FAIL block0: %s\n", err);
    return 1;
  }
  int finite = 1, changed = 0;
  double sq = 0, dx = 0;
  for (int i = 0; i < seq * H; i++) {
    if (!isfinite(y[i]))
      finite = 0;
    sq += (double)y[i] * y[i];
    dx += (double)(y[i] - x[i]) * (y[i] - x[i]);
    if (fabsf(y[i] - x[i]) > 1e-6f)
      changed = 1;
  }
  CHECK(finite && changed);
  double rms = sqrt(sq / (double)(seq * H));
  CHECK(rms > 1e-4 && rms < 200.0);
  printf("test_h3_dit_block pack OK seq=%d rms=%.6g delta_rms=%.6g\n", seq, rms,
         sqrt(dx / (double)(seq * H)));
  free(x);
  free(y);
  h3_st_store_free(st);
  return 0;
}

int main(void) {
  if (test_table_lerp() || test_sdpa_uniform())
    return 1;
  if (test_pack_block0())
    return 1;
  printf("test_h3_dit_block OK\n");
  return 0;
}
