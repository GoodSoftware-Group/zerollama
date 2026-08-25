/* ConvRot / FWHT rematch + optional pruned DiT I8 pack smoke. */
#include "h3_convrot.h"
#include "h3_dit_host.h"
#include "h3_st_store.h"

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

static int test_fwht_e0(void) {
  float x[4] = {1.f, 0.f, 0.f, 0.f};
  CHECK(h3_fwht_normalized(x, 4) == 0);
  for (int i = 0; i < 4; i++)
    CHECK(fabsf(x[i] - 0.5f) < 1e-6f);
  CHECK(h3_fwht_normalized(x, 4) == 0);
  CHECK(fabsf(x[0] - 1.f) < 1e-5f);
  CHECK(fabsf(x[1]) < 1e-5f && fabsf(x[2]) < 1e-5f && fabsf(x[3]) < 1e-5f);
  return 0;
}

static int test_involution(void) {
  float w[8] = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, -0.7f, 0.8f};
  float orig[8];
  memcpy(orig, w, sizeof(w));
  CHECK(h3_convrot_unrotate(w, 2, 4, 4) == 0);
  CHECK(h3_convrot_unrotate(w, 2, 4, 4) == 0);
  for (int i = 0; i < 8; i++)
    CHECK(fabsf(w[i] - orig[i]) < 1e-5f);
  return 0;
}

static int test_regular_not_sylvester(void) {
  /* ConvRot H4 first column is [0.5,0.5,0.5,-0.5], not FWHT [0.5,0.5,0.5,0.5]. */
  float w[4] = {1.f, 0.f, 0.f, 0.f};
  CHECK(h3_convrot_unrotate(w, 1, 4, 4) == 0);
  CHECK(fabsf(w[0] - 0.5f) < 1e-5f && fabsf(w[1] - 0.5f) < 1e-5f);
  CHECK(fabsf(w[2] - 0.5f) < 1e-5f && fabsf(w[3] + 0.5f) < 1e-5f);
  return 0;
}

static int test_quant_roundtrip(void) {
  const int rows = 3, cols = 8, gs = 4;
  float w[24];
  for (int i = 0; i < 24; i++)
    w[i] = sinf((float)(i + 1) * 0.37f);
  float rot[24];
  memcpy(rot, w, sizeof(w));
  CHECK(h3_convrot_unrotate(rot, rows, cols, gs) == 0);
  int8_t q[24];
  float scale[3];
  for (int r = 0; r < rows; r++) {
    float amax = 1e-30f;
    for (int c = 0; c < cols; c++) {
      float a = fabsf(rot[r * cols + c]);
      if (a > amax)
        amax = a;
    }
    scale[r] = amax / 127.f;
    for (int c = 0; c < cols; c++) {
      float v = rot[r * cols + c] / scale[r];
      if (v > 127.f)
        v = 127.f;
      if (v < -127.f)
        v = -127.f;
      q[r * cols + c] = (int8_t)lrintf(v);
    }
  }
  float rec[24];
  CHECK(h3_convrot_dequant_i8(q, rows, cols, scale, gs, rec) == 0);
  double dot = 0, na = 0, nb = 0;
  for (int i = 0; i < 24; i++) {
    dot += (double)w[i] * rec[i];
    na += (double)w[i] * w[i];
    nb += (double)rec[i] * rec[i];
  }
  double cos = dot / (sqrt(na) * sqrt(nb));
  CHECK(cos > 0.99);
  return 0;
}

static int test_comfy_json(void) {
  const char *j = "{\"format\": \"int8_tensorwise\", \"convrot\": true, "
                  "\"convrot_groupsize\": 256}";
  int gs = -1;
  CHECK(h3_comfy_quant_parse((const uint8_t *)j, strlen(j), &gs) == 0);
  CHECK(gs == 256);
  const char *j2 = "{\"format\": \"int8_tensorwise\"}";
  CHECK(h3_comfy_quant_parse((const uint8_t *)j2, strlen(j2), &gs) == 0);
  CHECK(gs == 0);
  return 0;
}

static int test_pruned_pack(void) {
  char path[768];
  const char *env = getenv("H3_DIT_ST");
  if (env && env[0])
    snprintf(path, sizeof(path), "%s", env);
  else {
    const char *home = getenv("HOME");
    if (!home)
      return 0;
    snprintf(path, sizeof(path),
             "%s/.zerollama/third_party/h3/dit/"
             "MiniMax-H3-FL2VA-pruned_int8_convrot.safetensors",
             home);
  }
  if (access(path, R_OK) != 0) {
    printf("test_h3_convrot pack SKIP (%s)\n", path);
    return 0;
  }
  char err[256];
  h3_st_store *st = h3_st_store_open(path, err, sizeof(err));
  if (!st) {
    fprintf(stderr, "FAIL open: %s\n", err);
    return 1;
  }
  CHECK(h3_st_store_tensors(st) >= 900);

  float *cp = (float *)malloc((size_t)H3_DIT_HIDDEN_SIZE * H3_DIT_TEXT_DIM *
                              sizeof(float));
  CHECK(cp);
  CHECK(h3_st_store_load_f32(st, "condition_proj.weight", cp,
                             (size_t)H3_DIT_HIDDEN_SIZE * H3_DIT_TEXT_DIM, err,
                             sizeof(err)) == 0);

  const int rows = H3_DIT_HIDDEN_SIZE;
  const int cols = H3_DIT_INNER_DIM;
  float *w = (float *)malloc((size_t)rows * (size_t)cols * sizeof(float));
  CHECK(w);
  CHECK(h3_st_store_load_f32(st, "blocks.0.attn.out_proj.weight", w,
                             (size_t)rows * (size_t)cols, err, sizeof(err)) ==
        0);
  int finite = 1, nonzero = 0;
  double sq = 0;
  size_t n = (size_t)rows * (size_t)cols;
  for (size_t i = 0; i < n; i++) {
    if (!isfinite(w[i]))
      finite = 0;
    if (fabsf(w[i]) > 1e-12f)
      nonzero = 1;
    sq += (double)w[i] * (double)w[i];
  }
  CHECK(finite && nonzero);
  double rms = sqrt(sq / (double)n);
  CHECK(rms > 1e-4 && rms < 10.0);

  float x[H3_DIT_INNER_DIM];
  float y[H3_DIT_HIDDEN_SIZE];
  for (int i = 0; i < H3_DIT_INNER_DIM; i++)
    x[i] = ((i % 17) - 8) * 0.01f;
  CHECK(h3_dit_linear(x, 1, H3_DIT_INNER_DIM, H3_DIT_HIDDEN_SIZE, w, NULL, y) ==
        0);
  int yfin = 1, ynz = 0;
  for (int i = 0; i < H3_DIT_HIDDEN_SIZE; i++) {
    if (!isfinite(y[i]))
      yfin = 0;
    if (fabsf(y[i]) > 1e-8f)
      ynz = 1;
  }
  CHECK(yfin && ynz);
  printf("test_h3_convrot pack OK tensors=%zu out_proj_rms=%.6g y0=%.6g\n",
         h3_st_store_tensors(st), rms, y[0]);
  free(w);
  free(cp);
  h3_st_store_free(st);
  return 0;
}

static int test_fakequant_act(void) {
  float x[8] = {0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, -0.7f, 0.8f};
  float orig[8];
  memcpy(orig, x, sizeof(x));
  CHECK(h3_convrot_fakequant_act(x, 2, 4, 4) == 0);
  int changed = 0;
  for (int i = 0; i < 8; i++) {
    CHECK(isfinite(x[i]));
    if (fabsf(x[i] - orig[i]) > 1e-8f)
      changed = 1;
  }
  CHECK(changed);
  /* gs=1: per-row max preserved */
  float y[4] = {1.f, 0.5f, -2.f, 0.25f};
  CHECK(h3_convrot_fakequant_act(y, 1, 4, 1) == 0);
  CHECK(fabsf(fabsf(y[2]) - 2.f) < 1e-5f);
  return 0;
}

int main(void) {
  if (test_fwht_e0() || test_involution() || test_regular_not_sylvester() ||
      test_quant_roundtrip() || test_comfy_json() || test_fakequant_act())
    return 1;
  if (!getenv("H3_CONVROT_PACK") || strcmp(getenv("H3_CONVROT_PACK"), "1") != 0) {
    printf("test_h3_convrot OK (math; set H3_CONVROT_PACK=1 for pruned DiT)\n");
    return 0;
  }
  if (test_pruned_pack())
    return 1;
  printf("test_h3_convrot OK\n");
  return 0;
}
