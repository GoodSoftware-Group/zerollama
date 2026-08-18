/* Weightless rematch of antirez Qwen3-VL TE RoPE / GQA / one layer. */
#include "h3_qwen_te_host.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int close_f(float a, float b, float eps) {
  return fabsf(a - b) <= eps;
}

static int test_rope_seq(void) {
  float cos[2 * 64], sin[2 * 64];
  CHECK(h3_qwen_te_rope_tables(2, 128, H3_QWEN_TE_ROPE_THETA, NULL, cos, sin) ==
        0);
  CHECK(close_f(cos[0], 1.f, 1e-6f));
  CHECK(close_f(sin[0], 0.f, 1e-6f));
  float inv0 = 1.f;
  float inv1 = 1.f / powf(H3_QWEN_TE_ROPE_THETA, 2.f / 128.f);
  CHECK(close_f(cos[64], cosf(1.f * inv0), 1e-5f));
  CHECK(close_f(sin[65], sinf(1.f * inv1), 1e-5f));
  return 0;
}

static int test_mrope(void) {
  /* axis-major [3, 1]: t=1 h=2 w=3 */
  uint32_t pos[3] = {1, 2, 3};
  float cos[64], sin[64];
  CHECK(h3_qwen_te_rope_tables(1, 128, H3_QWEN_TE_ROPE_THETA, pos, cos, sin) ==
        0);
  float inv0 = 1.f;
  float inv1 = 1.f / powf(H3_QWEN_TE_ROPE_THETA, 2.f / 128.f);
  float inv2 = 1.f / powf(H3_QWEN_TE_ROPE_THETA, 4.f / 128.f);
  /* i=0 axis 0 coord 1; i=1 axis 1 coord 2; i=2 axis 2 coord 3 */
  CHECK(close_f(cos[0], cosf(1.f * inv0), 1e-5f));
  CHECK(close_f(cos[1], cosf(2.f * inv1), 1e-5f));
  CHECK(close_f(cos[2], cosf(3.f * inv2), 1e-5f));
  return 0;
}

static int test_apply_rope(void) {
  float cos[2] = {0.6f, 0.8f};
  float sin[2] = {0.8f, 0.6f};
  float x[4] = {1.f, 2.f, 3.f, 4.f};
  CHECK(h3_qwen_te_apply_rope(x, 1, 1, 4, cos, sin) == 0);
  CHECK(close_f(x[0], 1.f * 0.6f - 3.f * 0.8f, 1e-5f));
  CHECK(close_f(x[2], 3.f * 0.6f + 1.f * 0.8f, 1e-5f));
  CHECK(close_f(x[1], 2.f * 0.8f - 4.f * 0.6f, 1e-5f));
  CHECK(close_f(x[3], 4.f * 0.8f + 2.f * 0.6f, 1e-5f));
  return 0;
}

static int test_repeat_kv(void) {
  float src[2 * 1 * 2] = {1.f, 2.f, 3.f, 4.f};
  float dst[2 * 2 * 2];
  CHECK(h3_qwen_te_repeat_kv(src, 2, 1, 2, 2, dst) == 0);
  CHECK(close_f(dst[0], 1.f, 1e-6f) && close_f(dst[2], 1.f, 1e-6f));
  CHECK(close_f(dst[4], 3.f, 1e-6f) && close_f(dst[6], 3.f, 1e-6f));
  return 0;
}

static void fill_eye(float *W, int out, int in) {
  memset(W, 0, (size_t)out * (size_t)in * sizeof(float));
  int n = out < in ? out : in;
  for (int i = 0; i < n; i++)
    W[i * in + i] = 1.f;
}

static int test_layer_finite(void) {
  const int T = 2, H = 8, nq = 2, nkv = 1, D = 4, F = 8;
  float hidden[16];
  for (int i = 0; i < 16; i++)
    hidden[i] = 0.1f * (float)(i + 1);
  float in_n[8], post_n[8], qn[4], kn[4];
  for (int i = 0; i < 8; i++) {
    in_n[i] = 1.f;
    post_n[i] = 1.f;
  }
  for (int i = 0; i < 4; i++) {
    qn[i] = 1.f;
    kn[i] = 1.f;
  }
  float Wq[8 * 8], Wk[4 * 8], Wv[4 * 8], Wo[8 * 8];
  float Wg[8 * 8], Wu[8 * 8], Wd[8 * 8];
  fill_eye(Wq, 8, 8);
  fill_eye(Wk, 4, 8);
  fill_eye(Wv, 4, 8);
  fill_eye(Wo, 8, 8);
  fill_eye(Wg, 8, 8);
  fill_eye(Wu, 8, 8);
  fill_eye(Wd, 8, 8);
  float cos[2 * 2], sin[2 * 2];
  CHECK(h3_qwen_te_rope_tables(2, 4, 10000.f, NULL, cos, sin) == 0);
  CHECK(h3_qwen_te_layer(hidden, T, H, nq, nkv, D, F, in_n, Wq, Wk, Wv, Wo, qn,
                         kn, post_n, Wg, Wu, Wd, cos, sin) == 0);
  for (int i = 0; i < 16; i++)
    CHECK(isfinite(hidden[i]));
  return 0;
}

static int test_hash_embed(void) {
  uint32_t ids[] = {32, 2518};
  float h[2 * 4];
  h3_qwen_te_hash_embed(ids, 2, 4, h);
  CHECK(fabsf(h[0] - h[4]) > 1e-6f);
  return 0;
}

int main(void) {
  if (test_rope_seq())
    return 1;
  if (test_mrope())
    return 1;
  if (test_apply_rope())
    return 1;
  if (test_repeat_kv())
    return 1;
  if (test_layer_finite())
    return 1;
  if (test_hash_embed())
    return 1;
  fprintf(stderr, "test_h3_qwen_te_host OK\n");
  return 0;
}
