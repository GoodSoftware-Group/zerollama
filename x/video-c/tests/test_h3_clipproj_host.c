/* ClipProj host rematch: affine formula + NicoLab28 control matrices if present. */
#include "h3_clipproj_host.h"

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

static int close_f(float a, float b, float eps) {
  return fabsf(a - b) <= eps;
}

static int test_affine_tiny(void) {
  const int seq = 2, din = 3, dout = 4;
  float h[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  float mean_in[] = {0.f, 0.f, 0.f};
  float std_in[] = {1.f, 1.f, 1.f};
  float mean_out[] = {0.1f, 0.2f, 0.3f, 0.4f};
  float std_out[] = {1.f, 1.f, 1.f, 1.f};
  /* W row-major [din, dout] */
  float W[12];
  for (int i = 0; i < 12; i++)
    W[i] = 0.1f * (float)(i + 1);
  float sink[4] = {9.f, 8.f, 7.f, 6.f};
  float out[8];
  CHECK(h3_clipproj_apply_affine(h, seq, din, dout, W, mean_in, std_in, mean_out,
                                 std_out, sink, out) == 0);
  /* Token 0 replaced by sink. */
  for (int j = 0; j < dout; j++)
    CHECK(close_f(out[j], sink[j], 1e-5f));
  /* Token 1: (h @ W) * std + mean with mean_in=0 std=1 */
  float yn[4] = {0};
  for (int j = 0; j < dout; j++) {
    float s = 0.f;
    for (int i = 0; i < din; i++)
      s += h[din + i] * W[i * dout + j];
    yn[j] = s;
  }
  for (int j = 0; j < dout; j++)
    CHECK(close_f(out[dout + j], yn[j] * std_out[j] + mean_out[j], 1e-4f));
  return 0;
}

static int test_control_file(const char *path, int expect_zero_like) {
  char err[256];
  h3_clipproj *p = h3_clipproj_load(path, err, sizeof(err));
  if (!p) {
    fprintf(stderr, "skip %s (%s)\n", path, err);
    return 0;
  }
  CHECK(h3_clipproj_din(p) == H3_CLIPPROJ_DIN_4B);
  CHECK(h3_clipproj_dout(p) == H3_CLIPPROJ_DOUT);
  CHECK(h3_clipproj_has_sink(p) == 1);

  const int seq = 3;
  float *h = calloc((size_t)seq * (size_t)H3_CLIPPROJ_DIN_4B, sizeof(float));
  float *out = calloc((size_t)seq * (size_t)H3_CLIPPROJ_DOUT, sizeof(float));
  CHECK(h && out);
  for (int i = 0; i < H3_CLIPPROJ_DIN_4B; i++) {
    h[H3_CLIPPROJ_DIN_4B + i] = 0.01f * (float)((i % 17) + 1);
    h[2 * H3_CLIPPROJ_DIN_4B + i] = -0.02f * (float)((i % 13) + 1);
  }
  CHECK(h3_clipproj_apply(p, h, seq, out, err, sizeof(err)) == 0);

  float *out2 = calloc((size_t)seq * (size_t)H3_CLIPPROJ_DOUT, sizeof(float));
  CHECK(out2);
  memset(h, 0, (size_t)seq * (size_t)H3_CLIPPROJ_DIN_4B * sizeof(float));
  CHECK(h3_clipproj_apply(p, h, seq, out2, err, sizeof(err)) == 0);
  for (int j = 0; j < H3_CLIPPROJ_DOUT; j++)
    CHECK(close_f(out[j], out2[j], 1e-5f));

  if (expect_zero_like) {
    for (int j = 0; j < H3_CLIPPROJ_DOUT; j++)
      CHECK(close_f(out2[H3_CLIPPROJ_DOUT + j],
                    out2[2 * H3_CLIPPROJ_DOUT + j], 1e-5f));
  }

  free(h);
  free(out);
  free(out2);
  h3_clipproj_free(p);
  printf("ok  %s\n", path);
  return 0;
}

/* Deterministic rematch vs numpy GELU + mlp.0/mlp.2 (celeb-mlp file). */
static int test_celeb_mlp(const char *path) {
  char err[256];
  h3_clipproj *p = h3_clipproj_load(path, err, sizeof(err));
  if (!p) {
    printf("skip celeb-mlp (%s)\n", err);
    return 0;
  }
  CHECK(h3_clipproj_has_mlp(p) == 1);
  CHECK(h3_clipproj_has_sink(p) == 1);

  const int seq = 3;
  float *h = calloc((size_t)seq * (size_t)H3_CLIPPROJ_DIN_4B, sizeof(float));
  float *out = calloc((size_t)seq * (size_t)H3_CLIPPROJ_DOUT, sizeof(float));
  CHECK(h && out);
  for (int s = 0; s < seq; s++)
    for (int i = 0; i < H3_CLIPPROJ_DIN_4B; i++)
      h[(size_t)s * H3_CLIPPROJ_DIN_4B + (size_t)i] =
          0.01f * (float)((s + 1) * ((i % 17) + 1));

  CHECK(h3_clipproj_apply(p, h, seq, out, err, sizeof(err)) == 0);

  /* sink_out prefix (stable across prompts) */
  CHECK(close_f(out[0], 0.6413230896f, 1e-4f));
  CHECK(close_f(out[1], -1.9392272234f, 1e-4f));
  CHECK(close_f(out[2], 1.9844863415f, 1e-4f));
  CHECK(close_f(out[3], -2.2252235413f, 1e-4f));

  const float *t1 = out + H3_CLIPPROJ_DOUT;
  const float *t2 = out + 2 * H3_CLIPPROJ_DOUT;
  CHECK(close_f(t1[0], -0.1058378518f, 2e-4f));
  CHECK(close_f(t1[1], 0.6174512506f, 2e-4f));
  CHECK(close_f(t1[2], -0.2079242468f, 2e-4f));
  CHECK(close_f(t1[3], -0.1166498140f, 2e-4f));
  CHECK(close_f(t2[0], 0.2274437845f, 2e-4f));
  CHECK(close_f(t2[1], 0.7661414146f, 2e-4f));

  float sum1 = 0.f, sum2 = 0.f;
  for (int j = 0; j < H3_CLIPPROJ_DOUT; j++) {
    sum1 += t1[j];
    sum2 += t2[j];
  }
  CHECK(close_f(sum1, 84.36067963f, 5e-2f));
  CHECK(close_f(sum2, 63.99664307f, 5e-2f));

  free(h);
  free(out);
  h3_clipproj_free(p);
  printf("ok  %s (mlp rematch)\n", path);
  return 0;
}

int main(void) {
  if (test_affine_tiny())
    return 1;

  const char *home = getenv("HOME");
  char zero[512], ident[512], mlp[512];
  if (home) {
    snprintf(zero, sizeof(zero),
             "%s/.zerollama/third_party/h3/clipproj/"
             "mmh3-ClipProj-control-zero.safetensors",
             home);
    snprintf(ident, sizeof(ident),
             "%s/.zerollama/third_party/h3/clipproj/"
             "mmh3-ClipProj-control-identity.safetensors",
             home);
    snprintf(mlp, sizeof(mlp),
             "%s/.zerollama/third_party/h3/clipproj/"
             "mmh3-4b-ClipProj-celeb-mlp.safetensors",
             home);
    if (access(zero, R_OK) == 0) {
      if (test_control_file(zero, 1))
        return 1;
    } else {
      printf("skip control-zero (not at %s)\n", zero);
    }
    if (access(ident, R_OK) == 0) {
      if (test_control_file(ident, 0))
        return 1;
    } else {
      printf("skip control-identity (not at %s)\n", ident);
    }
    if (access(mlp, R_OK) == 0) {
      if (test_celeb_mlp(mlp))
        return 1;
    } else {
      printf("skip celeb-mlp (not at %s)\n", mlp);
    }
  }

  printf("test_h3_clipproj_host OK\n");
  return 0;
}
