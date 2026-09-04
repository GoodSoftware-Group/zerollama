/*
 * test_wan_lora.c — weightless: minimal safetensors fixture → pair
 * classification (prefix/suffix/alpha), merge math vs naive B@A, no-target
 * and shape-mismatch paths, .weight suffix alignment.
 */
#include "safetensors_min.h"
#include "wan_lora.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s\n", msg);                                      \
      return 1;                                                                \
    }                                                                          \
    printf("PASS: %s\n", msg);                                                 \
  } while (0)

/* Minimal safetensors writer (F32 only). names/dims arrays parallel. */
static int write_st(const char *path, const char **names, const int *ndims,
                    const long long **dims, const float **data, int nt) {
  char hdr[16384];
  size_t off = 0;
  off += (size_t)snprintf(hdr + off, sizeof(hdr) - off, "{");
  size_t data_off = 0;
  for (int i = 0; i < nt; i++) {
    long long ne = 1;
    for (int d = 0; d < ndims[i]; d++)
      ne *= dims[i][d];
    off += (size_t)snprintf(hdr + off, sizeof(hdr) - off,
                            "%s\"%s\":{\"dtype\":\"F32\",\"shape\":[",
                            i ? "," : "", names[i]);
    for (int d = 0; d < ndims[i]; d++)
      off += (size_t)snprintf(hdr + off, sizeof(hdr) - off, "%s%lld",
                              d ? "," : "", dims[i][d]);
    off += (size_t)snprintf(
        hdr + off, sizeof(hdr) - off,
        "],\"data_offsets\":[%zu,%zu]}", data_off,
        data_off + (size_t)ne * 4);
    data_off += (size_t)ne * 4;
  }
  off += (size_t)snprintf(hdr + off, sizeof(hdr) - off, "}");
  /* pad header to multiple of 8 with spaces */
  while (off % 8)
    hdr[off++] = ' ';
  FILE *f = fopen(path, "wb");
  if (!f)
    return -1;
  uint64_t hlen = off;
  if (fwrite(&hlen, 8, 1, f) != 1 || fwrite(hdr, 1, off, f) != off)
    goto fail;
  for (int i = 0; i < nt; i++) {
    long long ne = 1;
    for (int d = 0; d < ndims[i]; d++)
      ne *= dims[i][d];
    if (fwrite(data[i], 4, (size_t)ne, f) != (size_t)ne)
      goto fail;
  }
  fclose(f);
  return 0;
fail:
  fclose(f);
  return -1;
}

int main(void) {
  char dir[] = "/tmp/wanlora_XXXXXX";
  if (!mkdtemp(dir))
    return 1;
  char path[512];
  snprintf(path, sizeof(path), "%s/lora.safetensors", dir);

  /* Flat tensors as they live in the file. */
  static const long long d_rank_in[2] = {2, 4};
  static const long long d_q_out[2] = {3, 2};   /* blocks.0.self_attn.q */
  static const long long d_f_out[2] = {5, 2};   /* blocks.1.ffn.0 */
  static const long long d_scalar[1] = {1};

  float qA[8] = {1.f, 2.f, -1.f, 0.5f, 2.f, 0.f, 1.f, 1.f}; /* rank=2,in=4 */
  float qB[6] = {1.f, -1.f, 0.f, 2.f, 0.5f, 0.25f};         /* out=3,rank=2 */
  float alpha = 8.f;
  float fA[8] = {1.f, 1.f, 1.f, 1.f, -1.f, 0.f, 0.f, -1.f};
  float fB[10] = {1.f, 1.f, 2.f, -1.f, 0.f, 0.f, 1.f, 1.f, -1.f, 2.f};
  float junk[3] = {9.f, 9.f, 9.f};

  const char *names[] = {
      "diffusion_model.blocks.0.self_attn.q.lora_A.weight",
      "diffusion_model.blocks.0.self_attn.q.lora_B.weight",
      "diffusion_model.blocks.0.self_attn.q.alpha",
      "blocks.1.ffn.0.lora_down.weight",
      "blocks.1.ffn.0.lora_up.weight",
      "unrelated.weight",
  };
  const int ndims[] = {2, 2, 1, 2, 2, 1};
  const long long *dims[] = {d_rank_in, d_q_out, d_scalar, d_rank_in,
                             d_f_out, d_scalar};
  const float *data[] = {qA, qB, &alpha, fA, fB, junk};

  CHECK(write_st(path, names, ndims, dims, data, 6) == 0, "write fixture");

  wan_lora *L = wan_lora_open(path);
  CHECK(L != NULL, "open");
  CHECK(wan_lora_targets(L) == 2, "two complete pairs");

  /* Target 1: base blocks.0.self_attn.q, alpha=8, rank=2 → s=cli·4. */
  enum { OUT = 3, IN = 4 };
  float w[OUT * IN], ref[OUT * IN];
  for (int i = 0; i < OUT * IN; i++)
    w[i] = (float)(i % 7) - 3.f;
  memcpy(ref, w, sizeof(ref));
  float s = 0.5f * alpha / 2.f;
  for (int o = 0; o < OUT; o++)
    for (int r = 0; r < 2; r++) {
      float bv = s * qB[o * 2 + r];
      for (int k = 0; k < IN; k++)
        ref[o * IN + k] += bv * qA[r * IN + k];
    }
  CHECK(wan_lora_apply(L, "blocks.0.self_attn.q.weight", w, OUT * IN, 0.5f) ==
            1,
        "apply cond pair (.weight suffix aligned)");
  int bad = 0;
  for (int i = 0; i < OUT * IN; i++)
    if (fabsf(w[i] - ref[i]) > 1e-5f)
      bad = 1;
  CHECK(!bad, "merge math matches naive s·B@A");

  /* No-alpha pair defaults to 1/rank. */
  float w2[5 * 4];
  for (int i = 0; i < 20; i++)
    w2[i] = 0.f;
  CHECK(wan_lora_apply(L, "blocks.1.ffn.0.weight", w2, 20, 1.f) == 1,
        "apply lowercase alias pair");
  bad = 0;
  for (int o = 0; o < 5; o++)
    for (int k = 0; k < 4; k++) {
      float e = 0.f;
      for (int r = 0; r < 2; r++)
        e += fB[o * 2 + r] * fA[r * 4 + k];
      if (fabsf(w2[o * 4 + k] - e * 0.5f) > 1e-5f)
        bad = 1;
    }
  CHECK(!bad, "no-alpha scale = 1/rank");

  CHECK(wan_lora_apply(L, "blocks.99.nope.weight", w, OUT * IN, 1.f) == 0,
        "non-target reports untouched");
  bad = 0;
  for (int i = 0; i < OUT * IN; i++)
    if (w[i] != ref[i])
      bad = 1;
  CHECK(!bad, "non-target left bytes alone");

  CHECK(wan_lora_apply(L, "blocks.0.self_attn.k.weight", w, OUT * IN, 1.f) ==
            0,
        "unmatched sibling name untouched");
  CHECK(wan_lora_apply(L, "blocks.0.self_attn.q.weight", w, 7, 1.f) == -1,
        "shape mismatch refused");

  wan_lora_close(L);
  printf("=== test_wan_lora OK ===\n");
  return 0;
}
