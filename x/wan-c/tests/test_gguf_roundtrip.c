/*
 * Write a tiny GGUF via the Python converter semantics (inline C writer),
 * then read it back with gguf_min and verify relative offsets + values.
 */
#include "gguf_min.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747u
#define GGUF_VERSION 3
#define ALIGN 32

static void pack_str(FILE *f, const char *s) {
  uint64_t n = (uint64_t)strlen(s);
  fwrite(&n, 8, 1, f);
  fwrite(s, 1, (size_t)n, f);
}

static size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

static int write_tiny_gguf(const char *path) {
  float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[2] = {5.5f, 6.5f};

  /* Build info blobs in memory to know meta size. */
  /* Tensor A: name "t.a", shape pytorch [2,2] → ggml [2,2] reversed same,
   * Tensor B: name "t.b", shape [2]. */

  FILE *f = fopen(path, "wb");
  if (!f)
    return -1;

  /* We'll write header+infos with placeholder offsets, then pad, then data
   * with relative offsets patched — same as convert_wan_to_gguf.py. */
  uint8_t *info_a = NULL;
  uint8_t *info_b = NULL;
  size_t info_a_n = 0, info_b_n = 0;

  /* Serialize infos into heap buffers. */
  {
    size_t cap = 256;
    info_a = calloc(1, cap);
    info_b = calloc(1, cap);
    if (!info_a || !info_b) {
      free(info_a);
      free(info_b);
      fclose(f);
      return -1;
    }
    /* manual: string + ndim + dims + dtype + offset */
#define PACK_INFO(buf, nref, name, ndim, d0, d1, dtype)                        \
  do {                                                                         \
    size_t pos = 0;                                                            \
    uint64_t sl = strlen(name);                                                \
    memcpy(buf + pos, &sl, 8);                                                 \
    pos += 8;                                                                  \
    memcpy(buf + pos, name, (size_t)sl);                                       \
    pos += (size_t)sl;                                                         \
    uint32_t nd = (ndim);                                                      \
    memcpy(buf + pos, &nd, 4);                                                 \
    pos += 4;                                                                  \
    uint64_t dim0 = (d0);                                                      \
    memcpy(buf + pos, &dim0, 8);                                               \
    pos += 8;                                                                  \
    if ((ndim) > 1) {                                                          \
      uint64_t dim1 = (d1);                                                    \
      memcpy(buf + pos, &dim1, 8);                                             \
      pos += 8;                                                                \
    }                                                                          \
    uint32_t dt = (dtype);                                                     \
    memcpy(buf + pos, &dt, 4);                                                 \
    pos += 4;                                                                  \
    uint64_t off = 0;                                                          \
    memcpy(buf + pos, &off, 8);                                                \
    pos += 8;                                                                  \
    (nref) = pos;                                                              \
  } while (0)

    PACK_INFO(info_a, info_a_n, "t.a", 2, 2, 2, 0); /* F32, ggml dims */
    PACK_INFO(info_b, info_b_n, "t.b", 1, 2, 0, 0);
#undef PACK_INFO
  }

  uint32_t magic = GGUF_MAGIC, ver = GGUF_VERSION;
  uint64_t n_tensors = 2, n_kv = 1;
  fwrite(&magic, 4, 1, f);
  fwrite(&ver, 4, 1, f);
  fwrite(&n_tensors, 8, 1, f);
  fwrite(&n_kv, 8, 1, f);
  pack_str(f, "general.architecture");
  uint32_t st = 8; /* STRING */
  fwrite(&st, 4, 1, f);
  pack_str(f, "wan.t2v");

  long info_start = ftell(f);
  fwrite(info_a, 1, info_a_n, f);
  fwrite(info_b, 1, info_b_n, f);
  long meta_end = ftell(f);
  size_t pad = align_up((size_t)meta_end, ALIGN) - (size_t)meta_end;
  for (size_t i = 0; i < pad; i++)
    fputc(0, f);
  long data_base = ftell(f);

  /* relative offsets */
  uint64_t off_a = 0;
  fwrite(a, sizeof(float), 4, f);
  while ((ftell(f) - data_base) % ALIGN)
    fputc(0, f);
  uint64_t off_b = (uint64_t)(ftell(f) - data_base);
  fwrite(b, sizeof(float), 2, f);

  /* patch offsets in info blobs (last 8 bytes) */
  fseek(f, info_start + (long)info_a_n - 8, SEEK_SET);
  fwrite(&off_a, 8, 1, f);
  fseek(f, info_start + (long)info_a_n + (long)info_b_n - 8, SEEK_SET);
  fwrite(&off_b, 8, 1, f);

  free(info_a);
  free(info_b);
  fclose(f);
  return 0;
}

int main(void) {
  char path[] = "/tmp/wan_gguf_rt_XXXXXX";
  int fd = mkstemp(path);
  if (fd < 0) {
    perror("mkstemp");
    return 1;
  }
  close(fd);
  unlink(path);

  if (write_tiny_gguf(path) != 0) {
    fprintf(stderr, "FAIL write\n");
    return 1;
  }

  gguf_file *gf = gguf_open(path);
  if (!gf) {
    fprintf(stderr, "FAIL open\n");
    unlink(path);
    return 1;
  }
  if (gguf_tensor_count(gf) != 2) {
    fprintf(stderr, "FAIL count\n");
    gguf_close(gf);
    unlink(path);
    return 1;
  }

  const gguf_tensor_t *ta = gguf_find_tensor(gf, "t.a");
  const gguf_tensor_t *tb = gguf_find_tensor(gf, "t.b");
  if (!ta || !tb) {
    fprintf(stderr, "FAIL find\n");
    gguf_close(gf);
    unlink(path);
    return 1;
  }

  float fa[4], fb[2];
  if (gguf_tensor_to_f32(gf, ta, fa, 4) != 0 ||
      gguf_tensor_to_f32(gf, tb, fb, 2) != 0) {
    fprintf(stderr, "FAIL to_f32 (NULL data?)\n");
    gguf_close(gf);
    unlink(path);
    return 1;
  }

  if (fabsf(fa[0] - 1.0f) > 1e-6f || fabsf(fa[3] - 4.0f) > 1e-6f ||
      fabsf(fb[0] - 5.5f) > 1e-6f || fabsf(fb[1] - 6.5f) > 1e-6f) {
    fprintf(stderr, "FAIL values %g %g %g %g | %g %g\n", fa[0], fa[1], fa[2],
            fa[3], fb[0], fb[1]);
    gguf_close(gf);
    unlink(path);
    return 1;
  }

  gguf_close(gf);
  unlink(path);
  printf("test_gguf_roundtrip OK\n");
  return 0;
}
