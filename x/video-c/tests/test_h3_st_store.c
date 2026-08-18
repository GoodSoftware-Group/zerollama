/* Multi-shard store smoke + MLX schedule rematch vs antirez h3_schedule_build. */
#include "h3_host.h"
#include "h3_st_store.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int close_enough(double a, double b, double eps) {
  return fabs(a - b) <= eps;
}

/* minimax-h3-mlx scheduler n=21 video/audio (dump_h3_mlx_schedule.py). */
static const float k_mlx_video_n21[] = {
    1.0f,           0.995633185f,  0.990825653f,  0.985507309f,  0.979591846f,
    0.972972989f,   0.965517223f,  0.957055211f,  0.947368383f,  0.936170220f,
    0.923076928f,   0.907562971f,  0.888888896f,  0.865979373f,  0.837209284f,
    0.800000012f,   0.75f,         0.679245293f,  0.571428597f,  0.387096792f,
    0.0f};
static const float k_mlx_audio_n21[] = {
    1.0f,          0.982758582f, 0.964285672f, 0.944444478f, 0.923076987f,
    0.899999976f,  0.874999940f, 0.847826064f, 0.818181813f, 0.785714388f,
    0.75f,         0.710526288f, 0.666666687f, 0.617646992f, 0.5625f,
    0.5f,          0.428571463f, 0.346153885f, 0.25f,        0.136363640f,
    0.0f};

static int test_mlx_schedule_rematch(void) {
  h3_sigma_schedule s;
  /* antirez steps=20 + terminal 0 ↔ MLX linspace n=21 */
  CHECK(h3_schedule_build(20, &s));
  CHECK(s.steps == 20);
  for (int i = 0; i <= 20; i++) {
    CHECK(close_enough(s.video[i], k_mlx_video_n21[i], 1e-6));
    CHECK(close_enough(s.audio[i], k_mlx_audio_n21[i], 1e-6));
  }
  return 0;
}

/* Minimal single-tensor F32 safetensors writer. */
static int write_st(const char *path, const char *name, const float *data,
                    size_t n) {
  char header[256];
  int hlen = snprintf(header, sizeof(header),
                      "{\"%s\":{\"dtype\":\"F32\",\"shape\":[%zu],\"data_"
                      "offsets\":[0,%zu]}}",
                      name, n, n * 4);
  if (hlen < 0 || (size_t)hlen >= sizeof(header))
    return -1;
  /* pad header to 8-byte alignment as some readers expect */
  while ((hlen + 8) % 8 != 0)
    header[hlen++] = ' ';
  header[hlen] = 0;
  FILE *f = fopen(path, "wb");
  if (!f)
    return -1;
  uint64_t hl = (uint64_t)hlen;
  fwrite(&hl, 1, 8, f);
  fwrite(header, 1, (size_t)hlen, f);
  fwrite(data, 4, n, f);
  fclose(f);
  return 0;
}

static int test_st_store(void) {
  char dir[] = "/tmp/video_c_h3_st_XXXXXX";
  if (!mkdtemp(dir))
    return 1;
  char a[300], b[300];
  snprintf(a, sizeof(a), "%s/a.safetensors", dir);
  snprintf(b, sizeof(b), "%s/b.safetensors", dir);
  float va[2] = {1.0f, 2.0f};
  float vb[1] = {3.0f};
  CHECK(write_st(a, "w.a", va, 2) == 0);
  CHECK(write_st(b, "w.b", vb, 1) == 0);

  char err[256];
  h3_st_store *store = h3_st_store_open(dir, err, sizeof(err));
  CHECK(store != NULL);
  CHECK(h3_st_store_shards(store) == 2);
  CHECK(h3_st_store_tensors(store) == 2);
  const st_file *sf = NULL;
  const st_tensor_t *t = h3_st_store_find(store, "w.b", &sf);
  CHECK(t && sf);
  float out[1];
  CHECK(st_tensor_to_f32(sf, t, out, 1) == 0);
  CHECK(out[0] == 3.0f);
  float via[2];
  CHECK(h3_st_store_load_f32(store, "w.a", via, 2, err, sizeof(err)) == 0);
  CHECK(via[0] == 1.0f && via[1] == 2.0f);
  CHECK(h3_st_store_load_f32(store, "missing", via, 2, err, sizeof(err)) != 0);
  CHECK(h3_st_store_find(store, "missing", NULL) == NULL);
  h3_st_store_free(store);
  return 0;
}

/* Minimal BF16 safetensors (1.0 → 0x3f80). */
static int write_st_bf16(const char *path, const char *name,
                         const uint16_t *data, size_t n) {
  char header[256];
  int hlen = snprintf(header, sizeof(header),
                      "{\"%s\":{\"dtype\":\"BF16\",\"shape\":[%zu],\"data_"
                      "offsets\":[0,%zu]}}",
                      name, n, n * 2);
  if (hlen < 0 || (size_t)hlen >= sizeof(header))
    return -1;
  while ((hlen + 8) % 8 != 0)
    header[hlen++] = ' ';
  header[hlen] = 0;
  FILE *f = fopen(path, "wb");
  if (!f)
    return -1;
  uint64_t hl = (uint64_t)hlen;
  fwrite(&hl, 1, 8, f);
  fwrite(header, 1, (size_t)hlen, f);
  fwrite(data, 2, n, f);
  fclose(f);
  return 0;
}

static int test_st_store_recursive_and_bf16(void) {
  char dir[] = "/tmp/video_c_h3_st_r_XXXXXX";
  if (!mkdtemp(dir))
    return 1;
  char nested[320], leaf[340], top[320];
  snprintf(nested, sizeof(nested), "%s/nested", dir);
  CHECK(mkdir(nested, 0700) == 0);
  snprintf(leaf, sizeof(leaf), "%s/c.safetensors", nested);
  snprintf(top, sizeof(top), "%s/top.safetensors", dir);
  float topv[1] = {9.0f};
  uint16_t bf[1] = {0x3f80}; /* BF16 1.0 */
  CHECK(write_st(top, "w.top", topv, 1) == 0);
  CHECK(write_st_bf16(leaf, "w.bf", bf, 1) == 0);

  char err[256];
  /* Non-recursive: only top-level shard. */
  h3_st_store *flat = h3_st_store_open(dir, err, sizeof(err));
  CHECK(flat != NULL);
  CHECK(h3_st_store_shards(flat) == 1);
  CHECK(h3_st_store_find(flat, "w.bf", NULL) == NULL);
  h3_st_store_free(flat);

  h3_st_store *deep = h3_st_store_open_ex(dir, 1, err, sizeof(err));
  CHECK(deep != NULL);
  CHECK(h3_st_store_shards(deep) == 2);
  float out[1];
  CHECK(h3_st_store_load_f32(deep, "w.bf", out, 1, err, sizeof(err)) == 0);
  CHECK(out[0] == 1.0f);
  CHECK(h3_st_store_load_f32(deep, "w.top", out, 1, err, sizeof(err)) == 0);
  CHECK(out[0] == 9.0f);
  h3_st_store_free(deep);
  return 0;
}

int main(void) {
  if (test_mlx_schedule_rematch())
    return 1;
  if (test_st_store())
    return 1;
  if (test_st_store_recursive_and_bf16())
    return 1;
  printf("test_h3_st_store OK (mlx schedule + multi-shard + load/recurse)\n");
  return 0;
}
