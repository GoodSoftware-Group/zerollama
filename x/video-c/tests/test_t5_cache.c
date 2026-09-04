/*
 * test_t5_cache.c — weightless: path resolution, store→get roundtrip,
 * header-mismatch invalidation, and cache disable via WAN_T5_CACHE=0.
 */
#include "t5_cache.h"
#include "wan_config.h"
#include "wan_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s\n", msg);                                      \
      return 1;                                                                \
    }                                                                          \
    printf("PASS: %s\n", msg);                                                 \
  } while (0)

int main(void) {
  char dir[] = "/tmp/wt5c_testXXXXXX";
  if (!mkdtemp(dir))
    return 1;
  setenv("WAN_T5_CACHE_DIR", dir, 1);
  unsetenv("WAN_T5_CACHE");

  wan_ctx ctx;
  memset(&ctx, 0, sizeof(ctx));
  snprintf(ctx.ckpt_dir, sizeof(ctx.ckpt_dir), "/fake/ckpt");
  ctx.cfg.text_len = 512;
  ctx.cfg.text_dim = 1536;

  char p1[4096], p2[4096];
  CHECK(wan_t5_cache_path(&ctx, "a red apple", p1, sizeof(p1)) == 0,
        "path resolve");
  CHECK(strstr(p1, dir) == p1, "path under override dir");
  CHECK(wan_t5_cache_path(&ctx, "a green apple", p2, sizeof(p2)) == 0,
        "path resolve 2");
  CHECK(strcmp(p1, p2) != 0, "different prompts → different keys");

  /* Same prompt, different ckpt dir → different key (no cross-pack hits). */
  {
    wan_ctx other = ctx;
    snprintf(other.ckpt_dir, sizeof(other.ckpt_dir), "/other/ckpt");
    char q[4096];
    CHECK(wan_t5_cache_path(&other, "a red apple", q, sizeof(q)) == 0,
          "path resolve ckpt2");
    CHECK(strcmp(p1, q) != 0, "different ckpts → different keys");
  }

  enum { N = 512 * 1536 };
  float *emb = calloc(N, sizeof(float));
  float *got = calloc(N, sizeof(float));
  CHECK(emb && got, "alloc");
  for (int i = 0; i < N; i++)
    emb[i] = (float)(i % 97) * 0.01f;

  CHECK(wan_t5_cache_get(&ctx, "a red apple", got, N) != 0, "miss before put");
  CHECK(wan_t5_cache_put(&ctx, "a red apple", emb, N) == 0, "put");
  memset(got, 0, sizeof(float) * N);
  CHECK(wan_t5_cache_get(&ctx, "a red apple", got, N) == 0, "hit after put");
  int bad = 0;
  for (int i = 0; i < N; i++)
    if (got[i] != emb[i]) {
      bad = 1;
      break;
    }
  CHECK(!bad, "roundtrip bytes identical");

  /* Shape mismatch must invalidate, not serve garbage. */
  ctx.cfg.text_len = 256;
  CHECK(wan_t5_cache_get(&ctx, "a red apple", got, 256 * 1536) != 0,
        "header shape mismatch refuses hit");

  /* Disable switch. */
  setenv("WAN_T5_CACHE", "0", 1);
  ctx.cfg.text_len = 512;
  CHECK(wan_t5_cache_get(&ctx, "a red apple", got, N) != 0,
        "WAN_T5_CACHE=0 disables get");
  CHECK(wan_t5_cache_path(&ctx, "x", p1, sizeof(p1)) != 0,
        "path resolve honors disable");

  /* Deep, not-yet-existing dir must be created recursively (the default
   * ~/.zerollama/cache/wan_t5 path hit this). */
  setenv("WAN_T5_CACHE", "", 1);
  unsetenv("WAN_T5_CACHE");
  char deep[512];
  snprintf(deep, sizeof(deep), "%s/a/b/c", dir);
  setenv("WAN_T5_CACHE_DIR", deep, 1);
  CHECK(wan_t5_cache_path(&ctx, "deep", p1, sizeof(p1)) == 0,
        "path resolve deep dir");
  CHECK(wan_t5_cache_put(&ctx, "deep", emb, N) == 0, "put creates parents");
  CHECK(wan_t5_cache_get(&ctx, "deep", got, N) == 0, "get from deep dir");

  free(emb);
  free(got);
  printf("=== test_t5_cache OK ===\n");
  return 0;
}
