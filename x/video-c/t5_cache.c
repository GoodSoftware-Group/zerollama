/*
 * t5_cache.c — see t5_cache.h for the why.
 */
#include "t5_cache.h"
#include "wan_internal.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define WT5C_MAGIC 0x57433554u /* 'T5CW' little-endian */
#define WT5C_VERSION 1u

static uint64_t fnv1a64_init(void) { return 1469598103934665603ull; }

static uint64_t fnv1a64_byte(uint64_t h, uint8_t b) {
  h ^= b;
  h *= 1099511628211ull;
  return h;
}

/* mkdir -p for the cache dir; best-effort, verified with stat(). */
static int ensure_dir(const char *dir) {
  char tmp[1024];
  size_t len = strlen(dir);
  if (len == 0 || len >= sizeof(tmp))
    return -1;
  snprintf(tmp, sizeof(tmp), "%s", dir);
  for (char *p = tmp + 1; *p; p++) {
    if (*p != '/')
      continue;
    *p = '\0';
    if (mkdir(tmp, 0777) != 0 && errno != EEXIST)
      return -1;
    *p = '/';
  }
  if (mkdir(tmp, 0777) != 0 && errno != EEXIST)
    return -1;
  struct stat st;
  return (stat(tmp, &st) == 0 && S_ISDIR(st.st_mode)) ? 0 : -1;
}

int wan_t5_cache_path(const struct wan_ctx *ctx, const char *prompt, char *out,
                      size_t cap) {
  if (!ctx || !prompt || !out || cap < 64)
    return -1;
  const char *env = getenv("WAN_T5_CACHE");
  if (env && env[0] && env[0] == '0')
    return -1;
  char dir[1024];
  env = getenv("WAN_T5_CACHE_DIR");
  if (env && env[0]) {
    snprintf(dir, sizeof(dir), "%s", env);
  } else {
    const char *home = getenv("HOME");
    if (!home || !home[0])
      return -1;
    snprintf(dir, sizeof(dir), "%s/.zerollama/cache/wan_t5", home);
  }
  if (ensure_dir(dir) != 0)
    return -1;
  /* Key spans prompt AND ckpt dir so different packs never collide. */
  uint64_t k = fnv1a64_init();
  for (const char *p = prompt; *p; p++)
    k = fnv1a64_byte(k, (uint8_t)*p);
  k = fnv1a64_byte(k, 0);
  for (const char *c = ctx->ckpt_dir; *c; c++)
    k = fnv1a64_byte(k, (uint8_t)*c);
  snprintf(out, cap, "%s/wt5c_%016llx.f32", dir, (unsigned long long)k);
  return 0;
}

int wan_t5_cache_get(const struct wan_ctx *ctx, const char *prompt, float *emb,
                     size_t n) {
  char path[4096];
  if (wan_t5_cache_path(ctx, prompt, path, sizeof(path)) != 0)
    return -1;
  FILE *f = fopen(path, "rb");
  if (!f)
    return -1;
  uint32_t hdr[4] = {0, 0, 0, 0};
  if (fread(hdr, sizeof(uint32_t), 4, f) != 4 || hdr[0] != WT5C_MAGIC ||
      hdr[1] != WT5C_VERSION || (size_t)hdr[2] * (size_t)hdr[3] != n ||
      n == 0) {
    fclose(f);
    return -1;
  }
  if (fread(emb, sizeof(float), n, f) != n) {
    fclose(f);
    return -1;
  }
  fclose(f);
  return 0;
}

int wan_t5_cache_put(const struct wan_ctx *ctx, const char *prompt,
                     const float *emb, size_t n) {
  char path[4096];
  if (wan_t5_cache_path(ctx, prompt, path, sizeof(path)) != 0 || n == 0)
    return -1;
  /* tmp + rename so a killed render never leaves a truncated entry. */
  char tmp[4200];
  snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, (int)getpid());
  FILE *f = fopen(tmp, "wb");
  if (!f)
    return -1;
  uint32_t hdr[4] = {WT5C_MAGIC, WT5C_VERSION,
                     (uint32_t)ctx->cfg.text_len,
                     (uint32_t)ctx->cfg.text_dim};
  if (fwrite(hdr, sizeof(uint32_t), 4, f) != 4 ||
      fwrite(emb, sizeof(float), n, f) != n) {
    fclose(f);
    unlink(tmp);
    return -1;
  }
  fclose(f);
  if (rename(tmp, path) != 0) {
    unlink(tmp);
    return -1;
  }
  return 0;
}
