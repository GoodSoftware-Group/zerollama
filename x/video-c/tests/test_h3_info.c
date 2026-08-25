/* Fixture smoke for h3_checkpoint_info (antirez h3_load_dir-shaped stubs). */
#include "h3_info.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int mk(const char *path) { return mkdir(path, 0755); }

static int touch(const char *path) {
  FILE *f = fopen(path, "wb");
  if (!f)
    return -1;
  fwrite("x", 1, 1, f);
  fclose(f);
  return 0;
}

static int touch_st(const char *dir, const char *name) {
  char path[768];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  return touch(path);
}

static int build_fixture(char *dir, size_t n) {
  snprintf(dir, n, "/tmp/video_c_h3_info_XXXXXX");
  if (!mkdtemp(dir))
    return -1;
  char p[768];
  snprintf(p, sizeof(p), "%s/FL2VA", dir);
  if (mk(p) != 0)
    return -1;
  snprintf(p, sizeof(p), "%s/FL2VA/transformer", dir);
  if (mk(p) != 0)
    return -1;
  snprintf(p, sizeof(p), "%s/FL2VA/transformer/config.json", dir);
  if (touch(p) != 0)
    return -1;
  snprintf(p, sizeof(p), "%s/FL2VA/transformer", dir);
  if (touch_st(p, "model.safetensors") != 0)
    return -1;

  snprintf(p, sizeof(p), "%s/FL2VA/tokenizer", dir);
  if (mk(p) != 0)
    return -1;
  snprintf(p, sizeof(p), "%s/FL2VA/tokenizer/tokenizer.json", dir);
  if (touch(p) != 0)
    return -1;

  snprintf(p, sizeof(p), "%s/FL2VA/text_encoder", dir);
  if (mk(p) != 0)
    return -1;
  if (touch_st(p, "model.safetensors") != 0)
    return -1;

  snprintf(p, sizeof(p), "%s/FL2VA/video_vae", dir);
  if (mk(p) != 0)
    return -1;
  snprintf(p, sizeof(p), "%s/FL2VA/video_vae/source", dir);
  if (mk(p) != 0)
    return -1;
  if (touch_st(p, "vae.safetensors") != 0)
    return -1;

  snprintf(p, sizeof(p), "%s/FL2VA/audio_vae", dir);
  if (mk(p) != 0)
    return -1;
  if (touch_st(p, "audio.safetensors") != 0)
    return -1;
  return 0;
}

/* Mac-lab shape: tokenizer + VAEs, no DiT/TE shards. */
static int build_vae_only_fixture(char *dir, size_t n) {
  snprintf(dir, n, "/tmp/video_c_h3_vae_XXXXXX");
  if (!mkdtemp(dir))
    return -1;
  char p[768];
  snprintf(p, sizeof(p), "%s/FL2VA", dir);
  if (mk(p) != 0)
    return -1;
  snprintf(p, sizeof(p), "%s/FL2VA/transformer", dir);
  if (mk(p) != 0)
    return -1;
  snprintf(p, sizeof(p), "%s/FL2VA/transformer/config.json", dir);
  if (touch(p) != 0)
    return -1;
  snprintf(p, sizeof(p), "%s/FL2VA/tokenizer", dir);
  if (mk(p) != 0)
    return -1;
  snprintf(p, sizeof(p), "%s/FL2VA/tokenizer/tokenizer.json", dir);
  if (touch(p) != 0)
    return -1;
  snprintf(p, sizeof(p), "%s/FL2VA/video_vae", dir);
  if (mk(p) != 0)
    return -1;
  snprintf(p, sizeof(p), "%s/FL2VA/video_vae/source", dir);
  if (mk(p) != 0)
    return -1;
  if (touch_st(p, "vae.safetensors") != 0)
    return -1;
  snprintf(p, sizeof(p), "%s/FL2VA/audio_vae", dir);
  if (mk(p) != 0)
    return -1;
  if (touch_st(p, "audio.safetensors") != 0)
    return -1;
  return 0;
}

int main(void) {
  if (h3_checkpoint_info(NULL) == 0) {
    fprintf(stderr, "FAIL: NULL dir should fail\n");
    return 1;
  }
  if (h3_checkpoint_info("/tmp/video_c_h3_info_missing_xyz") == 0) {
    fprintf(stderr, "FAIL: missing dir should fail\n");
    return 1;
  }

  char empty[256];
  snprintf(empty, sizeof(empty), "/tmp/video_c_h3_empty_XXXXXX");
  if (!mkdtemp(empty)) {
    perror("mkdtemp empty");
    return 1;
  }
  if (h3_checkpoint_info(empty) == 0) {
    fprintf(stderr, "FAIL: empty dir without FL2VA should fail\n");
    return 1;
  }

  char soft[256];
  snprintf(soft, sizeof(soft), "/tmp/video_c_h3_soft_XXXXXX");
  if (!mkdtemp(soft))
    return 1;
  char p[300];
  snprintf(p, sizeof(p), "%s/FL2VA", soft);
  mk(p);
  if (h3_checkpoint_info(soft) == 0) {
    fprintf(stderr, "FAIL: FL2VA without config/shards should fail\n");
    return 1;
  }

  char fix[256];
  if (build_fixture(fix, sizeof(fix)) != 0) {
    perror("fixture");
    return 1;
  }
  if (h3_checkpoint_info(fix) != 0) {
    fprintf(stderr, "FAIL: full FL2VA fixture should pass\n");
    return 1;
  }

  char vae[256];
  if (build_vae_only_fixture(vae, sizeof(vae)) != 0) {
    perror("vae-only fixture");
    return 1;
  }
  if (h3_checkpoint_info(vae) != 0) {
    fprintf(stderr, "FAIL: VAE-only snapshot (no DiT/TE) should probe ok\n");
    return 1;
  }

  printf("test_h3_info OK\n");
  return 0;
}
