#include "h3_text_cond.h"

#include <math.h>
#include <stdint.h>
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

int main(void) {
  const char *home = getenv("HOME");
  if (!home) {
    printf("test_h3_text_cond SKIP\n");
    return 0;
  }
  char clip[768];
  snprintf(clip, sizeof(clip),
           "%s/.zerollama/third_party/h3/clipproj/"
           "mmh3-4b-ClipProj-celeb-mlp.safetensors",
           home);
  if (access(clip, R_OK) != 0) {
    printf("test_h3_text_cond SKIP (no ClipProj)\n");
    return 0;
  }
  /* Force hash TE so this stays a short unit test. */
  setenv("H3_QWEN_TE_DIR", "/nonexistent", 1);
  char err[256];
  h3_text_cond c;
  if (h3_text_cond_from_prompt("A red fox", NULL, NULL, 0, clip, &c, err,
                               sizeof(err)) != 0) {
    fprintf(stderr, "FAIL text_cond: %s\n", err);
    return 1;
  }
  CHECK(c.nt >= 1 && c.cond);
  CHECK(c.used_4b == 0);
  CHECK(c.tags);
  CHECK(c.tags[0] == 1);
  int finite = 1;
  double sq = 0;
  size_t n = (size_t)c.nt * 5120;
  for (size_t i = 0; i < n; i++) {
    if (!isfinite(c.cond[i]))
      finite = 0;
    sq += (double)c.cond[i] * c.cond[i];
  }
  CHECK(finite);
  int nt_ok = c.nt;
  h3_text_cond_free(&c);

  char tmp[256];
  snprintf(tmp, sizeof(tmp), "/tmp/h3te_tags_%d.bin", (int)getpid());
  FILE *f = fopen(tmp, "wb");
  CHECK(f);
  uint32_t nt = 2, dim = 5120;
  fwrite("H3TE", 1, 4, f);
  fwrite(&nt, 4, 1, f);
  fwrite(&dim, 4, 1, f);
  float row[5120];
  memset(row, 0, sizeof(row));
  row[0] = 1.f;
  fwrite(row, sizeof(float), 5120, f);
  row[0] = 2.f;
  fwrite(row, sizeof(float), 5120, f);
  uint8_t tg[2] = {0, 1};
  fwrite(tg, 1, 2, f);
  fclose(f);
  h3_text_cond d;
  CHECK(h3_text_cond_from_bin(tmp, &d, err, sizeof(err)) == 0);
  CHECK(d.nt == 2 && d.used_dump == 1 && d.tags);
  CHECK(d.tags[0] == 0 && d.tags[1] == 1);
  h3_text_cond_free(&d);
  unlink(tmp);

  printf("test_h3_text_cond OK nt=%d rms=%.6g (hash TE)\n", nt_ok,
         sqrt(sq / (double)n));
  return 0;
}
