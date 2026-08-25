#include "encode_mp4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  const int W = 16, H = 16, F = 2, FPS = 24;
  size_t n = (size_t)W * H * F * 3;
  float *rgb = (float *)malloc(n * sizeof(float));
  if (!rgb)
    return 1;
  for (size_t i = 0; i < n; i++)
    rgb[i] = (i % 3 == 0) ? 255.f : 40.f;

  const char *path = "tests/tmp_encode_mp4.mp4";
  if (encode_mp4_from_rgb(path, W, H, F, FPS, rgb, n) != 0) {
    fprintf(stderr, "FAIL encode_mp4_from_rgb\n");
    free(rgb);
    return 1;
  }
  FILE *fp = fopen(path, "rb");
  const char *avi = "tests/tmp_encode_mp4.avi";
  if (!fp)
    fp = fopen(avi, "rb");
  if (!fp) {
    fprintf(stderr, "FAIL open mp4/avi\n");
    free(rgb);
    return 1;
  }
  unsigned char mag[12];
  if (fread(mag, 1, 12, fp) != 12) {
    fprintf(stderr, "FAIL short media header\n");
    fclose(fp);
    remove(path);
    remove(avi);
    free(rgb);
    return 1;
  }
  int is_mp4 = mag[4] == 'f' && mag[5] == 't' && mag[6] == 'y' && mag[7] == 'p';
  int is_avi = memcmp(mag, "RIFF", 4) == 0 && memcmp(mag + 8, "AVI ", 4) == 0;
  fclose(fp);
  if (!is_mp4 && !is_avi) {
    fprintf(stderr, "FAIL neither ftyp nor AVI\n");
    remove(path);
    remove(avi);
    free(rgb);
    return 1;
  }
  remove(path);
  remove(avi);
  free(rgb);
  printf("test_encode_mp4 OK (%s)\n", is_mp4 ? "mp4" : "avi-fallback");
  return 0;
}
