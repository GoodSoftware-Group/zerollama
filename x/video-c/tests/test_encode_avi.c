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
    rgb[i] = (i % 3 == 0) ? 255.f : 0.f;
  int pcm_n = 800;
  float *pcm = (float *)calloc((size_t)2 * pcm_n, sizeof(float));
  if (!pcm) {
    free(rgb);
    return 1;
  }
  pcm[0] = 0.1f;
  pcm[pcm_n] = -0.1f;

  const char *path = "tests/tmp_encode_avi.avi";
  if (encode_avi_from_rgb(path, W, H, F, FPS, rgb, n, pcm, 2, pcm_n, 32000) !=
      0) {
    fprintf(stderr, "FAIL encode_avi_from_rgb\n");
    free(rgb);
    free(pcm);
    return 1;
  }
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    fprintf(stderr, "FAIL open avi\n");
    free(rgb);
    free(pcm);
    return 1;
  }
  char mag[12];
  if (fread(mag, 1, 12, fp) != 12 || memcmp(mag, "RIFF", 4) != 0 ||
      memcmp(mag + 8, "AVI ", 4) != 0) {
    fprintf(stderr, "FAIL AVI magic\n");
    fclose(fp);
    remove(path);
    free(rgb);
    free(pcm);
    return 1;
  }
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    remove(path);
    free(rgb);
    free(pcm);
    return 1;
  }
  long sz = ftell(fp);
  fclose(fp);
  if (sz < 1000) {
    fprintf(stderr, "FAIL avi too small %ld\n", sz);
    remove(path);
    free(rgb);
    free(pcm);
    return 1;
  }
  remove(path);
  free(rgb);
  free(pcm);
  printf("test_encode_avi OK (%ld bytes)\n", sz);
  return 0;
}
