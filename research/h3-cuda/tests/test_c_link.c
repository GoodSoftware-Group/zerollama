#include "h3_gpu.h"
#include <stdio.h>
int main(void) {
  char err[128];
  h3_gpu *g = h3_gpu_create(NULL, err, sizeof(err));
  if (!g) { fprintf(stderr, "%s\n", err); return 1; }
  printf("ok C link create/free m5=%d int8=%d\n", h3_gpu_is_m5(g), h3_gpu_has_int8_mlp(g));
  h3_gpu_free(g);
  return 0;
}
