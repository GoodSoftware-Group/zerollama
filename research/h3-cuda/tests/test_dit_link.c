/* Compile/link check: portable h3_dit.h + CUDA backend. No weights. */
#include "h3_dit.h"
#include "h3_gpu.h"

#include <stdio.h>

int main(void) {
  char err[256];
  h3_gpu *gpu = h3_gpu_create(NULL, err, sizeof(err));
  if (!gpu) {
    fprintf(stderr, "create: %s\n", err);
    return 1;
  }
  /* Force portable DiT object into the link. */
  h3_dit_free(NULL);
  printf("ok dit link  gpu=%p  has_int8=%d  (portable h3_dit + libh3_cuda)\n",
         (void *)gpu, h3_gpu_has_int8_mlp(gpu));
  h3_gpu_free(gpu);
  return 0;
}
