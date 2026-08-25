#include <cstdio>
#include <cstring>
#include "h3_cuda.h"
#include "h3_gpu.h"

int main() {
  h3_device_info info{};
  char err[256]{};
  if (h3_cuda_probe(&info, err, sizeof(err)) != 0) {
    std::fprintf(stderr, "probe failed: %s\n", err);
    return 1;
  }
  std::printf("cuda_device=%s arch=%s vram_bytes=%llu unified=%d\n", info.name,
              info.architecture, (unsigned long long)info.physical_memory,
              info.unified_memory);

  char cerr[256]{};
  h3_gpu *gpu = h3_gpu_create(nullptr, cerr, sizeof(cerr));
  if (!gpu) {
    std::fprintf(stderr, "create failed: %s\n", cerr);
    return 1;
  }
  h3_gpu_free(gpu);
  return 0;
}
