/* Minimal CUDA device probe for MiniMax-H3 lab port. Does not load weights. */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include "../../include/h3_cuda.h"

extern "C" int h3_cuda_probe(h3_device_info *info, char *error, size_t error_size) {
  if (!info) {
    if (error && error_size) std::snprintf(error, error_size, "null info");
    return -1;
  }
  std::memset(info, 0, sizeof(*info));
  int n = 0;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (err != cudaSuccess || n < 1) {
    if (error && error_size)
      std::snprintf(error, error_size, "cudaGetDeviceCount: %s", cudaGetErrorString(err));
    return -1;
  }
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    if (error && error_size)
      std::snprintf(error, error_size, "cudaGetDeviceProperties: %s", cudaGetErrorString(err));
    return -1;
  }
  std::snprintf(info->name, sizeof(info->name), "%s", prop.name);
  std::snprintf(info->architecture, sizeof(info->architecture), "sm_%d%d",
                prop.major, prop.minor);
  info->physical_memory = prop.totalGlobalMem;
  info->unified_memory = prop.unifiedAddressing ? 1 : 0;
  info->metal4 = 0;
  info->apple_gpu_family = 0;
  return 0;
}
