#!/usr/bin/env bash
# Patch MLX CUDA allocator for tight 16GB VRAM hosts (RTX 5080 class).
# - Use synchronous cudaMalloc instead of cudaMallocAsync (avoids pool reservation overhead)
# - Lower default memory_limit from 95% to 90% of total VRAM
#
# Run after MLX source fetch, before: cmake --build build-mlx --target mlx --target mlxc
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ALLOC="${ROOT}/build-mlx/_deps/mlx-src/mlx/backend/cuda/allocator.cpp"

if [[ ! -f "$ALLOC" ]]; then
  echo "Run cmake configure first so ${ALLOC} exists" >&2
  exit 1
fi

python3 - "$ALLOC" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()

old_async = """        if (mem_pools_[device]) { // supports memory pools
          auto err = cudaMallocAsync(&data, size, stream);"""

if old_async in text and "Use synchronous cudaMalloc" not in text:
    start = text.index("      } else {\n        cu::device(device).make_current();")
    end = text.index("      if (!data) {", start)
    replacement = """      } else {
        cu::device(device).make_current();
        // Use synchronous cudaMalloc instead of cudaMallocAsync. The async memory pool
        // reserves address space that counts against physical VRAM on 16GB GPUs.
        auto err = cudaMalloc(&data, size);
        if (err != cudaSuccess) {
          size_t pool_reserved = 0;
          if (mem_pools_[device]) {
            cudaMemPoolGetAttribute(mem_pools_[device],
                cudaMemPoolAttrReservedMemCurrent, &pool_reserved);
          }
          size_t free_mem = 0, total_mem = 0;
          cudaMemGetInfo(&free_mem, &total_mem);
          fprintf(stderr,
              "[allocator] cudaMalloc OOM: size=%zu active=%zu pool_reserved=%zu free=%zu total=%zu err=%s\\n",
              size, active_memory_, pool_reserved, free_mem, total_mem,
              cudaGetErrorString(err));
          std::ostringstream msg;
          msg << "cudaMalloc(&data, " << size << ") failed: "
              << cudaGetErrorString(err);
          throw std::runtime_error(msg.str());
        }
        // device=-2: freed with plain cudaFree, not async pool.
        buf = new CudaBuffer{data, size, -2};
        lock.lock();
        active_memory_ += buf->size;
        peak_memory_ = std::max(active_memory_, peak_memory_);
        return Buffer{buf};
      }
"""
    text = text[:start] + replacement + text[end:]
    print("Patched cudaMallocAsync -> cudaMalloc")

text = text.replace("memory_limit_ = total_memory_ * 0.95;",
                      "memory_limit_ = total_memory_ * 0.90;")
text = text.replace("acteve", "active")  # fix typo if present

# Disable buffer cache recycle — always cudaFree to avoid heap corruption.
old_recycle = """  if (get_cache_memory() < max_pool_size_ && buf->device != -2) {
    buffer_cache_.recycle_to_cache(buf);
  } else {
    free_cuda_buffer(buf);
  }"""
new_recycle = """  // Always return memory to the driver (see zerollama imagegen checkpoint frees).
  free_cuda_buffer(buf);"""
if old_recycle in text:
    text = text.replace(old_recycle, new_recycle)
    print("Patched allocator: disable buffer cache recycle")

path.write_text(text)
print("Patched", path)
PY

echo "Rebuild: cmake --build build-mlx --target mlx --target mlxc --parallel"
echo "Install: cp build-mlx/lib/ollama/libmlx.so build-mlx/lib/ollama/libmlxc.so /usr/lib/ollama/mlx_cuda_v12/"
