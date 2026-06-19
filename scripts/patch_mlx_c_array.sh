#!/usr/bin/env bash
# Patch mlx-c (mlx/c/array.cpp) with two additions needed for CUDA imagegen:
#
#  1. mlx_array_detach  — detach an array from the MLX computation graph before
#     mlx_array_free. WHY: freeing model weight arrays without detaching first
#     leaves dangling sibling/parent graph links, causing use-after-free crashes
#     during the transformer reload on tight 16GB hosts.
#
#  2. mlx_go_export_latents_bin_d2h — write GPU latents to a ZLAT binary via a
#     direct cudaMemcpy D2H, bypassing mlx::core::copy. WHY: after the denoise
#     loop mlx::core::copy faults when the allocator is in a post-checkpoint
#     state; the direct CUDA path avoids this reliably.
#
# Run after cmake configure (so the source is fetched), before the rebuild:
#   cmake --build build-mlx --target mlx --target mlxc --parallel
#
# Idempotent: checks for marker strings before patching.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARRAY="${ROOT}/build-mlx/_deps/mlx-c-src/mlx/c/array.cpp"

if [[ ! -f "$ARRAY" ]]; then
  echo "Run cmake configure first so ${ARRAY} exists" >&2
  exit 1
fi

python3 - "$ARRAY" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()

changed = False

# ── 1. mlx_array_detach ────────────────────────────────────────────────────────
# Insert after mlx_array_free, before mlx_array_new.
# WHY: upstream mlx-c has no detach API; Go weight cleanup calls it via dlsym
# (falls back to a no-op if symbol missing) to break graph links before freeing.

DETACH_MARKER = "mlx_array_detach"
if DETACH_MARKER not in text:
    old = """extern "C" mlx_array mlx_array_new(void) {"""
    new = """\
extern "C" int mlx_array_detach(mlx_array arr) {
  // Detach the array from the MLX computation graph and clear sibling links.
  // Call before mlx_array_free when releasing model weights after inference to
  // prevent dangling references in shared ArrayDesc nodes.
  try {
    if (arr.ctx) {
      static_cast<mlx::core::array*>(arr.ctx)->detach();
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_array mlx_array_new(void) {"""
    if old in text:
        text = text.replace(old, new)
        print("Patched: added mlx_array_detach")
        changed = True
    else:
        print("WARN: insertion point for mlx_array_detach not found — check mlx-c version")
else:
    print("Skip: mlx_array_detach already present")

# ── 2. mlx_go_export_latents_bin_d2h ──────────────────────────────────────────
# Appended at end of file. WHY: mlx::core::copy faults post-denoise on CUDA when
# the allocator pool is in a partially-freed state. cudaMemcpy D2H avoids the
# MLX copy path entirely and writes a simple ZLAT binary (header + float32 data)
# that decode_latents reads for CPU VAE decode.

EXPORT_MARKER = "mlx_go_export_latents_bin_d2h"
if EXPORT_MARKER not in text:
    text = text.rstrip() + """

// mlx_go_export_latents_bin_d2h writes GPU latents to a ZLAT binary file.
// WHY: mlx::core::copy faults after the denoise loop on CUDA 16GB hosts when
// the allocator pool is partially freed. This function bypasses mlx::core::copy
// entirely and uses direct cudaMemcpy D2H, then writes a ZLAT file that the
// decode_latents subprocess reads for CPU VAE decode.
#include <vector>
#ifndef _WIN32
#include <dlfcn.h>
#endif
#include "mlx/device.h"

static float mlx_bf16_bits_to_f32(uint16_t v) {
  uint32_t bits = static_cast<uint32_t>(v) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

static bool mlx_go_array_on_gpu(const mlx::core::array& a) {
  return mlx::core::default_device().type == mlx::core::Device::gpu;
}

#ifndef _WIN32
static int mlx_go_cuda_sync(void) {
  using cuda_sync_fn = int (*)();
  static cuda_sync_fn sync_fn = nullptr;
  static int loaded = 0;
  if (!loaded) {
    loaded = 1;
    void* h = dlopen("libcudart.so.12", RTLD_NOW);
    if (!h) h = dlopen("libcudart.so", RTLD_NOW);
    if (h) sync_fn = reinterpret_cast<cuda_sync_fn>(dlsym(h, "cudaDeviceSynchronize"));
  }
  if (!sync_fn) return -1;
  return sync_fn();
}

static int mlx_go_cuda_d2h(void* dst, const void* src, size_t nbytes) {
  // cudaMemcpyDeviceToHost == 1
  using cuda_memcpy_fn = int (*)(void*, const void*, size_t, int);
  static cuda_memcpy_fn memcpy_fn = nullptr;
  static int loaded = 0;
  if (!loaded) {
    loaded = 1;
    void* h = dlopen("libcudart.so.12", RTLD_NOW);
    if (!h) h = dlopen("libcudart.so", RTLD_NOW);
    if (h) memcpy_fn = reinterpret_cast<cuda_memcpy_fn>(dlsym(h, "cudaMemcpy"));
  }
  if (mlx_go_cuda_sync() != 0) return -1;
  if (!memcpy_fn) return -1;
  return memcpy_fn(dst, src, nbytes, 1);
}
#endif

extern "C" int mlx_go_export_latents_bin_d2h(const char* path, const mlx_array arr_in) {
  // ZLAT binary layout (little-endian):
  //   [4]  magic "ZLAT"
  //   [4]  ndim  (int32)
  //   [16] shape (4 x int32, unused dims = 0)
  //   [4]  count (int32 = total elements)
  //   [n*4] float32 data (D2H copy from GPU, bf16 converted on the fly)
  try {
    auto& a = mlx_array_get_(arr_in);
    if (!a.is_available()) { a.eval(); a.wait(); }
    if (mlx_go_array_on_gpu(a)) {
#ifndef _WIN32
      if (mlx_go_cuda_sync() != 0) return 1;
#endif
    }

    size_t ndim = a.ndim();
    if (ndim == 0 || ndim > 4) return 4;
    size_t n = a.size();
    if (n == 0) return 5;

    int32_t shape[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < ndim; i++) shape[i] = static_cast<int32_t>(a.shape(i));

    FILE* f = fopen(path, "wb");
    if (!f) return 6;
    fwrite("ZLAT", 1, 4, f);
    int32_t nd = static_cast<int32_t>(ndim);
    fwrite(&nd, sizeof(int32_t), 1, f);
    fwrite(shape, sizeof(int32_t), 4, f);
    int32_t count = static_cast<int32_t>(n);
    fwrite(&count, sizeof(int32_t), 1, f);

    const void* src = a.data<void>();
    size_t nbytes = a.nbytes();
    if (!src || nbytes == 0) { fclose(f); return 7; }

    const bool gpu_d2h = mlx_go_array_on_gpu(a);

    if (a.dtype() == mlx::core::float32) {
      std::vector<float> host(n);
      if (gpu_d2h) {
#ifndef _WIN32
        if (mlx_go_cuda_d2h(host.data(), src, nbytes) != 0) { fclose(f); return 8; }
#else
        fclose(f); return 8;
#endif
      } else {
        std::memcpy(host.data(), src, nbytes);
      }
      fwrite(host.data(), sizeof(float), n, f);
    } else if (a.dtype() == mlx::core::bfloat16) {
      std::vector<uint16_t> raw(n);
      if (gpu_d2h) {
#ifndef _WIN32
        if (mlx_go_cuda_d2h(raw.data(), src, nbytes) != 0) { fclose(f); return 9; }
#else
        fclose(f); return 9;
#endif
      } else {
        std::memcpy(raw.data(), src, nbytes);
      }
      // Convert bfloat16 → float32 on CPU before writing.
      // WHY: VAE decode subprocess expects float32; doing conversion here avoids
      // a second pass in the Go decode_latents reader.
      for (size_t i = 0; i < n; i++) {
        float v = mlx_bf16_bits_to_f32(raw[i]);
        fwrite(&v, sizeof(float), 1, f);
      }
    } else {
      fclose(f); return 10;
    }

    fclose(f);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 11;
  }
}
"""
    print("Patched: added mlx_go_export_latents_bin_d2h")
    changed = True
else:
    print("Skip: mlx_go_export_latents_bin_d2h already present")

if changed:
    path.write_text(text)
    print("Written:", path)
else:
    print("No changes needed.")
PY

echo "Rebuild: cmake --build build-mlx --target mlx --target mlxc --parallel"
echo "Install: sudo cp build-mlx/lib/ollama/mlx_cuda_v12/{libmlx.so,libmlxc.so} /usr/lib/ollama/mlx_cuda_v12/"
