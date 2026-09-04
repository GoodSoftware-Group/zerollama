#!/usr/bin/env bash
# Patch MLX CUDA allocator for tight 16GB VRAM hosts (RTX 5080 class).
# - Force synchronous cudaMalloc / cudaFree (async pools reserve VA that counts as VRAM)
# - Lower default memory_limit from 95% to 90% of total VRAM
# - Disable buffer-cache recycle (avoids heap corruption after imagegen checkpoint frees)
#
# Compatible with MLX tip allocator.cpp (CHECK_CUDA_ERROR + mem_pools_ style).
#
# Run after MLX source fetch, before: cmake --build build-mlx --target mlx --target mlxc
#
# Paths (first existing wins):
#   1) $OLLAMA_MLX_ALLOCATOR
#   2) build-mlx/_deps/mlx-src/... (FetchContent)
#   3) $OLLAMA_MLX_SOURCE/mlx/backend/cuda/allocator.cpp
#   4) ../mlx/mlx/backend/cuda/allocator.cpp (sibling checkout)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_patch_mlx_cuda_vram.py"

if [[ -n "${OLLAMA_MLX_ALLOCATOR:-}" ]]; then
  ALLOC="${OLLAMA_MLX_ALLOCATOR}"
elif [[ -f "${ROOT}/build-mlx/_deps/mlx-src/mlx/backend/cuda/allocator.cpp" ]]; then
  ALLOC="${ROOT}/build-mlx/_deps/mlx-src/mlx/backend/cuda/allocator.cpp"
elif [[ -n "${OLLAMA_MLX_SOURCE:-}" && -f "${OLLAMA_MLX_SOURCE}/mlx/backend/cuda/allocator.cpp" ]]; then
  ALLOC="${OLLAMA_MLX_SOURCE}/mlx/backend/cuda/allocator.cpp"
elif [[ -f "${ROOT}/../mlx/mlx/backend/cuda/allocator.cpp" ]]; then
  ALLOC="${ROOT}/../mlx/mlx/backend/cuda/allocator.cpp"
else
  echo "MLX allocator.cpp not found. Run cmake configure, or set OLLAMA_MLX_SOURCE / OLLAMA_MLX_ALLOCATOR." >&2
  exit 1
fi

if [[ ! -f "$ALLOC" ]]; then
  echo "Missing allocator: ${ALLOC}" >&2
  exit 1
fi

python3 "$HELPER" "$ALLOC"

echo "Rebuild: cmake --build build-mlx --target mlx --target mlxc --parallel"
echo "Install: cp build-mlx/lib/ollama/{libmlx,libmlxc}.so /usr/lib/ollama/mlx_cuda_v12/  # or platform path"
