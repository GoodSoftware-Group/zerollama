#!/usr/bin/env bash
# Remove debug fprintf lines added to mlx-c/array.cpp during imagegen debugging.
# These print noise to stderr on every array free/data access and should not be
# in production builds.  Safe to run after patch_mlx_c_array.sh; idempotent.
#
# WHY a separate script: patch_mlx_c_array.sh adds the two required production
# changes (detach + export); this script only removes the debug instrumentation
# that was added while diagnosing OOM crashes and is not needed otherwise.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARRAY="${ROOT}/build-mlx/_deps/mlx-c-src/mlx/c/array.cpp"

if [[ ! -f "$ARRAY" ]]; then
  echo "Run cmake configure first so ${ARRAY} exists" >&2
  exit 1
fi

python3 - "$ARRAY" <<'PY'
import pathlib, sys, re
path = pathlib.Path(sys.argv[1])
text = path.read_text()
orig = text

# ── Remove debug block in mlx_array_free ───────────────────────────────────────
# Original debug block prints nbytes for every free of an array > 1MB.
old_free_debug = """\
    if (arr.ctx) {
      auto* a = static_cast<mlx::core::array*>(arr.ctx);
      // Print size to understand what's being freed
      size_t sz = a->nbytes();
      if (sz > 1024*1024) {  // Only print for sizeable arrays (>1MB)
        fprintf(stderr, "[mlx_array_free] ctx=%p nbytes=%zu\\n", arr.ctx, sz);
      }
    }
    mlx_array_free_(arr);\
"""
new_free = "    mlx_array_free_(arr);"
if old_free_debug in text:
    text = text.replace(old_free_debug, new_free)
    print("Cleaned: removed debug fprintf in mlx_array_free")
else:
    print("Skip: mlx_array_free debug block not present (already clean or different)")

# ── Remove debug block in mlx_array_data_float32 ──────────────────────────────
# Original function body was 1 line; debug version added null-checks + fprintfs.
old_float32_debug = """\
  try {
    auto& a = mlx_array_get_(arr);
    // If the array has no data yet (eval not complete), force a blocking eval.
    if (!a.data_shared_ptr()) {
      fprintf(stderr, "[MLX] data null pre-eval, calling a.eval()\\n");
      a.eval();
    }
    if (!a.data_shared_ptr()) {
      fprintf(stderr, "[MLX] data_float32: data still null after eval!\\n");
      return nullptr;
    }
    auto* ptr = a.data<float>();
    if (!ptr) {
      fprintf(stderr, "[MLX] data_float32: data() returned null! buffer=%p\\n",
              (void*)a.data_shared_ptr()->buffer.raw_ptr());
    }
    return ptr;
  } catch (std::exception& e) {
    fprintf(stderr, "[MLX] data_float32 exception: %s\\n", e.what());
    mlx_error(e.what());
    return nullptr;
  }\
"""
new_float32 = """\
  try {
    return mlx_array_get_(arr).data<float>();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return nullptr;
  }\
"""
if old_float32_debug in text:
    text = text.replace(old_float32_debug, new_float32)
    print("Cleaned: removed debug fprintf in mlx_array_data_float32")
else:
    print("Skip: mlx_array_data_float32 debug block not present (already clean or different)")

if text != orig:
    path.write_text(text)
    print("Written:", path)
else:
    print("No changes needed.")
PY

echo "Rebuild: cmake --build build-mlx --target mlx --target mlxc --parallel"
echo "Install: sudo cp build-mlx/lib/ollama/mlx_cuda_v12/{libmlx.so,libmlxc.so} /usr/lib/ollama/mlx_cuda_v12/"
