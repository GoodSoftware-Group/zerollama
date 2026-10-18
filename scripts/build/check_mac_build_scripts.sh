#!/usr/bin/env bash
# Fail if build_zerollama_mac.sh (or peers) source/call scripts that are missing.
# WHY: fresh clones used to die on uma_toolkit.sh / gen_ggml_metal_embed.sh gaps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

missing=0
required=(
  scripts/build/uma_toolkit.sh
  scripts/build/gen_ggml_metal_embed.sh
  scripts/build/build_mlx_dylibs_mac.sh
  scripts/build/build_llama_server.sh
  scripts/runtime/mac_cgo_env.sh
  scripts/runtime/mac_host_deps.sh
  scripts/runtime/runtime_uv_venv.sh
  scripts/install_mac.sh
  scripts/vendor/ensure_llama_cpp_sibling.sh
  scripts/vendor/ensure_llama_cgo_vendor.sh
  scripts/vendor/ensure_llama_vendor_patches.sh
  scripts/vendor/stage_llama_kv_ext_for_vendor.sh
  scripts/phase/ensure_ggml_version_h.sh
  scripts/phase/phase15_runtime_kv_env.sh
)

for f in "${required[@]}"; do
  if [[ ! -f "${f}" ]]; then
    echo "MISSING ${f}" >&2
    missing=1
  else
    echo "OK ${f}"
  fi
done

# Parse bash "${ROOT}/scripts/..." and source "${ROOT}/scripts/..." from the Mac build script.
while IFS= read -r rel; do
  [[ -z "${rel}" ]] && continue
  if [[ ! -f "${rel}" ]]; then
    echo "MISSING (referenced by build_zerollama_mac.sh): ${rel}" >&2
    missing=1
  fi
done < <(
  sed -n 's|.*"\${ROOT}/\([^"]*\)".*|\1|p' scripts/build/build_zerollama_mac.sh \
    | grep '^scripts/' \
    | sort -u
)

if [[ "${missing}" -ne 0 ]]; then
  echo "error: Mac build script surface incomplete" >&2
  exit 1
fi
echo "PASS: Mac build scripts present"
