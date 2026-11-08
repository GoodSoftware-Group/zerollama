#!/usr/bin/env bash
# Install TRELLIS.2 (+ optional TRELLIS text XL) for Ada sm_89 on astra (dual 4090).
#
# WHY not the Hugging Face Space / Comfy prebuilt wheels: those are sm_120 (Blackwell).
# A 4090 loads them and dies with "no kernel image is available".
#
# Lab only: separate venv, never on the serve LD_LIBRARY_PATH, never inside
# zerollama.service. Prefer CUDA_VISIBLE_DEVICES=1 when GPU 0 is serving.
#
# Usage:
#   TRELLIS_ROOT=/mnt/ssd2/zerollama/third_party/trellis \
#     CUDA_VISIBLE_DEVICES=1 ./scripts/mesh/install_trellis_astra.sh
#   … ./scripts/mesh/install_trellis_astra.sh --with-text
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: TRELLIS CUDA extensions are Linux-only (this host is $(uname -s))" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TRELLIS_ROOT="${TRELLIS_ROOT:-$HOME/.zerollama/third_party/trellis}"
VENV_DIR="${TRELLIS_VENV:-$TRELLIS_ROOT/.venv-trellis}"
VENV_TEXT_DIR="${TRELLIS_TEXT_VENV:-$TRELLIS_ROOT/.venv-trellis-text}"
SRC_DIR="$TRELLIS_ROOT/TRELLIS.2"
SRC_TEXT_DIR="$TRELLIS_ROOT/TRELLIS"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
WITH_TEXT=0
SKIP_FLASH=0
SKIP_NVDIFFREC=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-text) WITH_TEXT=1; shift ;;
    --skip-flash-attn) SKIP_FLASH=1; shift ;;
    --with-nvdiffrec) SKIP_NVDIFFREC=0; shift ;;
    -h|--help)
      echo "Usage: $0 [--with-text] [--skip-flash-attn] [--with-nvdiffrec]"
      echo ""
      echo "Env:"
      echo "  TRELLIS_ROOT              default: ~/.zerollama/third_party/trellis"
      echo "  TRELLIS_VENV              default: \$TRELLIS_ROOT/.venv-trellis"
      echo "  TRELLIS_TEXT_VENV         default: \$TRELLIS_ROOT/.venv-trellis-text"
      echo "  TORCH_CUDA_ARCH_LIST      default: 8.9 (Ada / RTX 4090)"
      echo "  CUDA_HOME                 toolkit whose nvcc can compile (prefer system; avoid broken shims)"
      echo "  CUDA_VISIBLE_DEVICES      prefer the free card (e.g. 1)"
      echo "  TRELLIS_TORCH_INDEX       default: cu124 (or cu130 if you already have matching wheels)"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
# Prefer a real toolkit. Do NOT prefer /mnt/ssd2/cuda-shim-* first — those have
# mismatched stubs and fail extension links. Unset CUDA_HOME to use PATH nvcc.
if [[ -z "${CUDA_HOME:-}" ]]; then
  for cand in /usr/local/cuda-12.4 /usr/local/cuda-12.6 /usr/local/cuda-13.0 /usr/local/cuda; do
    if [[ -x "$cand/bin/nvcc" ]]; then
      export CUDA_HOME="$cand"
      break
    fi
  done
fi
# Keep venv python ahead of /usr/bin after we prepend CUDA.
_CUDA_BIN=""
if [[ -n "${CUDA_HOME:-}" ]]; then
  _CUDA_BIN="$CUDA_HOME/bin"
  echo "CUDA_HOME=$CUDA_HOME"
else
  echo "note: CUDA_HOME unset — using nvcc from PATH ($(command -v nvcc || echo missing))"
fi

echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST (must be 8.9 on astra 4090 — never install sm_120 wheels)"
mkdir -p "$TRELLIS_ROOT"
# WHY TMPDIR on same FS as the venv: flash-attn / ninja wheel moves fail with
# "Invalid cross-device link" when /tmp is tmpfs and the venv lives on /mnt/ssd2.
export TMPDIR="${TMPDIR:-$TRELLIS_ROOT/tmp}"
mkdir -p "$TMPDIR"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$TRELLIS_ROOT/pip-cache}"
mkdir -p "$PIP_CACHE_DIR"
export HF_HOME="${HF_HOME:-$TRELLIS_ROOT/hf-cache}"
mkdir -p "$HF_HOME"

# Host compiler for recent torch headers (need C++20) + allow newer gcc with nvcc.
export CC="${CC:-gcc-15}"
export CXX="${CXX:-g++-15}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
export CXXFLAGS="${CXXFLAGS:--std=c++20}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--allow-unsupported-compiler -std=c++20}"

if [[ ! -d "$SRC_DIR/.git" ]]; then
  git clone --recursive -b main https://github.com/microsoft/TRELLIS.2.git "$SRC_DIR"
else
  git -C "$SRC_DIR" submodule update --init --recursive
fi

# Patch extension sources to C++20 (torch≥2.14 ATen requires it).
while IFS= read -r -d '' f; do
  if grep -q 'std=c++17' "$f" 2>/dev/null; then
    echo "patch c++20: $f"
    sed -i 's/std=c++17/std=c++20/g' "$f"
  fi
done < <(find "$SRC_DIR/o-voxel" "$TRELLIS_ROOT/extensions" -name 'setup.py' -print0 2>/dev/null || true)

PY="${TRELLIS_PYTHON:-python3.11}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python3
fi
"$PY" -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
# Venv first, then CUDA nvcc, then system — never put /usr/bin ahead of the venv.
export PATH="$VENV_DIR/bin${_CUDA_BIN:+:$_CUDA_BIN}:/usr/bin:/usr/local/bin:${PATH}"
hash -r

python -m pip install -U pip wheel packaging ninja
python -m pip install "setuptools>=70,<82"

TORCH_INDEX="${TRELLIS_TORCH_INDEX:-cu124}"
case "$TORCH_INDEX" in
  cu124)
    python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url "https://download.pytorch.org/whl/cu124"
    ;;
  cu130)
    # Working astra lab pin when system nvcc is 13.x and cu124 wheels refuse to build.
    python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/cu130"
    ;;
  *)
    echo "error: unknown TRELLIS_TORCH_INDEX=$TORCH_INDEX (use cu124 or cu130)" >&2
    exit 1
    ;;
esac
# BiRefNet (optional rembg) and misc pipeline deps.
python -m pip install einops

# Skip CUDA version mismatch checks when host nvcc ≠ torch.version.cuda (common on astra).
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
cat > "$SITE/sitecustomize.py" <<'PY'
try:
    import torch.utils.cpp_extension as _cpp_ext
    _cpp_ext._check_cuda_version = lambda *a, **k: None
except Exception:
    pass
PY

python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} arch_list_env={__import__('os').environ.get('TORCH_CUDA_ARCH_LIST')}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available in this venv — refuse install")
name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
print(f"gpu0={name} cc={cc[0]}.{cc[1]}")
if cc != (8, 9):
    print(f"warning: expected Ada sm_89; got {cc}. Continuing with TORCH_CUDA_ARCH_LIST as set.", flush=True)
PY

# Basic deps (from upstream setup.sh --basic), minus Gradio demo.
python -m pip install \
  imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
  trimesh "transformers>=4.45,<5.0" tensorboard pandas lpips zstandard \
  kornia timm pillow huggingface_hub
python -m pip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# Attention: prefer a wheel that matches this torch. Bare `pip install xformers`
# can upgrade torch and break the extension build.
if [[ "$SKIP_FLASH" -eq 0 ]]; then
  MAX_JOBS="${MAX_JOBS:-2}" NVCC_THREADS="${NVCC_THREADS:-1}" \
    python -m pip install flash-attn --no-build-isolation || {
      echo "warning: flash-attn build failed — installing matching xformers wheel" >&2
      python -m pip install xformers --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" || \
        python -m pip install xformers --no-deps || true
      export ATTN_BACKEND=xformers
    }
else
  python -m pip install xformers --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" || \
    python -m pip install xformers --no-deps || true
  export ATTN_BACKEND=xformers
fi
# Default sparse+dense attn backend that TRELLIS.2 sparse path accepts.
export ATTN_BACKEND="${ATTN_BACKEND:-xformers}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-$ATTN_BACKEND}"

EXT_ROOT="$TRELLIS_ROOT/extensions"
mkdir -p "$EXT_ROOT"

# Lab-only nvdiffrast (NVIDIA non-commercial). Needed for stock to_glb bake.
# Do not vendor into the zerollama tree. Service bake replacement is a later slice.
if [[ ! -d "$EXT_ROOT/nvdiffrast/.git" ]]; then
  git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git "$EXT_ROOT/nvdiffrast"
fi
python -m pip install "$EXT_ROOT/nvdiffrast" --no-build-isolation --force-reinstall --no-deps

# nvdiffrec is NOT required for o_voxel.postprocess.to_glb. Skip by default.
if [[ "$SKIP_NVDIFFREC" -eq 0 ]]; then
  if [[ ! -d "$EXT_ROOT/nvdiffrec/.git" ]]; then
    git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git "$EXT_ROOT/nvdiffrec"
  fi
  python -m pip install "$EXT_ROOT/nvdiffrec" --no-build-isolation --force-reinstall --no-deps || {
    echo "warning: nvdiffrec failed — continuing (GLB bake does not need it)" >&2
  }
fi

if [[ ! -d "$EXT_ROOT/CuMesh/.git" ]]; then
  git clone --recursive https://github.com/JeffreyXiang/CuMesh.git "$EXT_ROOT/CuMesh"
fi
# Re-apply C++20 after clone.
sed -i 's/std=c++17/std=c++20/g' "$EXT_ROOT/CuMesh/setup.py" 2>/dev/null || true
python -m pip install "$EXT_ROOT/CuMesh" --no-build-isolation --force-reinstall --no-deps

if [[ ! -d "$EXT_ROOT/FlexGEMM/.git" ]]; then
  git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git "$EXT_ROOT/FlexGEMM"
fi
sed -i 's/std=c++17/std=c++20/g' "$EXT_ROOT/FlexGEMM/setup.py" 2>/dev/null || true
python -m pip install "$EXT_ROOT/FlexGEMM" --no-build-isolation --force-reinstall --no-deps

# o-voxel ships inside the TRELLIS.2 checkout
sed -i 's/std=c++17/std=c++20/g' "$SRC_DIR/o-voxel/setup.py" 2>/dev/null || true
python -m pip install "$SRC_DIR/o-voxel" --no-build-isolation --force-reinstall --no-deps

echo "note: import trellis2 via PYTHONPATH=$SRC_DIR"

python - <<'PY'
import importlib
for m in ("cumesh", "flex_gemm", "o_voxel", "nvdiffrast"):
    importlib.import_module(m if m != "flex_gemm" else "flex_gemm")
    print(f"import ok: {m}")
print("TRELLIS.2 extension smoke: PASS")
PY

cat > "$TRELLIS_ROOT/env.sh" <<EOF
# Source before lab runs. Never export into zerollama serve.
export TRELLIS_ROOT="$TRELLIS_ROOT"
export TRELLIS_VENV="$VENV_DIR"
export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
export TMPDIR="\${TMPDIR:-$TRELLIS_ROOT/tmp}"
export HF_HOME="\${HF_HOME:-$TRELLIS_ROOT/hf-cache}"
export TRANSFORMERS_CACHE="\$HF_HOME"
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Sparse attention accepts xformers|flash_attn only (not sdpa).
export ATTN_BACKEND="\${ATTN_BACKEND:-xformers}"
export SPARSE_ATTN_BACKEND="\${SPARSE_ATTN_BACKEND:-\$ATTN_BACKEND}"
export OPENCV_IO_ENABLE_OPENEXR=1
# Prefer the free 4090 when GPU 0 holds chat / GLM.
export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-1}"
# Gated encoder / rembg remaps used by trellis_mesh_generate.py
export DINOV3_MODEL="\${DINOV3_MODEL:-camenduru/dinov3-vitl16-pretrain-lvd1689m}"
export RMBG_MODEL="\${RMBG_MODEL:-ZhengPeng7/BiRefNet}"
# Do not export an expired OAuth token from ~/.cache/huggingface/token.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
EOF

echo "TRELLIS.2 install complete → $VENV_DIR"
echo "source $TRELLIS_ROOT/env.sh"

if [[ "$WITH_TEXT" -eq 1 ]]; then
  echo "=== TRELLIS text XL (separate venv; different torch/spconv stack) ==="
  if [[ ! -d "$SRC_TEXT_DIR/.git" ]]; then
    git clone --recursive -b main https://github.com/microsoft/TRELLIS.git "$SRC_TEXT_DIR"
  else
    git -C "$SRC_TEXT_DIR" submodule update --init --recursive
  fi
  "$PY" -m venv "$VENV_TEXT_DIR"
  # shellcheck source=/dev/null
  source "$VENV_TEXT_DIR/bin/activate"
  export PATH="$VENV_TEXT_DIR/bin${_CUDA_BIN:+:$_CUDA_BIN}:/usr/bin:/usr/local/bin:${PATH}"
  hash -r
  python -m pip install -U pip wheel packaging ninja
  python -m pip install "setuptools>=70,<82"
  # Text XL stack tracks cu124 torch 2.5; keep it off the TRELLIS.2 venv.
  python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
  SITE_T="$(python -c 'import site; print(site.getsitepackages()[0])')"
  cat > "$SITE_T/sitecustomize.py" <<'PY'
try:
    import torch.utils.cpp_extension as _cpp_ext
    _cpp_ext._check_cuda_version = lambda *a, **k: None
except Exception:
    pass
PY
  export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
  export CC="${CC:-gcc-15}" CXX="${CXX:-g++-15}" CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
  export CXXFLAGS="${CXXFLAGS:--std=c++20}"
  export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--allow-unsupported-compiler -std=c++20}"
  cd "$SRC_TEXT_DIR"
  # Upstream setup often fails mid-way (flash-attn isolation, missing kaolin).
  # Drive the pieces we need with --no-build-isolation after a best-effort setup.sh.
  # shellcheck disable=SC1091
  . ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --nvdiffrast || {
    echo "warning: TRELLIS text setup.sh returned non-zero — finishing deps manually" >&2
  }
  python -m pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124 || true
  python -m pip install flash-attn --no-build-isolation || true
  python -m pip install spconv-cu124 || true
  python -m pip install "$EXT_ROOT/nvdiffrast" --no-build-isolation --force-reinstall --no-deps || true
  # kaolin wheel for torch 2.5.1+cu124 (required by FlexiCubes mesh extract)
  python -m pip install kaolin==0.17.0 \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html || {
    echo "warning: kaolin install failed — text mesh extract will not import" >&2
  }
  # transformers 5.x refuses torch 2.5; pin below 5.
  python -m pip install "transformers>=4.45,<5.0"
  # Optional CUDA extensions that setup.sh may have left half-built.
  if [[ -d "$EXT_ROOT/diffoctreerast" ]]; then
    python -m pip install "$EXT_ROOT/diffoctreerast" --no-build-isolation --force-reinstall --no-deps || true
  fi
  if [[ -d "$EXT_ROOT/mip-splatting/submodules/diff-gaussian-rasterization" ]]; then
    python -m pip install "$EXT_ROOT/mip-splatting/submodules/diff-gaussian-rasterization" \
      --no-build-isolation --force-reinstall --no-deps || true
  fi
  export PYTHONPATH="$SRC_TEXT_DIR${PYTHONPATH:+:$PYTHONPATH}"
  python - <<'PY' || echo "warning: text pipeline import failed — check $VENV_TEXT_DIR" >&2
from trellis.pipelines import TrellisTextTo3DPipeline
print("TRELLIS text pipeline import: OK")
PY
  cat >> "$TRELLIS_ROOT/env.sh" <<EOF
export TRELLIS_TEXT_VENV="$VENV_TEXT_DIR"
export TRELLIS_TEXT_SRC="$SRC_TEXT_DIR"
EOF
  echo "TRELLIS text XL install attempted → $VENV_TEXT_DIR"
fi

echo "Done. Weights download on first generate (HF cache under \$HF_HOME)."
echo "Repo scripts stay in: $REPO_ROOT/scripts/mesh/"
echo "DINOv3: gated facebook repo; wrapper defaults to camenduru mirror (same weights, DINOv3 license)."
echo "RMBG: pipeline pins gated briaai/RMBG-2.0; wrapper remaps to ZhengPeng7/BiRefNet (different model) or skips when the still already has alpha."
