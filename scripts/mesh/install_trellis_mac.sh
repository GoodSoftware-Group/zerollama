#!/usr/bin/env bash
# Install TRELLIS.2 on Darwin via shivampkumar/trellis-mac (Pedro mtl* Metal backends).
#
# Lab only: separate tree under ~/.zerollama/third_party/trellis-mac, never on the
# serve PYTHONPATH. Do not bind :11434 / :8081. Do not vendor the forks into git.
#
# Their setup.sh can print success after a Metal build failure. This script
# fails the install unless flex_gemm and the Metal o_voxel baker actually import.
#
# Usage:
#   ./scripts/mesh/install_trellis_mac.sh
#   TRELLIS_ROOT=~/.zerollama/third_party/trellis-mac ./scripts/mesh/install_trellis_mac.sh
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: Darwin arm64 only (this host is $(uname -s))" >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "error: Apple Silicon required (this host is $(uname -m))" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export TRELLIS_ROOT="${TRELLIS_ROOT:-$HOME/.zerollama/third_party/trellis-mac}"
TRELLIS_MAC_REPO="${TRELLIS_MAC_REPO:-https://github.com/shivampkumar/trellis-mac.git}"
# Optional: pin an already-known SHA. Empty → resolve main once and record it.
TRELLIS_MAC_COMMIT="${TRELLIS_MAC_COMMIT:-}"
# Known-good Metal dep pins from the M4 Max lab that produced trellis2-mps.glb.
# Override via TRELLIS_DEP_* env, or leave empty only when intentionally tracking HEAD.
TRELLIS_DEP_MTLBVH="${TRELLIS_DEP_MTLBVH:-fe6660bdb7de5a12ea00bd96df2e21bdcbdd40b1}"
TRELLIS_DEP_MTLDIFFRAST="${TRELLIS_DEP_MTLDIFFRAST:-c390a92aa296c5cf131f91253b8a11fe01ec11e4}"
TRELLIS_DEP_MTLMESH="${TRELLIS_DEP_MTLMESH:-5e6749356ba39b15489c99487f55840d49923da7}"
TRELLIS_DEP_MTLGEMM="${TRELLIS_DEP_MTLGEMM:-c2e5d3fa15152bc4aef65773cab1ded5a1855d38}"
TRELLIS_DEP_TRELLIS2_APPLE="${TRELLIS_DEP_TRELLIS2_APPLE:-698fb31113e2f79622c1f6df09cb2a03cf7da953}"
TRELLIS_DEP_TRELLIS2="${TRELLIS_DEP_TRELLIS2:-75fbf0183001ed9876c8dbb35de6b68552ee08bd}"

mkdir -p "$(dirname "$TRELLIS_ROOT")"
export HF_HOME="${HF_HOME:-$TRELLIS_ROOT/hf-cache}"
mkdir -p "$HF_HOME" "$TRELLIS_ROOT/lab/out"

echo "TRELLIS_ROOT=$TRELLIS_ROOT"
echo "HF_HOME=$HF_HOME"
echo "REPO_ROOT=$REPO_ROOT"

# Metal toolchain (required for mtldiffrast / mtlgemm metallibs).
if command -v xcodebuild >/dev/null 2>&1; then
  echo "Downloading Metal toolchain if missing (xcodebuild -downloadComponent MetalToolchain)..."
  xcodebuild -downloadComponent MetalToolchain || {
    echo "warning: MetalToolchain download failed or already present — continuing" >&2
  }
else
  echo "error: xcodebuild not found — install full Xcode" >&2
  exit 1
fi

if [[ ! -d "$TRELLIS_ROOT/.git" ]]; then
  echo "Cloning trellis-mac → $TRELLIS_ROOT"
  git clone "$TRELLIS_MAC_REPO" "$TRELLIS_ROOT"
fi

cd "$TRELLIS_ROOT"
git fetch --tags origin
if [[ -z "$TRELLIS_MAC_COMMIT" ]]; then
  TRELLIS_MAC_COMMIT="$(git rev-parse origin/main)"
fi
echo "Checking out trellis-mac @$TRELLIS_MAC_COMMIT"
git checkout --force "$TRELLIS_MAC_COMMIT"

# Never accept SKIP_METAL — KDTree bake is not the acceptance path.
unset SKIP_METAL
export SKIP_METAL=0
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Prefer a controlled install over blind setup.sh:
# - torch 2.15 ATen needs C++20 (their setup.py still says c++17)
# - o-voxel needs the Eigen submodule
# - setup.sh swallows Metal build failures; we verify imports below
echo "Installing Darwin TRELLIS.2 + Metal backends (controlled path)..."
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}"
export CXXFLAGS="${CXXFLAGS:--std=c++20}"

mkdir -p "$TRELLIS_ROOT/deps" "$TRELLIS_ROOT/hf-cache" "$TRELLIS_ROOT/lab/out"

clone_dep() {
  local url="$1" dir="$2" ref="${3:-}"
  if [[ ! -d "$TRELLIS_ROOT/deps/$dir/.git" ]]; then
    echo "Cloning $dir ..."
    # Full clone when pinning a SHA (shallow tip may not contain the pin).
    if [[ -n "$ref" ]]; then
      git clone "$url" "$TRELLIS_ROOT/deps/$dir"
    else
      git clone --depth 1 "$url" "$TRELLIS_ROOT/deps/$dir"
    fi
  else
    echo " $dir already cloned"
  fi
  if [[ -n "$ref" ]]; then
    echo "  checkout $dir @$ref"
    git -C "$TRELLIS_ROOT/deps/$dir" fetch --depth 1 origin "$ref" 2>/dev/null \
      || git -C "$TRELLIS_ROOT/deps/$dir" fetch origin "$ref" 2>/dev/null \
      || true
    git -C "$TRELLIS_ROOT/deps/$dir" checkout --force "$ref"
  fi
}

if [[ ! -d "$TRELLIS_ROOT/deps/utils3d/.git" ]]; then
  git clone https://github.com/EasternJournalist/utils3d.git "$TRELLIS_ROOT/deps/utils3d"
  git -C "$TRELLIS_ROOT/deps/utils3d" checkout 9a4eb15e4021b67b12c460c7057d642626897ec8
fi
clone_dep https://github.com/pedronaugusto/mtlbvh.git mtlbvh "$TRELLIS_DEP_MTLBVH"
clone_dep https://github.com/pedronaugusto/mtldiffrast.git mtldiffrast "$TRELLIS_DEP_MTLDIFFRAST"
clone_dep https://github.com/pedronaugusto/mtlmesh.git mtlmesh "$TRELLIS_DEP_MTLMESH"
clone_dep https://github.com/pedronaugusto/mtlgemm.git mtlgemm "$TRELLIS_DEP_MTLGEMM"
clone_dep https://github.com/pedronaugusto/trellis2-apple.git trellis2-apple "$TRELLIS_DEP_TRELLIS2_APPLE"
if [[ ! -d "$TRELLIS_ROOT/TRELLIS.2/.git" ]]; then
  git clone https://github.com/microsoft/TRELLIS.2.git "$TRELLIS_ROOT/TRELLIS.2"
fi
if [[ -n "$TRELLIS_DEP_TRELLIS2" ]]; then
  echo "checkout TRELLIS.2 @$TRELLIS_DEP_TRELLIS2"
  git -C "$TRELLIS_ROOT/TRELLIS.2" fetch origin "$TRELLIS_DEP_TRELLIS2" 2>/dev/null || true
  git -C "$TRELLIS_ROOT/TRELLIS.2" checkout --force "$TRELLIS_DEP_TRELLIS2"
fi

# Eigen submodule for o-voxel
if [[ ! -f "$TRELLIS_ROOT/deps/trellis2-apple/o-voxel/third_party/eigen/Eigen/Dense" ]]; then
  git -C "$TRELLIS_ROOT/deps/trellis2-apple" submodule update --init --recursive \
    || git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git \
         "$TRELLIS_ROOT/deps/trellis2-apple/o-voxel/third_party/eigen"
fi

# C++20 for torch≥2.14 ATen concepts
for pkg in mtlbvh mtldiffrast mtlmesh mtlgemm; do
  f="$TRELLIS_ROOT/deps/$pkg/setup.py"
  [[ -f "$f" ]] && sed -i.bak 's/-std=c++17/-std=c++20/g' "$f"
done
sed -i.bak 's/-std=c++17/-std=c++20/g' "$TRELLIS_ROOT/deps/trellis2-apple/o-voxel/setup.py"

if [[ ! -d "$TRELLIS_ROOT/.venv" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv "$TRELLIS_ROOT/.venv" --python python3.11
  else
    python3.11 -m venv "$TRELLIS_ROOT/.venv"
  fi
fi
# shellcheck disable=SC1091
source "$TRELLIS_ROOT/.venv/bin/activate"
export PATH="$TRELLIS_ROOT/.venv/bin:$PATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
export HF_HOME

if command -v uv >/dev/null 2>&1; then
  PIP=(uv pip install)
else
  PIP=(pip install)
fi
"${PIP[@]}" torch torchvision torchaudio transformers accelerate huggingface_hub \
  safetensors pillow numpy trimesh scipy scikit-image tqdm easydict kornia timm imageio \
  opencv-python-headless xatlas fast-simplification setuptools wheel pybind11
"${PIP[@]}" "$TRELLIS_ROOT/deps/utils3d"

# Metal packages — fail hard (no || continue)
"${PIP[@]}" --no-build-isolation "$TRELLIS_ROOT/deps/mtlbvh"
"${PIP[@]}" --no-build-isolation "$TRELLIS_ROOT/deps/mtldiffrast"
"${PIP[@]}" --no-build-isolation "$TRELLIS_ROOT/deps/mtlmesh"
"${PIP[@]}" --no-build-isolation "$TRELLIS_ROOT/deps/mtlgemm"
"${PIP[@]}" --no-build-isolation "$TRELLIS_ROOT/deps/trellis2-apple/o-voxel"

echo "Applying MPS compatibility patches..."
python3 "$TRELLIS_ROOT/patches/mps_compat.py"

# Import check with the same env preamble as generate.py.
python - <<'PY'
import os
import sys

# Match generate.py: try flex_gemm before falling back would be allowed —
# on this installer, fallback is a hard failure.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

root = os.environ.get("TRELLIS_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(root, "TRELLIS.2"))
sys.path.append(os.path.join(root, "stubs"))

errors = []
try:
    import flex_gemm  # noqa: F401
    print("ok: flex_gemm import")
    os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"
except Exception as e:
    errors.append(f"flex_gemm import failed: {e}")

try:
    import o_voxel.postprocess as pp
    backend = getattr(pp, "_BACKEND", None)
    has_dr = bool(getattr(pp, "_HAS_DR", False))
    has_fg = bool(getattr(pp, "_HAS_FLEX_GEMM", False))
    print(f"ok: o_voxel.postprocess _BACKEND={backend!r} _HAS_DR={has_dr} _HAS_FLEX_GEMM={has_fg}")
    if backend != "metal":
        errors.append(f"o_voxel.postprocess._BACKEND={backend!r} (want 'metal')")
    if not has_dr:
        errors.append("o_voxel.postprocess._HAS_DR is False (mtldiffrast missing)")
    if not has_fg:
        errors.append("o_voxel.postprocess._HAS_FLEX_GEMM is False (mtlgemm missing)")
except Exception as e:
    errors.append(f"o_voxel.postprocess import failed: {e}")

if errors:
    print("INSTALL FAILED — setup.sh may have printed success anyway:", file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)
print("Metal baker import checks passed")
PY

rev_parse() {
  local d="$1"
  if [[ -d "$d/.git" ]] || [[ -f "$d/.git" ]]; then
    git -C "$d" rev-parse HEAD
  else
    echo "missing"
  fi
}

TRELLIS_MAC_SHA="$(git -C "$TRELLIS_ROOT" rev-parse HEAD)"
MTLGEMM_SHA="$(rev_parse "$TRELLIS_ROOT/deps/mtlgemm")"
MTLMESH_SHA="$(rev_parse "$TRELLIS_ROOT/deps/mtlmesh")"
MTLBVH_SHA="$(rev_parse "$TRELLIS_ROOT/deps/mtlbvh")"
MTLDIFFRAST_SHA="$(rev_parse "$TRELLIS_ROOT/deps/mtldiffrast")"
TRELLIS2_APPLE_SHA="$(rev_parse "$TRELLIS_ROOT/deps/trellis2-apple")"
TRELLIS2_SHA="$(rev_parse "$TRELLIS_ROOT/TRELLIS.2")"

ENV_FILE="$TRELLIS_ROOT/env-mac.sh"
cat >"$ENV_FILE" <<EOF
# Generated by scripts/mesh/install_trellis_mac.sh — lab only, not committed.
# Source before running scripts/mesh/trellis_mesh_generate_mac.py
export TRELLIS_ROOT="$TRELLIS_ROOT"
export TRELLIS_VENV="$TRELLIS_ROOT/.venv"
export HF_HOME="\${HF_HOME:-$TRELLIS_ROOT/hf-cache}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
export SPARSE_CONV_BACKEND=flex_gemm
export DINOV3_MODEL="\${DINOV3_MODEL:-camenduru/dinov3-vitl16-pretrain-lvd1689m}"
# Prefer stub rembg when still already has alpha (wrapper does this).
unset HF_TOKEN
export PATH="\$TRELLIS_VENV/bin:\$PATH"

# Pins recorded at install time
export TRELLIS_MAC_COMMIT="$TRELLIS_MAC_SHA"
export TRELLIS_DEP_MTLGEMM="$MTLGEMM_SHA"
export TRELLIS_DEP_MTLMESH="$MTLMESH_SHA"
export TRELLIS_DEP_MTLBVH="$MTLBVH_SHA"
export TRELLIS_DEP_MTLDIFFRAST="$MTLDIFFRAST_SHA"
export TRELLIS_DEP_TRELLIS2_APPLE="$TRELLIS2_APPLE_SHA"
export TRELLIS_DEP_TRELLIS2="$TRELLIS2_SHA"
EOF

echo "Wrote $ENV_FILE"
echo "Pins:"
echo "  trellis-mac=$TRELLIS_MAC_SHA"
echo "  mtlgemm=$MTLGEMM_SHA"
echo "  mtlmesh=$MTLMESH_SHA"
echo "  mtlbvh=$MTLBVH_SHA"
echo "  mtldiffrast=$MTLDIFFRAST_SHA"
echo "  trellis2-apple=$TRELLIS2_APPLE_SHA"
echo "  TRELLIS.2=$TRELLIS2_SHA"
echo
echo "Next:"
echo "  source $ENV_FILE"
echo "  python $REPO_ROOT/scripts/mesh/make_connected_still.py --out /tmp/trellis-lab/still_mug_connected.png"
echo "  python $REPO_ROOT/scripts/mesh/trellis_mesh_generate_mac.py \\"
echo "    --image /tmp/trellis-lab/still_mug_connected.png \\"
echo "    --out $TRELLIS_ROOT/lab/out/trellis2-mps.glb"
