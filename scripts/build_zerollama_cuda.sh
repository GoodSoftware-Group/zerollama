#!/usr/bin/env bash
# Build zerollama on Linux CUDA (5080 / CT-class): Go binary + matching ggml CUDA
# runners + optional llama-server. Counterpart to scripts/build_zerollama_mac.sh.
#
# WHY not plain `go build`: the daemon dlopens /usr/lib/ollama/cuda_v*/libggml-cuda.so.
# Rebuilding only the Go binary against newer ggml headers (device_id props, etc.)
# while leaving stale system libs causes EnumerateGPUs panic / SIGSEGV on load.
#
# Default toolchain (driver 610+ / RTX 5080): CUDA 13.3 → OLLAMA_RUNNER_DIR=cuda_v13.
# Override for older hosts: CUDA_PRESET="CUDA 12" RUNNER_DIR=cuda_v12 BUILD_DIR=build-cuda12
#
# Usage:
#   ./scripts/build_zerollama_cuda.sh
#   BUILD_CUDA=0 ./scripts/build_zerollama_cuda.sh          # Go only (fast)
#   BUILD_CUDA=1 ./scripts/build_zerollama_cuda.sh          # force ggml CUDA rebuild
#   BUILD_LLAMA_SERVER=auto|0|1                            # default auto
#   INSTALL=1 ./scripts/build_zerollama_cuda.sh             # install to /usr/bin + /usr/lib/ollama
#   CUDA_ARCHS=120-real ./scripts/build_zerollama_cuda.sh   # 5080 native (default)
#   ./scripts/build_zerollama_cuda.sh /path/to/output/binary
#
# After install, serve with OLLAMA_LLM_LIBRARY=cuda_v13 and /root/nvidia-host first
# (see scripts/serve_gpu_example.sh / ~/bin/serve.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/zerollama}"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build-cuda13}"
CUDA_PRESET="${CUDA_PRESET:-CUDA 13}"
RUNNER_DIR="${RUNNER_DIR:-cuda_v13}"
INSTALL_ROOT="${INSTALL_ROOT:-/usr}"
# 5080 Blackwell: native SM120. Add e.g. 89 for multi-arch fleets.
CUDA_ARCHS="${CUDA_ARCHS:-120-real}"
BUILD_CUDA="${BUILD_CUDA:-auto}"
BUILD_LLAMA_SERVER="${BUILD_LLAMA_SERVER:-auto}"
INSTALL="${INSTALL:-0}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"

if [[ -z "${VERSION:-}" ]]; then
	if git -C "${ROOT}" describe --tags --first-parent --abbrev=7 --long --dirty --always &>/dev/null; then
		VERSION="$(git -C "${ROOT}" describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e 's/^v//')"
	else
		VERSION="$(grep 'var Version string' "${ROOT}/version/version.go" 2>/dev/null | sed -n 's/.*"\(.*\)".*/\1/p' || echo 0.0.0)"
	fi
fi

_die() { echo "error: $*" >&2; exit 1; }

_setup_cuda_path() {
	# Prefer default /usr/local/cuda (13.3 on this host after driver 610 upgrade).
	local cand
	for cand in \
		"${CUDAToolkit_ROOT:-}" \
		/usr/local/cuda \
		/usr/local/cuda-13.3 \
		/usr/local/cuda-13 \
		/usr/local/cuda-12.8 \
		/usr/local/cuda-12; do
		[[ -n "${cand}" ]] || continue
		if [[ -x "${cand}/bin/nvcc" ]]; then
			export PATH="${cand}/bin:${PATH}"
			export CUDACXX="${CUDACXX:-${cand}/bin/nvcc}"
			export CUDAToolkit_ROOT="${cand}"
			echo ">>> CUDA toolkit: ${cand} ($("${cand}/bin/nvcc" --version | sed -n 's/^.*release //p' | head -1))" >&2
			return 0
		fi
	done
	_die "nvcc not found (need CUDA 13.3 toolkit at /usr/local/cuda for cuda_v13)"
}

_toolkit_libdir() {
	local root="${CUDAToolkit_ROOT:-/usr/local/cuda}"
	if [[ -d "${root}/targets/x86_64-linux/lib" ]]; then
		echo "${root}/targets/x86_64-linux/lib"
	elif [[ -d "${root}/lib64" ]]; then
		echo "${root}/lib64"
	else
		echo "${root}/lib"
	fi
}

_ensure_httplib() {
	if [[ -d "${ROOT}/llama/llama.cpp/vendor/cpp-httplib" ]]; then
		return 0
	fi
	local sibling="${LLAMA_CPP_ROOT:-${ROOT}/../llama.cpp}/vendor/cpp-httplib"
	if [[ -d "${sibling}" ]]; then
		echo ">>> rsync cpp-httplib from ${sibling}" >&2
		rsync -a "${sibling}/" "${ROOT}/llama/llama.cpp/vendor/cpp-httplib/"
		return 0
	fi
	echo ">>> warn: missing llama/llama.cpp/vendor/cpp-httplib (some CGO paths may fail)" >&2
}

_setup_cgo() {
	export CGO_ENABLED=1
	export CGO_CFLAGS_ALLOW="${CGO_CFLAGS_ALLOW:--mfma|-mavx2|-O3}"
	if [[ -f "${ROOT}/scripts/training_embed_build_env.sh" ]] && pkg-config --exists python-3.11-embed 2>/dev/null; then
		# shellcheck source=scripts/training_embed_build_env.sh disable=SC1091
		source "${ROOT}/scripts/training_embed_build_env.sh" 3.11
	fi
}

_built_cuda_so() {
	# cmake may place the plugin at lib/ollama/libggml-cuda.so or .../cuda_v12/
	local f
	for f in \
		"${BUILD_DIR}/lib/ollama/${RUNNER_DIR}/libggml-cuda.so" \
		"${BUILD_DIR}/lib/ollama/libggml-cuda.so"; do
		if [[ -f "${f}" ]]; then
			echo "${f}"
			return 0
		fi
	done
	return 1
}

_installed_cuda_so() {
	echo "${INSTALL_ROOT}/lib/ollama/${RUNNER_DIR}/libggml-cuda.so"
}

_ggml_sources_newer_than() {
	local ref="$1"
	[[ -f "${ref}" ]] || return 0
	# Header / CUDA source churn that changes runner ABI vs Go CGO.
	local newest
	newest="$(find \
		"${ROOT}/ml/backend/ggml/ggml/include" \
		"${ROOT}/ml/backend/ggml/ggml/src/ggml-cuda" \
		"${ROOT}/llama/llama.go" \
		-type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' -o -name 'llama.go' \) \
		-newer "${ref}" 2>/dev/null | head -1 || true)"
	[[ -n "${newest}" ]]
}

_should_build_cuda() {
	case "${BUILD_CUDA}" in
	0) return 1 ;;
	1) return 0 ;;
	auto)
		# INSTALL=1 always refreshes runners with the binary (avoids the May-libs skew).
		if [[ "${INSTALL}" == "1" ]]; then
			echo ">>> BUILD_CUDA=auto: INSTALL=1 → rebuild runners with binary" >&2
			return 0
		fi
		local built installed
		built="$(_built_cuda_so 2>/dev/null || true)"
		installed="$(_installed_cuda_so)"
		if [[ -z "${built}" ]] || [[ ! -f "${built}" ]]; then
			echo ">>> BUILD_CUDA=auto: no built libggml-cuda.so under ${BUILD_DIR}" >&2
			return 0
		fi
		if [[ ! -f "${installed}" ]]; then
			echo ">>> BUILD_CUDA=auto: missing installed ${installed}" >&2
			return 0
		fi
		if _ggml_sources_newer_than "${installed}"; then
			echo ">>> BUILD_CUDA=auto: ggml/CUDA sources newer than ${installed}" >&2
			return 0
		fi
		echo ">>> BUILD_CUDA=auto: skip CUDA libs (up to date; INSTALL=1 to refresh)" >&2
		return 1
		;;
	*)
		_die "BUILD_CUDA must be 0, 1, or auto (got ${BUILD_CUDA})"
		;;
	esac
}

_build_cuda_libs() {
	_setup_cuda_path
	echo ">>> cmake configure (${CUDA_PRESET}, arches=${CUDA_ARCHS}) → ${BUILD_DIR}" >&2
	cmake -B "${BUILD_DIR}" --preset "${CUDA_PRESET}" \
		"-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"

	# CPU variants used on modern Intel hosts + portable fallbacks.
	local targets=(
		ggml-base
		ggml-cuda
		ggml-cpu-icelake
		ggml-cpu-skylakex
		ggml-cpu-haswell
		ggml-cpu-x64
		ggml-cpu-sse42
	)
	echo ">>> cmake build: ${targets[*]} (-j${JOBS})" >&2
	cmake --build "${BUILD_DIR}" --target "${targets[@]}" --parallel "${JOBS}"

	local so
	so="$(_built_cuda_so)" || _die "build finished but libggml-cuda.so not found under ${BUILD_DIR}/lib/ollama"
	echo ">>> built ${so}" >&2
}

_install_cuda_libs() {
	local so src_base dst_lib dst_cuda
	so="$(_built_cuda_so)" || _die "nothing to install — run with BUILD_CUDA=1 first"
	src_base="$(dirname "${so}")"
	# If so lives in .../cuda_v12/, base libs are one level up.
	if [[ "$(basename "${src_base}")" == "${RUNNER_DIR}" ]]; then
		src_base="$(dirname "${src_base}")"
	fi
	dst_lib="${INSTALL_ROOT}/lib/ollama"
	dst_cuda="${dst_lib}/${RUNNER_DIR}"
	mkdir -p "${dst_lib}" "${dst_cuda}"

	echo ">>> install ggml base/cpu → ${dst_lib}" >&2
	cp -a "${src_base}"/libggml-base.so* "${dst_lib}/"
	cp -a "${src_base}"/libggml-cpu-*.so "${dst_lib}/" 2>/dev/null || true

	echo ">>> install libggml-cuda.so → ${dst_cuda}/" >&2
	cp -a "${so}" "${dst_cuda}/libggml-cuda.so"

	# Refresh cudart/cublas from the active toolkit (13.3 math libs on driver 610).
	local tlib
	tlib="$(_toolkit_libdir)"
	if [[ -d "${tlib}" ]]; then
		echo ">>> sync cudart/cublas from ${tlib} → ${dst_cuda}/" >&2
		# Copy SONAME libs + versioned files the plugin links against.
		shopt -s nullglob
		local f
		for f in \
			"${tlib}"/libcudart.so* \
			"${tlib}"/libcublas.so* \
			"${tlib}"/libcublasLt.so*; do
			[[ -e "${f}" ]] || continue
			cp -a "${f}" "${dst_cuda}/"
		done
		shopt -u nullglob
	fi

	# Drop a stamp so operators can see which commit/toolkit built the runners.
	{
		echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
		echo "version=${VERSION}"
		echo "preset=${CUDA_PRESET}"
		echo "archs=${CUDA_ARCHS}"
		echo "runner=${RUNNER_DIR}"
		echo "toolkit=${CUDAToolkit_ROOT:-}"
		"${CUDACXX:-nvcc}" --version 2>/dev/null | sed -n 's/^.*release /nvcc_release=/p' | head -1 || true
		git -C "${ROOT}" rev-parse HEAD 2>/dev/null | sed 's/^/git=/' || true
	} >"${dst_cuda}/.zerollama-cuda-build"
	echo ">>> installed runners for OLLAMA_LLM_LIBRARY=${RUNNER_DIR}" >&2
}

_should_build_llama_server() {
	case "${BUILD_LLAMA_SERVER}" in
	0) return 1 ;;
	1) return 0 ;;
	auto)
		local bin="${LLAMA_SERVER_BIN:-}"
		if [[ -z "${bin}" ]]; then
			local pin sibling vendor
			pin="$(grep '^FETCH_HEAD=' "${ROOT}/Makefile.sync" 2>/dev/null | cut -d= -f2 || echo c84b3020)"
			vendor="${ROOT}/vendor/llama-cpp-${pin}/build/bin/llama-server"
			sibling="${ROOT}/../llama.cpp/build/bin/llama-server"
			intree="${ROOT}/llama/llama.cpp/build/bin/llama-server"
			for bin in "${vendor}" "${intree}" "${sibling}"; do
				if [[ -x "${bin}" ]]; then
					echo ">>> BUILD_LLAMA_SERVER=auto: found ${bin}" >&2
					return 1
				fi
			done
			echo ">>> BUILD_LLAMA_SERVER=auto: no llama-server on disk" >&2
			return 0
		fi
		[[ -x "${bin}" ]] && return 1
		echo ">>> BUILD_LLAMA_SERVER=auto: missing ${bin}" >&2
		return 0
		;;
	*)
		_die "BUILD_LLAMA_SERVER must be 0, 1, or auto (got ${BUILD_LLAMA_SERVER})"
		;;
	esac
}

_build_llama_server() {
	# Prefer scripts/build/ (current layout); fall back to legacy repo-root path.
	local script="${ROOT}/scripts/build/build_llama_server.sh"
	if [[ ! -x "${script}" ]]; then
		script="${ROOT}/scripts/build_llama_server.sh"
	fi
	[[ -x "${script}" ]] || _die "missing build_llama_server.sh under scripts/build/ or scripts/"
	echo ">>> building llama-server (${script})" >&2
	"${script}"
}

_build_go() {
	_ensure_httplib
	_setup_cgo
	cd "${ROOT}"
	echo ">>> go build → ${OUT} (version ${VERSION})" >&2
	GOFLAGS=-mod=mod go build -ldflags="-s -w -X=github.com/ollama/ollama/version.Version=${VERSION}" -o "${OUT}" .
	echo ">>> wrote ${OUT}" >&2
}

_install_binary() {
	local dest="${INSTALL_ROOT}/bin/zerollama"
	mkdir -p "$(dirname "${dest}")"
	cp -f "${OUT}" "${dest}"
	chmod 755 "${dest}"
	echo ">>> installed ${dest}" >&2
}

main() {
	cd "${ROOT}"
	echo ">>> build_zerollama_cuda (BUILD_CUDA=${BUILD_CUDA} BUILD_LLAMA_SERVER=${BUILD_LLAMA_SERVER} INSTALL=${INSTALL})" >&2

	if _should_build_cuda; then
		_build_cuda_libs
	fi

	if _should_build_llama_server; then
		_build_llama_server
	fi

	_build_go

	if [[ "${INSTALL}" == "1" ]]; then
		if [[ ! -w "${INSTALL_ROOT}/bin" ]] && [[ "$(id -u)" -ne 0 ]]; then
			_die "INSTALL=1 needs write access to ${INSTALL_ROOT} (re-run as root or set INSTALL_ROOT=...)"
		fi
		_install_binary
		# Always install libs when INSTALL=1 if we have a built so (even if BUILD_CUDA skipped).
		if _built_cuda_so >/dev/null 2>&1; then
			_install_cuda_libs
		elif [[ -f "$(_installed_cuda_so)" ]]; then
			echo ">>> INSTALL=1: keeping existing $(_installed_cuda_so)" >&2
		else
			echo ">>> warn: INSTALL=1 but no CUDA libs built/installed — set BUILD_CUDA=1" >&2
		fi
	fi

	echo ">>> done" >&2
	echo ">>>   binary: ${OUT}" >&2
	echo ">>>   serve:  OLLAMA_LLM_LIBRARY=${RUNNER_DIR} LD_LIBRARY_PATH=/root/nvidia-host:${INSTALL_ROOT}/lib/ollama:${INSTALL_ROOT}/lib/ollama/${RUNNER_DIR}:\$LD_LIBRARY_PATH" >&2
	if [[ "${INSTALL}" != "1" ]]; then
		echo ">>>   install: INSTALL=1 $0" >&2
	fi
}

main "$@"
