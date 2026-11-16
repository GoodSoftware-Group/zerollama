# vim: filetype=dockerfile

ARG FLAVOR=${TARGETARCH}

ARG ROCMVERSION=7.2.1
ARG JETPACK5VERSION=r35.4.1
ARG JETPACK6VERSION=r36.4.0
ARG CMAKEVERSION=3.31.2
ARG NINJAVERSION=1.12.1
ARG VULKANVERSION=1.4.321.1

# Default empty stages for local MLX source overrides.
# Override with: docker build --build-context local-mlx=../mlx --build-context local-mlx-c=../mlx-c
FROM scratch AS local-mlx
FROM scratch AS local-mlx-c

FROM --platform=linux/amd64 rocm/dev-almalinux-8:${ROCMVERSION}-complete AS base-amd64
RUN dnf install -y yum-utils ccache gcc-toolset-11-gcc gcc-toolset-11-gcc-c++ gcc-toolset-11-binutils \
    && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:$PATH

FROM --platform=linux/arm64 almalinux:8 AS base-arm64
# install epel-release for ccache
RUN yum install -y yum-utils epel-release \
    && dnf install -y clang ccache git \
    && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo
ENV CC=clang CXX=clang++

FROM base-${TARGETARCH} AS base
ARG CMAKEVERSION
ARG NINJAVERSION
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1
RUN dnf install -y unzip \
    && curl -fsSL -o /tmp/ninja.zip https://github.com/ninja-build/ninja/releases/download/v${NINJAVERSION}/ninja-linux$([ "$(uname -m)" = "aarch64" ] && echo "-aarch64").zip \
    && unzip /tmp/ninja.zip -d /usr/local/bin \
    && rm /tmp/ninja.zip
ENV CMAKE_GENERATOR=Ninja
ENV LDFLAGS=-s

FROM base AS cpu
RUN dnf install -y gcc-toolset-11-gcc gcc-toolset-11-gcc-c++
ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:$PATH
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'CPU' \
        && cmake --build --preset 'CPU' -- -l $(nproc) \
        && cmake --install build --component CPU --strip

FROM base AS cuda-11
ARG CUDA11VERSION=11.8
RUN dnf install -y cuda-toolkit-${CUDA11VERSION//./-}
ENV PATH=/usr/local/cuda-11/bin:$PATH
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'CUDA 11' \
        && cmake --build --preset 'CUDA 11' -- -l $(nproc) \
        && cmake --install build --component CUDA --strip

FROM base AS cuda-12
ARG CUDA12VERSION=12.8
RUN dnf install -y cuda-toolkit-${CUDA12VERSION//./-}
ENV PATH=/usr/local/cuda-12/bin:$PATH
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'CUDA 12' \
        && cmake --build --preset 'CUDA 12' -- -l $(nproc) \
        && cmake --install build --component CUDA --strip


FROM base AS cuda-13
ARG CUDA13VERSION=13.0
RUN dnf install -y cuda-toolkit-${CUDA13VERSION//./-}
ENV PATH=/usr/local/cuda-13/bin:$PATH
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'CUDA 13' \
        && cmake --build --preset 'CUDA 13' -- -l $(nproc) \
        && cmake --install build --component CUDA --strip


FROM base AS rocm-7
ENV PATH=/opt/rocm/hcc/bin:/opt/rocm/hip/bin:/opt/rocm/bin:/opt/rocm/hcc/bin:$PATH
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'ROCm 7' \
        && cmake --build --preset 'ROCm 7' -- -l $(nproc) \
        && cmake --install build --component HIP --strip
RUN rm -f dist/lib/ollama/rocm/rocblas/library/*gfx90[06]*

FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK5VERSION} AS jetpack-5
ARG CMAKEVERSION
ARG NINJAVERSION
RUN apt-get update && apt-get install -y curl ccache unzip \
    && curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1 \
    && curl -fsSL -o /tmp/ninja.zip https://github.com/ninja-build/ninja/releases/download/v${NINJAVERSION}/ninja-linux-aarch64.zip \
    && unzip /tmp/ninja.zip -d /usr/local/bin \
    && rm /tmp/ninja.zip
ENV CMAKE_GENERATOR=Ninja
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'JetPack 5' \
        && cmake --build --preset 'JetPack 5' -- -l $(nproc) \
        && cmake --install build --component CUDA --strip

FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK6VERSION} AS jetpack-6
ARG CMAKEVERSION
ARG NINJAVERSION
RUN apt-get update && apt-get install -y curl ccache unzip \
    && curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1 \
    && curl -fsSL -o /tmp/ninja.zip https://github.com/ninja-build/ninja/releases/download/v${NINJAVERSION}/ninja-linux-aarch64.zip \
    && unzip /tmp/ninja.zip -d /usr/local/bin \
    && rm /tmp/ninja.zip
ENV CMAKE_GENERATOR=Ninja
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'JetPack 6' \
        && cmake --build --preset 'JetPack 6' -- -l $(nproc) \
        && cmake --install build --component CUDA --strip

FROM base AS vulkan
ARG VULKANVERSION
RUN ln -s /usr/bin/python3 /usr/bin/python \
    && wget https://sdk.lunarg.com/sdk/download/${VULKANVERSION}/linux/vulkansdk-linux-x86_64-${VULKANVERSION}.tar.xz -O /tmp/vulkansdk.tar.xz \
    && tar xvf /tmp/vulkansdk.tar.xz -C /tmp \
    && /tmp/${VULKANVERSION}/vulkansdk -j 8 vulkan-headers \
    && /tmp/${VULKANVERSION}/vulkansdk -j 8 shaderc \
    && cp -r /tmp/${VULKANVERSION}/x86_64/include/* /usr/local/include/ \
    && cp -r /tmp/${VULKANVERSION}/x86_64/lib/* /usr/local/lib \
    && cp -r /tmp/${VULKANVERSION}/x86_64/bin/* /usr/local/bin/ \
    && rm -rf /tmp/${VULKANVERSION} /tmp/vulkansdk.tar.xz
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
RUN --mount=type=cache,target=/root/.ccache \
    cmake --preset 'Vulkan' \
        && cmake --build --preset 'Vulkan' -- -l $(nproc) \
        && cmake --install build --component Vulkan --strip

FROM base AS mlx
ARG CUDA13VERSION=13.0
RUN dnf install -y cuda-toolkit-${CUDA13VERSION//./-} \
    && dnf install -y openblas-devel lapack-devel \
    && dnf install -y libcudnn9-cuda-13 libcudnn9-devel-cuda-13 \
    && dnf install -y libnccl libnccl-devel
ENV PATH=/usr/local/cuda-13/bin:$PATH
ENV BLAS_INCLUDE_DIRS=/usr/include/openblas
ENV LAPACK_INCLUDE_DIRS=/usr/include/openblas
ENV CGO_LDFLAGS="-L/usr/local/cuda-13/lib64 -L/usr/local/cuda-13/targets/x86_64-linux/lib/stubs"
WORKDIR /go/src/github.com/ollama/ollama
COPY CMakeLists.txt CMakePresets.json .
COPY ml/backend/ggml/ggml ml/backend/ggml/ggml
COPY x/imagegen/mlx x/imagegen/mlx
COPY go.mod go.sum .
COPY MLX_VERSION MLX_C_VERSION .
RUN curl -fsSL https://golang.org/dl/go$(awk '/^go/ { print $2 }' go.mod).linux-$(case $(uname -m) in x86_64) echo amd64 ;; aarch64) echo arm64 ;; esac).tar.gz | tar xz -C /usr/local
ENV PATH=/usr/local/go/bin:$PATH
RUN go mod download
RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=bind,from=local-mlx,target=/tmp/local-mlx \
    --mount=type=bind,from=local-mlx-c,target=/tmp/local-mlx-c \
    if [ -f /tmp/local-mlx/CMakeLists.txt ]; then \
        export OLLAMA_MLX_SOURCE=/tmp/local-mlx; \
    fi \
    && if [ -f /tmp/local-mlx-c/CMakeLists.txt ]; then \
        export OLLAMA_MLX_C_SOURCE=/tmp/local-mlx-c; \
    fi \
    && cmake --preset 'MLX CUDA 13' -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas \
        && cmake --build --preset 'MLX CUDA 13' -- -l $(nproc) \
        && cmake --install build --component MLX --strip

FROM base AS build
# cmd imports the training and runtime embeds, which link pkg-config python3-embed
# (3.10+). Alma's default python3 is 3.6 and has no embed .pc, so pin 3.11 and
# expose it under the name cgo asks for. Stay on this Alma base: the linux
# tarball from --target archive still links glibc 2.28.
RUN dnf install -y python3.11 python3.11-devel pkgconf-pkg-config gawk \
    && pc="$(find /usr -name 'python-3.11-embed.pc' | head -n 1)" \
    && test -n "$pc" \
    && mkdir -p /opt/pc \
    && cp "$pc" /opt/pc/python3-embed.pc
ENV PKG_CONFIG_PATH=/opt/pc
WORKDIR /go/src/github.com/ollama/ollama
COPY go.mod go.sum .
RUN curl --retry 3 -fsSL https://golang.org/dl/go$(awk '/^go/ { print $2 }' go.mod).linux-$(case $(uname -m) in x86_64) echo amd64 ;; aarch64) echo arm64 ;; esac).tar.gz -o /tmp/go.tar.gz \
    && tar xzf /tmp/go.tar.gz -C /usr/local \
    && rm /tmp/go.tar.gz
ENV PATH=/usr/local/go/bin:$PATH
RUN go mod download
COPY . .
ARG GOFLAGS="'-ldflags=-w -s'"
ENV CGO_ENABLED=1
ARG CGO_CFLAGS
ARG CGO_CXXFLAGS
ENV CGO_CFLAGS="${CGO_CFLAGS}"
ENV CGO_CXXFLAGS="${CGO_CXXFLAGS}"
# RUNPATH, not RPATH: direct deps (libpython) resolve next to the ggml plugins,
# and dlopen of those plugins does not pick up the Alma OpenSSL copy.
ENV CGO_LDFLAGS="-Wl,-rpath,$$ORIGIN/../lib/ollama -Wl,--enable-new-dtags"
RUN --mount=type=cache,target=/root/.cache/go-build \
    go build -trimpath -buildmode=pie -o /bin/zerollama .
# Ship libpython and the Alma libs it needs (OpenSSL 1.1, libffi) inside
# lib/ollama. Skip glibc and libstdc++ so the host's copies are used.
RUN set -eu \
    && mkdir -p /opt/pyruntime \
    && libs="$(ldd /bin/zerollama | awk '/=> \// {print $3}')" \
    && for lib in $libs; do \
         base="$(basename "$lib")"; \
         case "$base" in \
           libc.so*|libm.so*|libdl.so*|libpthread.so*|librt.so*|libresolv.so*|ld-linux*|libstdc++.so*|libgcc_s.so*) continue ;; \
         esac; \
         cp -L "$lib" "/opt/pyruntime/$base"; \
       done \
    && ls /opt/pyruntime/libpython3.11.so* >/dev/null

FROM --platform=linux/amd64 scratch AS amd64
# COPY --from=cuda-11 dist/lib/ollama/ /lib/ollama/
COPY --from=cuda-12 dist/lib/ollama /lib/ollama/
COPY --from=cuda-13 dist/lib/ollama /lib/ollama/
COPY --from=vulkan  dist/lib/ollama  /lib/ollama/
COPY --from=mlx     /go/src/github.com/ollama/ollama/dist/lib/ollama /lib/ollama/

FROM --platform=linux/arm64 scratch AS arm64
# COPY --from=cuda-11 dist/lib/ollama/ /lib/ollama/
COPY --from=cuda-12 dist/lib/ollama /lib/ollama/
COPY --from=cuda-13 dist/lib/ollama/ /lib/ollama/
COPY --from=jetpack-5 dist/lib/ollama/ /lib/ollama/
COPY --from=jetpack-6 dist/lib/ollama/ /lib/ollama/

FROM scratch AS rocm
COPY --from=rocm-7 dist/lib/ollama /lib/ollama

FROM ${FLAVOR} AS archive
COPY --from=cpu dist/lib/ollama /lib/ollama
COPY --from=build /opt/pyruntime/ /lib/ollama/
COPY --from=build /bin/zerollama /bin/zerollama

FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y ca-certificates libvulkan1 libopenblas0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY --from=archive /bin/zerollama /usr/bin/zerollama
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY --from=archive /lib/ollama /usr/lib/ollama
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all
ENV OLLAMA_HOST=0.0.0.0:11434
# libpython is in the image so the binary loads. training.py, the runtime tree,
# and llama-server are not, so leave the embeds off. Mount the repo and override
# these to turn them back on.
ENV ZEROLLAMA_GPU_JOBS=false
ENV ZEROLLAMA_RUNTIME_EMBED=0
EXPOSE 11434
ENTRYPOINT ["/usr/bin/zerollama"]
CMD ["serve"]
