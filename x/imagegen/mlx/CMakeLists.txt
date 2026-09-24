include(FetchContent)

# Read MLX-C version from top-level file (shared with Dockerfile)
file(READ "${CMAKE_SOURCE_DIR}/MLX_C_VERSION" MLX_C_GIT_TAG)
string(STRIP "${MLX_C_GIT_TAG}" MLX_C_GIT_TAG)

# Read MLX version from top-level file
file(READ "${CMAKE_SOURCE_DIR}/MLX_VERSION" MLX_GIT_TAG)
string(STRIP "${MLX_GIT_TAG}" MLX_GIT_TAG)

set(MLX_C_BUILD_EXAMPLES OFF)

set(MLX_BUILD_GGUF OFF)
set(MLX_BUILD_SAFETENSORS ON)

function(set_target_output_directory _target)
    if(TARGET ${_target})
        set_target_properties(${_target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${OLLAMA_BUILD_DIR}
            LIBRARY_OUTPUT_DIRECTORY ${OLLAMA_BUILD_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${OLLAMA_BUILD_DIR}
        )
    endif()
endfunction()

# Check for Metal support (macOS only)
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    execute_process(
      COMMAND
        zsh "-c"
        "echo \"__METAL_VERSION__\" | xcrun -sdk macosx metal ${XCRUN_FLAGS} -E -x metal -P - | tail -1 | tr -d '\n'"
      OUTPUT_VARIABLE MLX_METAL_VERSION COMMAND_ERROR_IS_FATAL ANY)

    if(NOT MLX_METAL_VERSION)
        message(STATUS "`xcrun metal` error. Setting MLX_BUILD_METAL=OFF")
        set(MLX_BUILD_METAL OFF)
    endif()
else()
    # On Linux, disable Metal backend
    message(STATUS "Non-macOS platform detected. Setting MLX_BUILD_METAL=OFF")
    set(MLX_BUILD_METAL OFF)
endif()

# Map CMAKE_CUDA_ARCHITECTURES to MLX_CUDA_ARCHITECTURES if not explicitly set
if(NOT MLX_CUDA_ARCHITECTURES AND CMAKE_CUDA_ARCHITECTURES)
    set(MLX_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    message(STATUS "Using CMAKE_CUDA_ARCHITECTURES for MLX: ${MLX_CUDA_ARCHITECTURES}")
endif()

# Forward cuDNN environment variables to cmake variables so MLX's FindCUDNN.cmake
# can find them via HINTS ${CUDNN_INCLUDE_PATH} / ${CUDNN_LIBRARY_PATH}.
if(DEFINED ENV{CUDNN_INCLUDE_PATH} AND NOT CUDNN_INCLUDE_PATH)
    set(CUDNN_INCLUDE_PATH "$ENV{CUDNN_INCLUDE_PATH}" CACHE PATH "cuDNN include path")
    message(STATUS "Using CUDNN_INCLUDE_PATH from environment: ${CUDNN_INCLUDE_PATH}")
endif()
if(DEFINED ENV{CUDNN_LIBRARY_PATH} AND NOT CUDNN_LIBRARY_PATH)
    set(CUDNN_LIBRARY_PATH "$ENV{CUDNN_LIBRARY_PATH}" CACHE PATH "cuDNN library path")
    message(STATUS "Using CUDNN_LIBRARY_PATH from environment: ${CUDNN_LIBRARY_PATH}")
endif()

# Enable CUDA backend if CUDA architectures are specified and CUDA compiler is available
if(MLX_CUDA_ARCHITECTURES AND CMAKE_CUDA_COMPILER)
    set(MLX_BUILD_CUDA ON CACHE BOOL "Build CUDA backend for MLX" FORCE)
    message(STATUS "Enabling MLX CUDA backend with architectures: ${MLX_CUDA_ARCHITECTURES}")
elseif(MLX_CUDA_ARCHITECTURES)
    message(WARNING "MLX_CUDA_ARCHITECTURES specified but CUDA compiler not found, CUDA backend will be disabled")
endif()

# Allow local source overrides via environment variables
# Resolve to absolute paths so FetchContent doesn't break on relative dirs.
if(DEFINED ENV{OLLAMA_MLX_SOURCE})
    get_filename_component(_mlx_src "$ENV{OLLAMA_MLX_SOURCE}" ABSOLUTE BASE_DIR ${CMAKE_SOURCE_DIR})
    set(FETCHCONTENT_SOURCE_DIR_MLX "${_mlx_src}" CACHE PATH "" FORCE)
    message(STATUS "Using local MLX source: ${_mlx_src}")
endif()
if(DEFINED ENV{OLLAMA_MLX_C_SOURCE})
    get_filename_component(_mlx_c_src "$ENV{OLLAMA_MLX_C_SOURCE}" ABSOLUTE BASE_DIR ${CMAKE_SOURCE_DIR})
    set(FETCHCONTENT_SOURCE_DIR_MLX-C "${_mlx_c_src}" CACHE PATH "" FORCE)
    message(STATUS "Using local MLX-C source: ${_mlx_c_src}")
endif()

# Pre-declare mlx so our pinned version takes precedence over the one
# hardcoded in mlx-c's CMakeLists.txt (first FetchContent_Declare wins).
FetchContent_Declare(
    mlx
    GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
    GIT_TAG ${MLX_GIT_TAG}
)

FetchContent_Declare(
    mlx-c
    GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git"
    GIT_TAG ${MLX_C_GIT_TAG}
    # Carry patches until they land in ml-explore/mlx-c (qmm global_scale + gated_delta).
    PATCH_COMMAND ${CMAKE_COMMAND}
      -DPATCH_DIR=${CMAKE_SOURCE_DIR}/mlx/compat/mlx-c
      -DPATCH_LABEL=mlx/compat/mlx-c
      -P ${CMAKE_SOURCE_DIR}/mlx/compat/apply-git-patches.cmake
)
FetchContent_MakeAvailable(mlx-c)

# Sync vendored headers with fetched version
file(GLOB _mlx_c_hdrs "${mlx-c_SOURCE_DIR}/mlx/c/*.h")
file(COPY ${_mlx_c_hdrs} DESTINATION "${CMAKE_SOURCE_DIR}/x/mlxrunner/mlx/include/mlx/c/")

# Regenerate Go/C shim wrappers from the (possibly updated) headers.
# Skip during cross-compilation — the generated files are arch-independent.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR OR NOT APPLE)
    find_program(GO_EXECUTABLE go REQUIRED)
    message(STATUS "Regenerating MLX Go wrappers")

    # CGo (tree-sitter in x/mlxrunner/mlx/generator) needs a real SDK toolchain on macOS.
    # Clear CGO_CFLAGS/CGO_CXXFLAGS only — CC= breaks stdbool.h on Apple; on Windows CC
    # may contain spaces and breaks cgo parsing.
    set(_SAVE_CC "$ENV{CC}")
    set(_SAVE_SDKROOT "$ENV{SDKROOT}")
    set(_SAVE_CGO_CFLAGS "$ENV{CGO_CFLAGS}")
    set(_SAVE_CGO_CXXFLAGS "$ENV{CGO_CXXFLAGS}")
    set(_SAVE_GOFLAGS "$ENV{GOFLAGS}")

    set(_GO_GEN_ENVS GOFLAGS=-mod=mod CGO_CFLAGS= CGO_CXXFLAGS=)
    if(APPLE)
        execute_process(
            COMMAND xcrun --find clang
            OUTPUT_VARIABLE _MLX_GOGEN_CC
            OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY
        )
        execute_process(
            COMMAND xcrun --show-sdk-path
            OUTPUT_VARIABLE _MLX_GOGEN_SDK
            OUTPUT_STRIP_TRAILING_WHITESPACE
            COMMAND_ERROR_IS_FATAL ANY
        )
        list(APPEND _GO_GEN_ENVS "CC=${_MLX_GOGEN_CC}" "SDKROOT=${_MLX_GOGEN_SDK}")
    elseif(WIN32)
        list(APPEND _GO_GEN_ENVS CC=)
    endif()

    # WHY GOFLAGS=-mod=mod: incomplete vendor/ aborts configure (build_production_mac.sh).
    # WHY narrow paths: only mlxrunner/imagegen have //go:generate; ./x/... is unnecessary.
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env ${_GO_GEN_ENVS}
            ${GO_EXECUTABLE} generate ./x/mlxrunner/mlx ./x/imagegen/mlx
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMAND_ERROR_IS_FATAL ANY
    )

    set(ENV{CC} "${_SAVE_CC}")
    set(ENV{SDKROOT} "${_SAVE_SDKROOT}")
    set(ENV{CGO_CFLAGS} "${_SAVE_CGO_CFLAGS}")
    set(ENV{CGO_CXXFLAGS} "${_SAVE_CGO_CXXFLAGS}")
    set(ENV{GOFLAGS} "${_SAVE_GOFLAGS}")
else()
    message(STATUS "Skipping MLX Go wrapper generation (cross-compiling)")
endif()

# For local dev builds, override MLX_VERSION with git describe output
if(TARGET mlx_version AND DEFINED FETCHCONTENT_SOURCE_DIR_MLX)
    execute_process(
        COMMAND git describe --tags --first-parent --abbrev=7 --long --dirty --always
        WORKING_DIRECTORY ${mlx_SOURCE_DIR}
        OUTPUT_VARIABLE _mlx_git_version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _mlx_git_result
    )
    if(_mlx_git_result EQUAL 0 AND _mlx_git_version)
        # Strip leading "v" prefix for consistency
        string(REGEX REPLACE "^v" "" _mlx_git_version "${_mlx_git_version}")
        get_target_property(_mlx_defs mlx_version COMPILE_DEFINITIONS)
        list(FILTER _mlx_defs EXCLUDE REGEX "^MLX_VERSION=")
        set_target_properties(mlx_version PROPERTIES COMPILE_DEFINITIONS "${_mlx_defs}")
        target_compile_definitions(mlx_version PRIVATE "MLX_VERSION=\"${_mlx_git_version}\"")
        message(STATUS "MLX version (local dev): ${_mlx_git_version}")
    endif()
endif()

set_target_output_directory(mlx)
set_target_output_directory(mlxc)

# --- xgrammar (structured output for mlxrunner) ---
# Pin matches upstream Ollama tip. Override with OLLAMA_XGRAMMAR_SOURCE or
# FETCHCONTENT_SOURCE_DIR_XGRAMMAR for a sibling checkout.
set(XGRAMMAR_VERSION v0.2.5)
if(DEFINED ENV{OLLAMA_XGRAMMAR_SOURCE} AND NOT "$ENV{OLLAMA_XGRAMMAR_SOURCE}" STREQUAL "")
    get_filename_component(_xgrammar_src "$ENV{OLLAMA_XGRAMMAR_SOURCE}" ABSOLUTE BASE_DIR ${CMAKE_SOURCE_DIR})
    set(FETCHCONTENT_SOURCE_DIR_XGRAMMAR "${_xgrammar_src}" CACHE PATH "" FORCE)
    message(STATUS "Using local XGrammar source: ${_xgrammar_src}")
endif()
FetchContent_Declare(xgrammar
    GIT_REPOSITORY "https://github.com/mlc-ai/xgrammar.git"
    GIT_TAG ${XGRAMMAR_VERSION}
    GIT_SHALLOW TRUE
    GIT_SUBMODULES 3rdparty/dlpack
    # Do not add XGrammar's Python-oriented CMake project.
    SOURCE_SUBDIR cmake/ollama)
FetchContent_MakeAvailable(xgrammar)

file(GLOB_RECURSE XGRAMMAR_SOURCES CONFIGURE_DEPENDS "${xgrammar_SOURCE_DIR}/cpp/*.cc")
list(FILTER XGRAMMAR_SOURCES EXCLUDE REGEX "/cpp/tvm_ffi/.*\\.cc$")
add_library(xgrammar STATIC ${XGRAMMAR_SOURCES})
set_target_properties(xgrammar PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_include_directories(xgrammar PUBLIC "${xgrammar_SOURCE_DIR}/include")
target_include_directories(xgrammar SYSTEM PUBLIC
    "${xgrammar_SOURCE_DIR}/3rdparty/picojson"
    "${xgrammar_SOURCE_DIR}/3rdparty/dlpack/include")
target_compile_definitions(xgrammar PUBLIC
    XGRAMMAR_ENABLE_CPPTRACE=0
    XGRAMMAR_ENABLE_INTERNAL_CHECK=0)

add_library(ollama_xgrammar SHARED
    "${CMAKE_SOURCE_DIR}/x/mlxrunner/xgrammar/native/xgrammar.cpp")
target_include_directories(ollama_xgrammar PRIVATE
    "${CMAKE_SOURCE_DIR}/x/mlxrunner/xgrammar/native")
target_compile_definitions(ollama_xgrammar PRIVATE
    OLLAMA_XGRAMMAR_BUILD=1
    OLLAMA_XGRAMMAR_VERSION="${XGRAMMAR_VERSION}")
# Export only the ollama_xgrammar_* API; xgrammar's own C++ symbols stay hidden.
set_target_properties(ollama_xgrammar PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_link_libraries(ollama_xgrammar PRIVATE xgrammar Threads::Threads)
set_target_output_directory(ollama_xgrammar)

