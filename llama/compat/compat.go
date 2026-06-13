// Package compat links the Ollama GGUF compatibility shim into the in-process
// llamarunner CGO build (runner/llamarunner -> llama/llama.cpp).
//
// Why: llama/server CMake already applied this layer for llama-server subprocess
// loads. Mac default inference uses CGO directly; published blobs (qwen35moe M-RoPE
// sections, embedded vision/MTP tensors, etc.) need the same in-memory translation.
// Without this package, handle_qwen35moe never runs and loaders fail on metadata
// such as rope.dimension_sections expected length 4, got 3.
//
// Hook call sites live in vendored llama-model-loader.cpp and mtmd/clip.cpp
// (same as llama-cpp-hooks.patch). Disable at runtime: OLLAMA_LLAMA_CPP_COMPAT=0.
package compat

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/../llama.cpp/include
// #cgo CPPFLAGS: -I${SRCDIR}/../llama.cpp/src
// #cgo CPPFLAGS: -I${SRCDIR}/../llama.cpp/common
// #cgo CPPFLAGS: -I${SRCDIR}/../llama.cpp/vendor
// #cgo CPPFLAGS: -I${SRCDIR}/../llama.cpp/tools/mtmd
// #cgo CPPFLAGS: -I${SRCDIR}/../../ml/backend/ggml/ggml/include
import "C"
